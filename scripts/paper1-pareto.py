#!/usr/bin/env python3
"""Paper 1 B4 analysis — locks the fair-condition numbers.

Reads the (git-ignored) benchmark DB and reports, fully offline, stdlib only:
  - per-agent summary (gross reward, score, accuracy, abstention, cost vector)
  - product-order Pareto frontier over the 2-vector cost (tool/Q, api$/Q)
  - £/point scalarisation sweep (where bayesian_inferred stays undominated)
  - per-category duality table (bayesian_inferred vs greedy_inferred)
  - mix-reweighting crossover (VOI vs greedy under hypothetical category mixes)
  - price-of-inference (oracle vs inferred) — the dominant lever
  - paired-seed bootstrap CIs for the key contrasts

Framing (see papers/RESULTS.md header): per-category is the headline; the
aggregate is only its mix-weighted projection. Cost-efficiency, not parity.

Usage: python3 scripts/paper1-pareto.py [path/to/benchmark.db]
"""
import sqlite3, json, sys, os, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "..", "apps", "julia", "qa_benchmark", "results", "benchmark.db")
QPS = 50  # questions per seed

FAIR = ["bayesian_inferred", "greedy_inferred", "no_voi_inferred",
        "no_learning_inferred", "single_best", "random", "all_tools",
        "claude-haiku-4-5-20251001", "llama3.1"]
ORACLE = ["bayesian", "ablation_greedy", "ablation_no_voi",
          "ablation_no_learning", "ablation_no_abstain"]
CATS = ["factual", "numerical", "recent_events", "misconceptions", "reasoning"]
CAT_N = {"factual": 15, "numerical": 10, "recent_events": 8,
         "misconceptions": 7, "reasoning": 10}  # actual benchmark mix /50
TOOLNAME = {1: "web", 2: "KB", 3: "calc", 4: "llm"}

con = sqlite3.connect(DB); con.row_factory = sqlite3.Row


def run_rows(agent):
    return con.execute(
        "SELECT seed, total_score, total_reward, total_tool_cost, "
        "total_api_cost_usd FROM runs WHERE agent=? ORDER BY seed", (agent,)
    ).fetchall()


def q_rows(agent):
    return con.execute(
        "SELECT q.category AS cat, q.tools_queried AS tq, "
        "q.submitted AS sub, q.was_correct AS ok, q.reward AS reward, "
        "q.tool_cost AS cost FROM questions q JOIN runs r ON q.run_id=r.id "
        "WHERE r.agent=? ", (agent,)).fetchall()


def summarize(agent):
    rr = run_rows(agent)
    if not rr:
        return None
    n = len(rr)
    reward = st.mean(r["total_reward"] for r in rr)
    score = st.mean(r["total_score"] for r in rr)
    tcost = st.mean(r["total_tool_cost"] for r in rr)
    api = st.mean(r["total_api_cost_usd"] for r in rr)
    qr = q_rows(agent)
    nq = len(qr)
    answered = sum(1 for x in qr if x["sub"] is not None)
    correct = sum(1 for x in qr if x["ok"] == 1)
    return dict(agent=agent, seeds=n, reward=reward, score=score,
                tcost_q=tcost / QPS, api_q=api / QPS, api_total=api,
                acc=100 * correct / nq, abst=100 * (nq - answered) / nq,
                scores=[r["total_score"] for r in rr])


def dominates(a, b):
    """a dominates b in the product order (max reward, min tool/Q, min api/Q)."""
    ge = a["reward"] >= b["reward"] and a["tcost_q"] <= b["tcost_q"] and a["api_q"] <= b["api_q"]
    strict = a["reward"] > b["reward"] or a["tcost_q"] < b["tcost_q"] or a["api_q"] < b["api_q"]
    return ge and strict


def banner(s):
    print("\n" + "=" * 74 + f"\n{s}\n" + "=" * 74)


# ---------------------------------------------------------------- summary
banner("PER-AGENT SUMMARY (20 paired seeds)  [reward = gross, before tool cost]")
S = {a: summarize(a) for a in FAIR + ORACLE}
print(f"{'agent':24s} {'score':>7} {'reward':>7} {'acc%':>6} {'abst%':>6} "
      f"{'tool/Q':>7} {'api$/Q':>8}")
for a in FAIR + (["--oracle skyline--"] + ORACLE):
    if a.startswith("--"):
        print("  " + "-" * 20 + " price-of-inference skyline (NOT on fair frontier)")
        continue
    d = S[a]
    if d is None:
        print(f"{a:24s}   (no rows)"); continue
    print(f"{a:24s} {d['score']:>7.1f} {d['reward']:>7.1f} {d['acc']:>6.1f} "
          f"{d['abst']:>6.1f} {d['tcost_q']:>7.2f} {d['api_q']:>8.4f}")

# ---------------------------------------------------------------- product-order frontier
banner("PRODUCT-ORDER PARETO FRONTIER (fair agents; cost = (tool/Q, api$/Q))")
fair = [S[a] for a in FAIR if S[a]]
undom = [a for a in fair if not any(dominates(b, a) for b in fair if b is not a)]
print("undominated set:")
for a in sorted(undom, key=lambda d: -d["reward"]):
    print(f"  {a['agent']:24s} reward {a['reward']:7.1f}  tool/Q {a['tcost_q']:.2f}  api$/Q {a['api_q']:.4f}")
print(f"\n  bayesian_inferred undominated: {S['bayesian_inferred'] in undom}")
# explicit dominance checks
bi = S["bayesian_inferred"]
for other in ["llama3.1", "single_best", "random"]:
    if S[other]:
        print(f"  bayesian_inferred dominates {other}: {dominates(bi, S[other])}")
hk = S["claude-haiku-4-5-20251001"]
print(f"  Haiku dominates bayesian_inferred on (reward,tool/Q) but NOT product order "
      f"(api$ axis): reward {hk['reward']:.0f}>{bi['reward']:.0f}, tool/Q "
      f"{hk['tcost_q']:.2f}<{bi['tcost_q']:.2f}, api$ {hk['api_q']:.4f}>0")

# ---------------------------------------------------------------- £/point sweep
banner("£/POINT SCALARISATION SWEEP  (cost_$ = api$/Q + p·tool/Q)")
print("  honest robustness: under a $/tool-call rate p, is bayesian_inferred still")
print("  on the 1-D (reward vs cost_$) frontier?")
def undominated_scalar(target, p):
    t = S[target]
    tc = t["api_q"] + p * t["tcost_q"]
    for b in fair:
        if b is t:
            continue
        bc = b["api_q"] + p * b["tcost_q"]
        if b["reward"] >= t["reward"] and bc <= tc and (b["reward"] > t["reward"] or bc < tc):
            return False, b["agent"]
    return True, None
prev = None
for p in [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    ok, by = undominated_scalar("bayesian_inferred", p)
    tag = "undominated" if ok else f"DOMINATED by {by}"
    if tag != prev:
        print(f"  p={p:<6} $/unit-tool-cost : bayesian_inferred {tag}")
        prev = tag

# ---------------------------------------------------------------- per-category duality
banner("PER-CATEGORY DUALITY  (bayesian_inferred vs greedy_inferred)")
def per_cat(agent):
    out = {c: dict(tools={}, ok=0, n=0, net=0.0) for c in CATS}
    for r in q_rows(agent):
        c = r["cat"]; d = out[c]
        d["n"] += 1; d["ok"] += (r["ok"] == 1)
        d["net"] += r["reward"] - r["cost"]
        for t in json.loads(r["tq"]):
            d["tools"][int(t)] = d["tools"].get(int(t), 0) + 1
    return out
pc_b, pc_g = per_cat("bayesian_inferred"), per_cat("greedy_inferred")
BEST = {"factual": 2, "numerical": 3, "recent_events": 1, "misconceptions": 2, "reasoning": 4}
print(f"{'category':15s} {'best':5s} | {'VOI acc%':>8} {'VOI net/Q':>9} {'VOI top-tool':>13}"
      f" | {'grdy acc%':>9} {'grdy net/Q':>10} {'grdy top-tool':>13}")
for c in CATS:
    b, g = pc_b[c], pc_g[c]
    bt = max(b["tools"], key=b["tools"].get); gt = max(g["tools"], key=g["tools"].get)
    star = TOOLNAME[BEST[c]]
    print(f"{c:15s} {star:5s} | {100*b['ok']/b['n']:>8.1f} {b['net']/b['n']:>9.2f} "
          f"{TOOLNAME[bt]+'('+str(b['tools'][bt])+')':>13} | {100*g['ok']/g['n']:>9.1f} "
          f"{g['net']/g['n']:>10.2f} {TOOLNAME[gt]+'('+str(g['tools'][gt])+')':>13}")

# ---------------------------------------------------------------- mix-reweighting crossover
banner("MIX-REWEIGHTING  (aggregate net/Q under hypothetical category mixes)")
print("  net/Q per category (×50 = seed contribution at actual mix):")
net_b = {c: pc_b[c]["net"] / pc_b[c]["n"] for c in CATS}
net_g = {c: pc_g[c]["net"] / pc_g[c]["n"] for c in CATS}
for c in CATS:
    print(f"    {c:15s} VOI {net_b[c]:+6.2f}  greedy {net_g[c]:+6.2f}  Δ(VOI-grdy) {net_b[c]-net_g[c]:+6.2f}")
# actual-mix aggregate
def agg(net, w):
    tot = sum(w.values())
    return sum(net[c] * w[c] for c in CATS) / tot * QPS
print(f"\n  actual mix {CAT_N}: VOI {agg(net_b, CAT_N):.1f}  greedy {agg(net_g, CAT_N):.1f}")
# crossover: vary numerical (cheap-best, VOI's strength) share, hold others proportional
print("  sweep numerical share (cheap-best, VOI's strength); others held proportional:")
base_other = {c: CAT_N[c] for c in CATS if c != "numerical"}
bo = sum(base_other.values())
prev = None
for num_frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    w = {c: (1 - num_frac) * base_other[c] / bo for c in base_other}
    w["numerical"] = num_frac
    vb, vg = agg(net_b, w), agg(net_g, w)
    winner = "VOI" if vb > vg else "greedy"
    if winner != prev:
        print(f"    numerical={num_frac:.0%}: VOI {vb:6.1f}  greedy {vg:6.1f}  -> {winner} wins")
        prev = winner
actual_num = CAT_N["numerical"] / QPS
print(f"  (the benchmark's actual numerical share = {actual_num:.0%})")

# ---------------------------------------------------------------- price of inference
banner("PRICE OF INFERENCE  (oracle category -> inferred) — the DOMINANT lever")
pairs = [("VOI agent", "bayesian", "bayesian_inferred"),
         ("greedy", "ablation_greedy", "greedy_inferred")]
for name, o, i in pairs:
    if S[o] and S[i]:
        print(f"  {name:10s}: oracle {S[o]['score']:6.1f} -> inferred {S[i]['score']:6.1f}"
              f"   Δ = {S[i]['score']-S[o]['score']:+.1f}")
print("  cf. the max any exploration scheme buys over greedy: <= +16 (ceiling 306, reachable ~205)")

# ---------------------------------------------------------------- paired bootstrap
banner("PAIRED-SEED BOOTSTRAP 95% CI on score difference (B=10000)")
def boot(a, b, B=10000):
    da = {r["seed"]: r["total_score"] for r in run_rows(a)}
    db = {r["seed"]: r["total_score"] for r in run_rows(b)}
    seeds = sorted(set(da) & set(db)); diffs = [da[s] - db[s] for s in seeds]
    rng = random.Random(0); n = len(diffs)
    means = sorted(st.mean(rng.choice(diffs) for _ in range(n)) for _ in range(B))
    return st.mean(diffs), means[int(.025 * B)], means[int(.975 * B)]
for a, b in [("bayesian_inferred", "greedy_inferred"),
             ("bayesian_inferred", "llama3.1"),
             ("bayesian_inferred", "single_best"),
             ("greedy_inferred", "llama3.1")]:
    m, lo, hi = boot(a, b)
    sig = "" if lo <= 0 <= hi else "  *"
    print(f"  {a:20s} - {b:20s}: {m:+7.1f}  [{lo:+7.1f}, {hi:+7.1f}]{sig}")
print("\n(* = 95% CI excludes 0)")
