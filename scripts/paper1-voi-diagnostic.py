#!/usr/bin/env python3
"""Paper 1 B4 — the exploration-duality diagnostic (pillar iii evidence).

Shows WHY greedy beats VOI: per-category tool usage + within-seed temporal
exploration. The mechanism is single-question myopic VOI — a Bayesian *learner*,
not a Bayesian *explorer*: it economises onto cheap tools and never pays to
discover the expensive-but-reliable one. Offline, stdlib only.

  python3 scripts/paper1-voi-diagnostic.py [path/to/benchmark.db]
"""
import sqlite3, json, sys, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "..", "apps", "julia", "qa_benchmark", "results", "benchmark.db")
con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
TOOL = {1: "web(c1)", 2: "KB(c2)", 3: "calc(c1)", 4: "llm(c2)"}
CATS = ["factual", "numerical", "recent_events", "misconceptions", "reasoning"]
BEST = {"factual": 2, "numerical": 3, "recent_events": 1, "misconceptions": 2, "reasoning": 4}


def rows(agent):
    return con.execute(
        "SELECT q.question_idx AS qix, q.category AS cat, q.tools_queried AS tq, "
        "q.was_correct AS ok, q.submitted AS sub, q.reward AS reward, q.tool_cost AS cost "
        "FROM questions q JOIN runs r ON q.run_id=r.id WHERE r.agent=?", (agent,)).fetchall()


def overall(agent):
    rs = rows(agent)
    if not rs:
        print(f"  {agent}: no rows"); return
    tc = collections.Counter(); nt = collections.Counter()
    abst = correct = ans = 0
    for r in rs:
        tq = json.loads(r["tq"]); nt[len(tq)] += 1
        for t in tq: tc[int(t)] += 1
        if r["sub"] is None: abst += 1
        else:
            ans += 1; correct += (r["ok"] == 1)
    print(f"\n== {agent} ({len(rs)} q) == acc {100*correct/len(rs):.1f}% (answered {100*correct/max(ans,1):.1f}%) "
          f"abstain {100*abst/len(rs):.1f}%  #tools/Q {dict(sorted(nt.items()))}")
    print("   tool usage: " + ", ".join(f"{TOOL[t]}={tc[t]}" for t in sorted(tc)))


print("=" * 70 + "\nPER-AGENT TOOL USAGE\n" + "=" * 70)
for a in ["bayesian", "ablation_greedy", "bayesian_inferred", "greedy_inferred"]:
    overall(a)

print("\n" + "=" * 70 + "\nPER-CATEGORY (★ = best tool for category)\n" + "=" * 70)
def per_cat(agent):
    out = {c: dict(tools=collections.Counter(), ok=0, n=0, net=0.0) for c in CATS}
    for r in rows(agent):
        d = out[r["cat"]]; d["n"] += 1; d["ok"] += (r["ok"] == 1); d["net"] += r["reward"] - r["cost"]
        for t in json.loads(r["tq"]): d["tools"][int(t)] += 1
    return out
for a in ["bayesian_inferred", "greedy_inferred"]:
    pc = per_cat(a)
    print(f"\n {a}:")
    for c in CATS:
        d = pc[c]; top = max(d["tools"], key=d["tools"].get)
        print(f"   {c:15s} ★{TOOL[BEST[c]]:8s} acc {100*d['ok']/d['n']:5.1f}%  net/Q {d['net']/d['n']:+5.2f}  "
              f"top {TOOL[top]}={d['tools'][top]}")

print("\n" + "=" * 70 + "\nTEMPORAL: knowledge_base query-rate within a seed (KB-best cats)\n" + "=" * 70)
def kb_temporal(agent):
    buckets = {"q0-9": [0, 0], "q10-29": [0, 0], "q30-49": [0, 0]}
    for r in rows(agent):
        if r["cat"] not in ("factual", "misconceptions"): continue
        b = "q0-9" if r["qix"] < 10 else ("q10-29" if r["qix"] < 30 else "q30-49")
        buckets[b][1] += 1; buckets[b][0] += (2 in json.loads(r["tq"]))
    rate = {k: f"{100*v[0]/max(v[1],1):.0f}%" for k, v in buckets.items()}
    print(f"  {agent:18s} {rate}")
for a in ["bayesian", "ablation_greedy", "bayesian_inferred", "greedy_inferred"]:
    kb_temporal(a)
print("  (VOI crawls up slowly = under-exploration; greedy ramps fast = optimistic exploration)")
