#!/usr/bin/env python3
"""Paper 1 B4 figure — cost–reward frontier + the exploration duality.

Panel A: gross reward vs tool cost/Q. The Bayesian family (greedy, VOI) owns the
cost-conscious frontier; Haiku is the paid-$ frontier point; Llama is dominated.
Panel B: per-category net/Q, VOI vs greedy — the lopsided duality (greedy wins 4
of 5; VOI wins only numerical, the cheap-and-dominant-by-a-wide-margin tool).

  python3 scripts/paper1-pareto-figure.py
Writes papers/paper1/pareto.{pdf,png}.
"""
import sqlite3, json, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "apps", "julia", "qa_benchmark", "results", "benchmark.db")
OUT = os.path.join(HERE, "..", "papers", "paper1")
con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
QPS = 50
CATS = ["factual", "numerical", "recent_events", "misconceptions", "reasoning"]
CATLAB = ["factual", "numerical", "recent", "misconc", "reasoning"]


def agg(agent):
    rr = con.execute("SELECT total_reward, total_tool_cost, total_api_cost_usd "
                     "FROM runs WHERE agent=?", (agent,)).fetchall()
    rew = [r["total_reward"] for r in rr]
    return dict(reward=st.mean(rew), rsd=st.pstdev(rew),
                tcost=st.mean(r["total_tool_cost"] for r in rr) / QPS,
                api=st.mean(r["total_api_cost_usd"] for r in rr) / QPS)


def per_cat_net(agent):
    out = {c: [0.0, 0] for c in CATS}
    for r in con.execute("SELECT q.category c, q.reward rw, q.tool_cost tc FROM questions q "
                         "JOIN runs r ON q.run_id=r.id WHERE r.agent=?", (agent,)):
        out[r["c"]][0] += r["rw"] - r["tc"]; out[r["c"]][1] += 1
    return [out[c][0] / out[c][1] for c in CATS]


AG = {a: agg(a) for a in ["bayesian_inferred", "greedy_inferred", "llama3.1",
                          "claude-haiku-4-5-20251001", "single_best", "random",
                          "all_tools", "no_voi_inferred"]}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

# ---- Panel A: frontier ----
style = {
    "greedy_inferred":   ("Bayesian: greedy", "#1f77b4", "o", 90),
    "bayesian_inferred": ("Bayesian: VOI",    "#1f77b4", "D", 70),
    "claude-haiku-4-5-20251001": ("Haiku (paid)", "#d62728", "*", 240),
    "llama3.1":          ("Llama (free LLM)", "#ff7f0e", "s", 70),
    "single_best":       ("single-best", "#7f7f7f", "v", 50),
    "random":            ("random", "#7f7f7f", "^", 50),
    "all_tools":         ("all-tools", "#7f7f7f", "P", 50),
    "no_voi_inferred":   ("no-VOI (query-all)", "#7f7f7f", "X", 50),
}
for a, (lab, col, mk, sz) in style.items():
    d = AG[a]
    axA.errorbar(d["tcost"], d["reward"], yerr=d["rsd"], fmt=mk, color=col,
                 ms=(sz**0.5), capsize=3, alpha=0.9, label=lab,
                 markeredgecolor="black", markeredgewidth=0.4, zorder=3)
# family cheap-frontier line: single_best -> greedy (the $0 frontier), then up to Haiku
axA.plot([AG["single_best"]["tcost"], AG["greedy_inferred"]["tcost"]],
         [AG["single_best"]["reward"], AG["greedy_inferred"]["reward"]],
         "--", color="#1f77b4", alpha=0.5, zorder=1)
axA.annotate("Haiku pays\n$0.0032/Q", (AG["claude-haiku-4-5-20251001"]["tcost"],
             AG["claude-haiku-4-5-20251001"]["reward"]), textcoords="offset points",
             xytext=(8, -22), fontsize=8, color="#d62728")
axA.annotate("VOI dominated by its\ngreedy sibling", (AG["bayesian_inferred"]["tcost"],
             AG["bayesian_inferred"]["reward"]), textcoords="offset points",
             xytext=(6, -30), fontsize=7.5, color="#1f77b4")
axA.set_xlabel("tool cost per question"); axA.set_ylabel("gross reward (per seed)")
axA.set_title("(A) Cost–reward frontier: the family owns the cheap regime")
axA.legend(fontsize=7.5, loc="center right"); axA.grid(alpha=0.25)

# ---- Panel B: duality ----
nb, ng = per_cat_net("bayesian_inferred"), per_cat_net("greedy_inferred")
x = range(len(CATS)); w = 0.38
axB.bar([i - w/2 for i in x], nb, w, label="VOI", color="#1f77b4")
axB.bar([i + w/2 for i in x], ng, w, label="greedy", color="#9ecae1")
axB.axhline(0, color="black", lw=0.6)
axB.set_xticks(list(x)); axB.set_xticklabels(CATLAB, fontsize=8)
axB.set_ylabel("net score per question")
axB.set_title("(B) Exploration duality: greedy wins 4/5; VOI only numerical")
axB.annotate("VOI wins\n(cheap+dominant)", (1, max(nb[1], ng[1]) + 0.2),
             fontsize=7.5, ha="center", color="#1f77b4")
axB.legend(fontsize=8); axB.grid(alpha=0.25, axis="y")

fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"pareto.{ext}"), dpi=150, bbox_inches="tight")
print("wrote", os.path.join(OUT, "pareto.pdf"), "and pareto.png")
