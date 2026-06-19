#!/usr/bin/env python3
"""Paper 1 figure — the exploration--attribution contingency law + the cost frontier.

Panel A (headline): the decomposition as a slope chart. Horizon-aware VOI beats
optimism-greedy with *given* categories (+27) and loses under *inferred* ones; the
crossover between the two conditions IS the contingency law. Myopic VOI shown faint
(loses both). Decomposition numbers are locked in papers/RESULTS.md ("THE
DECOMPOSITION"): the oracle column and the inferred greedy/myopic points come from
the host DB; the inferred horizon point (134.2) comes from
scripts/paper1-horizon-inferred.jl. Hard-coded here because they span scripts.

Panel B: gross reward vs tool cost/Q. The Bayesian family (greedy, VOI) owns the
cost-conscious frontier; VOI is the frugal point that dominates the free local Llama
(fewer calls AND higher score); Haiku is the paid-$ frontier point; Llama dominated.

  python3 scripts/paper1-pareto-figure.py
Writes papers/paper1/pareto.{pdf,png}.
"""
import sqlite3, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "apps", "julia", "qa_benchmark", "results", "benchmark.db")
OUT = os.path.join(HERE, "..", "papers", "paper1")
con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
QPS = 50

# Locked decomposition (papers/RESULTS.md): (oracle/given, inferred/fair) score.
DECOMP = {
    "horizon-aware VOI": (216.8, 134.2),
    "optimism-greedy":   (189.4, 149.6),
    "myopic VOI":        (163.7, 110.4),
}


def agg(agent):
    rr = con.execute("SELECT total_reward, total_tool_cost, total_api_cost_usd "
                     "FROM runs WHERE agent=?", (agent,)).fetchall()
    rew = [r["total_reward"] for r in rr]
    return dict(reward=st.mean(rew), rsd=st.pstdev(rew),
                tcost=st.mean(r["total_tool_cost"] for r in rr) / QPS,
                api=st.mean(r["total_api_cost_usd"] for r in rr) / QPS)


AG = {a: agg(a) for a in ["bayesian_inferred", "greedy_inferred", "llama3.1",
                          "claude-haiku-4-5-20251001", "single_best", "random",
                          "all_tools", "no_voi_inferred"]}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

# ---- Panel A: the contingency law (slope chart) ----
xs = [0, 1]
pa_style = {  # colour, marker, linewidth, alpha, markersize
    "horizon-aware VOI": ("#2ca02c", "o", 2.4, 1.0, 8),
    "optimism-greedy":   ("#ff7f0e", "s", 2.4, 1.0, 8),
    "myopic VOI":        ("#9467bd", "D", 1.3, 0.45, 6),
}
for name, (y_oracle, y_inf) in DECOMP.items():
    col, mk, lw, al, ms = pa_style[name]
    axA.plot(xs, [y_oracle, y_inf], marker=mk, color=col, lw=lw, alpha=al, ms=ms,
             markeredgecolor="black", markeredgewidth=0.4, label=name, zorder=3)
# the two gaps that ARE the law, annotated at the line midpoints
axA.annotate("horizon $+27$", (0.0, (216.8 + 189.4) / 2), textcoords="offset points",
             xytext=(10, -3), fontsize=8.5, color="#2ca02c", fontweight="bold", ha="left")
axA.annotate("greedy $+15$", (1.0, (149.6 + 134.2) / 2), textcoords="offset points",
             xytext=(-10, -2), fontsize=8.5, color="#ff7f0e", fontweight="bold", ha="right")
axA.set_xlim(-0.4, 1.4)
axA.set_xticks(xs)
axA.set_xticklabels(["given\n(oracle)", "inferred\n(NB, LOO 0.78)"])
axA.set_ylabel("score (per seed)")
axA.set_title("(A) Contingency law: exploration's value $\\perp$ attribution quality",
              fontsize=10.5)
axA.legend(fontsize=8, loc="lower left"); axA.grid(alpha=0.25, axis="y")

# ---- Panel B: cost-reward frontier ----
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
    axB.errorbar(d["tcost"], d["reward"], yerr=d["rsd"], fmt=mk, color=col,
                 ms=(sz**0.5), capsize=3, alpha=0.9, label=lab,
                 markeredgecolor="black", markeredgewidth=0.4, zorder=3)
# family cheap-frontier line: single_best -> greedy (the $0 frontier)
axB.plot([AG["single_best"]["tcost"], AG["greedy_inferred"]["tcost"]],
         [AG["single_best"]["reward"], AG["greedy_inferred"]["reward"]],
         "--", color="#1f77b4", alpha=0.5, zorder=1)
axB.annotate("Haiku\n(only paid agent:\n$0.0032/Q)", (AG["claude-haiku-4-5-20251001"]["tcost"],
             AG["claude-haiku-4-5-20251001"]["reward"]), textcoords="offset points",
             xytext=(10, -30), fontsize=8, color="#d62728")
axB.annotate("myopic VOI: frugal $0 point\n(dominates free Llama)",
             (AG["bayesian_inferred"]["tcost"], AG["bayesian_inferred"]["reward"]),
             textcoords="offset points", xytext=(6, -30), fontsize=7.5, color="#1f77b4")
axB.set_xlabel("tool cost per question"); axB.set_ylabel("gross reward (per seed)")
axB.set_title("(B) Cost--reward frontier: the family owns the zero-dollar frontier",
              fontsize=10.5)
axB.legend(fontsize=7.5, loc="center right"); axB.grid(alpha=0.25)

fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"pareto.{ext}"), dpi=150, bbox_inches="tight")
print("wrote", os.path.join(OUT, "pareto.pdf"), "and pareto.png")
