# Benchmark Results — Paper 1 (fair, inferred-category condition)

Julia benchmark at `apps/julia/qa_benchmark/host.jl`; results in (git-ignored)
SQLite at `apps/julia/qa_benchmark/results/benchmark.db`. 20 paired seeds, 50
questions/seed, 4 simulated tools with category-dependent reliability. Categories
are **inferred** by a Gaussian Naive-Bayes classifier on committed offline
sentence embeddings (LOO ≈ 0.78), not given — the fair condition Phase B was
built to test. LLM agents (Haiku, Llama) are the saved no-category runs, reused
(both agents see only raw question text; see the OQ5 decision below). Fully
offline, $0. Regenerate the analysis with `scripts/paper1-pareto.py`.

---

## FRAMING DECISIONS (locked — the argument Phase C transcribes, does not relitigate)

> **This file is the design-doc to `credence.tex`'s rewrite.** It locks the
> *argument*, not just the arithmetic. Every commitment below is supported by the
> tables that follow.

### Headline — the inversion

In a Bayesian tool-selecting agent, **cost-efficiency is earned by the belief
substrate — reliability learning plus category inference — not by the
value-of-information action layer.** The sophisticated VOI policy in fact sits
*below* a trivial greedy rule (−39, significant); even an *optimal* action policy
adds at most +16 over greedy; meanwhile inference quality moves performance by
40–53. **The engine is the substrate, not the action machinery.**

This is the discovery, not an apology. Paper 1 set out (Phase B thesis, verbatim:
*"Bayesian VOI tool selection occupies a non-empty region of the cost-performance
Pareto frontier under fair conditions"*) to show the VOI machinery is what makes
tool selection cheap. It isn't. Quantifying the inversion is the contribution.

### Three pillars (all falsifiable, all supported)

1. **The substrate earns the frontier.** Both Bayesian action policies (VOI and
   greedy) bury the non-learning baselines and **significantly dominate the free
   local Llama**. The belief substrate — not the action rule — is what buys the
   cost-efficient frontier position.
2. **The action layer is not the engine.** VOI sits *below* the trivial greedy
   rule (−39 [−64, −14]); the gating experiments cap *any* action-policy
   improvement at ≤+16 over greedy (ceiling 306, reachable ~205, horizon-locked);
   and the inference lever (40–53) dwarfs both.
3. **The secondary structure is characterised, not hand-waved.** *Which* action
   policy wins, *when*, is governed by a per-category law (greedy wins where the
   best tool is expensive or only moderately separated; VOI wins only where a
   cheap tool is dominant by a wide margin) with a quantified mix-crossover (~40%).

### Vindicated / Revised / New

- **Vindicated:** the belief substrate (reliability learning + category
  inference), the framework spine (the five axioms / the Credence DSL), and the
  dominance of the free local LLM by a principled $0 agent.
- **Revised:** the prior that VOI's information-valuing is the *source* of
  efficiency. It isn't, at this horizon and mix. **Phase B's own premise is
  refuted**: the v1 greedy>VOI gap (+25.7) was explained away as a
  category-given artifact that fair conditions would correct; instead the gap
  *grows* to +39.2 under inferred categories, because VOI is *more* sensitive to
  category-inference noise than greedy (VOI loses −53 to inference, greedy −40).
  The experiment carried information: the prior was the right one to hold, and the
  data sharply corrects it.
- **New (the contribution):** the substrate/action decomposition; the lopsided
  exploration law; the mix-crossover; and the inference-dominates-action
  quantification.

### Scoping commitments (state plainly, scope precisely, do not soften or hedge)

- **Thesis broadened** from "Bayesian *VOI* tool selection" to "Bayesian tool
  selection." VOI keeps a *named, scoped* role, not vanishing: it is the
  cost-averse action policy that wins **only where the best tool is both cheap and
  dominant by a wide margin** — a cheap near-oracle for the sub-task (the
  calculator on arithmetic). Narrate as principled-prior → empirical-correction:
  a rational agent values information and exploration *ought* to emerge from
  EU-max; the finding is the precise correction that *myopic* VOI does not deliver
  it at short horizons. Not retreat — a revised prior.
- **The frontier claim is made at the family level, via greedy** as the robust
  cheap-frontier representative. **Drop the per-VOI frontier claim and the
  incommensurability rescue.** Under any scalarization that lets the cost axes
  trade off, VOI is dominated by its own greedy sibling except in a narrow band
  that is an artifact of the £/point rate. The **Bayesian family owns the
  cost-conscious ("cheap") regime** (greedy dominates Llama + baselines); **Haiku
  is the paid-frontier point**. At high £/point even greedy yields to Haiku, so
  the family's ownership is specifically the cost-conscious regime — which is the
  regime this paper is about.
- **Greedy is the stronger action policy on this benchmark — stated plainly**, a
  mechanistically-explained loss (mastery, not evasion). *Scoped*, not hedged: it
  wins on this mix and horizon, flips above ~40% cheap-decisive-best share, and
  the binding constraint is the *horizon*, not VOI-in-principle. One precision so
  the escape hatch isn't overclaimed: the deployed *myopic* VOI would not win at a
  longer horizon either — it doesn't explore regardless of horizon; only
  *horizon-aware* VOI (future work) converts a longer horizon into wins. Scope
  greedy's victory; leave myopic VOI's loss unhedged.
- **Cost-efficiency, never Haiku-parity.** Haiku answers 97.5% vs the Bayesian
  agents' 56–64% — state the subtraction plainly. The claim is purely
  cost-efficiency: dominates the free Llama outright; the only thing beating it on
  performance pays real dollars.

### Genre and venue

This is an **analysis/architecture paper** — an anatomy of what drives a Bayesian
agent's cost-efficiency, with a characterised exploration law — not a methods
paper claiming VOI is Pareto-optimal. Target: **arXiv cs.AI** (cross-list cs.LG,
cs.CL). The horizon-lock and mix-crossover are pitched as **laws that generalise**
(short-horizon exploration economics; cost × dominance-margin selection), not
facts about this benchmark; the framework's novelty (the Credence spine) carries
visible weight.

### OQ5 (LLM fairness) — resolved: no π-injection

The fair leveller is **input symmetry** (every agent sees the same raw question
text; each reasons in its own way), not information-surface symmetry. By the
data-processing inequality, the inferred posterior π is a deterministic lossy
function of the text and carries no category information the LLM (holding the full
text) lacks; injecting it would import the classifier's error rate into the LLM's
input. So the saved no-category LLM runs *are* the fair condition. "Category
known" (oracle) is dropped as a tested condition; it survives only as the
price-of-inference skyline below.

---

## Setup

50 questions across 5 categories (factual 15, numerical 10, recent_events 8,
misconceptions 7, reasoning 10). Scoring: correct +10, wrong −5, abstain 0; tool
costs below. Tool reliability by category (★ = best tool for that category):

| Tool (cost) | factual | numerical | recent | misconc | reasoning |
|---|---:|---:|---:|---:|---:|
| web_search (1) | ★0.70 | 0.20 | ★0.65 | 0.25 | 0.40 |
| knowledge_base (2) | ★0.92 | 0.40 | 0.55 | ★0.88 | 0.45 |
| calculator (1) | 0.25 | ★1.00 | 0.25 | 0.25 | 0.25 |
| llm_direct (2) | 0.65 | 0.50 | 0.45 | 0.40 | ★0.72 |

## Main results — fair condition (20 paired seeds)

`reward` is gross (Σ of +10/−5/0, before tool cost); `score` = reward − tool cost.

| Agent | Score | Gross reward | Acc% | Abst% | Tool/Q | API $/Q |
|---|---:|---:|---:|---:|---:|---:|
| claude-haiku-4-5 (reused) | +445.5 | 481.2 | 97.5 | 0.0 | 0.71 | 0.0032 |
| greedy_inferred | +149.6 | 230.8 | 64.1 | 0.0 | 1.62 | 0 |
| **bayesian_inferred** (VOI) | **+110.4** | 176.5 | 56.2 | 2.0 | 1.32 | 0 |
| llama3.1 (reused) | +79.3 | 151.5 | 45.9 | 22.9 | 1.44 | 0 |
| random | +55.4 | 131.8 | 50.9 | 0.0 | 1.53 | 0 |
| single_best | +44.2 | 94.2 | 45.9 | 0.0 | 1.00 | 0 |
| no_voi_inferred | −24.5 | 275.5 | 69.3 | 2.3 | 6.00 | 0 |
| no_learning_inferred | +3.8 | 53.8 | 40.5 | 0.0 | 1.00 | 0 |
| all_tools | −67.0 | 233.0 | 64.4 | 0.0 | 6.00 | 0 |

## Pillar (i) — the substrate earns the cost-efficient frontier

The **Bayesian family** (greedy + VOI) owns the cost-conscious frontier: greedy
dominates the free Llama and every non-learning baseline; Haiku is the
paid-frontier point. Removing learning collapses the agent (`no_learning` +3.8 ≈
chance-with-cost). Paired-seed bootstrap (95% CI, B=10000), score difference:

| contrast | Δ score | 95% CI |
|---|---:|---|
| greedy_inferred − llama3.1 | +70.2 | [+49.8, +91.2] |
| bayesian_inferred − single_best | +66.2 | [+44.4, +88.1] |
| bayesian_inferred − llama3.1 | +31.1 | [+6.2, +56.2] |

**The £/point honesty.** Under a scalarization `cost_$ = api$ + p·tool-calls`,
`bayesian_inferred` is undominated only in a narrow band p ≈ 0.001–0.005 $/call:
below it greedy dominates (tool-calls ~free → greedy is simply better at equal
$0 cost); above it Haiku dominates (fewer tool-calls *and* higher reward for a
tiny API$). So the frontier claim is made at the **family** level via greedy; VOI
specifically is **not a robust frontier point**. Under the decomposition thesis it
does not need to be.

## Pillar (ii) — the action layer is not the engine

| contrast | Δ score | 95% CI |
|---|---:|---|
| bayesian_inferred − greedy_inferred | **−39.2** | [−63.5, −13.9] |

VOI loses to the trivial greedy rule, significantly. The **gating experiments**
bound how much *any* action policy could recover (oracle category isolates this
from inference):

- **Known-θ ceiling** (true reliabilities, myopic-EU, no learning): **306.2** —
  the asymptote of perfect exploration. Headroom exists.
- **Reachable bound** (best explore-then-exploit, sweep forced-exploration k):
  peaks at **k=1 = 205.0**, then collapses (k=2=187 < k=0=189 — two rounds is
  worse than none). The horizon affords ~one exploration round; it cannot recover
  the 101-point ceiling gap. The bound is conservative (query-all,
  majority-vote-in-explore, fixed-k all slack it downward), but the *shape* —
  one-round optimum, collapse — is family-robust.
- So the most any action-policy improvement buys over free greedy is **≤+16**
  (205 vs 189), horizon-locked. (Reproduce: `scripts/paper1-ceiling.jl`,
  `scripts/paper1-reachable.jl`.)

**Price of inference — the dominant lever.** Oracle category → inferred:

| agent | oracle | inferred | Δ |
|---|---:|---:|---:|
| VOI (bayesian) | +163.7 | +110.4 | **−53.3** |
| greedy | +189.4 | +149.6 | **−39.8** |

Inference quality moves performance by 40–53 — an order of magnitude past the
≤+16 of action-policy headroom. **The 78%-LOO classifier, not horizon-aware VOI,
is the real improvement lever** and the natural future-work direction. (Phase B's
premise refuted: fair conditions *widened* greedy's edge, +25.7 → +39.2, because
VOI is more inference-sensitive.)

## Pillar (iii) — the exploration law (the secondary structure)

Per-category, `bayesian_inferred` (VOI) vs `greedy_inferred` (★ = best tool;
net/Q after cost):

| category | best tool | VOI acc / net (top tool) | greedy acc / net (top tool) | Δ(VOI−grdy) |
|---|---|---:|---:|---:|
| factual | ★KB (c2) | 70.7% / +4.11 (web) | 84.0% / +5.84 (KB) | −1.73 |
| misconceptions | ★KB (c2) | 38.6% / −0.52 (calc) | 57.9% / +1.99 (KB) | −2.51 |
| recent_events | ★web (c1) | 45.6% / +1.09 (web) | 55.6% / +1.98 (web) | −0.88 |
| reasoning | ★llm (c2) | 42.5% / +0.12 (calc) | 49.0% / +0.75 (KB) | −0.62 |
| **numerical** | ★calc (c1) | 69.0% / **+4.24** (calc) | 60.5% / +2.48 (KB) | **+1.76** |

**The law:** greedy (optimism-under-uncertainty) wins wherever the best tool is
**expensive** (the two knowledge_base categories — myopic VOI under-explores the
cost-2 tool, querying cheap web/calc instead) **or only moderately separated**
(recent_events: web is cheap but 0.65, so which tool is best is in genuine doubt —
exploration would help and VOI refuses it). VOI wins **only where the best tool is
both cheap and dominant by a wide margin** (numerical: calculator is cost-1 and
1.00-reliable — a cheap near-oracle, where cost-aversion is pure virtue and
exploration is unnecessary).

**Mechanism** (confirmed by per-category tool-usage + within-seed temporal data,
`scripts/paper1-voi-diagnostic.py`): the VOI in `agent-step` is single-step *and
single-question* — it values information only for the current decision, never
future ones. So it is a Bayesian *learner*, not a Bayesian *explorer*. With
Beta(1,1) priors every tool's net-VOI is `2.5 − cost`, so it economises onto cheap
tools and **never pays to discover the expensive-but-reliable tool**
(knowledge_base query-rate on KB-best categories crawls 4%→31% over a seed, vs
greedy's optimistic 25%→66%).

**Mix-crossover:** at the benchmark's category mix, greedy wins the aggregate
(149.6 vs 110.4). VOI overtakes only when the numerical (cheap-decisive-best)
share exceeds **~40%**; the actual share is **20%**. The aggregate is the
mix-weighted projection of the per-category law — report the law, not the
contingent aggregate.

## Oracle skyline — price-of-inference reference (NOT on the fair frontier)

The given-category condition Phase B exists to retire; kept only to quantify the
price of inference. Not comparable to the LLM agents (which never got categories).

| Agent | Score | Acc% | Tool/Q |
|---|---:|---:|---:|
| ablation_greedy (oracle) | +189.4 | 68.6 | 1.50 |
| bayesian (oracle VOI) | +163.7 | 62.0 | 1.25 |
| ablation_no_abstain (oracle) | +156.9 | 62.6 | 1.25 |
| ablation_no_voi (oracle) | +38.2 | 77.9 | 6.00 |
| ablation_no_learning (oracle) | +3.8 | 40.5 | 1.00 |

## Reproducibility

All offline from committed fixtures + the (git-ignored) DB:

- `scripts/paper1-pareto.py` — main table, product-order frontier, £/point sweep,
  per-category duality, mix-crossover, price-of-inference, paired bootstrap.
- `scripts/paper1-voi-diagnostic.py` — per-category tool usage + temporal
  exploration evidence.
- `scripts/paper1-ceiling.jl` / `scripts/paper1-reachable.jl` — the gating
  experiments (ceiling 306, reachable shape).
- `scripts/paper1-pairing-gate.jl` — proves the reused-LLM seeds reproduce the
  saved DB bit-exact (reuse validity).

```sql
-- per-agent summary
SELECT agent, ROUND(AVG(total_score),1) score, ROUND(AVG(total_reward),1) reward,
       ROUND(AVG(total_tool_cost),2) toolcost FROM runs GROUP BY agent ORDER BY score DESC;
-- per-category accuracy
SELECT r.agent, q.category, ROUND(100.0*SUM(q.was_correct)/COUNT(*),1) acc_pct
FROM questions q JOIN runs r ON q.run_id=r.id GROUP BY r.agent, q.category;
```
