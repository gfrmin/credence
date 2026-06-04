# Benchmark Results — Paper 1 (the exploration–attribution contingency)

Julia benchmark at `apps/julia/qa_benchmark/host.jl`; results in (git-ignored)
SQLite at `apps/julia/qa_benchmark/results/benchmark.db`. 20 paired seeds, 50
questions/seed, 4 simulated tools with category-dependent reliability. Scoring:
+10 correct, −5 wrong, 0 abstain, minus tool cost. Two conditions: **oracle**
(category given) and **inferred** (categories from a Gaussian Naive-Bayes
classifier on committed offline sentence embeddings, LOO ≈ 0.78 — the fair
condition Phase B was built to test). LLM agents (Haiku, Llama) are the saved
no-category runs, reused (raw question text only; OQ5 below). Fully offline, $0.

Reproduce: oracle policies via the host (`--seeds 20`, `--ablation`); the exact
ceiling via `scripts/paper1-horizon-gate.jl`; depth-d capture via
`scripts/paper1-horizon-depth.jl`; the inferred result + credit-rule de-confound
via `scripts/paper1-horizon-inferred.jl`; frontier analysis via
`scripts/paper1-pareto.py`.

---

## FRAMING DECISIONS (locked — the argument Phase C transcribes, does not relitigate)

> **This file is the design-doc to `credence.tex`'s rewrite.** It locks the
> *argument*, not just the arithmetic. Every commitment below is supported by the
> tables that follow.

### Headline — the contingency law

**In category-conditioned tool selection, the value of exploration is contingent
on category-attribution quality.** When categories are *given*, horizon-aware VOI
— valuing information over the remaining question horizon, pure EU-maximisation,
exploration emergent — fixes myopic VOI's under-exploration and **beats the
optimism-greedy rule (+27)**. When categories are *inferred* (fair), attribution
noise denies exploration the clean per-category signal it depends on, and
**minimal-query optimism wins**. The +27 oracle result and the fair loss are two
points on one curve; the curve is the contribution.

This refines, not abandons, Phase B's thesis (verbatim: *"Bayesian VOI tool
selection occupies a non-empty region of the cost-performance Pareto frontier
under fair conditions"*). The Bayesian *substrate* owns the cost-conscious
frontier in both conditions; VOI is the frugal frontier point that dominates the
free local Llama. What the experiment refuted is the narrower prior that the VOI
*action layer* beats greedy under fair conditions — and *why* it does not (the
attribution-noise mechanism, robust across every credit rule) is the finding.

### Three pillars (all falsifiable, all supported)

1. **The substrate earns the frontier — both conditions.** Reliability learning
   plus category inference is what buys the cost-efficient position: every learning
   policy buries the non-learning baselines, and **VOI dominates the free local
   Llama** (fewer tool-calls *and* higher score — the frugal frontier point).
   Optimism-greedy is itself a Bayesian-family member, so the *family* owns the
   cheap regime regardless of which action policy wins.
2. **Exploration is fixable, and beats greedy with given categories.** Myopic VOI
   is a Bayesian *learner*, not a Bayesian *explorer* — it never probes a tool
   whose answer it won't submit (Pillar iii). Horizon-aware VOI
   (`agent.bdsl:horizon-step`) values information over the remaining same-category
   questions and **beats greedy under oracle categories (216.8 vs 189.4)**. Exact
   decoupled ceiling 211.8; a depth-1 lookahead already captures it. A real,
   constitution-clean method result.
3. **Under fair categories the advantage evaporates — robustly.** Inferred-category
   noise makes every exploratory query inject misattribution noise into per-category
   reliability learning (B2c credits each category by the noisy posterior π).
   Horizon-VOI loses to greedy under **every** degree of freedom tested: submit
   policy, probe intensity, confidence gating, and all three credit rules
   (soft/hard/post). Greedy queries minimally → corrupts least → wins.

### Vindicated / Revised / New

- **Vindicated:** the belief substrate (reliability learning + category inference);
  the framework spine (the axioms / the Credence DSL); the dominance of the free
  local Llama by a principled $0 agent; and — newly — that a constitution-clean
  exploration policy (horizon-aware VOI) *can* beat greedy (with given categories).
- **Revised:** the prior that the VOI action layer beats greedy under *fair*
  conditions. It does not. The v1 greedy>VOI gap (+25.7, oracle) was read as a
  category-given artifact fair conditions would correct; instead it **persists and
  the mechanism is now identified** — not "VOI is worse" but "exploration's value is
  contingent on attribution quality, and the 0.78 classifier is below the bar." The
  prior was worth holding; its failure is mechanistic, not mysterious.
- **New:** (a) the **contingency law** — exploration's value ⟂ category-attribution
  quality, crossover between oracle and a 0.78-accurate classifier; (b) the
  **credit-rule × query-strategy interaction** — zero-leakage hard-credit is best
  for exploration-heavy policies, worst for minimal-query greedy. This is a
  *structural coupling*, **not** a route to exploration winning: even hard-credit
  (exploration's best case) leaves horizon-VOI 12 behind greedy. Subordinate to (a).
  (c) **B2c soft-credit is a measurably suboptimal silent approximation** to the
  proper posterior-weighted update (post-credit beats it even for deployed greedy)
  — disclosed in methods below; code migration tracked in issue #111.

### Scoping commitments (state plainly, scope precisely, do not soften)

- **Cost-efficiency, never Haiku-parity.** Haiku answers 97.5% vs the family's
  56–74% — state the subtraction plainly. The claim is purely cost-efficiency:
  dominates the free Llama outright; the only thing beating it on performance pays
  real dollars.
- **The frugal-frontier claim is VOI-specific and product-order;** the
  scalarized-frontier claim is family-level. VOI dominates Llama in the product
  order (fewer calls + higher score). Under a scalarization that trades the cost
  axes, VOI is dominated by its greedy sibling except in a narrow £/point band —
  so the *robust* cheap-frontier representative is the family (via greedy), with
  Haiku the paid-frontier point. Both stated; neither overclaimed.
- **Greedy wins fair, stated plainly** — a mechanistically-explained loss
  (attribution-noise contingency), not evasion. Scoped: it wins under inferred
  categories at this classifier accuracy; horizon-VOI wins once categories are
  given. The binding constraint is attribution quality, not VOI-in-principle.

### Genre and venue

Empirical study of *when* Bayesian VOI tool selection pays, with a
constitution-clean method (horizon-aware VOI) and a mechanism — not a methods
paper claiming VOI is Pareto-optimal, nor an apology. Target: **arXiv cs.AI**
(cross-list cs.LG, cs.CL); a method/workshop track is reachable on the oracle
result if the contingency framing lands. The contingency law and the credit-rule
interaction are pitched as **laws that generalise**, not facts about this
benchmark.

### OQ5 (LLM fairness) — resolved: no π-injection

The fair leveller is **input symmetry** (every agent sees the same raw question
text). By the data-processing inequality, the inferred posterior π is a
deterministic lossy function of the text and carries no category information the
LLM lacks; injecting it would import the classifier's error rate. So the saved
no-category LLM runs *are* the fair condition (pairing-reproduction gate:
`scripts/paper1-pairing-gate.jl`, bit-exact).

---

## Setup

50 questions across 5 categories (factual 15, numerical 10, recent_events 8,
misconceptions 7, reasoning 10). Tool reliability by category (★ = best tool):

| Tool (cost) | factual | numerical | recent | misconc | reasoning |
|---|---:|---:|---:|---:|---:|
| web_search (1) | ★0.70 | 0.20 | ★0.65 | 0.25 | 0.40 |
| knowledge_base (2) | ★0.92 | 0.40 | 0.55 | ★0.88 | 0.45 |
| calculator (1) | 0.25 | ★1.00 | 0.25 | 0.25 | 0.25 |
| llm_direct (2) | 0.65 | 0.50 | 0.45 | 0.40 | ★0.72 |

## THE DECOMPOSITION (the two points on the curve)

Same belief substrate (Beta-Bernoulli reliability learning) throughout; only the
action policy and the category condition vary.

| policy | **oracle** (given) | **inferred** (fair) | inference penalty |
|---|---:|---:|---:|
| myopic VOI (`bayesian` / `bayesian_inferred`) | 163.7 | 110.4 | −53.3 |
| optimism-greedy (`ablation_greedy` / `greedy_inferred`) | 189.4 | 149.6 | −39.6 |
| **horizon-aware VOI** (`horizon` / `paper1-horizon-inferred.jl`) | **216.8** | 134.2 | **−82.6** |
| winner | horizon-VOI **+27** | **greedy +15** | — |

Horizon-VOI is the *most* inference-sensitive policy (−82.6): precise exploration
is exactly what category noise destroys. Reference ceilings (oracle): exact
decoupled horizon-optimum **211.8** (`paper1-horizon-gate.jl`); depth-1 deployable
**216.8** (`paper1-horizon-depth.jl`); known-θ **306.2**.

## Main results — fair condition (20 paired seeds)

`reward` is gross (Σ +10/−5/0, before tool cost); `score` = reward − tool cost.

| Agent | Score | Gross reward | Acc% | Abst% | Tool/Q | API $/Q |
|---|---:|---:|---:|---:|---:|---:|
| claude-haiku-4-5 (reused) | +445.5 | 481.2 | 97.5 | 0.0 | 0.71 | 0.0032 |
| greedy_inferred | +149.6 | 230.8 | 64.1 | 0.0 | 1.62 | 0 |
| horizon-VOI (inferred) | +134.2 | — | 63.6 | 0.0 | 1.77 | 0 |
| **bayesian_inferred** (myopic VOI) | **+110.4** | 176.5 | 56.2 | 2.0 | 1.32 | 0 |
| llama3.1 (reused) | +79.3 | 151.5 | 45.9 | 22.9 | 1.44 | 0 |
| random | +55.4 | 131.8 | 50.9 | 0.0 | 1.53 | 0 |
| single_best | +44.2 | 94.2 | 45.9 | 0.0 | 1.00 | 0 |
| no_voi_inferred | −24.5 | 275.5 | 69.3 | 2.3 | 6.00 | 0 |
| no_learning_inferred | +3.8 | 53.8 | 40.5 | 0.0 | 1.00 | 0 |
| all_tools | −67.0 | 233.0 | 64.4 | 0.0 | 6.00 | 0 |

## Pillar (i) — the substrate earns the cost-efficient frontier

The Bayesian family owns the cost-conscious frontier; **VOI dominates the free
Llama in the product order** — fewer tool-calls (1.32 < 1.44) *and* higher score
(110.4 > 79.3) — the frugal frontier point. Greedy uses *more* calls than Llama
(1.62), so greedy is the high-score point, not the frugal one. Non-learners
collapse (`no_learning` +3.8 ≈ chance-with-cost). Paired-seed bootstrap (95% CI,
B=10000):

| contrast | Δ score | 95% CI |
|---|---:|---|
| greedy_inferred − llama3.1 | +70.2 | [+49.8, +91.2] |
| bayesian_inferred − single_best | +66.2 | [+44.4, +88.1] |
| bayesian_inferred − llama3.1 | +31.1 | [+6.2, +56.2] |

**The £/point honesty.** Under `cost_$ = api$ + p·tool-calls`, `bayesian_inferred`
is undominated only in a narrow band p ≈ 0.001–0.005 $/call: below it greedy
dominates, above it Haiku. So the *scalarized* frontier claim is made at the
family level via greedy; VOI's robust claim is the product-order domination of
Llama. Haiku is the paid-frontier point.

## Pillar (ii) — exploration is fixable, and contingent on attribution

Myopic VOI loses to greedy in both conditions (`bayesian` − `ablation_greedy` =
−25.7 oracle; `bayesian_inferred` − `greedy_inferred` = −39.2 [−63.5, −13.9]
fair). The cause is under-exploration (Pillar iii), and it is **fixable**:

**Horizon-aware VOI beats greedy with given categories.** Valuing information over
the remaining same-category questions (pure EU-max over the horizon — no
ε-greedy, no bonus) lifts the oracle score to **216.8 vs greedy 189.4 (+27)**. The
exact decoupled horizon-optimum is 211.8; a **depth-1** lookahead already captures
it (217.6 ≈ DSL 216.8), so the win is not an artifact of unbounded search.

**But the advantage is contingent on attribution quality.** Under inferred
categories, every exploratory query injects misattribution noise into per-category
reliability learning. Isolating exploration (greedy's cost-blind submit +
horizon-probing) across the credit-assignment rule:

| credit rule | greedy | horizon-VOI | gap | note |
|---|---:|---:|---:|---|
| soft (B2c, deployed) | 149.8 | 116.7 | −33.1 | fractional leak to every category |
| hard (argmax-π) | 142.5 | 130.4 | **−12.1** | zero leak; exploration's best case |
| post (π·likelihood) | 151.4 | 126.7 | −24.6 | the de Finettian rule B2c approximates |

Greedy wins fair under all three. Better attribution helps exploration (soft→hard
+13.7) and hurts minimal-query greedy (−7.3), narrowing the gap to −12 at best,
**never closing it** — the proper rule (`post`) lands worse than hard precisely
because it lifts greedy too. **Oracle (perfect attribution) is the only regime
where horizon-VOI wins.** The fair loss is a property of imperfect attribution,
not of any one credit rule or any submit/probe/gating choice.

**Methods caveat (disclosed, not deferred).** The deployed/headline numbers use
B2c soft-credit — a tractable approximation to the proper posterior-weighted
update the foundations endorse. The proper rule (`post`) lifts the greedy baseline
(149.8 → 151.4), and **greedy wins fair under it as under soft and hard** (table),
so the approximation is *not* what produces the contingency result — if anything
the proper rule favours the minimal-query baseline. We state this rather than have
it discovered: the deployed rule is an acknowledged, directionally-understood
approximation; the code migration is tracked in issue #111. (Detail:
`docs/paper1/b4-credit-rule-verdict.md`.)

## Pillar (iii) — why myopic VOI under-explores (the per-category mechanism)

Per-category, `bayesian_inferred` (myopic VOI) vs `greedy_inferred` (★ = best
tool; net/Q after cost):

| category | best tool | VOI acc / net (top tool) | greedy acc / net (top tool) | Δ(VOI−grdy) |
|---|---|---:|---:|---:|
| factual | ★KB (c2) | 70.7% / +4.11 (web) | 84.0% / +5.84 (KB) | −1.73 |
| misconceptions | ★KB (c2) | 38.6% / −0.52 (calc) | 57.9% / +1.99 (KB) | −2.51 |
| recent_events | ★web (c1) | 45.6% / +1.09 (web) | 55.6% / +1.98 (web) | −0.88 |
| reasoning | ★llm (c2) | 42.5% / +0.12 (calc) | 49.0% / +0.75 (KB) | −0.62 |
| **numerical** | ★calc (c1) | 69.0% / **+4.24** (calc) | 60.5% / +2.48 (KB) | **+1.76** |

Myopic VOI in `agent-step` is single-step *and single-question* — it values
information only for the current decision, never future ones. With Beta(1,1)
priors every tool's net-VOI is `2.5 − cost`, so it economises onto cheap tools and
**never pays to discover the expensive-but-reliable tool** (knowledge_base
query-rate on KB-best categories crawls 4%→31% over a seed, vs greedy's optimistic
25%→66%; `scripts/paper1-voi-diagnostic.py`). It wins only where the best tool is
both cheap and dominant by a wide margin (numerical: calculator is cost-1 and
1.00-reliable). This is the under-exploration horizon-aware VOI fixes (Pillar ii),
and the mix-crossover (VOI overtakes greedy only above ~40% numerical share;
actual 20%) is the mix-weighted projection of this per-category law.

## Oracle skyline — price-of-inference reference

| Agent | Score | Acc% | Tool/Q |
|---|---:|---:|---:|
| **horizon (oracle)** | **+216.8** | 74.0 | 1.19 |
| ablation_greedy (oracle) | +189.4 | 68.6 | 1.50 |
| bayesian (oracle myopic VOI) | +163.7 | 62.0 | 1.25 |
| ablation_no_abstain (oracle) | +156.9 | 62.6 | 1.25 |
| ablation_no_voi (oracle) | +38.2 | 77.9 | 6.00 |
| ablation_no_learning (oracle) | +3.8 | 40.5 | 1.00 |

## Reproducibility

All offline from committed fixtures + the (git-ignored) DB:

- Host `--seeds 20` (oracle `bayesian`/`horizon` + baselines + inferred roster);
  `--ablation` for the oracle greedy/skyline.
- `scripts/paper1-horizon-gate.jl` — exact decoupled ceiling (211.8) + single-query
  optimum (180.8) + matched-prior control.
- `scripts/paper1-horizon-depth.jl` — depth-d capture (depth-1 = 217.6).
- `scripts/paper1-horizon-inferred.jl` — inferred result + four-way de-confound +
  the soft/hard/post credit-rule table.
- `scripts/paper1-pareto.py` — frontier, £/point sweep, per-category law,
  mix-crossover, paired bootstrap. `scripts/paper1-voi-diagnostic.py` — temporal
  under-exploration evidence. `scripts/paper1-pairing-gate.jl` — LLM-reuse validity.

```sql
SELECT agent, ROUND(AVG(total_score),1) score, ROUND(AVG(total_reward),1) reward,
       ROUND(AVG(total_tool_cost),2) toolcost FROM runs GROUP BY agent ORDER BY score DESC;
```
