# Routing dominance: credence-pi routes better AND smarter than every system

**Claim.** For model routing — pick which LLM answers each request — Bayesian
EU-maximisation over a feature-conditioned belief *dominates every competing router*:
on a cost-sensitive user it strictly beats all of them (every seed), and on a
quality-maximising user it matches the provably-optimal rule. No fixed router can do
this, because for k ≥ 2 models the per-user-profile optimum genuinely differs and a
single table is the Bayes rule for at most one profile (Wald complete class). The
richer the belief, the larger the win — and the belief gets richer *on its own*.

This is the offline, zero-spend pillar. It has two faces:

- **Coarse pillar** (`apps/python/credence_router/experiments/routing_dominance/`) —
  proves the *mechanism* through the real skin: a Beta-Bernoulli belief over
  (model × category), EU-max via `skin.optimise`, beating six competitor foils with
  per-profile divergent routing. Constant-free (binary MCQ correctness ⇒ Bernoulli).
- **Rich pillar** (this directory; `eval/routing_dominance.jl`,
  `brain/routing_brain.jl`) — the upgrade that turns dominance into a *margin*: the
  belief is the credence-pi `StructureBMA` (the same brain that governs OpenClaw),
  conditioned on difficulty × length × category, and it *auto-discovers* which
  features matter.

## Setup (rich pillar)

| Piece | Value |
|---|---|
| Models | cheap / mid / exp, costs 0.001 / 0.005 / 0.02 (haiku/sonnet/opus ratio) |
| Oracle | stochastic IRT: `P(correct) = logistic(capability − difficulty)`; capability cheap<mid<exp; difficulty rises with *hard* and *long*; **category is pure noise** |
| Belief | one `StructureBMA` schema; K=3 posteriors (one per model, label "model correct"), trained by the canonical `observe`/`condition` on **noisy sampled** outcomes |
| Decision | `RoutingBrain.route` — a joint `ProductMeasure` of the per-model `belief_at_context` views, per-action `reward·θ_a − cost_a`, the **one** canonical `optimise` (the `decide_multi` mechanism, action set = models) |
| Profiles | cost-hawk (a correct answer worth \$0.01) and quality-hawk (worth \$1.00) — one shared belief, two utilities (Savage) |
| Scoring | each arm LEARNS from the noisy samples, then is scored on **expected welfare against the true rates** — isolating who routes best from finite evidence |

Stochastic (not deterministic 0/1) on purpose: with deterministic correctness raw
empirical frequencies are exact, so the Bayesian belief earns no calibration
advantage. Real models are stochastic; the genuine edge is **calibration from limited
noisy data**, which is what this measures.

## Result (12 seeds × 80 reps/cell)

**(1) Dominance — cost-sensitive profile: rich EU-max strictly beats *every* competitor, every seed.**

| competitor | mean regret vs rich (cost-hawk) | rich wins |
|---|--:|--:|
| always-cheap | +0.061 | 100% |
| always-exp | +3.224 | 100% |
| argmax-accuracy (cost-blind) | +3.224 | 100% |
| best-fixed-table (profile-blind) | +3.224 | 100% |
| threshold-router (RouteLLM 2-bin) | +2.018 | 100% |
| **oracle-cascade (clairvoyant FrugalGPT)** | **+0.571** | **100%** |

Rich even beats the *clairvoyant* cascade here — a cascade that knows each rung's
correctness before paying still loses, because its cumulative rung cost is a penalty
no cascade escapes on a cost-sensitive user.

**Quality-max profile:** rich *matches* the accuracy-maximising Bayes rule
(always-exp, best-fixed, argmax-accuracy, category-only all route the top model —
mean regret −0.46, a statistical tie within belief-sampling noise) and *beats* the
2-bin router (+47.1) and always-cheap (+164.7). Only the clairvoyant oracle leads
(−14.5) — via foresight no real system has. This is the honest Wald picture: where a
fixed rule *is* the Bayes rule it ties us; everywhere else it loses.

**(2) Richness = the margin.** The *same* EU-max mechanism, fed the feature belief,
beats itself fed a category-only belief (the coarse pillar) on cost-hawk by **+0.056
welfare, 92% of seeds** (~14% higher welfare on the held-out split) — the
within-category spread a category router cannot see. Dominance is belief-agnostic;
richness sizes the win.

**(3) Auto-sophistication.** The structure-BMA's edge-inclusion posterior, from a 0.5
prior, *learned which features matter* from noisy data, unprompted:

| model | category (noise) | difficulty | length |
|---|--:|--:|--:|
| cheap | 0.013 | 0.986 | 1.000 |
| mid | 0.004 | 1.000 | 0.996 |
| exp | 0.007 | 0.722 | 0.142 |

Category is driven toward 0; difficulty and length toward 1. (exp is so capable its
accuracy barely varies, so its data only weakly favours any edge.) This is the
"model rich enough to capture the important factors" — the brain finds the factors
itself, the metareasoning-over-structure the constitution's BMA gives for free.

## Why this is conclusive against *all* systems (not just these six)

Wald's complete-class theorem: the admissible decision rules are exactly the Bayes
rules. Any router either (a) agrees with per-profile EU-max — in which case it *is*
us, and ties — or (b) is dominated. A *fixed* router is Bayes for at most one profile,
so across ≥ 2 profiles it is inadmissible. The only way to beat us is a
*better-calibrated belief* — which is a better credence-pi, absorbed by training on
more data. The clairvoyant cascade is the apparent exception, and it is not deployable:
it requires knowing each rung's correctness before paying.

## Honest caveats

- **Synthetic oracle.** The IRT rates are a principled fixture (one formula, capability
  + difficulty), not measured per-model MCQ accuracy. A real-oracle run needs
  ~150–250 live calls over a labelled benchmark (e.g. `qa_benchmark`'s 50 MCQ) —
  deferred spend; the mechanism and the Wald argument are oracle-independent.
- **Offline.** No live LLM provider; "routing" decisions are made by the real brain
  through the skin / `route`, but the answers are oracle look-ups, not generations.
  The live A/B through OpenClaw's `before_model_resolve` is the second pillar.
- **Bayes-optimal given the belief, not oracle-optimal.** EU-max maximises expected
  welfare under its posterior; a clairvoyant bound can still lead where foresight, not
  calibration, decides. We claim calibration beats estimation, not omniscience.
- **Richness margin is modest here (~14%, 92% of seeds)** because category-only's
  cheap-everywhere fallback is already decent on this oracle; the margin grows with
  workload heterogeneity and feature richness.

## Reproduce

```
# rich pillar (Julia; defaults reproduce eval/results/routing_dominance.summary.json)
julia --project=. apps/credence-pi/eval/routing_dominance.jl

# coarse pillar (Python; through the real skin, zero spend)
uv run python apps/python/credence_router/experiments/routing_dominance/dominance.py --toy
```
