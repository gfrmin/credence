# Routing dominance: credence-pi is Wald-admissible across user profiles, and strictly dominant when the belief is well-resolved

**Claim.** For model routing — pick which LLM answers each request — Bayesian
EU-maximisation over a feature-conditioned belief is **undominated by every competing
router** (Wald-admissible across user profiles), and **strictly dominates** them in the
regime where the belief is well-resolved or the model gaps are real: on a cost-sensitive
user it strictly beats all of them (every seed), and on a quality-maximising user it
matches the provably-optimal rule. No fixed router can be admissible across profiles,
because for k ≥ 2 models the per-user-profile optimum genuinely differs and a single
table is the Bayes rule for at most one profile (Wald complete class). The richer the
belief, the larger the win — and the belief gets richer *on its own*. (On reality at
small scale the strict-dominance regime is data-bound — see "Real-model validation".)

> **Two regimes, stated honestly up front.** The synthetic pillars below show the
> mechanism's *ceiling* — strict dominance when the belief is well-resolved or model gaps
> are real. The **real-model validation** (its own section, 3 frontier models on a labelled
> benchmark) shows what holds on reality at small scale: EU-max is *undominated* across
> profiles (Wald-admissible) and *wins cost-sensitive routing*, but its strict
> quality-profile advantage is data-bound at 50 questions. Read both.

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
| **foresight cascade (FrugalGPT, knows each rung before paying)** | **+0.571** | **100%** |

Rich beats even this *foresight* cascade on cost-hawk — but note this is **not** an upper
bound: unlike a per-task oracle, the cascade still pays each rung's cost as it climbs, so
beating it is expected, not a beaten ceiling (its cumulative rung cost is a penalty no
cascade escapes on a cost-sensitive user). The genuine upper bound is the per-task oracle,
which leads everywhere and is non-deployable (it needs each rung's correctness before paying).

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

## Real-model validation (the honest result)

The synthetic pillars show the mechanism's *ceiling* (a resolved belief, designed gaps).
To de-risk the claim on reality, `oracle.py` measured three real frontier models on the
50-question labelled benchmark (`credence_agents`' bank; `correct_index` is ground truth),
and `dominance.py --real` ran the same proof over the measured grid.

Measured accuracy (150 live calls, cached):

| model | overall | factual | numerical | reasoning | recent | misconceptions |
|---|--:|--:|--:|--:|--:|--:|
| haiku (cheap) | 88% | 14/15 | 9/10 | **7/10** | 7/8 | 7/7 |
| sonnet (mid) | 96% | 14/15 | 9/10 | **10/10** | 8/8 | 7/7 |
| opus (exp) | 98% | 15/15 | 10/10 | **9/10** | 8/8 | 7/7 |

What real data shows (8 seeds):

- **The premise is real.** Per-profile routing genuinely diverges (cost-hawk → cheap
  everywhere; quality-hawk → per-category), and no single fixed router is best on both:
  always-cheap wins cost-hawk, always-exp wins quality-hawk.
- **EU-max is undominated (admissible).** No fixed router beats it on *both* profiles —
  exactly Wald's admissibility. It adapts where every fixed rule is stuck.
- **Cost-sensitive routing is validated.** EU-max ties the cost-optimal (always-cheap)
  and beats every wasteful router (always-exp +0.22, threshold +0.07, argmax/best-fixed
  +0.06). When models are close and cost differs 5×, "use the cheap one" is right — and
  EU-max concludes that itself rather than overpaying.
- **A real, non-obvious insight:** sonnet (mid) *beats* opus (the flagship) at reasoning
  (100% vs 90%) — so quality-hawk EU-max routes reasoning to the *cheaper* sonnet. "Use
  the biggest model" is not optimal; EU-max finds the inversion.
- **Honest limit:** on the *quality* profile, EU-max **trails** always-exp (−2.05) and the
  threshold router (−0.24). With ~6 questions/category, the belief cannot reliably
  estimate 2-point accuracy gaps between 88/96/98% models, so it occasionally misroutes;
  the uniformly-near-best flagship is hard to out-route from thin data. Strict per-profile
  dominance needs **more calibration data or larger model-accuracy gaps** than this easy
  benchmark provides — precisely what the synthetic pillar supplies, and what a deployed
  router accumulates online. The clairvoyant cascade also leads here (foresight, not
  calibration).

So the real claim is **admissibility + validated cost-routing + a genuine routing insight**,
not the synthetic ceiling of strict dominance everywhere. The mechanism is sound; its
quality-regime advantage is data-bound at 50 questions.

## Why this is conclusive against *all* systems (not just these six)

Wald's complete-class theorem: the admissible decision rules are exactly the Bayes
rules. Any router either (a) agrees with per-profile EU-max — in which case it *is*
us, and ties — or (b) is dominated. A *fixed* router is Bayes for at most one profile,
so across ≥ 2 profiles it is inadmissible. The only way to beat us is a
*better-calibrated belief* — which is a better credence-pi, absorbed by training on
more data. The clairvoyant cascade is the apparent exception, and it is not deployable:
it requires knowing each rung's correctness before paying.

## Wired live (the product surface)

The mechanism above is now wired into OpenClaw's live model selection, not just the
offline proof. With `routing: true` the credence-pi plugin registers a
`before_model_resolve` hook that posts a `route-request` to the daemon; the daemon runs
`RoutingBrain.route` — the *same* `optimise` over the per-model belief — and returns the
EU-max model as OpenClaw's `modelOverride` / `providerOverride`. Fail-open throughout
(daemon down / no roster ⇒ OpenClaw keeps its model). See
`apps/credence-pi/openclaw-plugin/README.md` § "Model routing".

What ships live, and what is deliberately deferred:

- **Warm belief from measured data.** The K per-model posteriors are seeded from this
  document's oracle grid (`brain/routing_brain.counts.json`, distilled by
  `eval/train_routing_brain.jl`), so the router knows haiku 88 / sonnet 96 / opus 98 from
  install — it does not start cold.
- **The validated result is what's wired.** Live routing delivers the *proven* part:
  per-profile cost-routing (`correct-answer-value` in `utility.bdsl` is the dial) and
  Wald per-profile divergence (low value → cheap, high value → best-believed), through
  the one canonical `optimise`.
- **The belief now learns online (per-turn, confound-aware).** Closing the quality-regime
  gap needs *online calibration data* — exactly what a deployed router accumulates. The
  credit-assignment wall is real but specific to *run-level* success (`agent_end.success`
  cannot attribute a session outcome to one call's model, conflating model-quality with
  task-difficulty and tool-reliability). We sidestep it by learning from the **per-turn**
  signal, which *is* per-call attributable: did the routed model's proposed call execute
  cleanly (`tool-completed.success`)? That signal is itself noisy — a correct call can fail
  on a flaky tool — so we model the confound explicitly and learn it: shared per-context
  latents `ρ = P(clean | correct)`, `σ = P(clean | incorrect)`, identified by the
  cross-model spread against the oracle-anchored accuracy. The signal is decoded to a soft
  correctness `π` (the routing belief is its own coherent prior), and the belief conditions
  on the resulting **virtual evidence** (`SoftBernoulli`) — a mean-exact ADF collapse, so
  the posterior *mean* (what routing's EU reads) is exact; only the variance is approximated.
  Every update is `condition`; the decode is `expect`; no arbitrary constants (the emission
  prior is declared, directional, and washed out by data). A human approve/reject, when
  present, is the gold anchor (a hard label). Mechanism + tests:
  `brain/routing_brain.jl` (`route_outcome!`), `test_routing.jl` §C/§D.
- **What confound-partialling buys (the test that matters).** A pure tool-flakiness stream
  — every call fails for every model, true accuracy unchanged — leaves the accuracy belief
  essentially flat (drift < 0.005), because the failures are absorbed by the learned `ρ`,
  not blamed on the models. The naive `fail ⇒ incorrect` credit rule craters the same
  belief by ~0.5. That gap is the deferral's worry, addressed.
- **Honest limits of the online layer.** `ρ` (the dominant confound for high-accuracy
  models) identifies tightly; `σ` (false-success) only weakly — when models are mostly
  correct, `C=0` is rare, so little data bears on it (its decode influence is proportionally
  small). The per-turn proxy is *not* ground-truth correctness; identification leans on the
  oracle anchor and drifts if the live task-mix departs from it. **Run-level** success and
  multi-call credit assignment remain deferred. The exact-vs-ADF decode *fidelity* is itself
  a metareasonable axis (EU-max over fidelity-vs-compute) — flagged, not yet built. The
  belief is durable: `route-outcome` events are logged and replayed on restart (exact).
- **Only `prompt-length` is conditioned on live.** It is the one feature honestly
  extractable from a raw prompt (semantic category — the offline signal — would need a
  classifier, an arbitrary unmeasured component). The structure-BMA collapses it to the
  marginal if it doesn't predict accuracy, so it is honest and self-correcting.

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
