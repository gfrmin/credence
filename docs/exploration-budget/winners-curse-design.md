# Winner's-curse pricing — selection-aware valuation of growth-op lookaheads

> Exploration-budget arc, follow-up to removal consumption (#192/#193) and the §8 measure–utility
> alignment (#194). Design-doc-before-code; ratify before any code lands. Master plan:
> `docs/exploration-budget/master-plan.md`. Constitutional grounding: `CONSTITUTION.md`
> (Tractatus Credentiae, 2nd ed.), cited inline as T-x.y. The acceptance criteria for this move
> were **fixed in advance** by dominance-design §8 (the measure–utility amendment): worst-seed
> (minimax) mean-rate gap ≥ 0 and the q10 gap vs `random_p005` and `fixed_k50`, with mean-parity
> CIs and the passing headline preserved. Authored 2026-07-04.

---

## 1. Purpose

Price the **cross-candidate selection bias** in growth-op valuation, so that the score a growth op
(`:gw_explore`, `:gw_add_feature`) carries into the meta-action argmax is an honest posterior
expectation of its future per-event gain — not the raw window fit of the argmax candidate
extrapolated to the full horizon. This is the round-4 gate's named remaining failure: the tails
(worst-seed mean-rate gap −0.5 / −0.381, q10 −0.476 / −0.333 vs the tuned baselines) are driven by
**early growth fires on tiny windows** (the instrumented shape: `add_feature` at steps 4–8), while
the means are already statistical ties and the headline passes.

**The bug, precisely located.** The growth score is (belief-derived-valuation §2a):

    score = plateau · (fit / n_buf) · H + prior_term − compute_cost

where `fit` is the **window-realised** Δℓ of the **argmax candidate** — `_best_threshold_refinement`
evaluates every midpoint candidate's marginal-log-loss reduction over the buffer and takes the max
(`exploration.jl`); `_best_feature_addition` does the same over the host-furnished feature set.
Two inferences in that formula are performed implicitly, with certainty, and are wrong exactly when
the window is small:

1. **Extrapolation.** `fit/n_buf` — a realised quantity over as few as 2–6 conditioning events — is
   treated as the *known* future per-event rate and multiplied by `H ≈ 200`. The window fit is a
   genuine log Bayes factor (the mll's parameter integration is the Bayesian Occam that handles
   *per-candidate* overfit), but a Bayes factor over n events is *evidence about* a rate, not the
   rate. Treating it as the rate asserts knowledge the agent does not have — a T-1.3 violation (the
   prior is the honest statement of ignorance) hiding in a multiplication.
2. **Selection.** The scored fit is an **order statistic**: the max over K candidates (tens of
   threshold midpoints; K ≈ 5 features) of noisy window fits. `E[max of K noisy fits] > max of K
   true rates` — on a window where *every* candidate chance-fits (n = 3–4 binary outcomes are
   perfectly separated by many splits), the argmax fit is large **because it was selected**, not
   because its candidate generalises. The mll's Occam prices each candidate's fit *given that
   candidate*; nothing prices the argmax *across* candidates.

Both errors scale as `H/n_buf` — negligible at n ≈ 30, catastrophic at n ≈ 4 with H ≈ 200 (the
multiplier is ~50×). This is why the pathology is invisible in the means (most fires are
mid-episode, informed) and owns the tails (a handful of seeds draw early encounter sequences that
make some candidate look separating).

**Why this is the reflective boundary again, and why "just compute the exact value" is degenerate.**
The tempting exact fix — score growth by the expected next-event predictive gain of the enlarged
belief — has a fixed sign by construction: under the union's own predictive the expected log-ratio
is `KL(P_union ‖ P_incumbent) ≥ 0` (growth always "helps"); under the incumbent's it is
`−KL(P_inc ‖ P_union) ≤ 0` (growth never does). The value of hypotheses you do not yet entertain is
not computable from the belief that excludes them — the same boundary that forced the escape ops
onto **learned** returns (belief-derived-valuation §2b). The window-realised fit is the right
*evidence* (it is data, not belief); the missing piece is an honest **inference** from that
evidence to the future rate — a posterior, not a point (T-2.81: derive it; T-3.53: this is a
fidelity refinement inside the one Δ log-evidence currency, not a new currency).

**The mechanism (recommended; alternatives and their defects in §5 Q1).** A two-group
(spike-and-slab) posterior over the winner's true per-event rate, conditioned on the **full
candidate ensemble**:

- Each candidate c has a latent rate `g_c ≥ 0`. Model: `g_c = 0` ("null" — the window fit is
  chance) with probability π, or `g_c > 0` ("real") with probability 1 − π.
- The K window fits `{f_1..f_K}` — **already computed** by the existing argmax loop, currently
  discarded except for the max — are the observations. The non-winning candidates, mostly null,
  calibrate what chance-level fit looks like *at this (n, grammar, buffer)*: the ensemble IS the
  null's scale, data-derived, no declared noise constant.
- The score's fit input becomes the posterior expectation `E[g* | f_1..f_K] · n_buf` (so
  `growth_value`'s interface is unchanged): spike-dominated ⇒ ≈ 0 ⇒ no fire; slab-dominated ⇒
  ≈ the raw fit ⇒ the current score. π is marginalised under a uniform prior (honest indifference
  — no baked constant; `decision-free-combinator` discipline).

Properties that make this the recommendation: **(i)** it fixes the *first* fire — the tail seeds
fire at steps 4–8, before any learned calibration could leave its prior (§5 Q1's alternative (c)
fails exactly there); **(ii)** it prices selection at the point where selection happens, using
numbers the argmax already computed (near-zero added compute); **(iii)** it converges to the
current score as evidence accumulates — a genuinely separating candidate's fit escapes the
ensemble's null scale, the slab takes the posterior, and the shrinkage vanishes (the current score
is the asymptotic limit of the honest one, not a competing rule); **(iv)** every step is
declared-model conditioning (Invariant 1) — the multiple-comparisons correction arrives as
posterior arithmetic, not a frequentist adjustment.

**What this move does NOT touch (resolved, recorded here).**

- The escape ops (`:gw_enumerate_more`, `:gw_deepen`): already priced by learned conjugate
  posteriors whose optimism decays under evidence — no raw-extrapolation term exists there.
- Perturb (`:gw_perturb_grammar`): exact realised Δ log-evidence (#193) — deterministic
  consequence, no extrapolation, no selection over noisy estimates (the candidate payoffs are
  description-length facts).
- The `plateau` factor: it prices *whether the regime still improves* (Move 2's residual-plateau
  posterior); this move prices *whether the measured fit is real*. Orthogonal—one is about the
  stream, the other about the candidate—no double count (§5 Q3 argues this explicitly).
- `growth_value`'s linear form and the §8 gate measures: unchanged. The fix is upstream of the
  valuation functional, in what "fit" honestly is.

## 2. Files touched

- **`src/program_space/selection_pricing.jl`** (new, ~70 lines). The two-group posterior over the
  argmax candidate's rate:
  - `selection_priced_fit(fits::Vector{Float64}, n_buf::Int) → Float64` — the posterior-expected
    window fit `E[g* | ensemble] · n_buf` of the argmax candidate given all K window fits. Pure,
    deterministic, no state. The declared model (exact form §5 Q1; the recommended (a) variant):
    null fits exchangeable around the ensemble's typical scale; slab carrying the winner's excess;
    π marginalised uniform. Total on any input (empty/singleton ensembles degrade honestly:
    K = 1 ⇒ no ensemble evidence ⇒ the π-marginal alone — no fabricated certainty either way).
  - Docstring carries the derivation and names the test (executable-documentation discipline).
- **`src/program_space/exploration.jl`** (modification, ~25 lines net).
  - `_best_threshold_refinement`: the candidate loop already computes every `fit`; collect them
    (currently discarded), select the argmax as today (selection by raw fit — the *decision* of
    which candidate is unchanged; only its *valuation* is priced), then
    `best_fit_priced = selection_priced_fit(fits, n)` feeds `growth_value`. The three projections
    (grammar / value / fit) keep one shared computation (Invariant 3); the `fit` projection
    returns the PRICED fit so host memoisation stays coherent (the cacheable pure component is
    still pure in (grammar, buffer, depth)).
  - `_best_feature_addition`: same change over the feature ensemble.
- **`apps/julia/grid_world/host.jl`**: **no change.** `exploration_fit`/`feature_discovery_fit`
  signatures and the score seam are untouched; the host's cached fit is simply the priced fit.
- **`test/test_selection_pricing.jl`** (new, ~5 sections): §1 the tiny-window chance ensemble —
  K fits at chance scale, winner typical ⇒ priced fit ≈ 0 (below the fire floor at H = 200; the
  seed-6 regression, exact fixture from §4); §2 the separated ensemble — one fit far above K−1
  chance fits ⇒ priced fit within a stated tolerance of the raw fit (slab-dominated); §3
  asymptotic convergence — fixed true separation, growing n ⇒ priced → raw monotonically; §4
  degenerate ensembles (K = 1, all-zero fits, empty) — total, honest, no throw; §5 determinism +
  the score/decision split (argmax candidate unchanged by pricing; only the value moves).
- **`test/test_threshold_explore.jl`, `test/test_feature_discovery.jl`,
  `test/test_grid_world_meta.jl`** (re-baselines): pins that assert exact small-window voi values
  change where the priced fit differs from the raw fit; each re-baselined pin is annotated with
  this design's rationale. The `fit > 0`-gate assertions and all structural behaviour unchanged.
  (The master plan predates the dominance rounds and has no winner's-curse entry to update; this
  doc and the §8 amendment are the frontier's record.)

## 3. Behaviour preserved

- **Selection identity:** which candidate a growth op applies is bit-identical (argmax by raw fit,
  unchanged) — only the op's *score* moves. `explore_grammar`/`explore_features` return the same
  grammar whenever they fire; what changes is *whether* the meta-argmax lets them fire on
  tiny-window evidence.
- **Asymptotic identity:** for ensembles where the winner's fit dominates the null scale (the
  informed-fire regime), the priced fit equals the raw fit within the §5 Q1 model's stated
  tolerance; `test_selection_pricing.jl` §2–3 pin this quantitatively. The current score is the
  limit, so mid-episode behaviour is expected to be near-identical (verified empirically by the
  gate re-run — mean-tier metrics should be statistically unmoved).
- All non-growth scoring (escape cells, perturb replacement value, do-nothing floor): bit-stable;
  existing suites (`test_growth_returns.jl`, `test_replacement.jl`, conjugate/threshold suites)
  must pass untouched.
- The skin wire and persistence surfaces: untouched (no schema, no protocol change).
- The dominance benchmark harness: untouched (measures fixed by §8 — this move is judged against
  them, it does not get to move them).

## 4. Worked end-to-end example

The tail-seed shape (seed 6's instrumented fire: `add_feature` at step 6, final rate gap −0.381).

1. **Step 6, buffer n = 4.** Two interactions and two adjacent observations so far; regime 1
   (colour-typed). `fit_explore = 0` (thresholds exhausted on the tiny window) ⇒ the
   `:gw_add_feature` gate opens. `_best_feature_addition` evaluates K = 5 candidate features.
   On 4 events, *every* candidate's refreshed program space contains splits consistent with the
   window: the five window fits come back at chance scale, say `{1.1, 0.9, 1.3, 1.0, 1.2}` nats
   (illustrative magnitudes; the fixture in `test_selection_pricing.jl` §1 uses exact captured
   values). Today: `fit = 1.3`, `H ≈ 195`, `plateau ≈ 0.5` ⇒
   `score ≈ 0.5 · (1.3/4) · 195 − log 2 − log 2 ≈ 30 nats` — fires, installs a wrong feature,
   injects ~80 diluted components, and the seed never recovers the lost ground (rate gap −0.381).
2. **Priced:** the winner's 1.3 sits inside an ensemble whose four other members — necessarily
   mostly null — span 0.9–1.2: the winner's excess over the ensemble's chance scale is ~0.1–0.2
   nats, indistinguishable from selection. The two-group posterior puts the bulk of `g*`'s mass on
   the spike; `E[g* | ensemble] · n ≈ 0.1` nat ⇒ `score ≈ 0.5 · (0.1/4) · 195 − 2·log 2 ≈ 1.0`
   nats ≈ the fire floor — and the *decision* margin against `do_nothing` collapses from ~30 nats
   of false certainty to noise. The op waits.
3. **Step ~90, buffer n = 30 (regime 2, `:speed` genuinely predicts).** The ensemble comes back
   shaped like `{6.2, 0.4, 0.7, 0.3, 0.5}`: the winner's fit is ~10× the null scale — no chance
   assignment of 30 outcomes produces it. The slab takes the posterior;
   `E[g* | ensemble] · n ≈ 6.0` ⇒ the score is the current score (the informed fire is preserved,
   within tolerance). The behaviour-verified inversion at step ~90 (the one that drives the
   headline's 19–1) is untouched.
4. Ownership: the ensemble is collected where it is already computed (`exploration.jl`, the argmax
   loops); the posterior arithmetic lives in `selection_pricing.jl`; the valuation functional
   (`growth_value`) and the host seam are unchanged.

## 5. Open design questions

1. **The observation model inside `selection_priced_fit`** — what exactly is declared for
   `f_c | g_c`, and where does the null's scale come from? Three candidates:
   **(a) Ensemble-empirical null (recommended):** null fits exchangeable with scale read from the
   ensemble itself (the K−1 losers are the null sample); slab = the winner's excess above it; π
   marginalised uniform. Cheapest (numbers already computed), data-derived (no declared noise
   constant — `decision-free-combinator` clean), and self-sharpening (bigger K ⇒ better null).
   Weakness to argue: at K = 2–3 the null sample is thin — the doc recommends the honest-degrade
   (posterior widens, shrinkage strengthens — small K *should* shrink harder, since a thin
   ensemble cannot certify the winner).
   **(b) Exact small-n permutation null:** the null distribution of window fit is exactly
   computable by deterministic enumeration of label reassignments weighted by the incumbent
   predictive (2ⁿ terms — affordable precisely when n is small, which is precisely when the
   correction matters; fidelity-switch to (a) above a declared cap, T-3.53). Exact and beautiful
   but a new enumeration machine (~150 lines) for a correction (a) approximates well; deferred
   unless (a)'s gate re-run under-shrinks.
   **(c) Learned realisation-ratio calibration** (the GrowthReturns pattern — cells over
   realised-vs-promised, the "realised next-window predictive delta" finer fidelity named in
   `growth_returns.jl`): REJECTED as the primary — the tails are **first** fires (steps 4–8), and
   a calibration cell sits at its prior exactly then; it cannot price a fire it has never seen.
   Named as the follow-on fidelity once fires accumulate (it prices whatever (a)'s model misses).
2. **Does the priced fit feed the ratchet gates as well as the score?** `fit_explore > 0`
   currently hard-gates `:gw_add_feature` (attribution). Pricing changes `fit`'s value, so the
   gate would consult the priced value. Recommended: **yes, one fit** — two fits (raw for gates,
   priced for scores) is exactly the score/transition split T-3.55 forbids, and the gate's
   attribution argument (a feature Δℓ measured against a coarse grid is confounded) applies to
   real fit, which is what the priced value estimates. Counter-position to argue: the gate is a
   measurement-ordering device, not a valuation, so raw is defensible — but then two numbers named
   `fit` circulate, and Invariant 3 says no.
3. **Interaction with `plateau` — is shrinkage double-counted?** No, and the doc wants this
   ratified explicitly: `plateau` is P(the residual stream still improves) — a property of the
   *learning trajectory*; the selection price is P(the winner's measured fit is real) — a property
   of the *candidate ensemble*. A plateaued stream with a genuinely separating new feature has
   plateau ≈ 1 AND slab-dominated pricing (both pass); a noisy early stream has plateau < 1 AND
   spike-dominated pricing — the factors multiply because the failure modes are independent, not
   because the same doubt is charged twice. If ratification disagrees, the alternative is to fold
   plateau into the slab prior — one mechanism, more entangled; not recommended.

## 6. Risk + mitigation

- **Under-exploration (over-shrink).** The failure the bracket detects: eu_max's mean rate falls
  toward never_explore's 0.274 because genuine early discoveries are also deferred. Mitigation:
  the asymptotic-identity pins (§3) bound the informed-fire regime; the gate re-run's headline
  assertion (rate CI vs never_explore > 0) is the acceptance-side tripwire; halt-the-line if the
  headline degrades.
- **The ensemble is not null-dominated** (K small AND several real candidates — e.g. two genuinely
  predictive features at once). (a)'s null scale is then inflated by real fits ⇒ over-shrink of
  the winner. Mitigation: the two-group model marginalises π rather than fixing it (a
  several-real ensemble drags π's posterior down, re-crediting the winner); §2 of the test pins a
  two-real-candidates fixture. Residual risk accepted and named — the follow-on fidelity (Q1(c))
  catches what the model misses, from data.
- **Score/decision drift.** The argmax candidate is selected by raw fit but valued by priced fit —
  deliberate (selection among candidates is invariant to the candidate-uniform pricing transform;
  valuation against OTHER ops is not), but it must stay one computation: the pricing consumes the
  same fits vector the argmax consumed, in the same call (`test_selection_pricing.jl` §5 pins the
  split). No re-derivation at the host.
- **Cache coherence.** Hosts memoise `exploration_fit` — the priced fit must be what's cached (it
  is: the projection returns priced), and it remains pure in (grammar, buffer, depth) since the
  ensemble is a deterministic function of those. No epoch semantics change.
- **Benchmark comparability.** Only eu_max (and clairvoyant's fallback path) read scores;
  score-blind baselines unchanged. Per-policy deltas vs round 4 are attributable to the pricing —
  cleaner than #193's all-policies transition change. Fresh results commit either way.

## 7. Verification cadence

    julia test/test_selection_pricing.jl        # new — chance-ensemble collapse, separation
                                                #   fidelity, asymptotic convergence, degeneracy
    julia test/test_threshold_explore.jl        # re-baselined pins
    julia test/test_feature_discovery.jl        # re-baselined pins
    julia test/test_grid_world_meta.jl          # re-baselined §1/§3/§4 score pins
    julia test/test_growth_returns.jl           # untouched pass
    julia test/test_replacement.jl              # untouched pass
    for f in test/test_*.jl; do julia "$f"; done
    JULIA_PROJECT=$PWD PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run python apps/skin/test_skin.py

Gate re-run (20 seeds, background), judged against the §8 measure set **fixed before this design**:

- **The target claims (what this move must fix):** minimax worst-seed mean-rate gap vs
  `random_p005` and `fixed_k50` rises from −0.5/−0.381 toward ≥ 0; q10 from −0.476/−0.333 toward
  ≥ 0. Falsifiable mechanism claim: **no eu_max growth fire before ~step 15 on any seed** whose
  ensemble was chance-shaped (checkable from the instrumented op log + the new per-fire priced/raw
  fit pair, which the run logs).
- **The must-not-degrade claims:** the headline (rate +0.150 CI > 0, 19–1) holds; the mean-tier
  CIs vs the tuned baselines do not become significantly negative; the bracket holds.
- Halt-the-line on any degrade; results commit honestly either way.
