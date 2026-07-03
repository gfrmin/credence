# Winner's-curse pricing — growth valuation as Bayesian model comparison

> Exploration-budget arc, follow-up to removal consumption (#192/#193) and the §8 measure–utility
> alignment (#194). Design-doc-before-code; ratify before any code lands. Constitutional
> grounding: `CONSTITUTION.md` (Tractatus Credentiae, 2nd ed.), cited inline as T-x.y. The
> acceptance criteria were **fixed in advance** by dominance-design §8: worst-seed (minimax)
> mean-rate gap ≥ 0 and the q10 gap vs `random_p005` and `fixed_k50`, with mean-parity CIs and
> the passing headline preserved. Authored 2026-07-04; **revised 2026-07-04 after constitutional
> review** — the first draft's spike-and-slab mechanism is REJECTED and recorded as such in §5
> Q1; the revision derives the score from the machinery the agent already has.

---

## 1. Purpose

Replace the growth ops' point-max-extrapolation score with the **Bayesian model comparison the
agent already performs everywhere else**, so that `:gw_explore` / `:gw_add_feature` carry an
honest posterior expectation into the meta-argmax. This is the round-4 gate's named remaining
failure: the tails (worst-seed mean-rate gap −0.5 / −0.381, q10 −0.476 / −0.333 vs the tuned
baselines) are early growth fires on tiny windows (the instrumented shape: `add_feature` at steps
4–8), while the means are ties and the headline passes.

**The bug, precisely located.** The growth score is (belief-derived-valuation §2a):

    score = plateau · (fit / n_buf) · H + prior_term − compute_cost

where `fit` is the **window-realised Δℓ of the argmax candidate** — `_best_threshold_refinement` /
`_best_feature_addition` evaluate every candidate's marginal-log-loss reduction over the buffer and
keep the max. Three inferences are performed implicitly, with certainty, and all three fail exactly
when the window is small:

1. **Selection (max where the machinery says sum).** The scored fit is an order statistic over K
   candidates. `E[max of K noisy fits] > max of K true rates` — on a window of 3–4 outcomes,
   *every* candidate chance-fits, and the argmax fit is large because it was selected. The mll's
   Occam prices each candidate's fit *given that candidate*; nothing prices the argmax across them.
2. **Extrapolation (a Bayes factor treated as a rate).** `fit/n_buf` — a realised log Bayes factor
   over as few as 2–6 events — is treated as the *known* future per-event rate and multiplied by
   `H ≈ 200`. Evidence about a rate is not the rate; asserting otherwise is a T-1.3 violation
   hiding in a multiplication. This error is independent of selection: even a K = 1 candidate with
   a chance-level 1.1-nat window fit scores `0.5 · (1.1/4) · 195 ≈ 26` nats and fires.
3. **Collapse (install-one scored as if certain).** The op installs the argmax candidate and the
   score credits that candidate's fit at face value — "pick the winning grammar", the
   `average-not-collapse` error applied at the meta-action seam.

**The fix is a deletion, not an addition (the constitutional review's finding, adopted).** The
first draft of this document built a bespoke two-group (spike-and-slab) posterior with a uniform π
in a new file — a second inference mechanism, stipulated rather than derived, reasoning *about*
the reasoner's lookaheads from outside it. The review's demolition is recorded in §5 Q1 and
accepted in full; its core is worth restating because it scopes the whole move:

- The posterior the draft reached for **already has a name**: the marginal likelihood of the
  enlarged hypothesis space under the complexity prior. `P(D | G⁺) = Σ_m π_m · P(D | G_m)` with
  `π_m ∝ 2^{−complexity(G_m)}` prices *everything* the spike-and-slab stipulated: **selection** is
  priced because the marginal likelihood *sums* over candidates where the score *maxes* — a proper
  prior over model space is the multiplicity correction (Scott & Berger; Jeffreys), the K−1 losers
  doing their work inside the integral rather than inside a bespoke empirical null; **the
  spike** is the complexity prior itself (`2^{−Δ|G|}` is the honest, informative prior odds the
  uniform π threw away); **asymptotic convergence** is the Bayes factor growing with n — derived,
  not engineered.
- The first draft's reflective-boundary argument (imported from the escape ops) was **scope
  abuse**: the boundary is real for *un-entertained* hypotheses (`:gw_enumerate_more` /
  `:gw_deepen`, where learned returns are the honest floor), but growth candidates are
  **enumerated and fit** — the K marginal likelihoods are in hand. That is not the reflective
  boundary; it is bog-standard Bayesian model comparison with an exact answer. Conflating the two
  is how a stipulated mechanism acquires constitutional cover it has not earned.
- The bellwether stakes: if a bespoke corrector gets in for the winner's curse, a bespoke detector
  gets in for drift, and the framework dies of a thousand locally-reasonable patches. The wager is
  that axioms + grammar + perturbation + the complexity prior are enough; holding this line is the
  rehearsal for the non-stationarity work.

**The two derived components of the honest score.** Stress-testing the model-comparison fix
against the tail arithmetic shows it must carry both parts — the sum alone closes error 1 but not
error 2 (§4 traces the numbers):

**(A) The evidence term — sum, not max.** The window evidence for growth is the marginal-likelihood
ratio of the enlarged space:

    bma_fit = log P(D | G⁺) − log P(D | G)
            = log( π̃_inc · 1 + Σ_c π̃_c · e^{fit_c} )      [nats]

computed **from numbers the existing argmax loops already produce** (`fit_c = baseline − mll_c`;
the K fits are currently discarded except the max), with `π̃` the complexity prior over the
entertained set `{G} ∪ {G_c}` normalised (`π_c/π_inc = 2^{−Δcomplexity_c}`; for features
Δcomplexity = +1, so the former `prior_term = −log 2` **moves inside the sum** and must not be
double-charged — §2). One lucky candidate at `e^{1.3}` among K rivals sharing normalised prior
mass contributes little; a genuine candidate at `e^{6}` dominates the sum and recovers the raw
score. No new file, no new prior, no new mechanism: replace a `max` with a prior-weighted
`logsumexp`.

**(B) The flow term — the posterior licenses the horizon, not the window rate.** `bma_fit` is a
*stock* (nats of realised window evidence); the score needs a *flow* (nats/event × H). The current
`fit/n_buf` is the window rate — the object error 2 lives in. The honest flow is the
**posterior-weighted expected per-event gain**, `average-not-collapse` applied to valuation:

    flow = Σ_m P(G_m | D) · E_{x ~ P_m}[ log P_{G⁺}(x | D) − log P_G(x | D) ]      [nats/event]

— for each entertained grammar-hypothesis m (incumbent included), the expected next-event
log-predictive advantage of holding the enlarged posterior, *if m is true*, weighted by m's
posterior. Every factor is machinery: `P(G_m | D)` from the same `fit_c`/`π̃` arithmetic as (A);
the inner expectations are `expect` against posterior predictives on the window's empirical
feature distribution (the declared event measure — a §5 question). This object has teeth in both
directions: the incumbent-is-true term is *negative* (carrying junk hedges your predictions —
the real cost the tail seeds pay), the candidate-is-true terms are positive — so on a chance
window, where the posterior barely leaves the incumbent, `flow ≈ small-or-negative` and no `×H`
can amplify it; on an informed window the candidate's posterior share carries its genuine rate.
The score becomes:

    score = plateau · flow · H − compute_cost

with `plateau` unchanged (it prices *whether the stream still improves* — a property of the
trajectory; `flow` prices *what the entertained posterior expects per event* — a property of the
candidates; independent failure modes, no double count).

**Fidelity note (T-3.53, the metareasoning door).** (A) is free (the mlls are computed). (B) costs
K+1 posterior-predictive passes over the window — the same order as the lookahead's existing K
mll replays; the exact form is affordable at the window sizes where it matters. If a host ever
finds it dear, the legitimate response is an approximation *to this quantity*, selected by the
compute cascade — never a different quantity with a flatter prior.

## 2. Files touched

- **`src/program_space/exploration.jl`** (modification, ~60 lines net; **no new file** — the
  first draft's `selection_pricing.jl` is not written).
  - `_best_threshold_refinement` / `_best_feature_addition`: the candidate loops already compute
    every `fit_c`; collect them (currently discarded). Selection of *which* candidate to install
    stays the raw-fit argmax (§5 Q2). The returned valuation becomes
    `growth_value(flow · n_buf, n_buf, plateau, h; compute_cost)` — i.e. the `fit` slot carries
    `flow · n_buf` so `growth_value`'s interface and the host seam are unchanged. The `prior_term`
    kwarg is **no longer passed by the feature path** (the complexity prior now enters through
    `π̃` inside the sum — passing both would double-charge; the kwarg itself stays for other
    callers).
  - New internal helpers (beside the loops, same file): `_entertained_posterior(fits, Δcs)` — the
    normalised posterior over `{G} ∪ {G_c}` from the window fits and complexity deltas (the (A)
    arithmetic); `_growth_flow(...)` — the (B) expectation, `expect`-canalised against the
    compiled kernels on the window's feature records. Docstrings carry the §1 derivations and
    name the tests.
- **`apps/julia/grid_world/host.jl`**: **no change** (signatures preserved; the cached fit is the
  flow-based fit, still pure in (grammar, buffer, depth)).
- **`test/test_growth_bma.jl`** (new, ~6 sections): §1 the sum-vs-max pin — a chance-shaped
  ensemble (K fits at chance scale) yields `bma_fit` ≪ max fit, against a hand-built logsumexp
  oracle; §2 the flow's sign teeth — an incumbent-dominated posterior gives `flow ≤ 0` (the hedge
  cost), a candidate-dominated one gives `flow` within tolerance of the candidate's true rate;
  §3 the seed-6 regression — the §4 fixture scores below the fire floor at H = 195 where the old
  score fired at ~26 nats; §4 asymptotic identity — fixed true separation, growing n: score →
  the raw-fit score (the current behaviour is the limit); §5 no-double-charge — the feature
  path's score equals the explicit two-axis arithmetic with the prior inside `π̃` (== against
  the manual oracle); §6 purity + determinism (same inputs ⇒ identical score; no state).
- **`test/test_threshold_explore.jl`, `test/test_feature_discovery.jl`,
  `test/test_grid_world_meta.jl`** (re-baselines): pins asserting exact small-window voi values
  move to the new score; each re-baselined pin annotated. Structural assertions (gates, no-op
  floors, argmax identity) unchanged.

## 3. Behaviour preserved

- **Selection identity:** which candidate an op installs when it fires is bit-identical (raw-fit
  argmax, unchanged — §5 Q2). Only *whether* the meta-argmax lets growth fire moves.
- **Asymptotic identity:** when one candidate's evidence dominates the entertained posterior, the
  score converges to the current raw score (`test_growth_bma.jl` §4 pins the convergence; the
  informed mid-episode fires that drive the headline's 19–1 are expected unmoved, verified by the
  gate re-run's mean-tier CIs).
- All non-growth scoring (escape cells, perturb replacement value, do-nothing floor): bit-stable;
  `test_growth_returns.jl`, `test_replacement.jl`, conjugate/threshold suites pass untouched.
- Skin wire, persistence, benchmark harness: untouched (measures fixed by §8 — this move is
  judged against them, it does not get to move them).

## 4. Worked end-to-end example

The tail-seed shape (seed 6's instrumented fire: `add_feature` at step 6, final rate gap −0.381),
traced through both components — including the stress-test that shows why (A) alone is not enough.

1. **Step 6, buffer n = 4, K = 5 feature candidates.** On 4 events every candidate's refreshed
   program space contains chance-consistent splits: window fits come back at chance scale, say
   `{1.1, 0.9, 1.3, 1.0, 1.2}` nats (illustrative; the test fixture uses captured values). The
   conjugate smoothing caps what 4 events can prove — a "perfect" program's Bayes factor is
   `(1/2·2/3·3/4·4/5)/(1/2)⁴ ≈ e^{1.6}` — so these are exactly the magnitudes selection luck
   produces. **Today:** `fit = 1.3`, `H ≈ 195`, `plateau ≈ 0.5` ⇒ score ≈ 26 nats — fires, installs
   a wrong feature, injects ~80 diluted components, and the seed never recovers.
2. **(A) alone — the honest failure of the first-order fix.** `π̃_c/π̃_inc = 1/2` each (one symbol);
   `bma_fit = log[(1 + Σ_c ½e^{f_c})/(1 + K/2)] ≈ log[(1+3.1)/(3.5)] ≈ 0.16` nats — the sum
   correctly refuses the max's 1.3. But fed to the *old* flow, `0.5 · (0.16/4) · 195 ≈ 3.9` nats —
   **still fires** (the floor is ~0.7). And with K = 1 the sum degenerates entirely and the old
   26-nat fire returns. The extrapolation error is independent of the selection error; the sum
   closes only the latter. This arithmetic is why (B) is load-bearing, and it is pinned as
   `test_growth_bma.jl` §3's counter-oracle.
3. **(B) on the same window.** The entertained posterior from those fits:
   `P(inc|D) ≈ 1/(1+3.1) ≈ 0.24`, each candidate ≈ 0.12–0.19. The flow: *if the incumbent is
   true*, holding the union hedges every prediction with ~0.7 posterior mass of junk — the inner
   expectation is **negative** (≈ −0.15 nats/event on the fixture); *if candidate c is true*, the
   union predicts better, but c's programs hold only their earned share of the union predictive —
   the inner expectation is small-positive (≈ +0.1). The posterior-weighted sum lands near zero
   (the fixture: `flow ≈ +0.01` nats/event) ⇒ `score ≈ 0.5 · 0.01 · 195 − log 2 ≈ 0.3` nats —
   **below the floor; the op waits.** No spike, no π, no ensemble null: the incumbent's posterior
   share and the hedging cost — both machinery-computed — do all the work.
4. **Step ~90, n = 30, regime 2 (`:speed` genuinely predicts).** Fits `{6.2, 0.4, 0.7, 0.3, 0.5}`:
   the winner dominates the sum (`bma_fit ≈ 5.4`), its posterior share ≈ 0.95, the flow ≈ its true
   per-event rate ≈ 0.2 nats/event ⇒ score ≈ current score ⇒ the informed fire (the one behind the
   headline's 19–1) is preserved.
5. Ownership: everything lives where the lookahead already lives (`exploration.jl`); `growth_value`
   and the host seam untouched; the event measure for (B)'s expectations is the window's feature
   records (data, host-provided — the brain does the arithmetic).

## 5. Open design questions

1. **[RESOLVED against the first draft — recorded per T-4.5's spirit].** The spike-and-slab
   ensemble-null mechanism (a bespoke two-group posterior, uniform π, new file) is rejected: it
   stipulated a second inference mechanism where the marginal likelihood under the complexity
   prior already computes the same quantity with a *better* prior (`2^{−Δ|G|}` is the honest prior
   odds; uniform π discards it); its multiplicity correction re-implemented what the BMA sum does
   inside the integral (Scott & Berger); and its constitutional cover (the reflective boundary)
   belongs to the *un-entertained* ops only — growth candidates are enumerated and fit, so exact
   model comparison is available and mandatory (T-2.32). A tractable approximation may only ever
   be *derived from* the exact object with the complexity prior in place, selected by the compute
   cascade (T-3.53) — never a different quantity with a flatter prior.
2. **The transition under an uncertain posterior — install the winner, or the union?** The score
   (B) values *holding the enlarged posterior*; the op today installs the argmax candidate only.
   Recommended: **install the winner, unchanged, this move** — the flow's incumbent-share teeth
   already suppress fires when the posterior is genuinely spread (precisely the case where
   install-one would collapse), so the score/transition gap is small exactly when it matters, and
   the blast radius stays contained. **Install-the-union** (inject all positive-posterior
   candidates, coherently, and let conditioning arbitrate — the full `average-not-collapse` form)
   is named as the finer fidelity: it is the greater-EU action when candidate uncertainty is real,
   at K× enumeration compute, and choosing between the two is itself an EU decision. Counter-
   position to argue at ratification: T-3.55 wants score = transition *now*, which favours either
   scoring only the winner's marginal contribution (weaker multiplicity pricing — the rivals'
   role shrinks to the posterior normalisation) or installing the union immediately (dearer).
3. **The event measure in (B)'s inner expectations.** The window's empirical feature distribution
   is the recommended declared measure (it is data the host already provides; no model of the
   feature process is invented). Alternatives: the temporal-recency-weighted window (privileges
   the current regime; adds a weighting choice that smells like a tunable), or a declared
   grid over feature space (invents a measure the world didn't provide). Recommend the plain
   window; revisit only if the gate shows regime-boundary staleness.
4. **Does the flow-based fit feed the `add_feature` attribution gate too?** Same recommendation
   as the first draft, now with the right object: **yes, one fit** — the gate's `fit_explore > 0`
   consults the same flow-based quantity the score uses; two circulating numbers named `fit` is
   the T-3.55 split. (The gate's attribution argument is about *real* fit, which the flow
   estimates better than the raw window max.)

## 6. Risk + mitigation

- **Under-exploration (over-shrink).** The flow's negative incumbent-term could suppress genuine
  early discovery. Mitigation: the asymptotic-identity pin (§3) bounds the informed regime; the
  headline assertion (rate CI vs never_explore > 0) is the acceptance-side tripwire; halt-the-line
  if the headline degrades. Note the flow is *derived*, not tuned — if it over-shrinks, the
  finding is about the model (e.g. the event measure, §5 Q3), and the response is a derivation
  fix, not a knob.
- **Compute.** (B) adds K+1 predictive passes over the window per growth evaluation — same order
  as the existing K mll replays; the voi_cache memoisation (pure in grammar/buffer/depth) applies
  unchanged. If profiling disagrees, the T-3.53 door (§1 fidelity note) is the only exit.
- **Double-charging the prior.** The feature path currently passes `prior_term = −log 2` to
  `growth_value` while (A) carries the same odds inside `π̃`. `test_growth_bma.jl` §5 pins the
  no-double-charge equality against a manual oracle; the kwarg is dropped at the call site in the
  same commit that adds the sum.
- **Score/transition gap under install-winner** (§5 Q2's accepted residual): small exactly when
  fires happen (dominated posterior), real when the posterior is spread — but spread posteriors
  now score near the floor, so the gap governs decisions that no longer occur. The union
  fidelity is the named successor if the gate's op logs show otherwise.
- **Benchmark comparability.** Only eu_max (and clairvoyant's fallback) read scores; score-blind
  baselines unchanged. Per-policy deltas vs round 4 attributable to the scoring change alone.

## 7. Verification cadence

    julia test/test_growth_bma.jl               # new — sum-vs-max, flow teeth, seed-6 regression,
                                                #   asymptotic identity, no-double-charge, purity
    julia test/test_threshold_explore.jl        # re-baselined pins
    julia test/test_feature_discovery.jl        # re-baselined pins
    julia test/test_grid_world_meta.jl          # re-baselined score pins
    julia test/test_growth_returns.jl           # untouched pass
    julia test/test_replacement.jl              # untouched pass
    for f in test/test_*.jl; do julia "$f"; done
    JULIA_PROJECT=$PWD PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run python apps/skin/test_skin.py

Gate re-run (20 seeds, background), judged against the §8 measure set **fixed before this design**:

- **Target claims:** minimax worst-seed mean-rate gap vs `random_p005` and `fixed_k50` rises from
  −0.5/−0.381 toward ≥ 0; q10 from −0.476/−0.333 toward ≥ 0. Falsifiable mechanism claim: **no
  eu_max growth fire whose entertained posterior was incumbent-dominated** (checkable from the op
  log's new per-fire (bma_fit, flow, P(inc|D)) triple, which the run logs).
- **Must-not-degrade:** the headline (rate +0.150 CI > 0, 19–1) holds; mean-tier CIs vs the tuned
  baselines do not become significantly negative; the bracket holds.
- Halt-the-line on any degrade; results commit honestly either way.
