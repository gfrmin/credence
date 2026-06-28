# Move 2 design doc — The saturation signal (the belief-aware meta-level entry)

> Move 2 of the `exploration-budget` arc (`docs/exploration-budget/master-plan.md`). Seven-section
> template. The **first belief-aware** meta computation — it establishes the seam (Q2's prior-only /
> belief-aware split) that exploration (Moves 3–4) hangs off. **Signal only**: Move 2 *computes* the
> saturation signal and the belief that backs it; the *consumer* (`explore_grammar`) lands in Move 3, so
> Move 2 is behaviour-preserving by construction (nothing reads the signal for a decision yet).

## 1. Purpose

The exploration budget must only spend expensive lookahead **when expansion can plausibly pay off** —
i.e. when the current discrete alphabet has *saturated* (master plan §3.2: "saturation gates *when*").
Move 2 defines and computes that signal:

> **`saturated` = compression-exhausted (prior-side) ∧ residual-plateaued (belief-side).**

- **Prior-side half — free, from Move 1.** "Compression exhausted" = `perturb_grammar` is a no-op: no
  `:add_rule` and no `:remove_rule` clears `net_voc > 0`. Move 1's deterministic-argmax body made this
  *definable* (a no-op iff *nothing* improves the prior). Move 2 exposes it as a pure predicate
  `compression_exhausted(g, freq_table; compute_cost)`, factored out of `perturb_grammar` (DRY) so the two
  can never disagree.
- **Belief-side half — the real work.** "Residual plateaued" = the belief's **predictive log-loss has
  stopped improving**. The per-step residual is `ℓ_t = −log_predictive(belief_before, k, obs)` — the exact
  Tier-1 predictive marginal of the *pre-conditioning* belief on the observation it then sees
  (`log_predictive(::MixturePrevision, k, obs)` already exists, `ontology.jl:1701`). While the alphabet is
  still explaining the data, conditioning keeps driving `ℓ_t` down; when it plateaus, the residual that
  remains is candidate evidence the alphabet is too coarse — exactly when expansion's VOI rises.

**The plateau is judged as a *carried belief*, not a threshold.** A naïve "EMA slope < ε" detector is a
hard-coded heuristic *alongside* EU-max — forbidden (Invariant 1; heuristics live *inside* EU-max). The
principled form is a **2-regime BMA** over the agent's own learning dynamics — `{:improving, :plateaued}`
— conditioned each step on the residual via the **existing** `condition(::MixturePrevision, k, obs)`. The
saturation signal is the *posterior weight on `:plateaued`* (carried, per `average-not-collapse`), never a
collapse-to-MAP or a thresholded point estimate. This is the constitution's own metareasoning framing
("beliefs about … expected improvement from further computation") instanced as a belief, updated by the
one learning mechanism. It also resolves *where the residual history lives*: it is **summarised into a
Measure** (the regime belief), conditioned incrementally — not a raw `Float64[]` buffer (Invariant 3 /
`state-is-measure`).

**The regime model is scale-free — the noise floor is *inferred*, not set (ratified, the load-bearing
refinement).** A fixed-σ plateaued Gaussian ("decrements near 0, σ hard-set") merely *relocates* the
threshold into the regime scale: a task whose genuine improvements run at 0.1 nat/step reads as plateaued
under a model calibrated to 0.3–1.3 nat drops — a **false plateau** that fires exploration before the
alphabet is exhausted. The fix is the model's principled completion, Bayesian signal-detection: the
plateaued regime is "improvement has fallen into the **inferred** obs-to-obs noise floor" — decrements
`~ N(0, σ²)` with `σ²` *learned from the residual series' own bounce* — and the improving regime is "a
drift **detectable above** that inferred noise." The noise scale is **marginalised** (the data's, not a
hyperparameter), so a slow-but-real improver (small drift, but consistently above its own noise) stays in
`:improving`. This is what makes the plateau genuinely threshold-free rather than threshold-relocated; a
fixed-scale regime is the version the stall gate (§6 R1) exists to refuse.

Move 2 delivers: `compression_exhausted`, the residual computation, the scale-free regime belief + its
per-step conditioning, and the saturation **evidence** (the components — see Q3, it is *not* a hard
`saturated` veto). It does **not** add `explore_grammar`, a wire verb, or any decision that reads the
signal — those are Move 3.

## 2. Files touched

- **`src/program_space/perturbation.jl`** — *modify (DRY extraction, bit-exact)*:
  - Factor the candidate-gathering + argmax out of `perturb_grammar` into
    `_best_compression_candidate(g, freq_table; compute_cost) → Union{Nothing, NamedTuple}` returning the
    winning `(kind, rule, net_voc)` or `nothing`. `perturb_grammar` becomes: compute the best candidate;
    `isnothing` ⇒ return `g`; else apply it. **No behaviour change** (capture-before-refactor: pin the
    current `perturb_grammar` outputs on the test_voc_gate/test_program_space fixtures, assert `==`).
  - **Add** `compression_exhausted(g, freq_table; compute_cost = 0.0)::Bool =
    isnothing(_best_compression_candidate(g, freq_table; compute_cost))` — the prior-side saturation half,
    sharing the *exact* logic `perturb_grammar` acts on (they can never drift).
- **`src/saturation.jl`** — *new file* (included from `ontology.jl` after the program-space includes):
  - The **scale-free** learning-regime model: a 2-regime BMA `{:improving, :plateaued}` over the per-step
    decrement `Δ_t = ℓ_{t−1} − ℓ_t` (>0 = loss fell = improvement), with the **noise scale inferred /
    marginalised** in both regimes (Bayesian signal-detection):
    - `:plateaued` — `Δ ~ N(0, σ²)`, `σ²` unknown (Gamma/inverse-gamma prior) ⇒ a zero-centred Student-t
      marginal. "Improvement has fallen into the inferred noise floor."
    - `:improving` — `Δ ~ N(μ, σ²)`, `μ > 0` (drift detectable *above* the noise), `σ²` unknown.
    Built on the conjugate NormalGamma path (collapse-towers Family-BMA + `condition(::MixturePrevision)` +
    the Phase-3 `log_predictive`/Student-t). **Code-time item:** the `:plateaued` component needs a
    *zero-mean, unknown-precision* marginal (Student-t at 0); if the conjugate roster lacks it, add it as a
    new exact named component (constitution-sanctioned — "named distributions may be added"). **Hard
    requirement:** the noise scale is the *data's*, inferred per regime — never a fixed σ (a fixed σ is the
    relocated threshold the stall gate refuses).
  - `initial_learning_regime() → MixturePrevision`: the two components above, uniform prior weights.
  - `update_learning_regime(regime, ℓ_prev, ℓ_now) → MixturePrevision`: form `Δ = ℓ_prev − ℓ_now`, build
    the regime kernel, return `condition(regime, k, Δ)`. **Pure** (host rebinds the field); routed entirely
    through Tier-1 `condition`.
  - `plateau_probability(regime)::Float64 = weights(regime)[plateaued_index]` — carried posterior weight on
    `:plateaued` (public `weights` accessor; never `.log_weights`).
  - **No hard `saturated` veto (Q3, ratified).** Move 2 returns the saturation **evidence** — the pair
    `(compression_exhausted::Bool, plateau_probability::Float64)` plus `residual_is_zero::Bool`. It does
    **not** define a boolean gate `explore iff saturated`, because that would let `plateau_probability`
    *block* a positive-EU explore (a feature can carry positive VOI mid-improvement) — the forbidden cap
    wearing a probability. The Move-3 contract (stated here so the seam is right): `plateau_probability` is
    a **soft, overridable prior** lowering exploration's expected value; the lookahead VOI **always runs and
    has final say**. The only **hard** cheap-defers are *provable-zero-VOI*: compute budget exhausted
    (`VOI − cost < 0` trivially) or residual already at zero (nothing left to explain). "Still improving" is
    not a proof of zero VOI, so it never hard-defers.
- **`src/program_space/agent_state.jl`** — *modify*: add two fields to `AgentState` —
  `learning_regime::Ontology.MixturePrevision` (the regime belief; a Measure-typed summary of the residual
  history) and `last_residual::Union{Nothing, Float64}` (the previous step's `ℓ`; `nothing` at cold start).
  `last_residual` is a *sufficient statistic* — one scalar is enough **precisely because** the history
  lives in the regime posterior, not a buffer (the `state-is-measure` payoff, Q4). Provide a
  constructor/default (`initial_learning_regime()`, `nothing`) so existing `AgentState(...)` call sites keep
  compiling (backward-compat). **Add `reset_learning_regime!(state)`** (Q1b): set
  `state.learning_regime = initial_learning_regime()` **and** `state.last_residual = nothing` — i.e. start
  the residual Measure *afresh*, not merely re-weight toward `:improving`. Pre-change residuals were
  generated under a superseded alphabet and are stale; carrying them would let old observations drag the
  fresh inference.
- **`apps/julia/grid_world/host.jl`, `apps/julia/email_agent/host.jl`** — *modify (additive, non-causal in
  Move 2)*: at the conditioning site, BEFORE `condition(state.belief, k, 1.0)`, compute
  `ℓ = −log_predictive(state.belief, k, 1.0)` (Tier 1), then after conditioning update the regime:
  `state.learning_regime = update_learning_regime(state.learning_regime, state.last_residual, ℓ)`;
  `state.last_residual = ℓ`. **Nothing reads `state.learning_regime` for a decision** in Move 2 — it is
  computed and recorded (optionally surfaced in the MetricsTracker for the figure) but does not gate any
  action until Move 3. The existing host-arithmetic `surprise` (telemetry `−log(p)` / `−log(EU)`,
  non-causal) is left untouched; the *causal* residual is the Tier-1 `log_predictive` value, kept separate.
  **At every grammar change** — `perturb_grammar` applies, a new grammar is added, or `deepen` changes the
  enumerated program set — call `reset_learning_regime!(state)` (Q1b): the residual is about the *current*
  ensemble's predictive performance, so a changed alphabet invalidates the prior residual history. The
  plateau signal then re-accumulates over the next few steps under the new alphabet — correct: you cannot
  know the new alphabet has saturated until you have watched it predict.
- **`test/test_saturation.jl`** — *new*: the §7 tests.
- **NOT touched:** `apps/skin/server.jl` (no wire verb — the consumer is Move 3; the signal is brain-side
  until something reads it). No protocol bump.

## 3. Behaviour preserved

- **`perturb_grammar` is bit-exact.** The `_best_compression_candidate` extraction is a pure refactor;
  capture canonical outputs PRE-change on the existing perturbation fixtures and assert `==`
  (capture-before-refactor). `compression_exhausted` is new and reuses the same path.
- **Host decisions are unchanged.** Move 2 only *adds* the residual/regime computation; no action selection
  or belief update reads it. Existing `test_email_agent`, `test_program_space`, grid_world tests stay green.
  (The regime belief is additive state; the domain `belief` and every `optimise`/`condition` over it are
  untouched.)
- **`state-is-measure` upheld.** The residual history is a `MixturePrevision`, not a scalar buffer; the one
  non-measure addition (`last_residual::Float64`) is a sufficient statistic, peer of the existing
  `current_max_depth::Int`.
- **AgentState schema-change safety.** New fields are defaulted (uninformative regime, `nothing` residual);
  pre-emptive grep `grep -rn 'AgentState(' src/ apps/ test/` to confirm every construction site gets the
  default, and `grep -n AgentState test/test_persistence.jl` to confirm AgentState is not serialised (if it
  is, a fixture bump per the commit-pinned-fixtures protocol).
- Tolerance: strata-1 `==`/`===` on the perturb refactor; seeded `==` for any sampling (the regime
  conditioning is conjugate/exact — no sampling expected).

## 4. Worked end-to-end example

Grid-world, post-Move-1 grammar `g` that has saturated compression (no add/remove candidate, so
`compression_exhausted(g, ft) == true`). Two phases of a run:

**Phase A — still learning.** Steps 1–8, the belief is rapidly improving: `ℓ_t = 2.1, 1.6, 1.3, 1.05,
0.9, 0.82, 0.78, 0.76` (predictive log-loss falling). Each step the decrement `ℓ_{t-1} − ℓ_t` is clearly
positive; the regime kernel scores these as far more likely under `:improving`, so
`condition` drives `plateau_probability(regime)` low (say `0.12`). `saturated` = `true ∧ (plateau 0.12)` —
compression is exhausted but the belief is still extracting value from the **current** alphabet, so the
carried plateau probability is low ⇒ Move 3 would *not* explore (the cheap screen says "keep conditioning").

**Phase B — plateaued.** Steps 9–16, `ℓ_t = 0.755, 0.758, 0.752, 0.757, 0.754, 0.756, 0.753, 0.755` —
bouncing around `0.755`, decrements ≈ 0 (mixed signs / tiny magnitude). The regime kernel now scores these
as far more likely under `:plateaued`; `condition` moves mass there, `plateau_probability(regime) → 0.94`.
Now `saturated` = `true ∧ (plateau 0.94)`: compression exhausted **and** the residual plateaued — the
belief-side precondition for exploration is met. Move 3's `explore_grammar` (not in this move) would, from
here, be allowed to spend lookahead on the residual-proposed candidates.

**Determinism / no decision in Move 2:** the regime belief is updated by exact conjugate `condition`; no
`rand`, no action gated on it this move. The two phases are just the signal coming online; the example
fixes the numbers the §7 test asserts.

## 5. Open design questions

> **ALL RESOLVED by ratification 2026-06-28 (author) — see §8 for the rulings + reasoning.** The prose
> below is retained as the rationale of record. Settled by the master plan (stated, not asked): the
> *valuation* is lookahead (Q1) and the *home* is a belief-aware entry (Q2) — both land in Move 3. The
> saturation *definition* is compression-exhausted **∧** residual-plateau (Q3). What was open is the
> **belief-side mechanism**, below.

1. **The plateau model (the central crux).** *Recommendation: a 2-regime BMA* `{:improving, :plateaued}`,
   conditioned each step via the existing `condition(::MixturePrevision, k, obs)`, saturation = carried
   `plateau_probability` (no collapse — `average-not-collapse`). This keeps the judgment a belief updated by
   the one learning mechanism, reuses existing machinery, and makes the history a Measure. **Alternatives to
   weigh:** (a) a single Beta-Bernoulli on "did the residual decrease?" (simplest conjugate, but loses
   decrement *magnitude* — a series of tiny decreases reads as "improving" forever); (b) the forbidden
   EMA-slope-threshold (named only to reject it). **Sub-question — is the regime non-stationary?** After an
   alphabet expansion (Move 4) the belief should be *able to re-enter `:improving`*. A static 2-regime BMA
   latches; a change-point/BOCPD treatment (or simply *resetting* the regime belief when the grammar
   changes) allows re-entry. *Recommendation: reset the regime belief on grammar change* (the cheapest sound
   re-entry; full BOCPD is heavier than Move 2 needs) — but flag for your call.
2. **The residual observable.** *Recommendation: the decrement magnitude* `Δ_t = ℓ_{t-1} − ℓ_t` modelled by
   regime-dependent Gaussians (`:improving` ~ `N(μ>0, ·)`, `:plateaued` ~ `N(0, small)`) — magnitude-aware,
   reuses the Gaussian/NormalGamma conjugate path. *Counter:* needs a noise-scale assumption per regime
   (a kernel parameter, not a decision threshold — defensible, and itself promotable to a latent later).
   The sign-only Bernoulli (OQ1-a) is the cheaper fallback if the Gaussian scale proves finicky.
3. **`saturated`: boolean gate vs carried probability.** §4 of the master plan says explore-vs-exploit is an
   *EU comparison*; §3.2 says saturation *gates when*. Reconciliation: saturation is a **conservative,
   one-sided screen** — it may only *defer* exploration when expansion VOI is cheaply ≤ 0; it must **never
   block a positive-EU explore** (else it is a cap/heuristic, forbidden). *Recommendation: Move 2 returns
   the **pair** (compression-exhausted bool, plateau probability)*, and Move 3's EU comparison consumes the
   probability — rather than Move 2 hard-committing a boolean. Decide whether you want a convenience boolean
   `saturated` at all in Move 2, or only the components.
4. **Residual state home + cold start.** *Recommendation: `learning_regime::MixturePrevision` +
   `last_residual::Union{Nothing,Float64}` in AgentState* (regime is a Measure ⇒ `state-is-measure`;
   `last_residual` a sufficient statistic, peer of `current_max_depth`). At cold start (`last_residual ===
   nothing`, < 2 observations) there is no decrement ⇒ the regime stays at its uninformative prior ⇒
   `plateau_probability` low ⇒ not saturated (correct: never explore before there is evidence the alphabet
   saturated). Confirm you are content with two new AgentState fields vs a single bundled sub-struct.
5. **Does the signal belong in the skin now?** *Recommendation: no* — Move 2 is brain-side; the only
   consumer is Move 3's `explore_grammar`. Exposing a wire verb before there is a consumer is premature
   surface. Move 3 adds the verb when it adds the consumer. (Flag if you want the signal observable over the
   wire earlier for the empirical figure.)

## 6. Risk + mitigation

- **R1 — the plateau detector smuggles in a non-EU heuristic (the arc-defining risk).** *Mitigation:* the
  regime-BMA keeps the judgment a belief updated by `condition`; the screen is *conservative one-sided*
  (defers, never blocks, a positive-EU explore). If no principled cheap plateau signal survives review →
  **STALL at this design doc** (the master plan §3.3 / §7 stall gate — do not ship a capped or
  threshold-gated guess).
- **R2 — "a second learning mechanism."** *Mitigation:* the regime belief is conditioned through the SAME
  `condition`; it is a legitimate metareasoning belief (constitution "On metareasoning"), not a parallel
  updater. No weights are touched outside `condition`. Add the `# credence-lint` audit surface check.
- **R3 — behaviour drift.** *Mitigation:* Move 2 is **signal-only / non-causal** — nothing reads the regime
  for a decision; existing host tests stay green unchanged. capture-before-refactor pins the
  `perturb_grammar` extraction `==`.
- **R4 — AgentState schema bump.** *Blast radius:* every `AgentState(...)` site + any serialisation.
  *Mitigation:* defaulted new fields; pre-emptive grep of construction sites and `test_persistence`
  (fixture bump only if AgentState is serialised — likely not; the belief is rebuilt per run).
- **R5 — predictive-residual ordering.** `log_predictive` MUST be computed on the **pre-conditioning**
  belief (it is the belief's prediction of the obs it then learns from). *Mitigation:* compute `ℓ` strictly
  before `condition(state.belief, k, 1.0)`; the test asserts the ordering via a known two-step sequence.
- **Lint:** `compression_exhausted` / `saturated` are stdlib compositions over `condition`/`weights` — no
  arithmetic-on-weights that feeds a decision (the only consumer is Move 3, and it routes through EU).
  Confirm corpus self-test + `check apps/` stay green; the host residual line reads `weights`/`log_predictive`
  (sanctioned accessors), no manual `logw`.

## 7. Verification cadence

End of Move-2 code (from repo root; Julia tests not CI-gated):
```
julia test/test_saturation.jl       # compression_exhausted, regime conditioning, the plateau example
julia test/test_voc_gate.jl         # perturb_grammar bit-exact through the _best_compression_candidate refactor
julia test/test_program_space.jl    # idem + :remove_rule cases unchanged
julia test/test_email_agent.jl      # behaviour-preserving (signal is non-causal in Move 2)
```
Then the **full** `test/test_*.jl` suite + lint corpus self-test + `check apps/`, and **stop and report**.
Skin smoke not required (skin untouched).

`test_saturation.jl` assertions (repo `check`/`@assert` idiom; tolerance inside the boolean):
- **compression_exhausted ≡ perturb no-op:** for a grammar/freq_table with a positive-net_voc candidate,
  `compression_exhausted == false` and `perturb_grammar` changes the grammar; for a saturated one,
  `compression_exhausted == true` and `perturb_grammar` returns `g` (same id). The two agree on a battery
  of cases (they share `_best_compression_candidate`).
- **perturb_grammar bit-exact:** the refactor preserves the Move-1 outputs `==` on the existing fixtures
  (capture-before-refactor).
- **regime comes online (the §4 example):** feeding the Phase-A decreasing series leaves
  `plateau_probability` low (< 0.3, a directional bound, not a magic equality); feeding the Phase-B flat
  series drives it high (> 0.7). Assert the *direction and ordering* (B ≫ A), not a hand-tuned point value.
- **scale-free — the slow-improver (the test a fixed-σ model fails):** a series improving at a *small but
  consistent* rate (e.g. `Δ ≈ 0.08`/step, well below the §4 example's 0.3–1.3 drops) but with smaller
  bounce stays in `:improving` (`plateau_probability` low) — because the inferred noise floor scales to the
  data, the small drift is still *detectable above* it. A fixed-σ plateaued regime calibrated to the §4
  scale would (wrongly) call this plateaued; assert it does **not**. This is the test that pins the
  scale-free requirement (R1).
- **reset clears history (Q1b):** after a non-trivial regime (high `plateau_probability`),
  `reset_learning_regime!(state)` returns `plateau_probability` to the uninformative prior **and**
  `state.last_residual === nothing`; a subsequent fresh series is unaffected by the pre-reset residuals.
- **cold start:** with `< 2` residuals, `plateau_probability` sits at the uninformative prior ⇒ not
  saturated.
- **conjugate exactness / determinism:** the regime conditioning is exact (no `rand`); two runs on the same
  residual series give identical `plateau_probability` (`==`).
- **saturation evidence, not a gate (Q3):** Move 2 exposes the *components* — `compression_exhausted`,
  `plateau_probability`, `residual_is_zero` — and **no** function that returns a hard `explore iff
  saturated` boolean. Assert the components are computed correctly across the four quadrants
  (compression-exhausted × plateau-high), and assert (by API surface) that no hard-gate veto is exported —
  the cap-free-by-construction contract for Move 3.

Halt-the-line: any failure at end-of-PR is a halt; the branch never sleeps red. If R1 cannot be discharged
(no principled cheap plateau signal), the move **stalls at this doc** rather than shipping a thresholded gate.

## 8. Ratification + refinements (2026-06-28)

Ratified by the owner; the core shape ("the plateau is a *carried belief* — regime posterior conditioned by
the engine's own `condition`, history-as-Measure — dissolves what the constitution forbids: a 2-regime BMA
is inference/one-learner/`average-not-collapse`, not a second decision rule bolted alongside EU-max")
affirmed. Three load-bearing refinements fold in (above), each guarding a way a cap/threshold could sneak
back:

1. **Q1/Q2 — scale-free, inferred noise floor (the decisive refinement).** A fixed-σ plateaued regime
   *relocates* a threshold into the regime scale and false-plateaus a slow-but-real improver (fires
   exploration early, burns lookahead on a still-improving belief). The principled completion is Bayesian
   signal-detection: `:plateaued` = `Δ ~ N(0, σ²)` with `σ²` **inferred from the series' own bounce**;
   `:improving` = drift **detectable above** that inferred noise. σ marginalised ⇒ scale-free ⇒ genuinely
   threshold-free (the noise floor is the data's). *This is what makes R1 not fire* — a fixed-scale regime
   is exactly the version the stall gate refuses. (§1, §2 saturation.jl, §7 scale-free test.)
2. **Q1b — reset on grammar change is the *principled* answer, and it clears the *history*.** A Move-4
   alphabet expansion is a change-point the agent *caused* — nothing to infer, so reset directly; BOCPD
   would over-model a known action's known consequence. The reset must **start the residual Measure
   afresh** (`learning_regime = initial`, `last_residual = nothing`), not merely re-weight toward
   `:improving` — pre-change residuals are stale under the new alphabet and would drag the fresh inference.
   (§2 agent_state + host bullets, §7 reset test.)
3. **Q3 — `plateau_prob` is a soft prior, never a gate (where a cap sneaks back).** Two kinds of "defer";
   only one is sound. A **hard cheap-defer** is legitimate *only* when it cheaply *proves* exploration
   VOI ≤ 0 — narrow: compute budget exhausted, or residual already zero. "Plateau prob low / still
   improving" is **not** such a proof — a feature can carry positive VOI mid-improvement (it captures
   structure the current alphabet can't, however well that alphabet is exploited). So `plateau_prob` enters
   Move 3's exploration EU as a **soft, overridable prior** (lowers the prior on exploration being
   worthwhile); the **lookahead VOI always runs and has final say**. Move 2 therefore exposes the
   *evidence components*, not a hard `saturated` veto — cap-free *by construction*. (§1, §2 saturation.jl,
   §7 evidence-not-gate test.)

**Q4 — two fields, ratified** (the elegance: `last_residual` suffices as one scalar *because* the history
lives in the regime posterior — `state-is-measure` earned). **Q5 — defer skin, ratified** (signal-only, no
consumer, no wire).

**PR strategy (ratified):** land **this design doc as its own PR** and merge it before the code PR — unlike
mechanical Move 1 (which bundled), this is the arc's architectural inflection (the prior-only → belief-aware
crossing), so a clean ratified-design checkpoint is cheap insurance on the move most worth getting right.

**Stall gate:** does not fire — but *only with* the noise-floor formulation. A fixed-scale regime would
quietly become the thresholded gate the stall gate exists to refuse. Built scale-free, with `plateau_prob`
a prior not a gate, Move 2 hands Move 3 a saturation read honest on both sides — demonstrated not assumed,
and incapable of capping.

### Code-time refinements (2026-06-28, discovered during implementation)

Two refinements the implementation forced, both consistent with the ratified intent:

1. **The scale-free priors are *near-Jeffreys*, and the new conjugate primitive `ZeroMeanGammaPrevision`
   was added.** Tracing the conjugacy showed `NormalGammaPrevision` cannot soundly encode μ≡0 (its μ
   carries a κ-weighted prior; κ→∞ is degenerate), so the `:plateaued` regime needed a new exact
   component — `ZeroMeanGammaPrevision(α, β)` (zero-mean Gaussian, unknown precision; α += 0.5, β += r²/2;
   zero-centred Student-t marginal) — added to the conjugate registry (constitution-sanctioned). More
   importantly, a `Gamma(1,1)` precision prior **pins the noise scale to ~1 and false-plateaus slow
   tasks** — the exact bug the stall gate refuses. The fix is a **near-Jeffreys diffuse precision prior
   `Gamma(0.01, 0.01) ≈ p(σ²) ∝ 1/σ²`** in both regimes, so the noise floor is the data's (inferred).
   The `:improving` regime is the clean nested Bayesian t-test (μ free, μ₀=0, κ=0.1 weak ⇒ σ-relative
   detection). Empirically: consistent-improving → 0.0, flat → 0.86, **slow (Δ≈0.08) → 0.0** (scale-free,
   the keystone), cold-start → 0.5, decelerating → 0.33 (graceful, intermediate). `test_saturation.jl`
   pins the slow-improver case as the scale-free guard.
2. **Host wiring is deferred to Move 3** (mirroring the Q5 skin-exposure deferral — "no consumer yet").
   The signal is non-causal in Move 2 (nothing reads it for a decision), so wiring it into the host
   hot-loops now then re-touching them in Move 3 (where `explore_grammar` consumes it) is wire-then-rewire
   with no payoff and real behaviour-preservation risk. Move 2 ships the **mechanism** —
   `compression_exhausted`, the scale-free regime model (`initial/update_learning_regime`,
   `plateau_probability`), the `AgentState` fields + `reset_learning_regime!` — all unit-tested on
   synthetic series; Move 3 wires `−log_predictive` into the conditioning sites and the reset into the
   grammar-change sites alongside the consumer. This makes Move 2 strictly additive (no host changes),
   and the §3 "existing host tests stay green" holds trivially.

### Worked example (corrected to the implemented model — supersedes §4's illustrative numbers)

§4's decelerating series (2.1→0.76) is genuinely *transitional* — its decrements shrink to ~0.02, so the
model correctly reads it as ~0.33 (leaning improving but approaching plateau), not the illustrative 0.12.
The clean signal is **consistency**, not mere decrease. Phase A = a *consistent* improver (Δ≈0.3 steady)
→ `plateau_probability` 0.0 (still improving); Phase B = bouncing flat (≈0.755) → 0.86 (plateaued). The
ordering and the scale-free slow case (Δ≈0.08 → 0.0) are what the tests assert (direction, not magic
points).
