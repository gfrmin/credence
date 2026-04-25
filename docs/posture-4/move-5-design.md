# Move 5 — Measure deleted from `src/`; `condition` produces concentrated Previsions

## 0. Final-state alignment

Move 5 is the point of no return: it converges the current tip with `master-plan.md` §"Final-state architecture" by deleting the entire Measure type hierarchy and rewriting `condition` to emit Prevision-primary outputs. After Move 5, `src/` matches the final-state §"Types deleted" exactly: nine Measure subtypes gone, the `getproperty` shields gone, the `Functional = TestFunction` alias gone. The 1860-line `src/ontology.jl` monolith retires, replaced by the six-file split in §"New module structure": `spaces.jl`, `events.jl`, `kernels.jl`, `test_functions.jl`, `conjugate.jl`, `stdlib.jl`. Two architectural inflexions complete the alignment: (1) `MixturePrevision.components::Vector{Prevision}` and `ProductPrevision.factors::Vector{Prevision}` tighten — the deferred Move 2 Phase 4 scope lands here paired with the `condition` rewrite that is the native Prevision-primary resolution to per-component space variation; (2) the `expect-through-accessor` lint slug enforces the §"Operational surface" claim that every numerical query at a call site routes through `expect`. The transient state Move 5 leaves: `apps/`, `examples/`, the BDSL stdlib, and the skin all still construct and consume Prevision (post-Move-4); they don't yet *speak* Prevision over the wire — Moves 6–8 carry the Prevision vocabulary outward. Move 5's `src/` is the foundation those moves build against.

## 1. Purpose

Delete the Measure surface. Rewrite `condition` to produce concentrated Previsions over the ambient space, replacing posture-3's reduced-space Measure outputs. Tighten the Prevision-internal vector types so `MixturePrevision.components::Vector{Prevision}` and `ProductPrevision.factors::Vector{Prevision}` hold actual Previsions, not Measure-wrapped Previsions. Split the 1860-line `src/ontology.jl` along the axiom / structure / representation seams in `master-plan.md` §"New module structure". Land the `expect-through-accessor` lint slug (extension of the existing `credence_lint.py` pass-two taint analysis from PR #40) and retire the twelve `posterior-iteration` pragma sites currently tracked under issue #39 by routing each through the new `src/stdlib.jl` one-liners (`mean`, `variance`, `probability`, `weights`, `marginal`).

## 2. Files touched

**Deleted:**
- `src/ontology.jl` — the 1860-line monolith retires entirely. Its content redistributes per the file map below.

**Created:**
- `src/spaces.jl` — `Space` abstract type, `Finite`, `Interval`, `ProductSpace`, `Simplex`, `Euclidean`, `PositiveReals`, `support`, `atoms`, `BOOLEAN_SPACE`. Lifted from current `ontology.jl:1-150` (approximately).
- `src/events.jl` — `Event` abstract type, `TagSet`, `FeatureEquals`, `FeatureInterval`, `Conjunction`, `Disjunction`, `Complement`, `indicator_kernel`, `feature_value`. Lifted from current `ontology.jl` event-related blocks.
- `src/kernels.jl` — `Kernel`, `FactorSelector`, `LikelihoodFamily` and concrete subtypes (`LeafFamily`, `PushOnly`, `BetaBernoulli`, `Flat`, `FiringByTag`, `DispatchByComponent`, `NormalNormal`, `Categorical`, `NormalGammaLikelihood`, `Exponential`), `kernel_source`, `kernel_target`, `density`, `DepthCapExceeded`. Lifted from current `ontology.jl` kernel blocks.
- `src/test_functions.jl` — `TestFunction` abstract type, `Identity`, `Projection`, `NestedProjection`, `Tabular`, `LinearCombination`, `OpaqueClosure`, `Indicator{E}`, plus the new `CenteredPower{n}` family (see §5.1). Lifted from `src/prevision.jl` and renamed; the `Functional = TestFunction` alias is removed.
- `src/conjugate.jl` — `ConjugatePrevision{Prior, Likelihood}`, the `maybe_conjugate` registry, `update`, `_dispatch_path`. Lifted from current `prevision.jl` and `ontology.jl` conjugate blocks.
- `src/stdlib.jl` — `mean`, `variance`, `probability`, `weights`, `marginal` as one-liners over `expect`. New file replacing the accessor soup currently scattered through `ontology.jl`.

**Modified:**
- `src/Credence.jl` — exports list updated: nine Measure subtypes removed, the `Functional` alias removed, the new file `include` directives added, the new `stdlib.jl` exports added. Roughly 30 lines change.
- `src/prevision.jl` — renamed to `src/previsions.jl` (plural; matches the module name `Previsions`). Internal struct definitions for `MixturePrevision.components` and `ProductPrevision.factors` tighten from `Vector{Measure}` to `Vector{Prevision}`. The `condition` method bodies for `MixturePrevision`, `ProductPrevision`, `CategoricalPrevision`, `BetaPrevision`, `GaussianPrevision`, `GammaPrevision`, `DirichletPrevision`, `NormalGammaPrevision` are added or rewritten so each returns a concentrated Prevision over the ambient space (no space reduction). The `expect` method bodies that read structural fields stay as-is — they are the legitimate internal accessor reads `expect-through-accessor` is calibrated *not* to flag.
- `src/persistence.jl` — Measure-typed dispatch branches retire (since the Measure type retires). v3 schema unchanged in structure but the values it serialises are now Prevision-typed; the v3 fixtures captured at the Move 3 tip will not round-trip through the Move 5 loader without re-capture (see §3 for the explicit scope of fixture re-capture).
- `src/eval.jl` — references to Measure constructors retire; BDSL stdlib still builds Previsions (Move 6 carries the BDSL surface change).
- `tools/credence-lint/credence_lint.py` — extends pass-two taint analysis with the `expect-through-accessor` slug (see §4 worked example and §5.4 false-positive mitigation).
- `tools/credence-lint/corpus/expect-through-accessor/` — new corpus directory with `bad1_*`, `bad2_*`, `good_*` exemplars matching the corpus shape established in PR #40.
- All twelve `apps/julia/*/host.jl` and `apps/skin/server.jl` sites currently pragma'd `precedent:posterior-iteration — tracked in issue #39` — rewritten to route through `src/stdlib.jl` per the per-site retirement table in §5.5.

**Not modified by Move 5:**
- `apps/python/*` (Move 8 owns the Python-side rewrite).
- `apps/skin/server.jl` *except* for the three issue-#39 sites (the wire-format rewrite is Move 7).
- `examples/*.bdsl`, `examples/host_credence_agent.jl` (Move 6).
- `src/persistence.jl` schema version — stays at v3; Move 9 bumps to v4 for production state.
- The Move 0 capture (`test/fixtures/posture-3-capture/`) — pinned, never re-captured.

## 3. Behaviour preserved

Move 0 fixture at `test/fixtures/posture-3-capture/` (6124 site×value tuples at branch-point `5c6a94e`) is the invariance target. Move 5 is the largest single diff in Posture 4 by line count, but its behavioural surface is precisely Move 0's: every `expect`, every `condition`, every `mean`/`probability` returns the value the Move 0 capture recorded, at the declared tolerance.

**Expected divergences:** none in captured assertion values. Stratum-1 (`==` and `atol=1e-14`), Stratum-2 (`atol=1e-12`), Stratum-3 (`atol=1e-10`) all hold. The structure and ordering of mass on Previsions matches the structure and ordering of mass on the Measures they replace, by the construction of the rewrite.

**Anticipated re-capture: persistence v3 fixtures.** `test/fixtures/agent_state_v3.jls` and `test/fixtures/email_agent_state_v3.jls` were captured at the Move 3 tip with `MixtureMeasure(ProductMeasure(BetaMeasure(...)))` payloads. Post-Move-5, the same payloads are `MixturePrevision(ProductPrevision(BetaPrevision(...)))`. The Julia stdlib `Serialization` round-trip is type-name-tagged: a v3 fixture serialised against `MixtureMeasure` cannot deserialise against a tip where `MixtureMeasure` has been deleted. Move 5 re-captures both v3 fixtures from the Move 5 code tip; the assertion content stays identical (same alpha/beta values, same component ordering); only the type tags shift. The provenance protocol in `test/fixtures/README.md` is followed: Move 5 records the post-rewrite SHA at which the v3 fixtures were re-captured. The `__schema_version` marker stays at 3; the schema *number* is preserved because the on-disk shape is unchanged structurally, only the type tags resolve to different concrete types.

**Move 0 fixture re-capture: NO.** The Move 0 capture is sacrosanct — pinned at `5c6a94e` per `test/fixtures/README.md`. Move 5 may not re-capture it. If a divergence surfaces between Move 5 output and the Move 0 capture, the rewrite is wrong and Move 5 halts.

**Pre-merge halting criteria.** Move 5 does not merge until:
1. All twelve issue-#39 sites have retired through `src/stdlib.jl` (zero `precedent:posterior-iteration` pragmas remain in `apps/julia/*/host.jl` or `apps/skin/server.jl`). The retirement table in §5.5 lists each site's mechanism.
2. `scripts/capture-invariance.jl --verify` passes against the Move 0 fixtures with `Dict("structural" => 206, "exact" => 5727, "directional" => 88, "failing" => 0, "total_captured" => 6124, "tolerance" => 103)`.
3. `grep -rE 'struct (Categorical|Beta|Tagged|Gaussian|Dirichlet|Gamma|NormalGamma|Product|Mixture)Measure' src/` returns nothing.
4. `grep -r 'Functional = TestFunction' src/` returns nothing.
5. `tools/credence-lint/credence_lint.py` reports zero `expect-through-accessor` violations (with all twelve issue-#39 retirement-PR sites no longer pragma'd).
6. The persistence v3 round-trip test passes against the re-captured v3 fixtures.

Failure of any criterion halts the merge. This is `master-plan.md` §"Move 5 as the point of no return" enforced as a gate, not a check.

## 4. Worked end-to-end example

The canonical `condition` rewrite: posterior concentration as Prevision weights, not space reduction.

**Before (post-Move-4 tip — concentration as space reduction):**

```julia
# Three hypotheses; observe evidence eliminating two
prior = CategoricalMeasure(Finite([:h1, :h2, :h3]),
                            CategoricalPrevision([log(1/3), log(1/3), log(1/3)]))
k = Kernel(prior.space, BOOLEAN_SPACE,
           h -> CategoricalMeasure(BOOLEAN_SPACE, [...]),
           likelihood_family = Categorical(...))

posterior = condition(prior, k, true)

# Posture 3/post-Move-4 outcome: posterior is over Finite([:h1])
# — a 1-element space, the eliminated atoms are gone from the support.
@assert support(posterior.space) == [:h1]
@assert weights(posterior) == [1.0]
```

**After (Move 5 tip — concentration as Prevision weights):**

```julia
prior = CategoricalPrevision([log(1/3), log(1/3), log(1/3)])
# Note: space carried separately; CategoricalPrevision doesn't hold its own
# space (the Finite([:h1, :h2, :h3]) lives wherever the test or app
# instantiated it). Per §5.5.2's distinction: spaces are observational;
# weights are the prevision's mathematical content.

k = Kernel(Finite([:h1, :h2, :h3]), BOOLEAN_SPACE,
           h -> CategoricalPrevision(...),
           likelihood_family = Categorical(...))

posterior = condition(prior, k, true)

# Move 5 outcome: posterior is over the ambient three-element space.
# Eliminated atoms have log_weights[i] = -Inf. The space stays unchanged.
@assert posterior.log_weights == [0.0, -Inf, -Inf]
# Equivalently:
@assert weights(posterior) == [1.0, 0.0, 0.0]
```

**Why this matters for `MixturePrevision.components::Vector{Prevision}`:**

```julia
# Two component priors, each a CategoricalPrevision over the same 3-atom space.
c1 = CategoricalPrevision([log(0.5), log(0.5), 0.0])  # supports h1, h2 only
c2 = CategoricalPrevision([0.0, 0.0, log(1.0)])       # supports h3 only

mp = MixturePrevision(Finite([:h1, :h2, :h3]),
                      Prevision[c1, c2],
                      [log(0.6), log(0.4)])

# Observation lands; condition routes per-component via the kernel's
# likelihood_family and updates the mixing weights via Bayes.
post = condition(mp, k, obs)

# Move 5 outcome: post.components is still Vector{Prevision} of length 2;
# each component CategoricalPrevision is independently conditioned and
# returned over the SAME ambient space. The Move 2 Phase 4 problem
# ("shield reconstruction needs per-component space info") doesn't arise
# because no component has a reduced space — concentration lives in the
# log_weights, not in the space.
@assert length(post.components) == 2
@assert all(c -> c isa Prevision, post.components)
@assert all(c -> support(mp.space) == [:h1, :h2, :h3], post.components)  # ambient
```

**Lint walker — the `expect-through-accessor` slug:**

```python
# tools/credence-lint/credence_lint.py — pass two extends with this rule.
# Before (legitimate, inside src/previsions.jl, expect method body):
function expect(p::BetaPrevision, ::Identity)
    return p.alpha / (p.alpha + p.beta)   # OK — internal read
end

# Before (illegal, in apps/julia/email_agent/host.jl):
score = w[j] * mean_j   # was: precedent:posterior-iteration pragma'd
# After (legal):
score = w[j] * mean(comp)   # mean dispatches expect(comp, Identity())

# Lint detection: a `.alpha`, `.beta`, `.log_weights`, `.mu`, `.sigma`,
# `.kappa` read on a Prevision-typed receiver outside `src/previsions.jl`
# (or `src/conjugate.jl`, where ConjugatePrevision update method bodies
# also legitimately read structural fields) flags as
# `expect-through-accessor`. Inside those two files, the read is allowed
# without pragma.
```

## 5. Open design questions

### 5.1 `variance`, `Square`, `Power{n}`, or `CenteredPower{n}` — which TestFunction subtype family?

Master plan §"Operational surface" specifies `variance(p::Prevision) = expect(p, CenteredSquare(mean(p)))` with `CenteredSquare` as a new TestFunction subtype. Prompt 9 raised three alternatives: `Square` (uncentred, requires two `expect` calls plus subtraction), `Power{n}` parametric (covers all moments), and `CenteredSquare(mean)` (one `expect` call, parameterised on a precomputed value).

The choice has structural consequences. `Square` is the simplest case but only covers variance via `variance = expect(p, Square()) - mean(p)^2`, which is two `expect` calls and a manual subtraction at the call site — every consumer reimplements the centring. `Power{n}` is more general but makes `variance` a `Power{2}`-then-subtract pattern; same problem. `CenteredSquare(μ)` parameterises on the precomputed mean and computes `(x - μ)^2` directly inside its `apply`, so `variance(p) = expect(p, CenteredSquare(mean(p)))` is one `expect` call.

The deeper question: do we want central moments beyond variance? Skewness needs the third central moment; kurtosis the fourth. If the answer is yes, `CenteredPower{n}` parametric covers all of them with a single declared TestFunction subtype: `apply(::CenteredPower{n}, x) = (x - μ)^n`. `CenteredSquare = CenteredPower{2}` becomes a const, not a separate type; variance, third central moment, fourth central moment all work with the same machinery.

My prior: **`CenteredPower{n} <: TestFunction` parametric, with `CenteredSquare` defined as `const CenteredSquare = CenteredPower{2}`.** This:
- Lands the de Finettian one-`expect`-call shape the master plan settled on.
- Generalises to any central moment without a new type per moment.
- Keeps the type system as the encoder of structure (n is a type parameter, not a runtime field).
- Matches the precedent of `Indicator{E}` where the TestFunction subtype is parametric on its specialising structure.

**Approved with constraint:** `CenteredSquare = CenteredPower{2}` is the *only* const alias landed in Move 5. No `CenteredCube`, `CenteredFourth`, etc. Const aliases are a forking discipline — once you have two, the question of which other moments deserve named constants becomes a recurring design conversation, and the answer is always "none of them; the parametric form is the API". One alias for variance (because variance has independent name-recognition value) is fine; two starts a list.

### 5.2 `weights` for non-categorical previsions

`weights(p::CategoricalPrevision)` returns `exp.(p.log_weights)` (normalised, since CategoricalPrevision normalises at construction). What does `weights(p::BetaPrevision)` return? The continuous analogue would be the density function evaluated on a grid — but a grid choice is observational, not mathematical content of the prevision. `weights(p::MixturePrevision)` over a mixture of continuous components — undefined component-wise, well-defined for the mixing weights `exp.(p.log_weights)`.

My prior: **`weights` is constrained to finite-support Previsions: `CategoricalPrevision`, `MixturePrevision`, `ParticlePrevision`, `QuadraturePrevision`, `EnumerationPrevision`.** For `MixturePrevision`, `weights` returns the mixing weights, not anything component-wise. For continuous Previsions (Beta, Gaussian, Gamma, Dirichlet, NormalGamma), `weights` raises a custom `WeightsDomainError` (not a bare `MethodError`) whose message reads: "weights is defined only for finite-support Previsions; for continuous Previsions, use `probability(p, e::Event)` with a declared Event to obtain a measure of an event's mass, or `expect(p, f)` for an integrated functional." A `MethodError` is a Julia-level failure, not a credence-level one — the custom error teaches the caller the right alternative.

**Approved. Foreclosure on the borderline case:** discretised approximations of continuous Previsions used as proposal distributions should be `ParticlePrevision` or `CategoricalPrevision` over a discretisation declared at construction time, not implicit grids inside `weights`. The discretisation grid is a declared Space, not a hidden parameter.

### 5.3 Module split granularity — six files or trim the monolith?

Master plan §"New module structure" specifies six files: `spaces.jl`, `events.jl`, `kernels.jl`, `test_functions.jl`, `conjugate.jl`, `stdlib.jl`. The alternative is a trimmed `ontology.jl` — gutted of Measures but keeping the structure-and-axiom-functions together.

The case for the split:
- `ontology.jl` at 2041 lines (post-Move-2) was a navigation problem; even before Move 5's deletions the file mixed three concerns (frozen types, axiom-constrained functions, conjugate dispatch). The split puts each concern in its own file.
- A future move that touches kernels (e.g., a new `LikelihoodFamily` subtype) opens `kernels.jl` — not `ontology.jl`. The diff signals scope.
- The Posture 3 author noted the monolith was a mistake to be unwound at the next opportunity. Move 5 *is* the next opportunity; the monolith is being gutted regardless.

The case against:
- File proliferation friction: more `include`s in `Credence.jl`, more files to navigate.
- Some concerns straddle: `LikelihoodFamily` lives in `kernels.jl` but `ConjugatePrevision` (in `conjugate.jl`) dispatches on it; cross-file coupling.

My prior: **do the split as master plan specifies.** The six-file structure has its own coherence — each file holds one concept, the cross-coupling is acceptable because `ConjugatePrevision` legitimately needs to know about `LikelihoodFamily`. The alternative (trimmed monolith) postpones a decision Move 5 is uniquely positioned to land.

Argue if the split should be coarser (four files) or finer (eight). Or argue that the trimmed monolith is the better choice and explain what the next-natural-split-point would be if Posture 5 ever opens.

### 5.4 `expect-through-accessor` lint — false-positive mitigation

PR #40's pass-two taint analysis already distinguishes "inside an axiom-constrained function's method body in `src/`" from "elsewhere" — the seed-and-stop machinery is built. The Move 5 slug extends that machinery: a `.alpha`, `.beta`, `.log_weights`, `.mu`, `.sigma`, `.kappa`, `.factors`, `.components` read on a Prevision-typed receiver flags as `expect-through-accessor` *outside* `src/previsions.jl` and `src/conjugate.jl`. Inside those files, the read is the legitimate internal accessor that the slug is built around.

The walker needs to distinguish "I am inside a Prevision-method body in `src/previsions.jl`" from "I am in `apps/julia/email_agent/host.jl`." Two implementation approaches:

(a) **File-scope rule.** `src/previsions.jl` and `src/conjugate.jl` are excluded from the slug entirely. The pass-two walker never tags `.alpha` reads in those files. Cheap, blunt; risks false negatives if a non-method-body computation in `src/previsions.jl` reads accessors outside an `expect`/`condition` method.

(b) **Method-body scope rule.** The pass-two walker tracks "currently inside a `function expect(...)` or `function condition(...)` body" via a stateful counter that increments at `function` and decrements at `end`. Inside those bodies, accessor reads are allowed. Outside (top-level `src/previsions.jl` code, helper functions), they flag.

My prior: **(a) — file-scope rule.** The slug's purpose is to prevent call-site accessor reads in apps and tests; legitimate `src/`-internal reads are by definition in the file that defines the type. The pass-two implementation cost is one line (a regex-based file-path filter, same shape as the Posture 3 `apps/julia/pomdp_agent/` exclusion already in `credence_lint.py`). Approach (b) is more precise but the imprecision of (a) doesn't matter — `src/previsions.jl` is the only file authoring Prevision-internal logic; if we ever need to author Prevision-internal logic elsewhere in `src/`, that file gets added to the exclusion list as a one-liner.

**Approved.** The file-scope exclusion is *upgradable* without a master-plan amendment if a concrete false-negative case arises. If a helper in `previsions.jl` is found to need the lint, the rule is promoted to body-scope or the helper is relocated; this is a maintenance decision, not a re-architecture.

### 5.5 Issue-#39 retirement table

The twelve sites currently pragma'd `precedent:posterior-iteration — tracked in issue #39`. Per §3 halting criterion 1, all twelve must retire through `src/stdlib.jl` (one of: `mean`, `variance`, `probability`, `weights`, `marginal`) or through a new `TestFunction` subtype before Move 5 merges. The table below is the design-doc-time plan; mid-implementation discoveries amend the design doc per the §5.5.1 cadence claim.

| File | Line | Current pragma reason | Retirement mechanism |
|------|------|-----------------------|----------------------|
| `apps/skin/server.jl` | 250 | inline Bernoulli log-density | `density(k, h, obs)` — kernel's declared `density` method covers Bernoulli log-density; no closure required |
| `apps/skin/server.jl` | 1108 | mixture label prob by hand (`w[j] * mean_j`) | `probability(mp, FeatureEquals(:label, val))` over the mixture, where `Indicator(FeatureEquals(...))` is the declared TestFunction |
| `apps/skin/server.jl` | 1111 | mixture label prob by hand | same as 1108 (paired computation) |
| `apps/julia/grid_world/host.jl` | 84 | inline Bernoulli log-density | `density(k, h, obs)` (same as skin.250) |
| `apps/julia/grid_world/host.jl` | 117 | mixture P(enemy) by hand (`w[j] * (rec == :enemy ? mean_j : 1 - mean_j)`) | `probability(mp, FeatureEquals(:rec, :enemy))` |
| `apps/julia/grid_world/host.jl` | 397 | mixture P(enemy) by hand | same as 117 |
| `apps/julia/email_agent/host.jl` | 141 | mixture EU by hand (`w[j] * p`) | `expect(mp, LinearCombination)` where the LinearCombination encodes the per-action utility coefficients; or `probability` if the EU is a probability of a labelled outcome |
| `apps/julia/email_agent/host.jl` | 510 | inline Bernoulli log-density | `density(k, h, obs)` |
| `apps/julia/email_agent/host.jl` | 549 | inline Bernoulli log-density | `density(k, h, obs)` |
| `apps/julia/qa_benchmark/host.jl` | 92 | EU of submit by hand from argmax weight | `expect(p, LinearCombination([(REWARD_CORRECT, Indicator(correct_event)), (PENALTY_WRONG, Indicator(wrong_event))]))` |
| `apps/julia/rss/host.jl` | 177 | inline Bernoulli log-density | `density(k, h, obs)` |
| `apps/julia/rss/host.jl` | 283 | mixture relevance score by hand | `expect(mp, LinearCombination)` over the per-component reliability×fires structure |

**Note: master plan said "thirteen" sites; current grep finds twelve.** The thirteenth site retired between drafting `claude-code-prompts.md` and Move 5 implementation — most likely Move 4's mechanical migration removed a pragma incidentally. The halting criterion is "zero issue-#39 pragmas at Move 5 tip", not "twelve retire" — the count is the count.

## 6. Risk + mitigation

**Risk (highest — point of no return):** A Stratum-2 or Stratum-3 deviation between Move 5 output and the Move 0 capture surfaces only after merge, when reverting requires a branch-level rollback. *Mitigation:* the §3 pre-merge halting criteria gate the merge; `--verify` runs in CI; the SHA at which v3 fixtures are re-captured is recorded in `test/fixtures/README.md` so the post-merge rollback target is unambiguous if the worst case happens.

**Risk (high — `condition` rewrite logic error):** The shift from reduced-space Measures to ambient-space concentrated Previsions changes the meaning of every posterior value reads downstream. A bug in the rewrite would manifest as Move 0 capture mismatches in the test suite — but only at Stratum-1 (`==` exact) for the simplest cases; mixture-of-product cases involve multi-step concentration and a logic error could pass at Stratum-1 while drifting at Stratum-2 (`atol=1e-12`). *Mitigation:* the worked example in §4 is the smallest non-trivial case (CategoricalPrevision); test suite coverage of mixture-of-product is provided by `test/test_flat_mixture.jl` and `test/test_prevision_mixture.jl` (both Move-4-migrated); persistence v3 round-trip tests structural integrity.

**Risk (medium — module-split breakage):** The six-file split changes `include` order; a subtle dependency cycle (e.g., `Event` declared in `events.jl` but referenced in `kernels.jl` for `indicator_kernel`'s codomain) could break load time. *Mitigation:* `include` order in `Credence.jl` follows the layering: `spaces.jl` first (no dependencies), `events.jl` second (depends on spaces), `kernels.jl` third (depends on spaces and events), `test_functions.jl` fourth, `previsions.jl` fifth (depends on all of the above), `conjugate.jl` sixth (depends on previsions and kernels), `stdlib.jl` seventh. Reordering is cheap; cycle detection is by load failure.

**Risk (medium — issue-#39 retirement uncovers stdlib gap):** A site whose retirement plan in §5.5 turns out not to compile (e.g., the `LinearCombination` encoding for `qa_benchmark/host.jl:92` doesn't have the right algebraic structure) signals the stdlib is incomplete. *Mitigation:* per §3 halting criterion 1, Move 5 does not merge until all twelve sites retire. If a site genuinely cannot retire through the stdlib, the design doc amends to introduce the missing TestFunction subtype or stdlib function, and Move 5 lands the addition before retiring the site. Per §5.5.1's cadence claim, mid-implementation amendment is expected.

**Risk (low — lint slug false negatives):** Approach (a) in §5.4 (file-scope rule) excludes `src/previsions.jl` and `src/conjugate.jl` entirely; a future helper function in those files that reads accessors illegitimately would not flag. *Mitigation:* the corpus self-test in `tools/credence-lint/corpus/expect-through-accessor/` includes `bad2_*` exemplars that exercise the boundary (e.g., a top-level statement in `src/previsions.jl` that is *not* inside a method body and reads accessors); if a future case demands precision we'll revisit, but the file-scope rule is sufficient under the current code shape.

**Process-failure risk: design-doc amendment cycle exceeds budget.** Per §5.5.1's cadence claim from Move 4, one amendment is expected. Move 5's blast radius is large enough that two or three amendments are conceivable. *Mitigation:* the master plan budgets accordingly; an amendment PR is first-class artefact, not a process exception. If three amendments land before code, the design-doc-vs-code phasing is doing its job — surfacing premise failures before they break behaviour.

## 7. Verification cadence

Run at the end of each code PR (Move 5 may split into sub-PRs along the lines of "split + rewrite", "lint slug", "issue-#39 retirements" — see §5.6 if introduced; otherwise one PR):

- `julia test/test_*.jl` for all 13 test files: full suite passes.
- `julia --project=scripts scripts/capture-invariance.jl --verify`: `Dict("structural" => 206, "exact" => 5727, "directional" => 88, "failing" => 0, "total_captured" => 6124, "tolerance" => 103)`; ✓ Verified: manifests identical (modulo timestamp).
- `python tools/credence-lint/credence_lint.py corpus`: corpus self-test passes including new `expect-through-accessor` exemplars.
- `python tools/credence-lint/credence_lint.py check apps/`: zero violations including zero `expect-through-accessor` violations and zero `posterior-iteration` pragmas.
- `julia test/test_persistence.jl`: v3 round-trip + re-captured v3 fixtures load.
- `apps/skin/test_skin.py` smoke: passes against the modified `apps/skin/server.jl` lines 250 / 1108 / 1111. The full skin rewrite (Move 7) is out of scope; the smoke test confirms the three issue-#39 retirements don't regress the wire format.

CI gates same as Move 4: `unit-tests` (Julia compile + Python pytest) and `smoke-build` (amd64 Docker).

## 8. de Finettian discipline self-audit

1. **Is every numerical query in this move routed through `expect`?** Yes. `mean`, `variance`, `probability`, `weights`, `marginal` are all one-liners over `expect`. The new `CenteredPower{n}` TestFunction subtype is the declared structure that lets `variance` be a one-liner. The twelve issue-#39 retirements all route through `expect` (or `density`, which is the kernel's declared method, not a numerical query on a prevision). No new function returns a `Float64` describing a probabilistic property without calling `expect`.

2. **Does this move hold a Prevision inside a Measure, or a Measure inside a Prevision, for any reason?** No. The Measure type retires entirely. `MixturePrevision.components::Vector{Prevision}` and `ProductPrevision.factors::Vector{Prevision}` tighten — no more `Vector{Measure}`-of-Prevision-wrappers. The `wrap_in_measure` helper Move 2 landed retires with the Measure surface (the `getproperty` shields it relied on are gone). Net: zero residency confusion at Move 5 tip.

3. **Does this move introduce an opaque closure where a declared structure would fit?** No. `CenteredPower{n}` is parametric on `n` (declared, not closure). `Indicator{E}` is parametric on the Event type (declared). `LinearCombination` carries `Vector{Tuple{Float64, TestFunction}}` (declared). `OpaqueClosure` remains as the explicit fallback for cases the declared structure doesn't cover; no new use of it is introduced. The lint slug enforces this externally.

4. **Does this move add a `getproperty` override on any Prevision subtype?** No. The Posture 3 shields retired with Measure; no new shields land. Direct field reads inside `expect` and `condition` method bodies in `src/previsions.jl` and `src/conjugate.jl` are by design — they are the legitimate internal accessors the `expect-through-accessor` slug is calibrated to allow.
