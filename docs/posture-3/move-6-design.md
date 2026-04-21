# Move 6 design — `ParticlePrevision`, `QuadraturePrevision`, `EnumerationPrevision`

Status: design doc (docs-only PR 6a). Corresponding code PR is 6b.

Template reference: `docs/posture-3/DESIGN-DOC-TEMPLATE.md`.

Master plan reference: `docs/posture-3/master-plan.md` § "Move 6 — Execution layer refactor (the high-risk move)".

Working reference: `docs/posture-3/precedents.md` — consulted throughout §5 and §6 drafting.

## 1. Purpose

Move 6 is the only high-risk move remaining in Posture 3. It refactors the execution layer — particle filtering, quadrature, and enumeration — into Prevision subtypes. Currently, three code paths produce `CategoricalMeasure(Finite(samples-or-grid), log_weights)` as the representation of a conditioned posterior when the conjugate registry (Move 4) doesn't match: (i) the importance-sampling fallback at `src/ontology.jl:1490,1498`; (ii) the grid quadrature at `src/ontology.jl:1242-1251,1254-1265`; (iii) the particle fallback for `ProductMeasure` at `src/ontology.jl:1487-1490`. Move 6 replaces each with a type-structural Prevision subtype — `ParticlePrevision`, `QuadraturePrevision`, `EnumerationPrevision` — each carrying its strategy's invariants at the type level (per Invariant 2 discipline).

The "high-risk" rating is load-bearing. Particle filtering is where bit-for-bit reproducibility under seeded RNG has its first real test: Move 2 pinned the seeded-MC `==` precedent as a principle; Moves 4 and 5 reaffirmed it without exercising it. Move 6 exercises it. Any sample-order change introduced by the refactor — whether from RNG consumption reordering, constructor-level arithmetic reassociation, or internal reduction reassociation — breaks the `==` assertion. The strata discipline names this as halt-the-line rather than tolerance-relaxation territory.

## 2. Files touched

**Modified:**

- `src/prevision.jl` — adds `ParticlePrevision(samples::Vector, log_weights::Vector{Float64}, seed::Int)`, `QuadraturePrevision(grid::Vector{Float64}, log_weights::Vector{Float64})`, `EnumerationPrevision(enumerated::Vector, log_weights::Vector{Float64})`. Constructors perform logsumexp normalisation on `log_weights` (same invariant as existing `MixturePrevision`). Exports extended.
- `src/ontology.jl:1487-1490` — refactor `_condition_product_fallback(m::ProductMeasure, k, obs)` to construct `ParticlePrevision(samples, log_weights, seed)` rather than `CategoricalMeasure(Finite(samples), log_weights)`. Wrap in a `CategoricalMeasure` facade for consumer compatibility (Move 3-style), since the Measure surface is what consumers see.
- `src/ontology.jl:1495-1498` — same refactor for the generic `condition(m::Measure, k, obs; n_particles)` fallback.
- `src/ontology.jl:1242-1251, 1254-1265` — `_condition_by_grid(m::BetaMeasure, …)` and `_condition_by_grid(m::GaussianMeasure, …)` refactor to construct `QuadraturePrevision(grid, log_weights)` then wrap.
- `src/program_space/enumeration.jl` — program enumeration constructs `EnumerationPrevision` rather than ad-hoc accumulator; this is cosmetic (the site is the only change needed).
- `src/Credence.jl` — add `ParticlePrevision`, `QuadraturePrevision`, `EnumerationPrevision` to exports.
- `apps/skin/server.jl` — extend `_dispatch_path` handler to return the extended vocabulary (§5.3). The RPC return shape stays `{"path": "<symbol>"}`; the string values extend from `{"conjugate", "particle", "mixed"}` to `{"conjugate", "particle", "quadrature", "enumeration", "mixed"}`.

**New:**

- `test/test_prevision_particle.jl` — Stratum-1/Stratum-2 corpus for particle/quadrature/enumeration paths. Opens with the canonical-seed particle test (sample order invariance under `==`), extends to quadrature bit-for-bit invariance, closes with the `_dispatch_path` vocabulary pins for all three strategies.
- `apps/skin/test_skin.py` — adds `test_particle_path_roundtrip` and `test_grid_fallback_roundtrip` per the Move 0 skin surface audit's Move 6 section. `_set_seed(seed::Int)` RPC extension added if a test needs explicit RNG control from the Python side (per audit); otherwise tests rely on `Random.seed!` at the top of each skin test (matching existing test_skin.py pattern).

**Not touched:**

- The conjugate registry (Move 4) and MixturePrevision (Move 5) are unchanged. `maybe_conjugate` still returns `nothing` for unregistered pairs; the post-Move-6 effect is that the fallback construction produces a typed Prevision subtype rather than a CategoricalMeasure.
- Consumer code: no site constructs `CategoricalMeasure(Finite(samples), log_weights)` outside the particle path (confirmed by grep — §6 R2). Consumer reads of particle-state posteriors go through `weights()`, `mean()`, `support()` — all covered by the Measure surface. No `.samples` field access anywhere in `src/`, `test/`, or `apps/`.

## 3. Behaviour preserved

### Stratum-2 tolerances for particle / quadrature / enumeration paths

Per `precedents.md` §4:

- **Particle under deterministic seeding:** `==`. Not `rtol=1e-12`. The only legitimate drift is floating-point reassociation from constructor changes, bounded by ~1e-12 — but tighter in practice under `==`, because the pre-refactor sample sequence, log-density evaluation, and weight array construction are all sequentially deterministic under a fixed seed. Any `==` failure is a sample-order change, which is posterior-changing by definition.
- **Quadrature:** `atol=1e-14`. Pairwise-reduction-legal arithmetic budgets. `sum(exp(logw[i])*density[i])` and logsumexp over a fixed grid of 64 points have reorder room; `1e-14` covers that without masking a grid-ordering regression.
- **Enumeration:** `==` under deterministic iteration order. Program enumeration is a depth-first walk over a grammar; the walk order is deterministic; the weight accumulation is a deterministic sum. Any deviation at `==` is an enumeration-order bug, not a tolerance issue.

### Verification invariants

1. The pre-Move-6 test suite must pass with identical results post-refactor: every `test/test_*.jl` file that exercises a particle or grid path (grid_world, email_agent, rss, program_space) asserts equality at `rtol=1e-10` today; Move 6 tightens where possible to `==`, but a test that passes at `rtol=1e-10` is the minimum.
2. The new `test/test_prevision_particle.jl` asserts `_dispatch_path == :particle` on any non-conjugate fallback path (importance-sampling, grid quadrature, or program-space enumeration — all three return the uniform fallback label per §5.3). Per-strategy drilldown, if a specific test needs it, queries the concrete Prevision subtype directly (`isa ParticlePrevision`, `isa QuadraturePrevision`, `isa EnumerationPrevision`) rather than pattern-matching on distinct Symbols.
3. Per-seed reproducibility: `Random.seed!(42); condition(m, k, obs)` produces the same `ParticlePrevision` at byte-level across runs. The seed-consumption order is part of the invariant; shifting a `randn()` call inside the refactor breaks the test.

### Behaviour NOT preserved

None of the samples / grid points / enumerated paths change. None of the weight arithmetic changes. The refactor is type-structural — the representation shifts from `CategoricalMeasure{Finite}` to a typed Prevision subtype — but the underlying arrays stored in the new type are bit-identical to what the old `CategoricalMeasure` held.

## 4. Worked end-to-end example

**Inputs:** an `ExchangeablePrevision` prior over a categorical outcome space; decompose to a MixturePrevision; condition with a kernel that doesn't match any registered conjugate pair for the decomposed components. Canonical seed: `Random.seed!(42)`.

```julia
# Setup: a 3-category exchangeable prior. decompose produces a 3-component
# MixturePrevision of degenerate CategoricalMeasures weighted by α/Σα.
ep = ExchangeablePrevision(Finite([:a, :b, :c]),
                           DirichletPrevision([2.0, 3.0, 5.0]))
mp = decompose(ep)   # MixturePrevision with 3 CategoricalMeasure components,
                     # log_weights = log.([0.2, 0.3, 0.5]).
mix = MixtureMeasure(Finite([:a, :b, :c]), mp.components, mp.log_weights)

# A kernel that's deliberately NOT a Categorical-pair conjugate. Source/target
# mismatched to any registered family — forces fallback to particle.
k = Kernel(Finite([:a, :b, :c]), Euclidean(1),
           cat -> error("generate not used"),
           (cat, o) -> cat === :a ? -0.5 * (o - 0.0)^2 :
                       cat === :b ? -0.5 * (o - 1.0)^2 :
                                    -0.5 * (o - 2.0)^2;
           likelihood_family = PushOnly())   # no conjugate claim

obs = 1.3
Random.seed!(42)
```

**Step-by-step dispatch:**

```julia
# 1. condition(mix::MixtureMeasure, k, obs) — Move 5 facade (src/ontology.jl).
condition(mix, k, obs)
  ↓
# 2. MixtureMeasure facade delegates to condition(mix.prevision, k, obs).
#    That's the MixturePrevision-level per-component coordinator from Move 5
#    (src/ontology.jl, owned semantically by MixturePrevision).

# 3. Per-component loop (Move 5 § condition(p::MixturePrevision, k, obs)):
#    - _resolve_likelihood_family(k.likelihood_family, comp_i) → PushOnly for
#      all three components (the unwrap finds no FiringByTag/DispatchByComponent).
#    - _predictive_ll(comp_i, k, obs) — importance-weight estimate.
#    - condition(comp_i::CategoricalMeasure, k, obs) — delegates to the
#      generic fallback (Move 4 facade: maybe_conjugate returns nothing
#      because no (CategoricalPrevision, PushOnly) pair is registered).

# 4. Per-component dispatch path:
#    _dispatch_path(comp_i.prevision, k)  →  :particle   (Move 6 vocabulary)
#    Rollup at the MixturePrevision level:
#    _dispatch_path(mix.prevision, k)     →  :mixed       (Move 5 §5.4: not
#                                                          all :conjugate, so
#                                                          :mixed fires).
#
#    Halt-the-line: if any component returns :conjugate unexpectedly, the
#    registry fired for a pair that shouldn't have — bug. Per §5.3 the
#    fallback label is uniform :particle across the three strategies;
#    tests that need to assert "this specific path took quadrature, not
#    importance-sampling" check the concrete Prevision subtype via
#    isa ParticlePrevision / isa QuadraturePrevision rather than the
#    Symbol.

# 5. Inside each per-component condition call (src/ontology.jl generic
#    fallback, lines 1495-1498 post-Move-6):
#    - samples = [draw(comp_i) for _ in 1:1000]
#      comp_i is a degenerate CategoricalMeasure (point mass on one of
#      :a/:b/:c), so every draw returns the same symbol. Deterministic
#      under the seed.
#    - log_weights = Float64[k.log_density(s, obs) for s in samples]
#      Evaluated at each sample; all log-weights within a component are
#      identical (component is degenerate), equal to the kernel's density
#      at that category with obs=1.3.
#    - Construct ParticlePrevision(samples, log_weights, seed=42).
#    - Wrap in CategoricalMeasure(Finite(samples), log_weights) — the
#      Measure-surface facade (Move 3 shield pattern extended to the new
#      ParticlePrevision subtype).

# 6. Reassemble at the MixturePrevision level: three conditioned components,
#    each a CategoricalMeasure wrapping ParticlePrevision. New log_weights
#    are the prior log_weights + per-component pred_ll. Logsumexp normalise.

# 7. MixtureMeasure(mix.space, new_components, new_log_weights) returned.
#    Consumer code reading weights(result), mean(result) sees the expected
#    posterior shape.
```

**Dual-residency trace — what moves from `src/ontology.jl` to `src/prevision.jl` and what stays.** The master plan's framing "particle-path arithmetic moves from `src/ontology.jl` to `src/prevision.jl`" is directionally correct but needs nuance: the *arithmetic* (the `draw` loop, the `k.log_density` evaluation) stays in `src/ontology.jl` because it depends on Measure-level types (`draw(m::Measure)`, `k.log_density(s, obs)` both live Measure-side). What relocates is the **result-representation**: pre-Move-6, the fallback body constructs a `CategoricalMeasure` directly (Measure-side); post-Move-6, the body constructs a `ParticlePrevision` (prevision-side) then wraps it in a `CategoricalMeasure` for consumer-surface compatibility. The wrapping is a Move 3 shield — consumer code reading `.logw`, `.space.values`, `weights()`, `mean()` sees the same values, resolving through the shield to the underlying Prevision's fields.

The trace across the module boundary, step by step: in §4.5 above, the `samples` and `log_weights` Vectors are constructed ontology-side using `draw` and `k.log_density`; they are then handed to the `ParticlePrevision(samples, log_weights, seed)` constructor, which lives in `src/prevision.jl`; the returned `ParticlePrevision` is wrapped by a `CategoricalMeasure(Finite(samples), log_weights)` facade — again ontology-side. Two module crossings per fallback call: ontology → prevision at construction, prevision → ontology at wrapping. Construction is prevision-side (the new struct lives there); wrapping is ontology-side (CategoricalMeasure lives there); arithmetic is ontology-side (draws and densities live there).

**Result bit-invariance trace.** Pre-Move-6, the 1000 samples drawn under seed 42 are stored in `CategoricalMeasure.space.values`. Post-Move-6, the same 1000 samples are stored in `ParticlePrevision.samples`, then referenced (by the shared-reference contract, precedent #2) as `CategoricalMeasure.space.values` via the wrapper's `getproperty` shield. Byte-identical arrays, byte-identical log-weights, byte-identical logsumexp normalisation. The `==` assertion in the Stratum-2 test holds. Any `==` failure at this boundary is halt-the-line: a sample-order change, an RNG consumption reorder, or a constructor-level reassociation.

**Not vestigial.** The `CategoricalMeasure` facade wrapping `ParticlePrevision` is not a vestige — it's the Move 3 shield pattern extended to a new Prevision subtype. Consumer code accessing `.logw`, `.space.values`, `weights()`, `mean()` reads through the shield; the underlying storage is the new Prevision's fields. Deleting the facade would require rewriting every consumer of particle-path posteriors. Move 6 explicitly preserves the Measure-surface API; the facade is load-bearing for that preservation, not leftover.

## 5. Open design questions

### 5.1 (substantive) `ParticlePrevision.update` contract — in-place vs new-value return

A `condition` call on a `ParticlePrevision` constructs a new posterior. The question: does the new posterior share `ParticlePrevision.samples` with the prior (in-place update on log_weights, samples-reference-shared), or is it a freshly-allocated `ParticlePrevision` with new `samples` and `log_weights` arrays?

- **Option A (new-value return):** each `condition` returns `ParticlePrevision(new_samples, new_log_weights, seed)` — fresh Vectors, no shared state with the prior.
- **Option B (in-place mutation, shared-reference):** `update(pp::ParticlePrevision, obs)` reuses the `samples` Vector (samples don't change under importance reweighting — only weights do) and mutates `log_weights` in place. The returned `ParticlePrevision` shares `samples` with the prior; `log_weights` is updated in-place or by fresh allocation depending on the path.

**Recommendation: A (new-value return).**

Two technical reasons:

1. **The wrapper-allocation cost is dominated by the sample array construction itself.** An importance-sampling condition constructs a new `samples` Vector of length `n_particles` (typically 1000). Each sample is a `draw` call that allocates per-sample state (e.g. `Tuple{Float64, Float64}` for NormalGammaMeasure's `(μ, σ²)` pair). The log-weights Vector is another n-length allocation. A fresh `ParticlePrevision` wrapper is O(1) storage relative to the sample-and-weights payload. The "avoid allocation on hot paths" argument for in-place is weak — the hot-path allocation cost is paid by the sample/weights arrays, not the wrapper.

2. **Move 6 introduces a new type whose invariants are fresh.** Opting for immutability keeps the mental model simple: `condition(pp, k, obs)` returns a new `ParticlePrevision`; there's no shared-reference asterisk to reason about. The Move 3 shared-reference contract (precedent #2) existed to preserve existing skin-server `push!` patterns on `MixtureMeasure.components` — a specific, pre-existing consumer. No equivalent consumer exists for `ParticlePrevision.samples`; the grep confirms zero sites access `.samples` today.

**Invitation to argue.** Profiling on a specific particle-heavy workload (e.g. `grid_world`'s tight inner loop of 1000-particle conditions) may show the wrapper allocation is measurable relative to the sample/weights construction. If a profile surfaces a non-trivial fraction of time in the wrapper allocation — not the dominant cost, but a cost above noise — in-place becomes the right call. Commit to new-value; reverse if profiling contradicts. Measuring here is cheaper than assuming.

### 5.2 (substantive) Quadrature and enumeration as Prevision subtypes — one type or three

The refactor introduces new Prevision subtypes to carry strategy-specific state. Two shapes:

- **Option A (three distinct types):** `ParticlePrevision`, `QuadraturePrevision`, `EnumerationPrevision`. Each has its own constructor, fields, and methods on `condition` / `_dispatch_path` / `update`. Julia's method-table dispatches on the concrete type.
- **Option B (one unified type with strategy tag):** `FallbackPrevision(strategy::Symbol, state::NamedTuple)`. The strategy tag (`:particle | :quadrature | :enumeration`) is read at dispatch time; methods branch on it.

**Recommendation: A (three types).**

Two technical reasons:

1. **Invariant 2 compliance.** The Posture 3 reconstruction's core pattern is "structure declared at the type level." Move 4's conjugate registry dispatches on `(Prior, Likelihood)` type pairs via Julia's method table; Move 5's MixturePrevision routing loop resolves `FiringByTag`/`DispatchByComponent` at the type level. A single unified `FallbackPrevision` with a Symbol tag reintroduces the dynamic-tag dispatch Invariant 2 rejected for `Kernel.likelihood_family` (the `PushOnly` vs `BetaBernoulli` distinction is type-level, not tag-level). Three distinct types preserve the discipline; one tagged type breaks it.

2. **The three strategies have genuinely different structural invariants.** Particle carries `(samples, log_weights, seed)` — the seed is load-bearing for reproducibility. Quadrature carries `(grid, log_weights)` — no seed, deterministic grid. Enumeration carries `(enumerated_paths, log_weights)` — paths are depth-bounded AST shapes with their own grammar-derived invariants. Collapsing to a single `state::NamedTuple` field forces runtime checks for "does this strategy have a seed?" / "does this strategy have a grid?" that the type system can express directly with three types.

**Invitation to argue.** Option B becomes correct if Julia's multiple-dispatch on three distinct types proves measurable at particle-heavy hot paths under profiling. Not expected — dispatch is a compile-time-resolvable method-table lookup, and the three types' methods are non-overlapping — but profile if Move 6 surfaces a hot-path bottleneck at the dispatch level. §5.2 is independent of §5.3: the type decision reflects structural invariants; the Symbol decision reflects observability needs. A change to one doesn't cascade to the other.

### 5.3 (substantive) `_dispatch_path` vocabulary extension

Move 4 committed `:conjugate` / `:particle` for the base Prevision hook; Move 5 committed `:mixed` for MixturePrevision rollup. Move 6 introduces three distinct fallback strategies. Three shapes:

- **Option A (uniform `:particle` for any non-conjugate):** the current Move 4 scheme. `_dispatch_path` returns `:conjugate` or `:particle`; the "which fallback" question isn't surfaced at the hook.
- **Option B (distinct labels):** `:conjugate` / `:particle` / `:quadrature` / `:enumeration` / `:mixed`. Each strategy gets its own label.
- **Option C (two-tier):** `:conjugate` / `:fallback` at the rollup level; tests that need fallback-type drill down via per-Prevision `_dispatch_path` on the concrete type.

**Recommendation: A (uniform `:particle`).**

Two technical reasons:

1. **No planned test in Moves 6, 7, or 8 requires distinguishing `:quadrature` from `:particle` from `:enumeration`.** The assertions that tests actually want to make against `_dispatch_path` are "did the conjugate path fire?" (halt-the-line on silent registry miss, per precedent #5) and "did a fallback fire?" (for non-conjugate cases). The strategies share the "non-conjugate fallback" category that tests assert against; the further distinction between quadrature and particle and enumeration doesn't correspond to a test condition anyone has named. A speculative "future test might want to pin quadrature specifically" is not a current consumer, and precedent #7 (§5 reasoning strength) rules against committing vocabulary on speculative need.

2. **Adding labels later is a one-line change; removing them is breaking.** If after Move 6 a consumer genuinely needs to distinguish fallback strategies, extending the vocabulary is additive — new Symbols that no existing test pattern-matches on. Conversely, committing to three distinct labels now and having downstream tests pattern-match on `:particle` vs `:quadrature` locks a vocabulary that can't be collapsed without rewriting tests. Defer the specificity until a consumer actually needs it.

**Decoupling from §5.2.** The type system already distinguishes the three strategies (§5.2 Option A: three distinct Prevision subtypes). Observability and dispatch are separate concerns — the Symbol label summarises "did the conjugate path miss?", the type tells you which fallback fired if anyone needs to drill down. Drilldown is already available via `_dispatch_path(p.components[i], k)` on the concrete component, same as Move 5's per-component pattern. Three types + one fallback label is coherent; the additional labels in Option B would duplicate information the type system already carries.

**Invitation to argue.** Option B becomes correct if a concrete consumer surfaces within Move 6's code PR or Moves 7/8's design docs — a test case whose assertion cannot be expressed without distinguishing the fallback strategies at the observability level. Extending the vocabulary in that PR is additive; no prior test pattern-matches on the extended labels, so the extension is one-line. Default to A; promote to B when a consumer demands the distinction, not when the distinction sounds useful.

## 6. Risk + mitigation

**Risk R1 (main risk, per master plan): seed-consumption order regression.** Particle filtering under deterministic seeding produces bit-identical results iff (i) the seed is set at the same call-chain point pre- and post-refactor, and (ii) the RNG is consumed in the same order. If Move 6's refactor introduces a `randn()` call that the pre-refactor code didn't have, or reorders `draw` calls, the post-refactor sample sequence differs — silently. The `==` assertion catches this loudly if it fires; *the work is ensuring the assertion fires when the sequence drifts, not only when the mean does.* *Investigation posture if breached:* halt. Read the refactored code paths side-by-side with the pre-refactor versions. Identify any `Random` call that moved; identify any new array construction that could have internal RNG consumption. Do not loosen the tolerance. *Caught by:* `test/test_prevision_particle.jl`'s canonical-seed byte-exact assertion on the post-condition `ParticlePrevision.samples` and `.log_weights`; plus existing tests under `Random.seed!(42)` in `test_grid_world.jl` and `test_email_agent.jl` that assert value equality.

**Risk R2 (low): pre-emptive grep for the Move 6 surfaces.** Pattern search 2026-04-21 across `src/`, `test/`, `apps/`, `docs/` for the four target surfaces named in the Move 6 prompt:

| Target | Total hits | Category (a) | Category (b) | Category (c) |
|--------|-----------|--------------|--------------|--------------|
| `CategoricalMeasure(Finite(samples), log_weights)` particle-path constructions | 2 src + 0 consumer | 2 relocation targets (src/ontology.jl:1490, 1498) | 0 | 0 |
| `_condition_by_grid` quadrature-path constructions | 2 src + 2 callers | 2 relocation targets (src/ontology.jl:1242, 1254); 2 upstream calls (lines 1086, 1195) unchanged | 0 | 0 |
| `.samples` on particle-state posteriors | 0 | 0 | 0 | 0 |
| `.log_weights` on particle-state posteriors | ~15 across src/test/apps | All covered by Move 3's `getproperty` shield (precedent #2); post-Move-6 the shield forwards `.log_weights` through the wrapper to `ParticlePrevision.log_weights`. Invariant preserved. | 0 | 0 |
| `Random.seed!` threading through conditioning | 20+ in tests; 1 in apps/julia/grid_world/host.jl; 1 in apps/julia/rss/features.jl (MersenneTwister) | All test-scope; seed set before conditioning calls; no site passes an RNG object *through* `condition` as an argument | 0 | 0 |

**Category (a) — covered unchanged: all hits.** All construction sites stay valid; the wrapper facade preserves Measure-surface field access; no consumer uses `.samples` today.

**Category (b) — minor adaptation: 0.**

**Category (c) — mutations or plan-amending hits: 0.**

Go/no-go gate: **GO.** One naming drift flagged below as R4 — non-blocking but design-doc should name it explicitly.

**Risk R3 (medium): Prevision-subtype wrapper-getproperty consistency.** Move 3's `getproperty` shield on `MixtureMeasure` forwards `.components` and `.log_weights` to the underlying `MixturePrevision`. Move 6 introduces three new wrapping cases: `CategoricalMeasure` wrapping `ParticlePrevision` (samples → space.values + log_weights → logw), `CategoricalMeasure` wrapping `QuadraturePrevision` (grid → space.values + log_weights → logw), `CategoricalMeasure` wrapping `EnumerationPrevision`. Each needs a shield entry; each needs a contract test in `test/test_prevision_particle.jl` (precedent #3: invariant-comment-names-test). *Caught by:* contract tests per the precedent #2 pattern — construct, read-through-shield, mutate-in-place (log_weights only — samples don't mutate under importance reweighting), re-read, assert. If Move 6 opts for Option A in §5.1 (new-value return, not in-place), the contract test narrows to reference-identity between wrapper `.logw` and underlying `.log_weights`.

**Risk R4 (low): naming drift between master plan (`_condition_particle`) and master code (`_condition_product_fallback` + generic fallback).** The master plan references `_condition_particle` as the name of the fallback function; the actual master code at `src/ontology.jl:1487-1490, 1495-1498` uses `_condition_product_fallback` (for ProductMeasure) and an unnamed `condition(m::Measure, k, obs; n_particles)` (the generic one). Move 6 can either (a) rename to `_condition_particle` as part of the refactor — aligning master-plan vocabulary with code — or (b) keep the existing names and update the master plan to reflect what the code actually says. *Recommendation: rename to `_condition_particle` in the code PR.* Two-line change; restores master-plan-to-code parity; the vocabulary-pins precedent (precedents.md §6) argues for doing it at the refactor moment rather than deferring.

**Rename scope, explicit.** Master-plan-to-code parity isn't preserved by renaming only the code — the plan document is consumed by future sessions reading master, and a stale plan re-introduces the drift when those sessions re-read. The rename commit touches both: `src/ontology.jl` (rename the two fallback functions + any internal call sites) *and* `docs/posture-3/master-plan.md` (currently references `_condition_particle` at lines 136 and 195; either update the line to name the new function or leave the plan's forward-looking reference pointing at the now-existing name). Pre-rename grep to run at code-PR time: `grep -n '_condition_particle\|_condition_product_fallback' src/ docs/ test/` and include every hit's resolution in the same commit. A rename commit that leaves the plan stale is a half-fix that the next session reads and re-drifts from.

**Risk R5 (medium): skin smoke extensions arrive under existing test_skin.py surface.** The Move 0 skin audit names `test_particle_path`, `test_grid_fallback`, `test_particle_snapshot` as Move 6 additions plus a `_set_seed(seed)` RPC for deterministic Python-side seeding. The audit explicitly calls out that deferring any of these to follow-up is "a recipe for a silent break post-merge." *Caught by:* skin smoke tests extended in the Move 6 code PR (6b), not as follow-up; halt-the-line condition in §7. *Investigation posture if skin smoke fails:* check whether (a) the JSON-RPC shape of the particle posterior changed (post-Move-6 `CategoricalMeasure` wrapping `ParticlePrevision` should serialise identically), (b) the seed threading through the RPC boundary is deterministic, (c) the shield for the new wrapping case fires correctly on `weights`/`mean`/field access over the wire.

## 7. Verification cadence

At end of Move 6's code PR (6b):

```bash
# Stratum 1 / Stratum 2 — new particle/quadrature/enumeration corpus.
julia test/test_prevision_particle.jl

# Stratum 2 inherited: conjugate registry must still fire for registered pairs.
julia test/test_prevision_conjugate.jl

# Stratum 2 inherited: mixture routing must still compose with the new
# Prevision subtypes at the fallback slots.
julia test/test_prevision_mixture.jl

# Stratum 1 inherited.
julia test/test_prevision_unit.jl
julia test/test_persistence.jl

# Existing test suite — must pass unchanged; particle paths in grid_world,
# email_agent, and program_space are where seeded-MC `==` has its first
# real exercise.
julia test/test_core.jl
julia test/test_program_space.jl
julia test/test_email_agent.jl
julia test/test_flat_mixture.jl
julia test/test_grid_world.jl
julia test/test_host.jl
julia test/test_rss.jl
julia test/test_events.jl

# POMDP agent — program-space factored models exercise particle paths
# indirectly.
cd apps/julia/pomdp_agent && julia --project=. -e 'using Pkg; Pkg.test()'

# Skin smoke — MANDATORY at Move 6 (wire-path changes, particle posterior
# serialisation). Includes new test_particle_path, test_grid_fallback,
# test_particle_snapshot per Move 0 audit.
python -m skin.test_skin
```

**Halt-the-line conditions:**

- Any `test/test_prevision_particle.jl` Stratum-2 `==` failure on seeded-MC bit-invariance — the R1 class; investigate per R1 posture.
- Any existing test regression below `rtol=1e-10` (Stratum 3).
- Any `_dispatch_path` vocabulary assertion failure — per §5.3 the labels are `:conjugate` / `:particle` / `:mixed`; a silent registry miss flips `:conjugate` to `:particle` or `:mixed`.
- Any skin smoke failure at the extended RPC surface.
- Any POMDP agent test failure — the POMDP package exercises particle paths through MCTS rollouts; its continued-green signal is one of the strongest indicators a refactor didn't break anything at a distance.

Per the `precedents.md` §8 (checkpoint-per-phase) and §5 (`_dispatch_path` assertion-before-value) conventions: each phase of the code PR lands as its own commit, each commit leaves the branch green, and every conjugate-vs-fallback assertion pins `_dispatch_path` before the value check.
