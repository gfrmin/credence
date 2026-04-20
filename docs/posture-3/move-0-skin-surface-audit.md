# Move-0 audit — skin smoke-test surface vs Moves 3/4/6/7 needs

## Purpose

The master plan declares the JSON-RPC API surface (`apps/skin/server.jl`) preserved bit-for-bit on this branch and asserts that proof via a `python -m skin.test_skin` smoke run at end of Moves 3, 4, 6, and 7. This audit checks what `apps/skin/test_skin.py` actually exercises today against what each of those four moves will change in the underlying Julia code, so we know up front which moves can lean on the existing test surface and which need the smoke test extended as a sub-task in their design doc.

The risk this audit prevents: discovering at Move 3 (or 4, or 6, or 7) that the smoke test never covered the affected wire path, then trying to build a smoke test under deadline pressure while the refactor is mid-flight. Better to enumerate gaps at Move 0 and attach each gap to the move that needs it.

## Today's smoke-test coverage

`apps/skin/test_skin.py` (verified — 395 lines, 8 test functions):

| Test | Exercises | RPC methods called |
|------|-----------|--------------------|
| `test_basic_inference` | Beta + Bernoulli conditioning | `create_state(beta)`, `mean`, `condition(bernoulli)`, `destroy_state` |
| `test_categorical` | Categorical + tabular preference | `create_state(categorical)`, `weights`, `optimise(tabular_2d)`, `destroy_state` |
| `test_router_roundtrip` | Nested ProductMeasure + functional_per_action + linear_combination of nested_projections + factor/replace_factor + DSL kernel | `initialize(dsl_files)`, `create_state(product)`, `optimise(functional_per_action)`, `factor`, `condition(quality)`, `replace_factor`, `expect(projection)` |
| `test_snapshot_restore` | BetaMeasure round-trip | `snapshot_state`, `restore_state`, `mean`, `condition` |
| `test_unknown_state_id` | Error path -32000 | `mean`, `weights`, `condition` on bogus IDs |
| `test_unknown_method` | Error path -32601 | `_call("nonexistent_method")` |
| `test_factor_on_non_product_measure` | Error path on type mismatch | `factor(beta_state)` |
| `test_replace_factor_identity_pin` | Sibling-factor preservation under replace | `factor`, `replace_factor`, `mean` |

RPC methods covered today (✓) vs exposed in `client.py` but uncovered (✗):

| Method | Covered? | Notes |
|--------|----------|-------|
| `initialize`, `shutdown` | ✓ | Every test |
| `create_state` | ✓ | beta, categorical, product, beta-inside-product |
| `destroy_state` | ✓ | basic_inference, categorical |
| `snapshot_state`, `restore_state` | ✓ | BetaMeasure only |
| `transfer_beliefs` | ✗ | Not exercised |
| `condition` | ✓ | Bernoulli kernel, DSL `quality` kernel |
| `weights` | ✓ | CategoricalMeasure |
| `mean` | ✓ | BetaMeasure |
| `expect` | ✓ | NestedProjection + Projection |
| `optimise` | ✓ | tabular_2d, functional_per_action |
| `value` | ✗ | Not exercised separately from optimise |
| `draw` | ✗ | Not exercised |
| `factor`, `replace_factor` | ✓ | ProductMeasure |
| `n_factors` | ✗ | Not exercised |
| `enumerate`, `perturb_grammar`, `add_programs` | ✗ | Tier 2 — not exercised in skin smoke |
| `sync_prune`, `sync_truncate` | ✗ | Not exercised |
| `top_grammars`, `belief_summary` | ✗ | Not exercised |
| `condition_and_prune` | ✗ | Not exercised |
| `eu_interact` | ✗ | Not exercised |
| `call_dsl` | ✗ (indirectly via DSL kernel) | Not exercised as a primary method |

## Per-move gap analysis

### Move 3 — Measure as derived view over Prevision

**Wire path changes:** the JSON-encoded shape of every `create_state` reply, `snapshot_state` blob, and `restore_state` input changes if Measure becomes a wrapper around Prevision. The `getproperty` forwarding shield keeps existing field access (`m.alpha`, `m.beta`, `m.logw`) working in Julia consumer code, but JSON3's struct-serialisation respects the actual struct layout — so the snapshot blob format will change unless explicit `JSON3.StructTypes` overrides preserve the v1 layout.

**Covered today:**
- `test_basic_inference` round-trip on BetaMeasure — exercises mean/condition.
- `test_categorical` round-trip on CategoricalMeasure — exercises weights/optimise.
- `test_router_roundtrip` round-trip on ProductMeasure of (Beta, Gamma) and ProductMeasure of factors.
- `test_snapshot_restore` round-trip on BetaMeasure with serialisation byte-blob.
- `test_replace_factor_identity_pin` confirms sibling-factor invariance under structural updates.

**Gaps for Move 3:**

| Gap | Move-3 design doc sub-task |
|-----|----------------------------|
| No MixtureMeasure round-trip over the wire. | Add `test_mixture_roundtrip`: create a small MixtureMeasure (3 components of TaggedBetaMeasure), assert weights + per-component means; snapshot + restore; assert weights + means survive bit-exactly. |
| No NormalGammaMeasure round-trip. | Add `test_normal_gamma_roundtrip`: create with explicit κ, μ, α, β; assert mean; snapshot + restore; assert mean survives. |
| No GammaMeasure round-trip. | Add `test_gamma_roundtrip` analogously. |
| No DirichletMeasure round-trip. | Add `test_dirichlet_roundtrip` analogously. |
| `snapshot_state` blob format invariance not pinned. | Pin a v1 fixture file inside `apps/skin/test_skin_fixtures/beta_v1.b64` (separate from `test/fixtures/` because skin-side serialisation is JSON3, not Julia `Serialization`); `test_snapshot_restore` decodes the fixture and asserts the loaded state's mean. |

### Move 4 — Conjugate dispatch as type-structural registry

**Wire path changes:** the JSON shape of `condition` results does not change for users (a state ID is returned); but the Julia code path now goes through `maybe_conjugate` lookup. If the lookup fails to match for a registered conjugate pair (e.g. a type mismatch in the registry key), the call silently falls through to particle filtering — which would pass `test_basic_inference`'s tolerance check but represent a silent regression.

**Covered today:**
- BetaBernoulli conjugate path — `test_basic_inference` (beta+bernoulli) and `test_router_roundtrip` (DSL quality kernel on Beta factor).

**Gaps for Move 4:**

| Gap | Move-4 design doc sub-task |
|-----|----------------------------|
| No GaussianNormal (Normal-Normal) conjugate test over the wire. | Add `test_gaussian_normal_conjugate`: create GaussianMeasure(0, 1), condition on a kernel with `sigma_obs` param, observe a single value, assert posterior mean shifts toward observation by the closed-form amount (not a tolerance check — closed-form). |
| No DirichletCategorical conjugate test. | Add `test_dirichlet_categorical_conjugate` analogously: Dirichlet(1,1,1,1), condition on a categorical kernel with one observation, assert α at observed index increments. |
| No NormalGammaLikelihood conjugate test. | Add `test_normal_gamma_conjugate`: create NormalGamma with explicit hyperparameters, condition on real-valued obs, assert closed-form posterior κ/μ/α/β update. |
| No `Flat` likelihood-family no-op test. | Add `test_flat_likelihood_no_op`: create BetaMeasure, condition with kernel of family `Flat`, assert posterior mean is bit-exactly equal to prior mean. |
| No GammaExponential (introduced in Move 4 as a new fast-path) test. | Add `test_gamma_exponential_conjugate` once the registry entry lands. |
| No assertion that the registry actually fires. | Add a debug RPC method `_dispatch_path` (Move 4 internal) returning `:conjugate` or `:particle` for a (state, kernel) pair, and have each conjugate test assert the path is `:conjugate`. Without this, a silent regression to particle is undetectable from the public RPC surface. |

### Move 6 — Execution layer refactor (ParticlePrevision)

**Wire path changes:** any condition that falls through to the particle path now constructs a `ParticlePrevision` rather than a `CategoricalMeasure(Finite(samples), log_weights)`. The sample's posterior mean and weight distribution must be bit-exactly identical to pre-refactor under the same RNG seed; a single-sample-reorder is a posterior-changing bug, not a tolerance issue.

**Covered today:**
- *Nothing.* All current smoke tests hit conjugate fast-paths.

**Gaps for Move 6:**

| Gap | Move-6 design doc sub-task |
|-----|----------------------------|
| No particle-path test exists in the skin smoke at all. | Add `test_particle_path`: create a state and kernel pair that has *no* conjugate registry entry (e.g. BetaMeasure with a custom Bernoulli-shaped kernel that the registry doesn't recognise), condition; assert the resulting state's mean matches the closed-form posterior to within `rtol=1e-12` (the particle estimate at the canonical seed is deterministic and reproducible). |
| No deterministic-seed contract for particle paths over the wire. | Move-6 PR adds an RPC method `_set_seed(seed::Int)` that is called at the start of each particle-path test; without this, JSON-RPC test reproducibility depends on global RNG state which is wire-invisible. |
| No ParticlePrevision snapshot/restore. | Add `test_particle_snapshot`: condition into a ParticlePrevision, snapshot, restore, assert mean and weight distribution survive bit-exactly. |
| No grid-fallback test. | Add `test_grid_fallback`: a GaussianMeasure conditioned on a kernel without `sigma_obs` (forces grid path), assert posterior mean matches the closed-form Normal-Normal answer to within `rtol=1e-12`. |

### Move 7 — `condition` as conditional prevision (event-primary)

**Wire path changes:** `condition(p, k, obs)` becomes derived (sugar for `condition(p, ObservationEvent(k, obs))`). The wire shape of the existing `condition` RPC is unchanged. A new `condition_on_event` RPC is added for direct event-form conditioning.

**Covered today:**
- The existing `condition` smoke continues to validate the derived path.

**Gaps for Move 7:**

| Gap | Move-7 design doc sub-task |
|-----|----------------------------|
| No `condition_on_event` RPC test. | Add `test_condition_on_event`: create a MixtureMeasure with several tagged components, call the new `condition_on_event` RPC with a TagSet event, assert the posterior weights are the same as `condition(m, indicator_kernel(TagSet(...)), true)` would produce — bit-exact, since Posture 2's gate-4 (`5c7f63f`, now on master) already established this equivalence. |
| No event-equivalence test on the wire. | Add `test_event_kernel_equivalence`: two side-by-side conditions of the same prior, one via `condition` with the indicator kernel and one via `condition_on_event` with the bare event; assert resulting state IDs have bit-exactly equal weights. |
| The Move 7 design-doc Socratic ("does ObservationEvent belong in the Event hierarchy?") may resolve in either direction. | If the Socratic resolves toward "ObservationEvent is *not* an Event, parametric update is a sibling primitive," then `condition_on_event` does not subsume the kernel form and the test design above is correct as stated. If it resolves toward "ObservationEvent is an Event, kernel form is sugar," then the test design adds a third case asserting that `condition` via the kernel form goes through the event path internally. |

## Summary — what Move 0 ships vs what each later move owns

This audit is the Move 0 deliverable. It does not ship test code. Each gap above is attached to the corresponding move's design doc as an "Open design questions"-adjacent sub-task ("which gaps from the audit must be closed in this PR vs deferred to a follow-up?"). The expectation:

- **Move 3** owns: 4 round-trip tests + 1 fixture-pinned snapshot test.
- **Move 4** owns: 4 conjugate-path tests + 1 dispatch-path observability hook.
- **Move 6** owns: 4 particle-path tests + a deterministic-seed RPC.
- **Move 7** owns: 2 event-conditioning tests, one of them resolving the ObservationEvent Socratic.

Total skin-smoke additions across the four moves: 14 new tests + 2 new RPC methods. None block Move 0; all are tracked here so the moves know the work is theirs.

## Things this audit deliberately does NOT do

- **Does not propose changes to the JSON-RPC protocol shape.** The protocol is preserved; only test coverage changes.
- **Does not rewrite `apps/skin/test_skin.py` upfront.** Each move adds its own tests when its design doc lands; consolidation (if needed) is a follow-up after Move 8.
- **Does not extend coverage to `transfer_beliefs`, `eu_interact`, `enumerate`, `perturb_grammar`, etc.** These are Tier-2 / convenience methods whose wire shape is unchanged by Posture 3. They should be tested for their own sake but that work is orthogonal to this branch.
