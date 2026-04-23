# Test fixtures — provenance protocol

This directory holds frozen reference state used by tests that verify schema-version migrations and other point-in-time invariants. Every fixture is **commit-pinned**: it is captured from a named SHA, that SHA is recorded in this README, and the fixture is never regenerated to fix a loading bug. If a future-discovered bug affects how the fixture is loaded, the fix goes in the load code; the fixture stays as-is.

This protocol exists because fixture regeneration silently invalidates migration tests. If `test/fixtures/agent_state_v1.jls` is regenerated from a v2-aware codebase to "fix" a load failure, the test passes — and the migration codepath the test was supposed to verify is no longer exercised. Real users with v1 state on disk discover the corruption in production. The protocol prevents this by making regeneration a procedural violation rather than a one-line script.

## Captured fixtures

### `agent_state_v1.jls`

**Source SHA:** `bf74f985821c37b89fa4321e74b07092d1f63b65` (master tip at Move 3 code PR opening; post-Move-3-design-doc merge)
**Capture date:** 2026-04-20
**Size:** small (< 1 KB)
**Represents:** a `MixtureMeasure` of 3 `TaggedBetaMeasure` components with non-uniform posterior weights after synthetic `FiringByTag` conditioning. Wrapped in a `Dict(:belief => m, :note => ...)` to match the Dict-wrapping convention of `save_state`; serialised via `Serialization.serialize`. The Measure structs inside are raw pre-Move-3 layout (BetaMeasure has `space, alpha, beta` fields directly; no `prevision` wrapper).

**Construction script (verbatim, for provenance):**

```julia
# capture_agent_state_v1.jl — one-off script, run once from master SHA bf74f98.
push!(LOAD_PATH, "src")
using Credence
using Serialization

c1 = TaggedBetaMeasure(Interval(0.0, 1.0), 1, BetaMeasure(Interval(0.0, 1.0), 1.0, 1.0))
c2 = TaggedBetaMeasure(Interval(0.0, 1.0), 2, BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0))
c3 = TaggedBetaMeasure(Interval(0.0, 1.0), 3, BetaMeasure(Interval(0.0, 1.0), 5.0, 2.0))

m = MixtureMeasure(Interval(0.0, 1.0), Measure[c1, c2, c3], [log(1.0), log(1.0), log(1.0)])

k_fire12 = Kernel(Interval(0.0, 1.0), Finite([0, 1]),
                  h -> CategoricalMeasure(Finite([0, 1])), (h, o) -> 0.0;
                  likelihood_family = FiringByTag(Set([1, 2]), BetaBernoulli(), Flat()))

m = condition(m, k_fire12, 1)
m = condition(m, k_fire12, 1)
m = condition(m, k_fire12, 0)

state = Dict(
    :belief => m,
    :note => "agent_state_v1 fixture; captured from master bf74f98; 3 TaggedBetaMeasure components with posterior after 2 pos + 1 neg FiringByTag(1,2) observations",
)
open(io -> serialize(io, state), "test/fixtures/agent_state_v1.jls", "w")
```

**Expected loaded values (what the migration test asserts against):**

After loading and migrating v1 → v2:

- `state[:belief]` is a `MixtureMeasure` of 3 components (pre-migration type) OR a `MixtureMeasure` wrapping a `MixturePrevision` of 3 components (post-migration; the test asserts the final shape).
- `length(state[:belief].components) == 3`.
- `state[:belief].log_weights == [-1.0986122886681098, -1.0986122886681098, -1.0986122886681098]` (uniform: 2 positive observations fired components 1 and 2, 1 negative observation fired the same; net effect on log-weights depends on the conditioning mechanics — under the Flat / BetaBernoulli per-tag dispatch the mixture log-weights remain uniform because the _predictive_ll contributions cancel across the FiringByTag Flat non-firing path).
- Component 1: `tag == 1`, `beta.alpha == 3.0`, `beta.beta == 2.0` (was Beta(1,1); +2 positive obs, +1 negative obs).
- Component 2: `tag == 2`, `beta.alpha == 4.0`, `beta.beta == 4.0` (was Beta(2,3); +2 positive obs, +1 negative obs).
- Component 3: `tag == 3`, `beta.alpha == 5.0`, `beta.beta == 2.0` (was Beta(5,2); unchanged because tag 3 is not in the `fires` set).

Assertion tolerances: `==` on all α/β (integer-accumulated); `==` on all log_weights (same arithmetic applied to the same inputs — not reassociation-sensitive).

**Invalidation conditions:** the v1 schema's struct layout changes upstream of Move 3 (would require a v0 fixture for that older shape). A change to the v2 schema does *not* invalidate this fixture — that's the whole point.

### `email_agent_state_v1.jls`

**Source SHA:** `bf74f985821c37b89fa4321e74b07092d1f63b65` (same as above)
**Capture date:** 2026-04-20
**Size:** small (< 1 KB)
**Represents:** the email-agent persistence shape — `rel_beliefs` and `cov_beliefs` as single-component `MixtureMeasure`s of `ProductMeasure` of `BetaMeasure` (3 categories each), a `cat_belief` `CategoricalMeasure`, plus `total_score` and `total_cost` scalars. Serialised via the existing `save_state(path; rel_beliefs, cov_beliefs, cat_belief, total_score, total_cost)` entry point in `src/persistence.jl` — which wraps everything in a Dict and `Serialization.serialize`s it.

**Construction script (verbatim, for provenance):**

```julia
push!(LOAD_PATH, "src")
using Credence

n_cats = 3
rel_factors = Measure[BetaMeasure(Interval(0.0, 1.0), 1.0, 1.0) for _ in 1:n_cats]
rel_factors[1] = BetaMeasure(Interval(0.0, 1.0), 3.0, 2.0)
rel_factors[2] = BetaMeasure(Interval(0.0, 1.0), 5.0, 1.0)
rel_prod = ProductMeasure(rel_factors)
rel_beliefs = MixtureMeasure(rel_prod.space, Measure[rel_prod], [0.0])

cov_factors = Measure[BetaMeasure(Interval(0.0, 1.0), 2.0, 2.0) for _ in 1:n_cats]
cov_prod = ProductMeasure(cov_factors)
cov_beliefs = MixtureMeasure(cov_prod.space, Measure[cov_prod], [0.0])

cat_belief = CategoricalMeasure(Finite([:urgent, :routine, :spam]),
                                 [log(3.0), log(5.0), log(2.0)])

save_state("test/fixtures/email_agent_state_v1.jls";
           rel_beliefs = rel_beliefs,
           cov_beliefs = cov_beliefs,
           cat_belief  = cat_belief,
           total_score = 42.5,
           total_cost  = 0.123)
```

**Expected loaded values (what the migration test asserts against):**

- `state[:rel_beliefs].components[1].factors[1]`: `α == 3.0`, `β == 2.0`.
- `state[:rel_beliefs].components[1].factors[2]`: `α == 5.0`, `β == 1.0`.
- `state[:rel_beliefs].components[1].factors[3]`: `α == 1.0`, `β == 1.0`.
- `state[:cov_beliefs].components[1].factors[i]`: `α == 2.0`, `β == 2.0` for all `i ∈ 1:3`.
- `weights(state[:cat_belief])`: `[0.3, 0.5, 0.2]` (to `atol=1e-14` — reassociation-sensitive through the `weights()` normaliser).
- `state[:total_score] == 42.5`.
- `state[:total_cost] == 0.123`.

Assertion tolerances: `==` on α/β (integer/literal values from construction); `atol=1e-14` on normalised weights (the log-weight normaliser uses logsumexp which is reassociation-sensitive); `==` on scalars.

**Invalidation conditions:** as above.

---

### `particle_canonical_v1.jls` — Move 6 capture-before-refactor

**Source SHA:** `173411b` (Move 5 tip, pre-Move-6 refactor).
**Captured:** 2026-04-22.
**Julia version:** 1.11.x (CI-pinned).
**Purpose:** Canonical particle-path and grid-quadrature outputs under `Random.seed!(42)`, captured before Move 6's particle/quadrature refactor begins. The test `test/test_prevision_particle.jl` asserts `==` against these values throughout the Move 6 code PR; any `==` failure in a subsequent Move 6 commit is a halt-the-line signal that the refactor introduced a seed-consumption reorder or arithmetic reassociation.

Contents (a Dict{Symbol, Any}):
- `:source_sha` — string pinning the capture SHA.
- `:julia_version` — Julia version used to capture; canonical values are bit-identical only under this version.
- `:gamma_generic_samples` — 50 Float64s, from `condition(GammaMeasure(2.0, 3.0), k_pushonly, 2.5; n_particles=50)` under seed 42.
- `:gamma_generic_logw` — 50 Float64s, matching log_weights.
- `:beta_grid_values` — 64 Float64s from `_condition_by_grid(BetaMeasure(2.0, 3.0), k_pushonly, 0.5)`.
- `:beta_grid_logw` — 64 Float64s, matching log_weights.
- `:gaussian_grid_values` — 64 Float64s from `_condition_by_grid(GaussianMeasure(Euclidean(1), 0.0, 1.0), k_pushonly, 1.5)`.
- `:gaussian_grid_logw` — 64 Float64s, matching log_weights.

Assertion tolerance: `==`. Seeded Monte Carlo under fixed seed produces bit-identical outputs run-to-run; the only legitimate drift is floating-point reassociation from constructor changes, which Move 6's refactor is specifically designed to avoid.

**Invalidation conditions:** Julia version change (RNG implementation differences); intentional change to the `draw` / log-density evaluation order (which would be a Move 6 design-doc amendment); any change to the canonical GammaMeasure / BetaMeasure / GaussianMeasure constructors that affects storage layout. If invalidated, a new fixture (`particle_canonical_v2.jls`) captures at the SHA that introduced the change; the v1 file stays as-is for backward-compat verification.

### `posture-3-capture/` — Posture 4 Move 0 invariance target

**Source SHA:** `5c6a94e464225776e996d6f1f690219a0728ff35` (master tip after PR #43 merge — Posture 3 complete + Move 0 design-doc amendment).
**Capture date:** 2026-04-24.
**Julia version:** recorded in `posture-3-capture/manifest.toml[capture.julia_version]`.
**Purpose:** Whole-suite behavioural invariance target for Posture 4 (`de-finetti/complete`). Every assertion in `test/test_*.jl` was captured by `scripts/capture-invariance.jl` at this SHA under the three-idiom / four-shape classification from `docs/posture-4/move-0-design.md` §3. Moves 1–10 verify against this capture at the declared per-shape tolerance semantics; any divergence halts the move.

Categorically different from the other fixtures in this directory: the migration fixtures (`agent_state_v1.jls`, `email_agent_state_v1.jls`, `particle_canonical_v1.jls`) capture pre-migration *state shape* so load codepaths can be verified. This directory captures pre-branch *assertion values* so every test's post-condition can be verified across a ten-move refactor. Both follow the same commit-pinning discipline (never regenerate to fix a later-move bug); the contents are disjoint.

Contents (see `posture-3-capture/README.md` for schema detail):
- `strata-1.jls` / `strata-2.jls` / `strata-3.jls` — Exact + Tolerance assertions by stratum.
- `directional.jls` — directional assertions (bare `<`, `<=`, `>`, `>=`).
- `structural.jls` — structural assertions (`isa`, membership, predicate-form).
- `failing.jls` — latent broken assertions (per Q4).
- `manifest.toml` — per-idiom sorted listing + capture metadata + `bad2_corpus` inventory.

Size: ~1.8 MB total. Larger than the KB-range ceiling the "Rules" section below sets for migration fixtures, but justified: this is one fixture covering 6118 unique site×value tuples across 13 test files, not a granular migration fixture. Splitting further would impose an artificial per-file structure on an inherently whole-suite invariance target.

**Invalidation conditions:** a Move 0 follow-up PR amending the capture protocol (see `docs/posture-4/move-0-design.md`); Move 10's paper-reconciliation PR upgrading any `failing` assertions to passing (per Q4). Otherwise immutable.

## Loading these fixtures

`test/test_persistence.jl` (created in Move 3) loads each fixture in v2 code and asserts the resulting object's weights/parameters/structure match the recorded expected values. The test file documents which fixture covers which load codepath; if a new load codepath is added later, a new fixture covers it (do not extend an existing fixture's expectations).

## Rules

- Fixtures are **read-only** in the test suite. Tests load them and assert; tests never write back.
- Fixtures are **never regenerated** to fix loading bugs. The fix goes in load code.
- A new schema version (v3, v4, …) gets a new set of fixtures captured at the SHA that introduced that version. Old fixtures remain to verify backward-compat load.
- Fixture binary blobs are checked into git. They should be small (KB range, not MB); if a fixture grows large, that signals it's capturing too much — split it.
