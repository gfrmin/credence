# Move 4 design — Conjugate dispatch as a type-structural registry

Status: design doc (docs-only PR 4a). Corresponding code PR is 4b.

Template reference: `docs/posture-3/DESIGN-DOC-TEMPLATE.md`.

Master plan reference: `docs/posture-3/master-plan.md` § "Move 4 — Conjugate dispatch as a type-structural registry".

## 1. Purpose

Move 4 replaces the case-analytic `condition` dispatch at `src/ontology.jl:545-860` with a single dispatch path through a type-structural `ConjugatePrevision{Prior, Likelihood}` registry. The current dispatch inspects `k.likelihood_family` and branches case-by-case; each new conjugate pair adds a new branch scattered across nine Measure subtypes and five LikelihoodFamily routing types. Under the registry: adding a conjugate pair adds one `update` method; dispatch is parametric; the case-analytic tree collapses to a thin facade.

Move 4 is the first move where Move 3's template (Prevision subtypes + `getproperty` shield) pays operational rent. The nine concrete `Prior` types Move 3 landed are the types the registry keys on.

**Pre-committed resolution from PR #19's Move 1 revision addendum (commit [6dce5e4](https://github.com/gfrmin/credence/commit/6dce5e4)).** The `TaggedBetaMeasure` routing loop currently at `src/ontology.jl:584-617` (iterates components, dispatches per-tag through `FiringByTag` / `DispatchByComponent`) moves to `MixturePrevision` as a per-component operation that calls `update` on each component's `ConjugatePrevision`. It does **not** become a compound registry entry keyed on `(TaggedBetaPrevision, FiringByTag)`. The master plan's invitation to argue (a) vs (b) was resolved in PR #19; nothing Move 3 shipped changes the analysis. Move 4 implements (a). §5 does not reopen the question.

## 2. Files touched

**New:**
- `test/test_prevision_conjugate.jl` — Stratum-2 test corpus. For each registered conjugate pair, `condition(Measure(prior), k, obs) == <closed-form result>` at `==` (conjugate arithmetic is bit-exact: α increments by 1 for positive Beta-Bernoulli obs; μ updates by closed-form Normal-Normal formula; etc.). Plus `_dispatch_path(p, k) == :conjugate` assertions (per §5.2). Per the Move 2 precedent: `==` for closed-form conjugate arithmetic; `1e-12` for any numerically-sensitive derived quantities.

**Modified:**
- `src/prevision.jl` — adds `ConjugatePrevision{Prior, Likelihood}` parametric struct, `update(p::ConjugatePrevision{...}, obs)` methods (one per pair), and `maybe_conjugate(p::Prevision, k::Kernel) → Union{ConjugatePrevision, Nothing}` lookup function. Exports extended.
- `src/ontology.jl:545-860` — refactor `condition(p::Prevision, k::Kernel, obs)` into a thin facade:
  ```julia
  function condition(p::Prevision, k::Kernel, obs)
      cp = maybe_conjugate(p, k)
      cp === nothing && return _condition_particle(p, k, obs)
      update(cp, obs).prior
  end
  ```
  Delete inline case-analytic dispatches at `src/ontology.jl:561-654` (Beta/Gaussian/Gamma/Dirichlet/NormalGamma specialisations). Keep the TaggedBetaMeasure `condition` method but simplify it to delegate to MixturePrevision's per-component update (per the pre-committed resolution).
- `src/ontology.jl:704-770` — the existing `expect` method bodies stay unchanged (Move 2 unified them onto the TestFunction hierarchy; Move 4 doesn't touch them).
- `apps/skin/server.jl` — new `_dispatch_path` RPC (underscore-prefixed per the repo-conventions internal-hook convention; see `CLAUDE.md` §Repo conventions §Merge authority and §Scope boundary notes about underscore-prefixed internal hooks). Takes a `state_id` and a kernel spec; returns `"conjugate"` or `"particle"` without executing the update. For use by test code and observability.
- `apps/skin/test_skin.py` — 4 conjugate-path tests (per the Move 0 skin surface audit): `test_beta_bernoulli_conjugate`, `test_gaussian_normal_conjugate`, `test_dirichlet_categorical_conjugate`, `test_normal_gamma_conjugate`. Each asserts closed-form posterior AND `_dispatch_path == "conjugate"`. Plus `test_flat_likelihood_no_op` (no-op conjugate entry).

**Not touched in Move 4:**
- `MixtureMeasure` / `MixturePrevision` — the per-component update machinery that TaggedBetaMeasure routing migrates to is Move 5's scope (`MixturePrevision` and its `decompose` method). Move 4 keeps a temporary bridge in `condition(::TaggedBetaMeasure, ...)` that does the current per-tag routing loop, but delegated in a way that Move 5's `MixturePrevision` can drop into without re-architecting. Noted explicitly as transitional scaffolding.

## 3. Behaviour preserved

### Registered conjugate pairs (six)

| Prior | Likelihood | Replaces | Test |
|-------|------------|----------|------|
| `BetaPrevision` | `BetaBernoulli` | `src/ontology.jl:606-611` | `test_beta_bernoulli_conjugate` |
| `BetaPrevision` | `Flat` | `src/ontology.jl:612-613` (no-op) | `test_flat_likelihood_no_op` |
| `GaussianPrevision` | `NormalNormal` | `src/ontology.jl:619-629` | `test_gaussian_normal_conjugate` |
| `GammaPrevision` | `Exponential` | *(new fast-path)* | `test_gamma_exponential_conjugate` |
| `DirichletPrevision` | `Categorical` | `src/ontology.jl:634-644` | `test_dirichlet_categorical_conjugate` |
| `NormalGammaPrevision` | `NormalGammaLikelihood` | `src/ontology.jl:646-654` | `test_normal_gamma_conjugate` |

`GammaPrevision + Exponential` is net-new — currently no fast-path exists (falls to particle). Move 4 adds it.

### Tolerances (Stratum-2, inheriting Move 2 precedent)

- **Conjugate path cases**: `==` where the arithmetic is exact (α increments by 1 for Beta-Bernoulli positive obs, etc.). `rtol=1e-12` where intermediate floating-point arithmetic admits reassociation (GaussianPrevision posterior μ via `(κ*μ + τ*x) / (κ+τ)`).
- **Particle fallback cases** (e.g. the non-conjugate Beta kernel branch that falls through `maybe_conjugate` → `_condition_particle`): `==` under deterministic seeding. Move 2 established this precedent; conflating with the 1e-12 quadrature tolerance would misframe Move 6.

### Verification invariant

For every registered conjugate pair, the Stratum-2 test asserts both:
1. `condition(m, k, obs) == <closed-form expected>`.
2. `_dispatch_path(m, k) == :conjugate`.

Assertion (2) is load-bearing: without it, a registry miss silently falls through to particle, which still converges to the correct value at enough samples, but the `_dispatch_path` check catches the silent-fallback bug (§5.2).

## 4. Worked end-to-end example

**Inputs:** `p = BetaPrevision(2.0, 3.0)`, `k = Kernel(Interval(0, 1), Finite([0, 1]), ...; likelihood_family = BetaBernoulli())`, `obs = true` (or `1`).

**Step-by-step dispatch:**

```julia
# 1. Caller invokes condition(p::BetaPrevision, k::Kernel, obs::Bool).
#    Julia resolves to the thin-facade method in src/ontology.jl.

condition(p, k, obs)
  ↓
# 2. Facade calls maybe_conjugate(p, k).
cp = maybe_conjugate(p, k)
#    maybe_conjugate dispatches on (typeof(p), k.likelihood_family):
#      maybe_conjugate(p::BetaPrevision, k::Kernel) where k.likelihood_family isa BetaBernoulli
#        = ConjugatePrevision{BetaPrevision, Bernoulli}(p, Dict())
#    Returns: ConjugatePrevision{BetaPrevision, Bernoulli}(BetaPrevision(2.0, 3.0), Dict())

# 3. Facade checks cp !== nothing → TRUE → proceed.

# 4. Facade calls update(cp, obs).
updated = update(cp, obs)
#    update dispatches on (typeof(cp), typeof(obs)):
#      update(cp::ConjugatePrevision{BetaPrevision, Bernoulli}, obs::Bool)
#    Body: α' = cp.prior.alpha + (obs ? 1 : 0)   # = 2 + 1 = 3
#          β' = cp.prior.beta  + (obs ? 0 : 1)   # = 3 + 0 = 3
#    Returns: ConjugatePrevision{BetaPrevision, Bernoulli}(BetaPrevision(3.0, 3.0), Dict())

# 5. Facade returns updated.prior = BetaPrevision(3.0, 3.0).

# Result: BetaPrevision(3.0, 3.0) — bit-exact α + 1.
```

**Observability trace:**

```julia
# _dispatch_path RPC (new; skin server surface).
_dispatch_path(p, k)
#    Internally:
#      cp = maybe_conjugate(p, k)
#      cp === nothing ? :particle : :conjugate
#    For BetaPrevision + BetaBernoulli kernel: returns :conjugate.

# Test assertion:
@assert _dispatch_path(p, k) == :conjugate
@assert condition(p, k, true).alpha == 3.0
@assert condition(p, k, true).beta == 3.0
```

The `_dispatch_path` RPC does NOT mutate state; it runs `maybe_conjugate` alone and returns a tag. Tests use it to pin the registry's dispatch explicitly — without it, a silent registry miss would fall through to particle and the result would converge correctly but the test would pass for the wrong reason.

## 5. Open design questions

### 5.1 (substantive) Registry shape — explicit Dict vs method-based dispatch

- **Option A (method-based, Julia-idiomatic):** `maybe_conjugate` is a generic function with methods:
  ```julia
  maybe_conjugate(p::BetaPrevision, k::Kernel) =
      k.likelihood_family isa BetaBernoulli ?
          ConjugatePrevision{BetaPrevision, Bernoulli}(p, Dict()) :
      k.likelihood_family isa Flat ?
          ConjugatePrevision{BetaPrevision, Flat}(p, Dict()) :
      nothing
  maybe_conjugate(p::GaussianPrevision, k::Kernel) = ...
  # etc.
  ```
  Adding a pair = adding a method branch (or a new method). Dispatch is Julia's method-table lookup; zero additional machinery. `update` dispatches on the `ConjugatePrevision{P, L}` parametric type.

- **Option B (explicit Dict registry):** a module-level `const CONJUGATE_PAIRS = Dict{Tuple{Type, Type}, Type}(...)` keyed on `(PriorType, LikelihoodFamilyType)`. `maybe_conjugate(p, k)` does `get(CONJUGATE_PAIRS, (typeof(p), typeof(k.likelihood_family)), nothing)` and constructs a `ConjugatePrevision` if matched. Adding a pair = adding a row to the Dict plus an `update` method.

**Recommendation: A (method-based).** Julia's method table *is* a registry. Option B layers an explicit Dict on top of a machinery that already does the same lookup; it adds indirection without adding expressiveness. Option A also gets compile-time error surface (if you misspell a Prior type, the method won't compile) where Option B's misspelling is a silent Dict-miss at runtime.

**Invitation to argue.** Option B is more introspectable from outside Julia (e.g. a Python client or observability tool could dump the registry). If Posture 3's paper draft's operational-consequences section benefits from "we emit the conjugate-pair table as JSON for comparison with other frameworks", Option B earns its indirection. If the paper doesn't need that, A wins on clarity.

### 5.2 (substantive) Observability contract for dispatch decisions

`_dispatch_path(p, k) → Symbol` returning `:conjugate` or `:particle` is called out in the Move 0 skin surface audit as Move 4's observability deliverable. The design question: what's the contract?

- **(a) State-free, query-only.** `_dispatch_path` runs `maybe_conjugate` and returns a tag without updating. Cheap, side-effect-free, callable from test code. The version the audit sketched.
- **(b) Log on every condition call.** Every `condition(p, k, obs)` emits a debug-level log line with the dispatch path. Tests can pattern-match on log output. Invasive (every real call is noisy) but always-on.
- **(c) Counter-based.** A module-level counter per dispatch path that tests can read and reset. `CONDITION_DISPATCH_COUNTS[:conjugate] += 1` on each conjugate call. Tests reset, call, read, assert.

**Recommendation: (a) only, for Move 4.** `_dispatch_path` is an underscore-prefixed test-only hook (per the repo-conventions note in `docs/posture-3/decision-log.md` §Decision 3). Production callers do not use it; production logging is separate concern (not Move 4's scope). Tests that need to assert dispatch path call `_dispatch_path(p, k)` pre-execution and assert the return; or call `condition(p, k, obs)` then call `_dispatch_path(p, k)` post-execution and assert consistency.

**Invitation to argue.** (b) would catch a specific failure mode (a test passes because the correct answer happened to emerge from particle with high n_samples, even though the registry was supposed to fire). But (a) + assertion pinning at test-write time catches the same class. If the paper draft wants to report "95% of email-agent condition calls are conjugate in production," something like (c) would be needed — but that's a production-observability concern, not a Move 4 concern.

### 5.3 (calibrating) Strict-conjugate marker — fail loudly vs silent particle fallback

When `maybe_conjugate` returns `nothing`, the current design falls through to `_condition_particle` silently. Some (Prior, Likelihood) combinations that the author *declared conjugate-intended* may be registry-missed due to a typo or incomplete registry. Today's `PushOnly` likelihood family does something similar at the TaggedBetaMeasure level (raises on misuse).

- **Option A (silent fallback, current master plan):** registry miss → particle. Conjugate-pair oversights surface only if the test suite is thorough (e.g. `_dispatch_path` assertions).
- **Option B (strict marker):** `Kernel(..., likelihood_family = Strict(BetaBernoulli()))` wraps the family in a marker that says "I intend this to be conjugate; fail if the registry doesn't match." `maybe_conjugate` checks for the marker and raises on miss.
- **Option C (both):** default is silent fallback; authors who want strict behaviour opt-in via `Strict(…)`.

**Recommendation: A for Move 4; revisit at Move 6.** The strict-marker is genuine value but adds surface area. The grep-gate result (80 hits, 0 category (c)) showed consumer code doesn't pattern-match on likelihood_family from outside, so in-the-wild strict-intent is expressed today through declaration alone. Move 6's particle refactor is the natural moment to introduce Strict as part of the particle-vs-quadrature-vs-conjugate dispatch semantics; there it has scope-aligned motivation.

**Invitation to argue.** If a reviewer identifies a specific case in Moves 4-5 where silent fallback would hide a real bug (e.g. a specific paper-case-study kernel that MUST be conjugate or the comparison breaks), B becomes cheaper now than defer.

## 6. Risk + mitigation

**Risk R1 (main risk): silent drift at Stratum-2 `==` boundary for conjugate pairs.** Same class as Move 2 R1 — the method-table routing through the new facade could introduce an implicit arithmetic reorder (e.g. `α + (obs ? 1 : 0)` vs `(obs ? α + 1 : α)` evaluates identically at the Float64 level but Julia's code generation could order differently under extremely specific conditions). Stratum-2 `==` assertions catch it. *Investigation posture:* halt. Read the `update` method body. Check for any implicit reordering introduced by the facade. Do not relax to `rtol=1e-12` on what should be integer-accumulated α/β.

**Risk R2 (low): unanticipated consumer dispatch on `k.likelihood_family`.** Pre-emptive grep run 2026-04-21, pattern `\.likelihood_family` across `src/`, `test/`, `apps/`. **80 total hits across 18 files.** Disposition:

- **Category (a) — declarations and internal reads (covered transparently): all 80.** Declarations at kernel construction (the biggest chunk, e.g. `Kernel(..., likelihood_family = BetaBernoulli())` in test setup and app-layer domain files) are Invariant-2 conformant and unchanged by Move 4. Internal reads inside `condition` dispatch (`src/ontology.jl:913`, `src/ontology.jl:1158`) are exactly what Move 4 refactors away — they move into `maybe_conjugate`. Test assertions `k.likelihood_family isa Flat` (`test/test_events.jl:47`) and `classify(m) = k_ref.likelihood_family` (`test/test_core.jl:1293`) are reads for verification / classifier closures, not reads-for-own-dispatch-decisions.
- **Category (b) — reads that need explicit forwarding: 0.** No consumer pattern-matches on `k.likelihood_family` from outside the `condition` implementation.
- **Category (c) — mutations or out-of-dispatch decision-making: 0.** No `kernel.likelihood_family = …` assignments anywhere; kernels are immutable by convention and the grep confirms no consumer violates it.

Go/no-go: **GO.** Ratio is effectively 100% (a); no scope amendment needed. Move 4 refactors the `condition` dispatch without touching any consumer site.

**Risk R3 (low, pre-committed): TaggedBetaMeasure routing relocation.** The master plan's Move 4 section named two options: (a) MixturePrevision per-component operation vs (b) compound registry entry. **Pre-committed (a)** in PR #19's Move 1 revision addendum (commit `6dce5e4`); nothing Move 3 shipped changes the analysis. §1 cites the resolution; §5 does not reopen. The implementation caveat: Move 4's `condition(::TaggedBetaMeasure, k, obs)` keeps the current per-tag loop as transitional scaffolding (MixturePrevision doesn't yet exist — that's Move 5). Move 5 removes the scaffolding.

**Risk R4 (low): `_dispatch_path` RPC introduces a new public surface.** The RPC is underscore-prefixed, test-only, documented in `docs/posture-3/decision-log.md` §Decision 3. A future change to the dispatch algorithm that breaks `_dispatch_path`'s contract would break Stratum-2 tests. *Mitigation:* the contract test (part of `test_prevision_conjugate.jl`) asserts `_dispatch_path` consistency; any algorithm change that invalidates it must update the test explicitly (not silently).

## 7. Verification cadence

At end of Move 4's code PR (4b):

```bash
# Stratum 2 opens (new)
julia test/test_prevision_conjugate.jl

# Stratum 1 from Move 2 and 3 must still pass (registry refactor is
# transparent to expect dispatch)
julia test/test_prevision_unit.jl
julia test/test_persistence.jl

# Existing test suite must pass unchanged
julia test/test_core.jl
julia test/test_program_space.jl
julia test/test_email_agent.jl
julia test/test_flat_mixture.jl
julia test/test_grid_world.jl
julia test/test_host.jl
julia test/test_rss.jl
julia test/test_events.jl

# POMDP agent
cd apps/julia/pomdp_agent && julia --project=. -e 'using Pkg; Pkg.test()'

# Skin smoke — MANDATORY at Move 4 (dispatch surface changes)
# Includes 5 new conjugate-path tests + _dispatch_path assertions.
python -m skin.test_skin
```

**Halt-the-line conditions:**
- Any Stratum-2 `==` failure on conjugate-arithmetic paths — silent drift signal per R1.
- Any `_dispatch_path(p, k) == :conjugate` assertion failure — silent registry miss per §5.2.
- Any skin smoke failure (bounded teardown per issue #22 still applies; flakes in the Julia lifecycle class are a third-occurrence signal per the established precedent).
- Any existing test regression (Stratum-3 end-to-end at `rtol=1e-10`).

Per the Move 3 review posture: the registry refactor and the `_dispatch_path` RPC must land in the same PR as the facade replacement. Deferring the RPC to a follow-up leaves the dispatch decision unobservable from tests, which is the exact failure mode §5.2 identifies.
