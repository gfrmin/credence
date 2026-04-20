# Move 3 design — `Measure` as derived view over `Prevision`

Status: design doc (docs-only PR 3a). Corresponding code PR is 3b.

Template reference: `docs/posture-3/DESIGN-DOC-TEMPLATE.md`.

Master plan reference: `docs/posture-3/master-plan.md` § "Move 3 — `Measure` as derived view over `Prevision`".

## 1. Purpose

Move 3 is the (b)-decision-shaped move. Per `docs/posture-3/decision-log.md` Decision 2, existing consumers keep the Measure surface; new domain code writes against Previsions. This move is what makes that decision coherent in the type system: every `Measure` subtype becomes a thin wrapper around a concrete `Prevision` subtype, with a `Base.getproperty` forwarding shield so that existing `m.alpha`, `m.beta`, `m.logw`, `m.factors`, `m.components`, `m.log_weights` reads across the codebase work unchanged.

Three concrete pieces land together:

1. **New concrete `Prevision` subtypes** in `src/prevision.jl` — `BetaPrevision`, `CategoricalPrevision{T}`, `TaggedBetaPrevision`, `GaussianPrevision`, `GammaPrevision`, `DirichletPrevision`, `NormalGammaPrevision`, `ProductPrevision`, `MixturePrevision`. Each carries the same parameter fields the current `Measure` subtype holds (α, β, μ, σ, logw, factors, etc.).
2. **`Measure` subtypes refactored to wrap their `Prevision`** in `src/ontology.jl`. Each Measure subtype becomes `struct XMeasure <: Measure; prevision::XPrevision; space::XSpace; end` (or similar). `Base.getproperty(m::XMeasure, s::Symbol)` forwards scalar/vector field reads to `m.prevision.<s>`; the `space` field stays on the Measure. `Base.propertynames` extended so tab-completion and introspection still list the forwarded names.
3. **Persistence schema v2** in `src/persistence.jl`. `save_state` writes a schema-version marker; `load_state` detects v1 (raw Measure) and migrates to v2 (wrapped) on read. Commit-pinned v1 fixtures (`test/fixtures/agent_state_v1.jls`, `test/fixtures/email_agent_state_v1.jls`) land in the Move 3 PR, captured from master at the SHA immediately preceding PR 3b's opening.

This is the move where "Measure as view" becomes operational. Moves 4–7 extend the `Prevision` hierarchy (conjugate registry, mixture/exchangeability, particle, condition-as-conditional-prevision) without changing the view surface; apps continue to call `mean(m)`, `weights(m)`, `m.alpha`, etc.

## 2. Files touched

**New:**
- `test/test_persistence.jl` — loads the v1 fixtures, asserts round-trip correctness against recorded expected values. Per the master plan, a round-trip-in-same-process test is not sufficient (it doesn't exercise the migration codepath); the fixtures are the only adequate test.
- `test/fixtures/agent_state_v1.jls` — AgentState captured from master at the SHA preceding PR 3b's opening. Contents: a `MixtureMeasure` of `TaggedBetaMeasure` components with non-trivial posterior weights, produced by a reproducible construction script recorded in `test/fixtures/README.md`.
- `test/fixtures/email_agent_state_v1.jls` — email-agent shape (`MixtureMeasure` of `ProductMeasure` of `BetaMeasure` triples), captured from the same SHA.
- `test/fixtures/README.md` — updated with the exact capture SHA, capture date, construction script per fixture, and the expected load-side values (weights, parameters) the test asserts against.

**Modified:**
- `src/prevision.jl` — adds concrete `Prevision` subtypes (~150 new lines; one struct per Measure type). Each is parametric where its underlying space/parameter type varies (e.g. `CategoricalPrevision{T}`, `ProductPrevision{S <: ProductSpace}`). Constructors mirror the existing `Measure` constructors including argument validation.
- `src/ontology.jl` — Measure subtypes at lines 64-232 refactored to wrap `Prevision` subtypes. Each gets:
  - A `prevision::<X>Prevision` field (replacing the direct parameter fields).
  - A `space::<X>Space` field (kept on the Measure for clean `support(m::Measure)` dispatch).
  - A validating outer constructor that takes the same arguments as today, constructs the Prevision internally, and wraps.
  - `Base.getproperty(m::<X>Measure, s::Symbol)` forwarding `:alpha`, `:beta`, etc. to `getfield(m, :prevision).<s>`. Explicit `if-elseif-else` chain per Measure subtype; default branch is `getfield(m, s)` to preserve `m.prevision`, `m.space` access. **Returns references, not copies — see §3's shared-reference contract and R4's mitigation**; a comment on each `getproperty` definition points at `test/test_prevision_unit.jl`'s contract test so the invariant is visible to anyone modifying the shield.
  - `Base.propertynames(m::<X>Measure)` extended with the forwarded names.
- `test/test_prevision_unit.jl` — extended with the shared-reference contract test (§3): construct a MixtureMeasure, read `.components` via the shield, `push!` in place, assert the new length visible on a second read. Guards against the defensive-copy regression class named in R4.
- `src/ontology.jl` method bodies at lines 128-180 (`mean`, `variance`, `weights` etc.) and 604-770 (`expect` methods) — unchanged: they continue to reference `m.alpha`, `m.beta`, etc., which now resolve through the forwarding shield.
- `src/persistence.jl` — adds schema-version handling:
  - `save_state(path, state)` includes `__SCHEMA_VERSION = 2` in the serialised payload.
  - `load_state(path)` detects v1 (no version marker, raw Measure fields) and reconstructs v2 (wrapped) via a `_migrate_v1_to_v2(raw)` helper. One-shot migration on next save; v1 never written going forward.
- `docs/posture-3/decision-log.md` §Decision 2 Fallback-to-(a) conditions — add a note: "grep-and-categorise gate run 2026-04-20 on `de-finetti/p3-move-3-design`. Result: 349/2/0 split (a/b/c) out of 351 hits. Shield strategy holds; no fallback triggered."

**Not touched in Move 3:**
- Any consumer site (tests, apps/julia/*, apps/python/*, apps/skin/*). The grep in §6 R2 confirms 0 mutations and ~2 edge cases that the shield covers transparently.
- Any existing `expect` or `condition` method body. Dispatch resolves to the same types (Measure subtypes are still concrete; methods attached to them still fire); field references now resolve through the shield.

## 3. Behaviour preserved

### Shared-reference contract (reusable shield property)

The shield returns **references to underlying fields, not copies.** `Base.getproperty(m::BetaMeasure, :alpha)` returns the `Float64` field directly; `Base.getproperty(m::MixtureMeasure, :components)` returns the underlying `Vector{Measure}` by reference; mutations to that vector (`push!`, `empty!`, index assignment) are visible to both the Measure view and the underlying Prevision because they share the same backing vector.

**This is a contract, not an implementation detail.** Two sites today depend on it:

- `apps/skin/server.jl:549` — `push!(state.belief.components, <new_component>)` during a mixture manipulation handler.
- `apps/skin/server.jl:552` — `push!(state.belief.log_weights, <new_logw>)` on the same handler path.

If the shield ever changes to return defensive copies — e.g. `getproperty(m, :components)` returns `copy(m.prevision.components)` — these `push!` calls succeed on the copy and the original state is silently unchanged. No error, no test failure at the mutation site; the corruption surfaces much later when the state is read and the components are missing or stale. This is the category of silent-corruption bug that's hardest to diagnose; the shield must not introduce it.

**Reusable property for future moves touching the shield:** Move 5's `MixturePrevision` will add component-update methods; Move 6's `ParticlePrevision` will want in-place reweighting on sample arrays. Both sit under the same shield. Framing "shield preserves shared-reference semantics" as a contract now means future moves inherit the pattern and don't re-litigate defensive-copying per subtype.

**Contract test** (new in this PR, extending `test/test_prevision_unit.jl`):

```julia
# Construct a MixtureMeasure with two components.
c1 = BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0)
c2 = BetaMeasure(Interval(0.0, 1.0), 5.0, 5.0)
m = MixtureMeasure(Interval(0.0, 1.0), [c1, c2], [log(1.0), log(1.0)])
# Read .components through the shield; mutate in place; read again.
comps = m.components
push!(comps, BetaMeasure(Interval(0.0, 1.0), 1.0, 1.0))
@assert length(m.components) == 3  # mutation visible through the shield
# Same for .log_weights.
lws = m.log_weights
push!(lws, log(1.0))
@assert length(m.log_weights) == 3
```

Four lines of actual logic, guards against the entire defensive-copy-regression class.

### Tolerances

Per §3 of the Move 2 design doc (extended for Move 3):

- **Stratum-1 closed-form cases:** `==` unchanged. `expect(::BetaMeasure, ::Identity)` still returns `m.alpha / (m.alpha + m.beta)`; the field lookup resolves through the shield to the same Float64 values; same arithmetic; same result.
- **Stratum-1 quadrature cases:** `isapprox(atol=1e-14)` unchanged.
- **Stratum-1 seeded Monte Carlo:** `==` unchanged. Critically: the RNG consumption order must be preserved. Any path that does `draw(m)` → internally uses `m.alpha`, `m.beta` etc. — those calls go through the shield and must not introduce extra allocations or ordering changes that affect RNG consumption.
- **Stratum-3 end-to-end (`test/test_*.jl` + POMDP agent):** `isapprox(rtol=1e-10)` unchanged.

**New Stratum-1 cases opened in Move 3:**

- For every `Measure` subtype M, `M(args...).prevision isa XPrevision` is true (wrap is sound).
- For every `Measure` subtype M and each declared parameter field φ, `M(args...).φ == <recorded-value>` (forwarding shield works on reads).
- `propertynames(M(args...))` includes all declared parameter fields (introspection works).

**Persistence migration cases (new in `test/test_persistence.jl`):**

- `load_state("test/fixtures/agent_state_v1.jls")` produces a v2-wrapped AgentState with weights, α, β per component matching recorded expected values (`==` where the values are exact integers; `atol=1e-14` for any derived quantities).
- Same for `email_agent_state_v1.jls`.
- `save_state(roundtrip_path, loaded_state)` then `load_state(roundtrip_path)` returns a bit-identical state (v2-to-v2 round-trip, established by the migration codepath alone on first save).

## 4. Worked end-to-end example

**Constructing and using a BetaMeasure post-Move-3:**

```julia
# Construction: same surface syntax as today.
m = BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0)
# Post-Move-3 internal shape:
#   m isa BetaMeasure
#   m.prevision isa BetaPrevision    (new field)
#   m.space :: Interval               (kept)
#   getfield(m, :prevision).alpha == 2.0
#   getfield(m, :prevision).beta  == 3.0

# Consumer read through the shield: same syntax, same value.
m.alpha   # → 2.0, via Base.getproperty(m, :alpha) → m.prevision.alpha
m.beta    # → 3.0, via Base.getproperty(m, :beta)  → m.prevision.beta

# Method dispatch: unchanged.
mean(m)   # → m.alpha / (m.alpha + m.beta) = 0.4
          # (method body at src/ontology.jl:129; `m.alpha` goes through
          #  getproperty; `m.alpha + m.beta` is the same arithmetic; bit-exact)

# Expect dispatch: unchanged method table.
expect(m, Identity())  # → 2.0 / 5.0 = 0.4 (Stratum-1 `==`)

# Persistence round-trip through v2 schema.
save_state("/tmp/state.jls", AgentState(belief = MixtureMeasure(...)))
loaded = load_state("/tmp/state.jls")  # reads v2; if it were v1, migrates.
loaded.belief.components[1].alpha  # → same value, shield forwards
```

**Getproperty shield trace (the centrepiece):**

```julia
# BetaMeasure's getproperty override:
function Base.getproperty(m::BetaMeasure, s::Symbol)
    if s === :alpha || s === :beta
        return getproperty(getfield(m, :prevision), s)
    else
        return getfield(m, s)
    end
end

# Call `m.alpha`:
#   1. Julia invokes Base.getproperty(m, :alpha).
#   2. Branch `s === :alpha`: returns getproperty(getfield(m, :prevision), :alpha).
#   3. getfield(m, :prevision) returns the BetaPrevision (bypasses the override to avoid recursion).
#   4. getproperty(<BetaPrevision>, :alpha) resolves to the Prevision's :alpha field (default getproperty).
#   5. Returns Float64(2.0).
# Total: two function calls, both inlined by Julia's compiler under normal dispatch.
# Overhead: ~0ns at runtime after compilation.
```

**Persistence v1-to-v2 migration trace:**

```julia
# Loading a v1 file (serialised pre-Move-3):
raw = Serialization.deserialize(io)
# raw is a v1 AgentState: raw.belief isa MixtureMeasure where components are
# direct TaggedBetaMeasure structs with raw.α, raw.β fields (no :prevision).

if !haskey(raw, :__SCHEMA_VERSION)
    # v1 detected. Migrate.
    state = _migrate_v1_to_v2(raw)
end

# _migrate_v1_to_v2 walks the AgentState, replacing each old Measure with
# the new wrapped form: TaggedBetaMeasure(space, tag, BetaMeasure(inner))
# becomes TaggedBetaMeasure(prevision = TaggedBetaPrevision(tag, BetaPrevision(α, β)), space).
# The migration is structural: same leaf values, new wrapping.
```

## 5. Open design questions

### 5.1 (substantive) Defensive copying on getproperty for Vector-typed fields?

**The question.** When `Base.getproperty(m::MixtureMeasure, :components)` forwards to `getfield(m, :prevision).components`, it returns the Vector by reference. Consumers that mutate the returned vector (e.g. `push!(state.belief.components, new_component)` in `apps/skin/server.jl:549`) mutate the prevision's internal vector directly. Should the shield defensively copy the vector on return?

- *Option A (no defensive copy, current behaviour):* `getproperty` returns the vector by reference. Existing `push!` patterns work. Mutation visible to both the Measure view and the underlying Prevision. Relies on the convention that Measure parameters are treated as immutable by consumer code, enforced by author discipline rather than the type system.
- *Option B (defensive copy):* `getproperty` returns `copy(getfield(m, :prevision).components)`. Mutations to the returned vector don't affect the Measure. Breaks existing `push!` patterns (they would mutate a copy, with no effect). Type-system-enforced immutability of Measure.
- *Option C (immutable view):* return an `ImmutableVector` or `ReadOnlyArray` wrapper. Existing `push!` patterns raise a clear error rather than silently mutating a copy. Explicit rejection of mutation through the shield.

**Recommendation: A.** The grep in §6 R2 confirms zero `setfield!` / assignment patterns on Measure fields — mutation doesn't happen today through direct assignment. The only mutation-through-field-reference patterns are the `push!` cases at `apps/skin/server.jl:549,552`, which are part of the skin server's mixture-manipulation handlers and are semantically legitimate (they modify the mixture's composition). Option B breaks those sites silently. Option C surfaces them as errors but requires a consumer edit to every site that push!-es into a Measure-viewed vector.

The immutability-enforcement concern that motivates B/C is real, but the master plan already scopes "Measure is immutable by convention" as a norm, and a future hardening pass (post-Posture-3) can promote it to a type-system invariant if the lax convention ever bites. For Move 3 specifically, A matches the 349/2/0 grep result: the existing code pattern doesn't need enforcement, and preserving it costs less than breaking it.

**Invitation to argue.** The strongest counter-case is that Measure-as-view is supposed to clean up the semantics around what's mutable and what's not; accepting A means the shield is "read-only in practice but not in principle." If a reviewer wants the type-system guarantee now rather than later, B is the defensible alternative — at the cost of the two skin-server `push!` sites, which would need rewriting to something like `replace_factor`-style structural updates.

### 5.2 (calibrating) Schema-version detection strategy — embedded marker vs type-based?

**The question.** How does `load_state` know it's loading a v1 vs v2 file?

- *Option A (embedded version marker):* `save_state` writes a `__SCHEMA_VERSION = 2` sentinel as part of the serialised payload. `load_state` checks for the sentinel first. If absent or < 2, migrate.
- *Option B (type-based):* `load_state` deserialises the payload, then checks whether the top-level Measure's struct layout matches the v1 shape (has `:alpha` as a direct field) or v2 (has `:prevision`). Dispatch the migration on that.

**Recommendation: A.** Explicit versioning is cheaper to maintain than structural introspection. Future schema bumps (v3, v4) can gate on the same sentinel; with B, each migration would need to re-discover the shape via runtime checks, which is error-prone.

**Invitation to argue.** B's advantage is that it handles state files written by forks of the codebase that never had the sentinel. That's a non-use-case for this single-maintainer repo.

### 5.3 (calibrating) Does `space` live on the Measure or on the Prevision?

**The question.** The current `BetaMeasure` struct has fields `(space, alpha, beta)`. Post-Move-3, does `space` move into the Prevision (so `BetaPrevision` has `(space, alpha, beta)` and `BetaMeasure` just wraps `prevision`), or stay on the Measure (so `BetaPrevision` has `(alpha, beta)` and `BetaMeasure` has `(prevision, space)`)?

- *Option A (`space` on the Measure):* Measure carries its own space field; Prevision is purely parameter-holding. `support(m::BetaMeasure) = m.space` resolves directly.
- *Option B (`space` on the Prevision):* Prevision holds the space; Measure is pure wrap. `support(m::BetaMeasure) = m.prevision.space` goes through the forwarding path.

**Recommendation: A.** Two reasons: (i) `support(m)` is called frequently enough that an extra indirection is visible; (ii) future `Prevision` subtypes that don't have a natural Kolmogorov space (e.g. an abstract prevision on a test function space without a representing measure) wouldn't have `space` as a meaningful field, but Measure always has one.

**Invitation to argue.** B's appeal is parsimony — the Measure becomes a single-field wrapper. That's aesthetically pleasing but pays the indirection cost on every `support()` call for no functional benefit.

## 6. Risk + mitigation

**Risk R1 (main risk): silent value drift through the shield.** The method body of `mean(m::BetaMeasure) = m.alpha / (m.alpha + m.beta)` (src/ontology.jl:129) compiles post-Move-3 with `m.alpha` going through `Base.getproperty`. If `getproperty` inadvertently returns a type different from `Float64` (e.g. wraps the result in an `Some{Float64}` or a promotion path kicks in), the arithmetic result differs. Stratum-1 `==` assertions catch this immediately. *Investigation posture if breached:* halt. Check `typeof(m.alpha)` post-refactor vs pre-refactor; it must be `Float64`. Likely fix: the `getproperty` branch for `:alpha` should return `getfield(m.prevision, :alpha)` (no wrapping, no conversion).

**Risk R2 (low): unanticipated consumer field access.** Pre-emptive grep run 2026-04-20, pattern `\.(alpha|beta|logw|factors|components|log_weights|mu|sigma)\b` across `src/`, `test/`, `apps/` (including `apps/julia/*`, `apps/python/*`, `apps/skin/*`). Hits:

- **Total: 351 across 29 files.**
- **Category (a) — covered by `getproperty` shield transparently: ~349 (99.4%).** The overwhelming majority: `m.alpha`, `m.beta`, `m.components[i]`, `state.belief.log_weights`, `comp.beta.alpha`, etc., throughout `src/ontology.jl` (75), `src/host_helpers.jl` (14), `src/program_space/agent_state.jl` (14), `src/eval.jl` (1), tests (~159), `apps/julia/*` (~55), `apps/python/*` (~14), `apps/skin/*` (18). All through `getproperty`.
- **Category (a) with explicit reasoning (worth pinning):**
  1. `push!(state.belief.components, ...)` at `apps/skin/server.jl:549` and `push!(state.belief.log_weights, ...)` at `apps/skin/server.jl:552`. **Reasoning:** the shield returns the underlying Vector by reference, not a copy (per §3's shared-reference contract). `push!` on the returned vector mutates the prevision's backing storage; the mutation is visible on subsequent reads. Covered by (a); the contract test in §3 guards against future regression.
  2. `getfield(cpd, :measure).alpha` at `apps/julia/pomdp_agent/src/probability/cpd.jl:67`. **Reasoning:** the `getfield` deliberately escapes the CPD wrapper's own `getproperty` (that's exactly what `getfield` is for — it avoids the `getproperty(cpd, :counts)` branch's recursion risk). Once past that, `.alpha` is invoked on the returned `DirichletMeasure` and resolves through the Measure-level `getproperty` chain — which the Move 3 shield covers normally. Two-stage access, both stages covered, but by different shields. Covered by (a).
- **Category (c) — mutations: 0.** No `m.alpha = x` patterns; `grep -n '\.(field)\s*=[^=]'` returned no matches; `setfield!` on Measure types not found.

Go/no-go gate result: **GO.** The 99.4% / ~0% / 0% ratio clears the plan's ≥90%/<15%/0 thresholds with margin.

**Risk R3 (medium): persistence migration fails on a v1 fixture.** The v1 → v2 migration walks the AgentState replacing old Measure structs with wrapped forms. Possible failure modes: (i) a field is renamed in the migration that consumers expect to find; (ii) the `MixtureMeasure` component reconstruction drops a tag; (iii) Julia's `Serialization` module handles the renamed struct as-if-unknown. *Caught by:* `test/test_persistence.jl` loading the commit-pinned v1 fixtures and asserting recorded expected values — `==` on integer-accumulated α/β, `atol=1e-14` on derived quantities. *Mitigation now:* the fixture-driven test is mandatory in this PR (per `test/fixtures/README.md` provenance protocol); synthetic round-trip tests do not catch migration bugs because they don't exercise the migration codepath.

**Risk R4 (medium): future refactor defensively copies in the shield without realising two consumer sites depend on shared-reference semantics.** A later session, seeing `Base.getproperty(m, :components)` returning an internal mutable Vector, decides to "harden encapsulation" by returning `copy(m.prevision.components)`. The `push!` sites at `apps/skin/server.jl:549,552` silently break — `push!` succeeds on the copy, the original state is unchanged, no error fires. Corruption surfaces later when a read through the shield returns stale components. *Caught by:* the contract test added to `test/test_prevision_unit.jl` in §3 (construct MixtureMeasure, read .components via shield, `push!` in place, assert new length visible through a second shield read). Four lines; guards against the entire class. *Mitigation now:* the §3 shared-reference contract is named explicitly with a code comment on the `Base.getproperty` definitions pointing at the test file, so the invariant is visible to anyone modifying the shield. A future move touching the shield (Move 5 MixturePrevision updates, Move 6 ParticlePrevision reweighting) inherits the contract and the test rather than re-deriving it.

## 7. Verification cadence

At end of Move 3's code PR (3b):

```bash
# Stratum 1 — extended with Measure-as-view cases
julia test/test_prevision_unit.jl

# Persistence migration — mandatory; loads v1 fixtures
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

# POMDP agent (separate package; exercises the cpd.jl getfield pattern)
cd apps/julia/pomdp_agent && julia --project=. -e 'using Pkg; Pkg.test()'

# Skin smoke test — MANDATORY at Move 3 (per master plan §Verification)
# Move 3 changes what crosses the JSON-RPC boundary (Measure shape);
# skin smoke proves the shield covers the serialisation surface.
python -m skin.test_skin
```

**Halt-the-line conditions:**
- Any Stratum-1 assertion failure (closed-form `!=`, seeded MC `!=`, quadrature outside `atol=1e-14`).
- Any persistence-migration assertion failure — the v1 fixtures are the only test that exercises the migration codepath; a failure here is a silent-corruption signal.
- Any skin smoke test failure — either functional (wrong value over the wire), teardown (the #22 bounded-shutdown path broke), or serialisation (JSON3 sees the Prevision wrapper shape instead of the Measure shape and over-exposes implementation details).

All three are investigate-don't-patch. Per master plan Move 3's risk section, consumer-site refactoring during Move 3 is out of scope — the shield is meant to cover those sites. If the skin smoke fails because JSON3 serialisation traverses `:prevision`, the fix is a JSON3 `StructType` override on the Measure subtypes to present the legacy shape, *not* a refactor of consumer code reading the JSON output.
