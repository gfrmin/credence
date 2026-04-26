# Posture 4 — Move 8b design doc: substrate field-type tightening + construction-site discipline

## 0. Final-state alignment

Move 8b closes a gap the stricter Posture 4 audit surfaced: Move 5's design doc committed to `MixturePrevision.components::Vector{Prevision}` and `ProductPrevision.factors::Vector{Prevision}`, but the implementation never landed. The current substrate has four bare unparameterised `Vector` fields — looser than either Move 2's deferred-language description or Move 5's committed-target description. Ten construction sites across `apps/` and `src/` use `Any[]` or `Measure[]` containers for uniformly-typed contents, which the existing lint does not detect. Beyond the Mixture/Product fields, `ParticlePrevision.samples::Vector` and `EnumerationPrevision.enumerated::Vector` each conflate algebraic and carrier content in ways that violate the Prevision/Measure split that Posture 4 established.

### End-state properties

After Move 8b, every Prevision type in the substrate satisfies five properties:

1. **Every Prevision holds only algebraic content** — the parameters and weights that enable `expect` to compute. No carrier-space objects, no domain elements, no observational data.
2. **Every Measure is `(Prevision, carrier-space)`** — observational carrier binding happens at the Measure layer, not inside Prevision fields.
3. **Parametric typing for fields whose element type varies across instances but is uniform within each instance** — `ParticlePrevision{T}` uses `Vector{T}`, not `Vector{Any}`.
4. **No type named for what it isn't** — `EnumerationPrevision` holds carrier objects (programs, domain elements) inside a Prevision; it is retired and replaced by `EnumerationMeasure{T}`, whose simplex content lives in a `CategoricalPrevision`.
5. **Lint covers container element-type discipline** — the `untyped-mixture-construction` slug prevents the blind spot from recurring.

Every construction site uses typed containers. Posture 4 then closes properly and Posture 5 opens against a verified-clean substrate.

## 1. Purpose

Land the substrate tightening Move 5 designed but didn't implement, and complete the Prevision/Measure split for the two types that conflate algebraic and carrier content. Four deliverables:

1. **Phase A (substrate — Mixture/Product):** Tighten `MixturePrevision.components` and `ProductPrevision.factors` to `Vector{Prevision}`. Update `MixtureMeasure`/`ProductMeasure` constructors to extract `.prevision` from Measure components. Clean up the three substrate `Any[]` sites.
2. **Phase A′ (substrate — Particle/Enumeration):** Make `ParticlePrevision` parametric (`ParticlePrevision{T}`), retire `EnumerationPrevision`, introduce `EnumerationMeasure{T}`.
3. **Phase B (apps + examples):** Tighten the seven app/skin construction sites from `Any[]`/`Measure[]` to `TaggedBetaPrevision[]`. Migrate the six email_agent secondary-file sites and `examples/host_credence_agent.jl` vocabulary to Prevision.
4. **Phase C (lint):** Land the `untyped-mixture-construction` slug that catches `Any[]` for uniform-type mixture/product construction.

## 2. Files touched

### Phase A — substrate

**Modified:**

- `src/prevision.jl` — `MixturePrevision.components::Vector` → `::Vector{Prevision}`; `ProductPrevision.factors::Vector` → `::Vector{Prevision}`. Constructor signatures updated.
- `src/ontology.jl` — `MixtureMeasure` constructor (line 355–357): extract `.prevision` from each Measure component before passing to `MixturePrevision`. `ProductMeasure` constructor (line 322–325): same extraction. Internal construction sites in `condition`, `prune`, `truncate` updated to pass `Prevision[]` rather than `Measure[]`.
- `src/host_helpers.jl` — Line 47: `Any[]` → typed Prevision container in `marginalize_betas` Prevision dispatch.
- `src/program_space/agent_state.jl` — Lines 118, 140: `Any[]` → `TaggedBetaPrevision[]` in `add_programs!` and the accumulator site.

### Phase A′ — Particle/Enumeration substrate

**Modified:**

- `src/prevision.jl` — `ParticlePrevision` becomes `ParticlePrevision{T}` with `samples::Vector{T}`. Inner constructor parameterised. `EnumerationPrevision` struct deleted.
- `src/ontology.jl` — New `EnumerationMeasure{T}` struct: `(CategoricalPrevision, carrier::Vector{T}, space::Space)`. `_condition_particle` (line 1122–1127) unchanged (already constructs `CategoricalMeasure(Finite(pp.samples), pp)` — `pp.samples` becomes `Vector{T}`). Three `expect` dispatches updated:
  - `expect(p::ParticlePrevision, f::Function)` (line 625) — unchanged; `p.samples[i]` works regardless of `T`.
  - `expect(p::ParticlePrevision, tf::TestFunction)` (line 672) — unchanged.
  - `expect(p::EnumerationPrevision, f::Function)` (line 641) → becomes `expect(m::EnumerationMeasure, f::Function)` accessing `m.carrier[i]`.
  - `expect(p::EnumerationPrevision, tf::TestFunction)` (line 674) → becomes `expect(m::EnumerationMeasure, tf::TestFunction)`.
- `src/stdlib.jl` — `weights(p::EnumerationPrevision)` (line 62) deleted; `weights(p::CategoricalPrevision)` already exists (line 34) and covers the simplex. `weights(p::ParticlePrevision)` (line 48) unchanged.
- `src/program_space/enumeration.jl` — `enumerate_programs_as_prevision` (line 171) → `enumerate_programs_as_measure`. Returns `EnumerationMeasure{Program}(CategoricalPrevision(log_weights), programs, program_space)` instead of `EnumerationPrevision(convert(Vector{Any}, programs), log_weights)`. Function renamed to reflect the return type change.

**Tests updated:**

- Any test that constructs `EnumerationPrevision` directly → constructs `EnumerationMeasure{T}` instead.
- Any test that dispatches on `::EnumerationPrevision` → dispatches on `::EnumerationMeasure` or `::CategoricalPrevision`.

### Phase B — apps + examples

**Modified:**

- `apps/julia/email_agent/host.jl` — Line 749: `Any[]` → `TaggedBetaPrevision[]`.
- `apps/julia/email_agent/live.jl` — Lines 170, 183, 191: Measure pathway → Prevision pathway (`Measure[]` → `TaggedBetaPrevision[]`, `TaggedBetaMeasure(...)` → `TaggedBetaPrevision(...)`, `MixtureMeasure(...)` → `MixturePrevision(...)`).
- `apps/julia/email_agent/eval_retrospective.jl` — Lines 17, 137, 150, 159, 295: same migration pattern as live.jl.
- `apps/julia/grid_world/host.jl` — Line 264: `Any[]` → `TaggedBetaPrevision[]`.
- `apps/julia/rss/host.jl` — Line 58: `Any[]` → `TaggedBetaPrevision[]`.
- `apps/skin/server.jl` — Line 643: `Any[]` → `TaggedBetaPrevision[]`.
- `examples/host_credence_agent.jl` — Lines 16, 102, 135: `MixtureMeasure` → `MixturePrevision` in imports, type checks, comments.

### Phase C — lint

**Modified:**

- `tools/credence-lint/credence_lint.py` — New slug `untyped-mixture-construction`.
- `tools/credence-lint/corpus/untyped-mixture-construction/` — Corpus files (good + bad examples).

## 3. Behaviour preserved

Move 8b is a type-discipline tightening. No numerical output changes.

**Mixture/Product (Phase A):** Every `MixturePrevision` and `ProductPrevision` that previously held its components/factors as bare `Vector` now holds them as `Vector{Prevision}`. The values stored are identical; only the Julia type system's enforcement changes. The `MixtureMeasure`/`ProductMeasure` constructors extract `.prevision` from Measure arguments — every Measure subtype has a `.prevision` field (verified in the audit's Measure shape check; DirichletMeasure has an extra `categories::Finite` but its `.prevision` is a standard `DirichletPrevision`).

**Particle (Phase A′):** `ParticlePrevision{T}` replaces `ParticlePrevision`. Same struct, same fields, same methods — the type parameter is inferred from the samples vector. Julia's existing `isa(x, ParticlePrevision)` and `::ParticlePrevision` dispatch match parametric instances.

**Enumeration (Phase A′):** `EnumerationPrevision` is retired. `EnumerationMeasure{T}` takes its place. The numerical behaviour is identical: `expect(m::EnumerationMeasure, f)` computes the same weighted sum as `expect(p::EnumerationPrevision, f)`, with weights from `m.prevision.log_weights` and domain elements from `m.carrier`. The only API change is the constructor shape and the type name in dispatch; all consumers are in `src/` or `apps/julia/` and are migrated in this move.

## 4. Worked end-to-end example

### Before (current):

```julia
# MixtureMeasure constructor passes Measure objects into MixturePrevision
function MixtureMeasure(space::Space, components::Vector{<:Measure}, log_weights::Vector{Float64})
    new(MixturePrevision(Vector{Measure}(components), log_weights), space)
end

# App code constructs with untyped container
components = Any[]
for (pi, p) in enumerate(programs)
    idx += 1
    push!(components, TaggedBetaPrevision(idx, BetaPrevision(1.0, 1.0)))
end
belief = MixturePrevision(components, log_prior_weights)
# typeof(belief.components) == Vector{Any}
```

### After (Move 8b):

```julia
# MixtureMeasure constructor extracts .prevision from each Measure
function MixtureMeasure(space::Space, components::Vector{<:Measure}, log_weights::Vector{Float64})
    previsions = Prevision[c.prevision for c in components]
    new(MixturePrevision(previsions, log_weights), space)
end

# App code constructs with typed container
components = TaggedBetaPrevision[]
for (pi, p) in enumerate(programs)
    idx += 1
    push!(components, TaggedBetaPrevision(idx, BetaPrevision(1.0, 1.0)))
end
belief = MixturePrevision(components, log_prior_weights)
# typeof(belief.components) == Vector{TaggedBetaPrevision}  (subtype of Vector{Prevision})
```

The extraction `c.prevision` works on every Measure subtype because every Measure is `(Prevision, carrier-space)` shape (verified empirically; see §5.1).

## 5. Open design questions

### 5.1 Field-type target for MixturePrevision and ProductPrevision

**The question.** Should the fields tighten to `Vector{Prevision}` (Move 5's committed type) or `Vector{Union{Prevision, Measure}}` (honest about the current dual-pathway reality)?

**Prior: `Vector{Prevision}`.** The Move 2/5 design lineage framed the dual-pathway as transitional. `Vector{Union{Prevision, Measure}}` would ratify the dual-pathway as permanent, amending Move 5's architecture retroactively. The design lineage should land as designed.

**Implementation consequence.** The `MixtureMeasure` constructor currently passes `Vector{Measure}(components)` into `MixturePrevision`. With `Vector{Prevision}`, it must extract `.prevision` from each Measure component: `Prevision[c.prevision for c in components]`. The `ProductMeasure` constructor has the same pattern (`Vector{Measure}(factors)` at line 324).

**Measure subtype shape verification.** All 9 Measure subtypes have the `(Prevision, carrier-space)` shape:

| Measure | prevision field type | Extra fields |
|---------|---------------------|-------------|
| BetaMeasure | BetaPrevision | None |
| TaggedBetaMeasure | TaggedBetaPrevision | None |
| GaussianMeasure | GaussianPrevision | None |
| GammaMeasure | GammaPrevision | None |
| NormalGammaMeasure | NormalGammaPrevision | None |
| CategoricalMeasure | Prevision (generic) | None |
| DirichletMeasure | DirichletPrevision | `categories::Finite` |
| ProductMeasure | ProductPrevision | None |
| MixtureMeasure | MixturePrevision | None |

DirichletMeasure's extra `categories::Finite` is carrier-binding context (the Finite support over which the Dirichlet is defined). It doesn't affect `.prevision` extraction — `DirichletMeasure.prevision` is a standard `DirichletPrevision`.

**No contested case.** Every Measure subtype reduces cleanly to `(Prevision, carrier-space)`. The extraction is straightforward. No premise failure.

**Internal construction sites in `src/ontology.jl`.** Several `condition` paths construct new `MixturePrevision`/`ProductPrevision` from existing components. These currently pass component arrays that may contain either Measures or Previsions depending on the code path. Each must be audited during implementation:

- `condition(p::MixturePrevision, k, obs)` (line ~1023) — constructs from conditioned components
- `prune(p::MixturePrevision)` (line ~1343) — constructs from subset of existing components
- `truncate(p::MixturePrevision)` (line ~1357) — constructs from subset of existing components
- `replace_factor(p::ProductPrevision, ...)` (line ~1051) — constructs from modified factors
- `condition(m::MixtureMeasure, ...)` (line ~1028) — wraps conditioned MixturePrevision in MixtureMeasure

These are all within `src/` and operate on components that are already Previsions (post-condition output) or already in the existing `components` field. The type tightening should be transparent at these sites — they just need the vector literal typed as `Prevision[]` rather than untyped.

### 5.2 ParticlePrevision and EnumerationPrevision

The initial version of this design doc deferred these two types as "residual debt." That framing repeats the Move 5 pattern — designed-but-not-landed — that this entire audit arc exists to retire. The three first-principles below yield direct answers.

**First-principles:**

1. **Prevision = algebraic content enabling `expect`.** A Prevision holds the parameters and weights that `expect` integrates over. It does not hold the objects being integrated over — those are domain elements or carrier-space objects.
2. **Measure = (Prevision, carrier-space).** Observational carrier binding happens at the Measure layer. When a type needs to store domain elements alongside a simplex, it is a Measure, not a Prevision.
3. **Parametric typing for uniform-within-instance, varying-across-instance fields.** When a field's element type is `Float64` at one construction site and `Vector{Float64}` at another, but every instance holds a uniform-type vector, the correct answer is a type parameter `{T}`, not `Vector{Any}`.

#### 5.2.1 ParticlePrevision → ParticlePrevision{T}

**Current state.** `ParticlePrevision` (prevision.jl:465) holds `samples::Vector` (bare) and `log_weights::Vector{Float64}`. The samples are draws from the underlying Measure's domain: `Float64` for Beta/Gaussian, `Vector{Float64}` for Dirichlet, tuples for NormalGamma, nested vectors for Product. Each instance's samples are uniform in type; the type varies across instances.

**Analysis.** The samples are algebraic content — they are the support points of an empirical distribution, the parameters `expect` needs to compute. `expect(p::ParticlePrevision, f)` (ontology.jl:625) evaluates `f(p.samples[i])` weighted by the simplex. The samples do not bind an external carrier space — they *are* the distribution's representation. This is parallel to `QuadraturePrevision.grid::Vector{Float64}` (already typed), which is a special case of `T = Float64`.

**Decision: parametric `ParticlePrevision{T}`.** The struct becomes:

```julia
struct ParticlePrevision{T} <: Prevision
    samples::Vector{T}
    log_weights::Vector{Float64}
    seed::Int
end
```

**Dispatch consequence.** All three methods on `ParticlePrevision` work generically over any `T`:

- `expect(p::ParticlePrevision, f::Function)` (ontology.jl:625) — `f(p.samples[i])` does not depend on `T`
- `expect(p::ParticlePrevision, tf::TestFunction)` (ontology.jl:672) — delegates to the above
- `weights(p::ParticlePrevision)` (stdlib.jl:48) — simplex-only, does not touch `samples`

Unparameterised `ParticlePrevision` in dispatch signatures (`p::ParticlePrevision`) matches `ParticlePrevision{T}` for any `T` in Julia — no method signatures need to change. `isa(x, ParticlePrevision)` checks work against the unparameterised generic. The conversion is entirely mechanical.

**Construction site.** `_condition_particle` (ontology.jl:1122–1127) calls `draw(m)` which returns `T`-typed values. The comprehension `[draw(m) for _ in 1:n_particles]` already produces `Vector{T}` by Julia's type inference. The only change is the struct accepting `Vector{T}` instead of `Vector{Any}`.

#### 5.2.2 EnumerationPrevision → EnumerationMeasure{T}

**Current state.** `EnumerationPrevision` (prevision.jl:524) holds `enumerated::Vector` (domain objects — programs, symbols) and `log_weights::Vector{Float64}` (simplex weights). The single production construction site (`enumerate_programs_as_prevision` at program_space/enumeration.jl:180) explicitly converts to `Vector{Any}`: `convert(Vector{Any}, programs)`.

**Analysis.** The `enumerated` field holds carrier-space objects, not algebraic content. `expect(p::EnumerationPrevision, f)` (ontology.jl:641) evaluates `f(p.enumerated[i])` — it applies a function to domain elements, not to distribution parameters. The simplex (`log_weights`) is the algebraic content; the enumerated objects are the carrier binding. This type conflates Prevision and Measure content in one struct, violating the Prevision/Measure split.

A Prevision that holds programs is named for what it isn't — the programs are the carrier, and the Prevision is just a `CategoricalPrevision` (simplex weights over a finite support). The existing `CategoricalPrevision` already holds `log_weights::Vector{Float64}` with a `weights()` method (stdlib.jl:34).

**Decision: retire `EnumerationPrevision`, introduce `EnumerationMeasure{T}`.** The replacement is:

```julia
struct EnumerationMeasure{T} <: Measure
    prevision::CategoricalPrevision
    carrier::Vector{T}
    space::Finite{T}
end
```

The `CategoricalPrevision` holds the simplex (algebraic content). The `carrier::Vector{T}` holds the enumerated domain objects. The `space::Finite{T}` is the carrier space — typed to match the carrier, because the space *is* the set of possibilities the carrier elements live in. This is the standard `(Prevision, carrier-space)` shape.

**Space-carrier type alignment.** The `space` field is `Finite{T}`, not bare `Space`, because the carrier type and the space type must agree: if `carrier::Vector{Program}`, then the space is `Finite{Program}` — the finite set whose elements are those programs. `Finite{T}` already accepts `Vector{T}` (spaces.jl:10–12), so `Finite(programs)` yields `Finite{Program}` directly. Substituting a `Symbol`-projection (`Finite(Symbol[p.name for p in programs])`) would misalign the space type (`Finite{Symbol}`) with the carrier type (`Vector{Program}`), creating a type error that the constructor should reject.

**Method migration.** EnumerationPrevision has exactly three methods:

| Current dispatch | Migrates to | Rationale |
|-----------------|-------------|-----------|
| `expect(::EnumerationPrevision, f::Function)` (ontology.jl:641) | `expect(m::EnumerationMeasure, f::Function)` — accesses `m.carrier[i]`, weights from `m.prevision` | `f` applied to carrier objects; the Measure holds both |
| `expect(::EnumerationPrevision, tf::TestFunction)` (ontology.jl:674) | `expect(m::EnumerationMeasure, tf::TestFunction)` — delegates | Same pattern |
| `weights(::EnumerationPrevision)` (stdlib.jl:62) | Deleted; `weights(m.prevision)` via existing `weights(::CategoricalPrevision)` | Simplex lives in CategoricalPrevision |

**Construction site migration.** `enumerate_programs_as_prevision` (enumeration.jl:171) becomes `enumerate_programs_as_measure`:

```julia
function enumerate_programs_as_measure(g::Grammar, max_depth::Int; ...)
    programs = enumerate_programs(g, max_depth; ...)
    log_weights = Float64[-g.complexity * log(2) - p.complexity * log(2) for p in programs]
    EnumerationMeasure{Program}(CategoricalPrevision(log_weights), programs, Finite(programs))
end
```

**CategoricalMeasure exception.** `CategoricalMeasure` stays as a Measure — it binds `Finite{T}` carrier space to `CategoricalPrevision`, which is the correct Measure/Prevision split. `EnumerationMeasure{T}` follows the same pattern: carrier-binding Measure wrapping an algebraic Prevision.

#### 5.2.3 Consumer migration

Consumers that dispatch on `EnumerationPrevision` (outside `src/`) must be updated. The migration is mechanical: replace `::EnumerationPrevision` with `::EnumerationMeasure`, access weights via `m.prevision`, access domain elements via `m.carrier`.

Consumers that dispatch on `ParticlePrevision` need no changes — unparameterised dispatch matches the parametric type.

Consumers that construct `EnumerationPrevision(items, weights)` → `EnumerationMeasure{T}(CategoricalPrevision(weights), items, space)`. The constructor gains a `space` argument (consistent with all other Measure types).

#### 5.2.4 Lint slug extension

The `untyped-mixture-construction` slug (§5.5) covers `MixturePrevision` and `ProductPrevision` construction. Extend its scope to also catch:

- `convert(Vector{Any}, ...)` feeding `EnumerationMeasure` construction (catches the retired pattern)
- Explicit `Any[]` feeding `ParticlePrevision` construction (Julia's type inference usually produces `Vector{T}` from comprehensions, but explicit `Any[]` is a discipline violation regardless)

The slug's taint-analysis pattern generalises: seed = untyped container literal; sink = Prevision/Measure constructor. No new slug needed; the existing slug's sink list expands.

### 5.3 Substrate `Any[]` sites

**The question.** Should the three substrate `Any[]` sites tighten in Phase A or Phase B?

**Prior: Phase A.** These are in `src/program_space/agent_state.jl` (lines 118, 140) and `src/host_helpers.jl` (line 47). All three construct uniformly-`TaggedBetaPrevision` contents. They are substrate code that downstream hosts mirror — fixing the example is a prerequisite for the app cleanup being principled rather than arbitrary.

**Verification.** `agent_state.jl:118` — `Any[]` populated by `Ontology.TaggedBetaPrevision(base_idx, Ontology.BetaPrevision(1.0, 1.0))`. Line 140 — `Any[state.belief.components..., new_components...]` concatenating two uniformly-TaggedBetaPrevision vectors. `host_helpers.jl:47` — `Any[]` populated by `prod.factors[c]` extracted from `ProductPrevision` factors, which are `BetaPrevision` objects. The disciplined type for host_helpers.jl:47 is `BetaPrevision[]` (not `TaggedBetaPrevision[]` — these are bare Beta factors, not tagged).

### 5.4 Phase ordering

**The question.** Should Phase A and Phase B land as one PR or two?

**Prior: two PRs, Phase A first.** Phase A is a substrate-only change that should pass all existing tests — the type tightening is a no-op behaviourally (the same objects flow through the same paths; only the Julia type system's enforcement changes). Phase B depends on Phase A: app construction sites that pass `TaggedBetaPrevision[]` into `MixturePrevision` type-check against `Vector{Prevision}` (which accepts any `Vector{<:Prevision}`) but would also type-check against bare `Vector`. Substrate-first means the contract is enforced before apps comply with it.

**Phase A vs Phase A′: two PRs.** Phase A lands Move 5's committed Mixture/Product tightening. Phase A′ is audit-surfaced architectural cleanup (Particle parametricity, Enumeration retirement) — logically separable, independently revertable, different narrative. Two-PR shape gives each clean provenance in `git log`.

Phase C (lint slug) lands with Phase B or as a small follow-up. It doesn't depend on Phase A semantically but should run against the post-Phase-B codebase to verify zero violations.

### 5.5 Lint slug coverage

**The question.** Should Move 8b include the `untyped-mixture-construction` lint slug?

**Prior: yes.** The whole reason Move 8b exists is that the lint missed this gap. Closing the gap without closing the lint blind spot means the next equivalent regression surfaces the same way. Landing the slug as part of Move 8b is the discipline-completing step.

**Slug design.** `untyped-mixture-construction` catches:
- Vector literal `Any[]` or bare `[]` assigned to a variable that is subsequently (a) populated via `push!` calls of a single uniform type, AND (b) passed to a `MixturePrevision`, `ProductPrevision`, `MixtureMeasure`, or `ProductMeasure` constructor.
- Implementation: extend pass-two taint analysis. Seed: `Any[]` or untyped `[]` assignment. Propagation: track `push!` targets and element types. Sink: tainted variable in Mixture/Product constructor call.
- Scope: all `.jl` files under `apps/` and `src/`. Exclude `test/`.
- Opt-out: `# credence-lint: allow — precedent:untyped-mixture-construction — <reason>` for justified heterogeneous construction (e.g., skin deserialisation). Or explicit `Vector{Union{TypeA, TypeB}}` annotation, which documents the heterogeneity in the type system.

**The other slug (`measure-vocabulary-in-prevision-app`) is deferred.** It catches stale vocabulary in apps but doesn't close a substrate-level discipline gap. One slug per move; the `untyped-mixture-construction` slug is the one that prevents the blind spot from recurring.

## 6. Risk + mitigation

**Risk 1: The `MixtureMeasure`/`ProductMeasure` extraction breaks something.**

The Phase A change requires extracting `.prevision` from each Measure component. Move 5 designed this and didn't implement it. The most likely explanation is scope: Move 5 delivered the module split, stdlib, and lint slug — substantial work that consumed the move's budget — and the field-type tightening, though designed, was never coded.

Evidence against a hidden technical blocker: (a) every Measure subtype has a `.prevision` field (verified empirically); (b) `state_persistence.jl:83` already constructs `MixturePrevision` with `TaggedBetaPrevision[]`, demonstrating the Prevision-direct pathway works; (c) the `condition(p::MixturePrevision, ...)` code path already operates on the Prevision-level components, not the Measure wrappers.

Mitigation: Phase A's test plan is "run the existing test suite and confirm no regressions." If any test fails, the failure identifies a code path that depends on `MixturePrevision.components` containing Measure objects — that would be a finding to address, not a reason to abandon the tightening.

**Risk 2: Internal `condition`/`prune`/`truncate` paths construct components from mixed types.**

Several `src/ontology.jl` code paths construct new `MixturePrevision` objects from conditioned/pruned/truncated component lists. If any of these paths produce a mix of Prevision and Measure objects in the same list, the `Vector{Prevision}` field type would reject it at runtime.

Mitigation: each internal construction site will be inspected during Phase A implementation. The likely outcome is that all internal paths already produce Prevision objects (since `condition` on a Prevision returns a Prevision), but verification is required.

**Risk 3: EnumerationPrevision retirement breaks downstream callers.**

Any code that dispatches on `::EnumerationPrevision` or constructs it directly will break. The blast radius is narrow: one production construction site (`enumerate_programs_as_prevision` in `src/program_space/enumeration.jl`), three `expect`/`weights` dispatches in `src/`, and test files. No `apps/` code dispatches on `EnumerationPrevision` directly.

Mitigation: grep for `EnumerationPrevision` before and after the refactor. Every hit must be migrated or deleted.

**Risk 4: `ParticlePrevision{T}` breaks serialisation or persistence.**

If any persistence code (state_persistence.jl, production_persistence.jl) serialises `ParticlePrevision` objects, the parametric type changes the serialised representation.

Mitigation: grep for `ParticlePrevision` in persistence code. If found, update the deserialisation to handle both the old unparameterised and new parametric form during migration.

**Risk 5: `EnumerationMeasure{T}` space-carrier type alignment.**

The `EnumerationMeasure{T}` constructor requires `space::Finite{T}` matching `carrier::Vector{T}`. If any construction site builds the space from a projection (e.g. `Symbol` names) rather than the carrier objects themselves, the types will mismatch at compile time. The design doc's migration example uses `Finite(programs)` directly, which is correct. The risk is that an untouched construction site or test uses the projection pattern.

Mitigation: the `space::Finite{T}` field type enforces alignment — a `Finite{Symbol}` will not fit `Finite{Program}`. Type errors surface at construction, not at use.

**Risk 6: The lint slug surfaces unexpected violations.**

When `untyped-mixture-construction` runs against the post-Phase-B codebase, it may find construction sites beyond the ten the audit named. These are findings worth investigating before merging.

Mitigation: run the slug locally during Phase C development and diagnose any unexpected hits before opening the PR.

## 7. Verification cadence

### Phase A (substrate)

1. `julia test/test_core.jl` — DSL core unchanged numerically
2. `julia test/test_prevision_unit.jl` — Prevision unit tests pass
3. `julia test/test_prevision_mixture.jl` — mixture-specific tests pass
4. `julia test/test_host.jl` — host integration tests pass (exercises `MixtureMeasure` → `MixturePrevision` extraction via `host_helpers.jl`)
5. `julia test/test_flat_mixture.jl` — flat mixture conditioning paths
6. `julia test/test_persistence.jl` — serialisation round-trip (exercises `agent_state.jl` construction)
7. `julia test/test_program_space.jl` — program enumeration paths
8. `python tools/credence-lint/credence_lint.py check apps/` — 0 violations
9. `python tools/credence-lint/credence_lint.py test` — corpus 14/10/5
10. `PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run pytest apps/python/` — Python tests unaffected

### Phase A′ (Particle/Enumeration substrate)

1. All Phase A tests still pass
2. `julia test/test_program_space.jl` — program enumeration returns `EnumerationMeasure` not `EnumerationPrevision`
3. `julia test/test_core.jl` — particle conditioning produces `ParticlePrevision{T}` (inferred)
4. `grep -r 'EnumerationPrevision' src/ apps/ test/ examples/` — zero hits (type fully retired)
5. `grep -rn 'samples::Vector[^{]' src/prevision.jl` — zero hits (bare `samples::Vector` replaced by `samples::Vector{T}`)
6. `python tools/credence-lint/credence_lint.py check apps/` — 0 violations

### Phase B (apps)

1. All Phase A tests still pass
2. `julia apps/julia/email_agent/host.jl` with test corpus — simulation output unchanged
3. `python tools/credence-lint/credence_lint.py check apps/` — 0 violations
4. `uv run python apps/skin/test_skin.py` — skin unchanged

### Phase C (lint)

1. `python tools/credence-lint/credence_lint.py test` — corpus updated (14+N good / 10+M bad-pass-one / 5+P bad-pass-two)
2. `python tools/credence-lint/credence_lint.py check apps/` — 0 violations against post-Phase-B codebase

## 8. de Finettian discipline self-audit

1. **Is every numerical query in this move routed through `expect`?** N/A — Move 8b changes type annotations and container types, not numerical computation.

2. **Does this move hold a Prevision inside a Measure, or a Measure inside a Prevision?** After Move 8b, `MixturePrevision.components` holds `Vector{Prevision}`, not `Vector{Measure}`. The "Measure inside a Prevision" anti-pattern that the current bare `Vector` allows (via `MixtureMeasure` passing `Vector{Measure}`) is eliminated. The "carrier inside a Prevision" anti-pattern (`EnumerationPrevision.enumerated` holding domain objects) is eliminated by retiring `EnumerationPrevision` and replacing it with `EnumerationMeasure{T}`, which correctly separates the simplex (`CategoricalPrevision`) from the carrier (`Vector{T}`). This is the point.

3. **Does this move introduce an opaque closure where a declared structure would fit?** No. The lint slug uses the existing pass-two taint analysis framework.

4. **Does this move add a `getproperty` override on any Prevision subtype?** No.

## Commit sequence

### Phase A (PR 1: substrate)

1. **Tighten `MixturePrevision.components` and `ProductPrevision.factors` to `Vector{Prevision}`.** Update struct definitions in `src/prevision.jl`. Update `MixtureMeasure`/`ProductMeasure` constructors in `src/ontology.jl` to extract `.prevision`. Update internal construction sites in `condition`/`prune`/`truncate`.
2. **Tighten substrate `Any[]` sites.** `src/program_space/agent_state.jl` lines 118, 140 → `TaggedBetaPrevision[]`. `src/host_helpers.jl` line 47 → `BetaPrevision[]`.
3. **Full test suite verification.**

### Phase A′ (PR 2: Particle/Enumeration substrate)

4. **Make `ParticlePrevision` parametric.** `struct ParticlePrevision{T} <: Prevision` with `samples::Vector{T}`. Inner constructor parameterised. No method signature changes needed (unparameterised dispatch matches).
5. **Retire `EnumerationPrevision`.** Delete the struct from `src/prevision.jl`. Delete `weights(::EnumerationPrevision)` from `src/stdlib.jl`. Delete `expect(::EnumerationPrevision, ...)` methods from `src/ontology.jl`.
6. **Introduce `EnumerationMeasure{T}`.** New struct in `src/ontology.jl`: `(CategoricalPrevision, carrier::Vector{T}, space::Space)`. Add `expect(::EnumerationMeasure, f::Function)` and `expect(::EnumerationMeasure, tf::TestFunction)` methods.
7. **Migrate `enumerate_programs_as_prevision` → `enumerate_programs_as_measure`.** Update `src/program_space/enumeration.jl` and all callers.
8. **Full test suite verification.**

### Phase B (PR 3: apps + examples)

9. **Tighten app/skin `Any[]` sites.** `email_agent/host.jl`, `grid_world/host.jl`, `rss/host.jl`, `skin/server.jl` — `Any[]` → `TaggedBetaPrevision[]`.
10. **Migrate email_agent secondary files.** `live.jl`, `eval_retrospective.jl` — Measure pathway → Prevision pathway.
11. **Migrate `examples/host_credence_agent.jl` vocabulary.** MixtureMeasure → MixturePrevision in imports, type checks, comments.
12. **Full test suite + lint verification.**

### Phase C (PR 4, or folded into PR 3: lint slug)

13. **Add `untyped-mixture-construction` slug.** Pattern, corpus files, scope exclusions.
14. **Corpus self-test + `check apps/` verification.**

## Closure ritual

After Move 8b lands (all three phases merged):

1. **Posture 4 closure confirmation.** Either amend the existing completion audit PR (#69) or open a brief closure-confirmation PR that states: "Posture 4 complete. Substrate field types match Move 5's committed architecture. All construction sites use typed containers. Lint blind spot closed."
2. **Posture 5 master plan stub.** Open `docs/posture-5/master-plan.md` with the provisional move list from the architectural-review conversation (cache-discipline audit, benchmark methodology, benchmark execution, release engineering, distribution). Mark it provisional; Posture 5's first design conversation refines it.
3. **Shelved Move 9 relocation.** Move `docs/posture-4/move-9-design.md` to `docs/posture-6-prep/personal-agent-priors.md` with a header marking it provisional and citing the strategic reframe.
4. **pomdp_agent status determination.** Per the previous audit's Task 2 — deferred until the closing arc completes, which it now has. Determination can proceed as part of Posture 5's opening or as a standalone PR.

## Out of scope

- Full Measure-type retirement (deleting `MixtureMeasure`, `ProductMeasure`, etc. from `src/ontology.jl`). Move 5 designed this; Move 8b does not attempt it. The Measure types remain as declared views over Prevision, which is architecturally sound.
- Schema v4, Connection abstraction, Maildir, Telegram, server loop (shelved Move 9 scope).
- Any Posture 5 move implementation (cache audit, benchmark methodology, etc.).
- pomdp_agent migration (46 Measure-vocabulary sites; status determination is sequenced after Move 8b).
- The `measure-vocabulary-in-prevision-app` lint slug (deferred to follow-up; one slug per move).
