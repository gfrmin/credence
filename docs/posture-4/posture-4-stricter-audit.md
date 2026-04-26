# Posture 4 Stricter Completion Audit

**Date:** 2026-04-26

The previous completion audit (PR #69) reported 144/0 lint violations and concluded "Posture 4 complete at Move 8." Investigation of the first remediation task — migrating six Measure-vocabulary sites in `apps/julia/email_agent/live.jl` and `eval_retrospective.jl` — surfaced a deeper problem: the migrated `host.jl` uses `components = Any[]` to construct `MixturePrevision` despite all components being uniformly `TaggedBetaPrevision`. The lint enforces accessor patterns and Measure-side operation discipline but does not check container element-type discipline at construction sites. The lint's "144/0" headline was therefore misleading — it certified one form of Prevision-primary discipline while a different form of typing discipline went unchecked.

The Move 2 → Move 5 design lineage is directly relevant. Move 2's PR #51 amendment (§5.1.1–5.1.2) deferred `MixturePrevision.components::Vector{Prevision}` and `ProductPrevision.factors::Vector{Prevision}` tightening to Move 5, citing architectural coupling to the `condition` rewrite. Move 5's design doc (§0, §1, §2, §4) explicitly committed to landing this tightening. Whether it actually landed is the question this audit answers empirically.

## Part 1: Substrate field types

### Current state

Four composite Prevision types in `src/prevision.jl` have bare unparameterized `Vector` fields:

| Type | Field | Line | Current type |
|------|-------|------|-------------|
| `ProductPrevision` | `factors` | 368 | `::Vector` |
| `MixturePrevision` | `components` | 393 | `::Vector` |
| `ParticlePrevision` | `samples` | 466 | `::Vector` |
| `EnumerationPrevision` | `enumerated` | 525 | `::Vector` |

The `MixturePrevision` inner constructor (`src/prevision.jl:396`) accepts `components::Vector` — no type constraint on elements. The `ProductPrevision` has no inner constructor at all; Julia's default constructor accepts `Vector` of anything.

### What passes through these fields

The `MixtureMeasure` constructor (`src/ontology.jl:355-357`) calls:
```julia
MixturePrevision(Vector{Measure}(components), log_weights)
```

So through the Measure pathway, `MixturePrevision.components` holds `Measure` objects. Through the direct Prevision pathway (app code constructing `MixturePrevision` directly), it holds whatever the caller passes — `TaggedBetaPrevision` in practice, typed as `Any[]` or inferred from comprehensions.

### Design doc lineage

**Move 2 design doc (§5.1.1, line 129):**
> **Option C chosen.** The element-type tightening on `MixturePrevision.components` and `ProductPrevision.factors` is architecturally coupled to Move 5's `condition` rewrite. The master plan was wrong to partition them.

**Move 2 design doc (§5.1.2, line 137):**
> **`MixturePrevision.components::Vector{Prevision}`, `ProductPrevision.factors::Vector{Prevision}`, and shield reconstruction activation defer to Move 5** concurrent with the `condition` rewrite.

**Move 5 design doc (§0, line 5):**
> Two architectural inflexions complete the alignment: (1) `MixturePrevision.components::Vector{Prevision}` and `ProductPrevision.factors::Vector{Prevision}` tighten — the deferred Move 2 Phase 4 scope lands here paired with the `condition` rewrite [...]

**Move 5 design doc (§1, line 9):**
> Tighten the Prevision-internal vector types so `MixturePrevision.components::Vector{Prevision}` and `ProductPrevision.factors::Vector{Prevision}` hold actual Previsions, not Measure-wrapped Previsions.

**Move 5 design doc (§2, line 26):**
> Internal struct definitions for `MixturePrevision.components` and `ProductPrevision.factors` tighten from `Vector{Measure}` to `Vector{Prevision}`.

**Move 5 design doc (§8, line 261):**
> `MixturePrevision.components::Vector{Prevision}` and `ProductPrevision.factors::Vector{Prevision}` tighten — no more `Vector{Measure}`-of-Prevision-wrappers.

### What Move 5 actually delivered

Move 5 delivered:
- The six-file module split (spaces.jl, events.jl, kernels.jl, prevision.jl, conjugate.jl, stdlib.jl extracted from ontology.jl)
- The `expect-through-accessor` lint slug with corpus
- The stdlib one-liners (`mean`, `variance`, `probability`, `weights`, `marginal`)
- Retirement of 12 `posterior-iteration` pragma sites

Move 5 did NOT deliver:
- Measure type deletion (`src/ontology.jl` is 1381 lines, reduced from 1860 by extraction, not deleted; all Measure subtypes still exist)
- Field-type tightening (`components::Vector` and `factors::Vector` remain bare)
- Shield removal (`getproperty` overrides on `MixtureMeasure` and `ProductMeasure` still exist)

### Verdict

**(c) Intermediate state.** The substrate is neither Move 2's deferred state (which described the fields as `Vector{Measure}` internally) nor Move 5's committed tightened state (`Vector{Prevision}`). It is bare unparameterized `Vector` — looser than either design doc described. The tightening was explicitly designed and committed in both Move 2's deferral and Move 5's purpose statement, but never implemented. This is a Posture 4 substrate-completion finding that the previous audit missed entirely.

## Part 2: Construction-site typing discipline

### `Any[]` for uniformly-typed components (8 sites)

All sites below construct a `MixturePrevision` where every `push!` adds a `TaggedBetaPrevision`. The disciplined type is `TaggedBetaPrevision[]`.

| File | Line | Layer |
|------|------|-------|
| `apps/julia/email_agent/host.jl` | 749 | Host |
| `apps/julia/grid_world/host.jl` | 264 | Host |
| `apps/julia/rss/host.jl` | 58 | Host |
| `apps/skin/server.jl` | 643 | Skin |
| `src/program_space/agent_state.jl` | 118 | Substrate |
| `src/program_space/agent_state.jl` | 140 | Substrate |
| `src/host_helpers.jl` | 47 | Substrate |

Note: two of these are in `src/` — substrate code, not just application-level sloppiness.

### `Measure[]` for uniformly-typed components, old vocabulary (2 sites)

These construct `MixtureMeasure` where every `push!` adds a `TaggedBetaMeasure`. They use the correct element type for the Measure pathway but should be migrated to the Prevision pathway.

| File | Line | Layer |
|------|------|-------|
| `apps/julia/email_agent/live.jl` | 170 | Driver |
| `apps/julia/email_agent/eval_retrospective.jl` | 137 | Eval script |

### Correctly typed (1 site)

| File | Line | Type used |
|------|------|-----------|
| `apps/julia/email_agent/state_persistence.jl` | 83 | `TaggedBetaPrevision[]` |

This is the only construction site in the apps tree that uses the disciplined element type.

### Heterogeneous construction (justified)

| File | Lines | Reason |
|------|-------|--------|
| `apps/skin/server.jl` | 176, 180, 213, 217 | JSON deserialisation; component types determined at runtime |
| `src/host_helpers.jl` | 104 | Prepending `CategoricalMeasure` to `ProductMeasure` factors |
| `src/ontology.jl` | condition internals | Constructing posteriors from mixed component types |

These are genuine heterogeneous-element cases where `Measure[]` or untyped containers are defensible.

### Measure-pathway construction (correct for that pathway)

| File | Lines | Pattern |
|------|-------|---------|
| `src/host_helpers.jl` | 19, 20, 34, 54, 55 | `Measure[]` through `MixtureMeasure` constructors |
| `src/ontology.jl` | 324, 344, 356 | Internal Measure-side construction |

These use `Measure[]` because they construct `MixtureMeasure`/`ProductMeasure`, which expects `Vector{<:Measure}`. Correct for the Measure pathway.

### Additional Measure vocabulary in examples

`examples/host_credence_agent.jl` (lines 16, 102, 135) still imports and type-checks against `MixtureMeasure` vocabulary. No direct Mixture/Product construction (delegates to `host_helpers.jl`), but the vocabulary is stale.

### Summary

| Category | Count | Disciplined alternative |
|----------|-------|------------------------|
| `Any[]` for uniform `TaggedBetaPrevision` | 8 | `TaggedBetaPrevision[]` |
| `Measure[]` for uniform `TaggedBetaMeasure` (old vocab) | 2 | `TaggedBetaPrevision[]` + Prevision pathway |
| Correctly typed | 1 | (already correct) |
| Heterogeneous (justified) | 7 | (no change needed) |
| Measure-pathway (correct) | 8 | (no change needed) |

## Part 3: Other type-discipline holes

### Vector element types in non-mixture contexts

`Dict{Symbol, Any}` for temporal state appears in `email_agent/host.jl`, `grid_world/host.jl`, `rss/host.jl`. **Justified** — these are heterogeneous session dictionaries with dynamically-keyed runtime data (recent email history, thread counts, feature caches). The keys and value types change per domain.

`Dict{Tuple{Any,String}, Vector{Any}}` in `apps/julia/pomdp_agent/src/models/factored_world_model.jl:26` erases the observable state space type. This is a typing discipline gap — the keys are `(MinimalState, String)` tuples and the values are `Vector{MinimalState}` — but it is inside the pomdp_agent package which has its own invariants.

### Function fields

All `::Function` fields in the codebase are documented and justified:
- `src/kernels.jl:10,41,47,48` — kernel algebra closures (mathematical necessity)
- `src/prevision.jl:169` — `OpaqueClosure` (documented escape hatch with warning)
- `src/program_space/types.jl:199` — `CompiledKernel.evaluate` (closures cannot be serialised; documented)
- `apps/julia/email_agent/features.jl:188` — domain-specific decision closure
- `apps/julia/pomdp_agent/` — domain-specific sensor/preprocessor closures

None trigger the Invariant 2 concern at the axiom layer.

### Untyped struct fields

None found. All struct fields across the codebase have explicit type annotations.

### Missing return type annotations

Public app-level functions missing return types:
- `apps/julia/email_agent/host.jl:718` — `run_agent` returns NamedTuple
- `apps/julia/grid_world/host.jl:243` — `run_agent` returns NamedTuple
- `apps/julia/email_agent/host.jl:160` — `select_action_eu` returns NamedTuple
- `apps/julia/email_agent/eval_retrospective.jl:133` — `init_agent` returns `AgentState`

Low severity — Julia relies on inference for return types, and these are app-level functions, not substrate. But they are the public surface readers encounter first.

### Verdict

The dominant discipline gap is the `Any[]`-for-uniform-contents pattern (Part 2). The other categories are either justified (`Dict{Symbol,Any}` for session state, `::Function` for kernel closures) or low-severity (missing return type annotations). The pomdp_agent typing gaps are internal to that package.

## Part 4: Lint coverage extension proposals

### Proposal 1: `untyped-mixture-construction`

**Pattern caught:** Vector literal `Any[]` (or bare `[]` with no element type) assigned to a variable that is subsequently:
1. Populated via `push!` calls of a single uniform type, AND
2. Passed to a `MixturePrevision`, `ProductPrevision`, `MixtureMeasure`, or `ProductMeasure` constructor.

**Scope:** All `.jl` files under `apps/` and `src/` (including `src/program_space/`). Exclude `test/`.

**Implementation approach:** Extend the existing pass-two taint analysis. Seed rule: `Any[]` or untyped `[]` assignments are tainted. Propagation: track `push!` calls on tainted variables and record element types. Sink rule: tainted variable passed to Mixture/Product constructor triggers a violation if all pushed elements share a single type.

**False-positive rate:** Low. The genuine-heterogeneity case (e.g., skin deserialisation where component types are determined at runtime) would need a pragma. Propose: `# credence-lint: allow — precedent:untyped-mixture-construction — <reason>` for sites where heterogeneous construction is justified.

**Opt-out for justified heterogeneity:** Either an explicit `Vector{Union{TypeA, TypeB}}` type annotation (which documents the heterogeneity in the type system), or the pragma. The pragma should name the reason the heterogeneity is unavoidable.

**Estimated scope:** Would catch 8 `Any[]` sites (Part 2 inventory) plus the 2 `Measure[]` sites that are stale vocabulary. The skin deserialisation sites and `host_helpers.jl:104` would need pragmas.

### Proposal 2: `measure-vocabulary-in-prevision-app`

**Pattern caught:** References to Measure type names (`BetaMeasure`, `TaggedBetaMeasure`, `MixtureMeasure`, `ProductMeasure`, `GaussianMeasure`, `GammaMeasure`) in `using`/`import` statements, type assertions (`::TaggedBetaMeasure`), or constructor calls — in files within `apps/julia/` directories whose `host.jl` already speaks Prevision vocabulary.

**Scope:** `apps/julia/*/` files, excluding `apps/julia/pomdp_agent/` (own invariants). `CategoricalMeasure` is excluded (principled exception — it binds carrier space).

**Implementation approach:** Pass-one regex scan (like the existing pass-one). Pattern: `\b(Beta|TaggedBeta|Mixture|Product|Gaussian|Gamma)Measure\b` in non-comment, non-pragma lines. File-scope exclusion for any file containing `# credence-lint: file-exclude — measure-vocabulary-in-prevision-app`.

**False-positive rate:** Very low within scope. The Measure types are well-named and distinct from Prevision types. The only edge case is files that legitimately work at the Measure level (e.g., persistence code loading v3 fixtures) — these would use file-scope exclusion.

**Estimated scope:** Would catch the 6 email_agent secondary-file sites, the `examples/host_credence_agent.jl` vocabulary, and any future regressions.

## Overall verdict

**Posture 4 is not complete at Move 8.** The previous audit's conclusion was premature. Three findings, in descending order of significance:

**1. Substrate field-type tightening never landed (Part 1).** Move 5's design doc explicitly committed to `MixturePrevision.components::Vector{Prevision}` and `ProductPrevision.factors::Vector{Prevision}`. The current fields are bare `Vector` — looser than either Move 2's deferred description or Move 5's committed target. This is a substrate-level gap, not application-level cleanup. The `MixtureMeasure` constructor passes `Vector{Measure}(components)` into `MixturePrevision`, meaning the "Prevision-primary" type currently holds `Measure` objects when constructed through the Measure pathway. Additionally, `ParticlePrevision.samples` and `EnumerationPrevision.enumerated` have the same bare `Vector` pattern, though these were not named in the Move 2/5 design docs.

**2. Ten construction sites use untyped containers for uniform contents (Part 2).** Eight `Any[]` sites and two `Measure[]` sites construct mixtures from uniformly `TaggedBetaPrevision`/`TaggedBetaMeasure` components. Two of the `Any[]` sites are in `src/` (substrate), not just `apps/`. The one correctly-typed site (`state_persistence.jl:83` using `TaggedBetaPrevision[]`) demonstrates that the disciplined pattern works and is not blocked by any substrate limitation.

**3. The lint has a structural blind spot (Part 4).** Container element-type discipline is an unchecked category. The two proposed slugs (`untyped-mixture-construction` and `measure-vocabulary-in-prevision-app`) would close the gap but are not yet implemented.

### What closing Posture 4 properly now requires

The finishing work is a real small move — call it Move 8b — with two phases:

**Phase A (substrate):** Tighten the four `::Vector` fields in `src/prevision.jl` to their committed types. This requires deciding whether the field type is `Vector{Prevision}` (as Move 5 designed) or whether the Measure-pathway usage means `Vector{Union{Prevision, Measure}}` (which would be honest about the current dual-pathway reality). If `Vector{Prevision}` is the target, the `MixtureMeasure` constructor must extract `.prevision` from each Measure component before passing to `MixturePrevision` — which is exactly what Move 5 designed but didn't implement. This is a substrate change that needs test verification.

**Phase B (apps):** Tighten the 10 construction sites to use typed containers. Mechanical: `Any[]` → `TaggedBetaPrevision[]` at the 8 uniform sites, `Measure[]` → `TaggedBetaPrevision[]` + Prevision pathway at the 2 old-vocabulary sites. Plus the Measure vocabulary cleanup in `examples/host_credence_agent.jl` and the 6 email_agent secondary-file sites.

The lint slug proposals (Part 4) are post-Move-8b and can land as Posture 5 infrastructure or as a follow-up — they prevent regression but aren't blocking for closure.

Total scope estimate: ~50 lines of substrate changes (Phase A) + ~30 lines of app changes (Phase B) + test verification. Half a day to a day, depending on whether the Measure-pathway `Vector{Measure}` → `Vector{Prevision}` extraction surfaces edge cases in `condition` or `host_helpers.jl`.
