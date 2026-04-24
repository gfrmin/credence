# Move 2 — Previsions hold Previsions

## 0. Final-state alignment

Move 2 retires two of the "internal representation invariants" in `master-plan.md` §"Final-state architecture" — the **untyped-element problem** in `TaggedBetaPrevision.beta::Any` and the `Vector` fields on `ProductPrevision` / `MixturePrevision` get replaced with typed analogues (`::BetaPrevision`, `::Vector{Prevision}`, `::Vector{Prevision}`). The `decompose(::ExchangeablePrevision) → MixturePrevision` path becomes structurally typed. Measure subtypes remain alive and their `getproperty` shields retire in Move 5; this move leaves explicit transient state documented in §2's Risk section — specifically, the shield on `MixtureMeasure` / `ProductMeasure` now returns a reconstructed `Vector{Measure}` on read rather than the stored vector, which breaks the shared-reference contract for `push!`-through-shield patterns in `apps/skin/server.jl:611-614`. That breakage is **bounded to non-CI-tested code** and is fixed by Move 7's skin rewrite (which replaces the `push!` pattern with explicit mutation APIs per `master-plan.md` §Move 7). This move makes the tradeoff explicit rather than glossed.

## 1. Purpose

Tighten the three concrete Prevision subtypes that currently hold Measures-inside-Previsions to hold Previsions-inside-Previsions. Retire the pragmatic `Vector` / `Any` field types in favour of declared parametric subtypes. The `decompose` path that returns `MixturePrevision{...}` becomes structurally type-preserving. External consumers continue to read `m.components` / `m.factors` / `m.beta` via the Measure-level shields, which reconstruct Measure-shaped wrappers on the fly.

## 2. Files touched

Creates: none.

Modifies:
- `src/prevision.jl`
  - `TaggedBetaPrevision` struct (`src/prevision.jl:226-229`): `beta::Any` → `beta::BetaPrevision`. Constructor takes `BetaPrevision` directly (not `BetaMeasure`).
  - `ProductPrevision` struct (`src/prevision.jl:353-355`): `factors::Vector` → `factors::Vector{Prevision}`.
  - `MixturePrevision` struct (`src/prevision.jl:378-392`): `components::Vector` → `components::Vector{Prevision}`. Constructor signature + body updated; inner `new(components, log_weights .- log_total)` unchanged structurally but now typed.
  - Add two new exported functions for explicit mutation at the Prevision level:
    - `push_component!(p::MixturePrevision, c::Prevision, log_weight::Float64)` — appends to `p.components` and `p.log_weights`, re-normalises.
    - `replace_component!(p::MixturePrevision, i::Int, c::Prevision)` — replaces element `i`.

    These exist to give Move 7's skin rewrite a disciplined target; they are not consumed by `src/` internally today. Skin's existing `push!`-through-shield pattern is actively guarded against at the shield level per the executable-guard mechanism described below.
  - Add `FrozenVectorView{T}` — a thin read-only wrapper produced by the Measure-level shield on `:components` / `:factors` / `:log_weights` reads. Methods: `getindex`, `length`, `iterate`, `eachindex`, `firstindex`, `lastindex`, `size`, `axes`. **Throws on `push!`, `setindex!`, `append!`, `pop!`, `resize!`, `empty!`** with a message pointing at `push_component!` / `replace_component!` as the migration target. This is the executable guard that converts silent `push!`-through-shield breakage into loud runtime errors — skin's line 611 fails at the first invocation on a Move-2+ tip, with a pointer to the fix. See §5.1 for the rationale; this guard is the load-bearing mitigation, not the §6 prose flag.
  - Export list: `push_component!`, `replace_component!`, `FrozenVectorView` added (the last is exported so external consumers get a legible error-source rather than `Main.Previsions.FrozenVectorView` in stack traces).
- `src/ontology.jl`
  - `TaggedBetaMeasure` constructor (`src/ontology.jl:210-212`): update to wrap `beta::BetaMeasure` as `BetaPrevision` when constructing the `TaggedBetaPrevision`. (Constructor accepts `BetaMeasure` for external compatibility; extracts `.prevision` internally.)
  - `ProductMeasure` constructor (`src/ontology.jl:363-367`): update `Vector{Measure}(factors)` to `Vector{Prevision}([f.prevision for f in factors])`.
  - `MixtureMeasure` constructor (`src/ontology.jl:403-405`): similar — extract `.prevision` from each Measure component when constructing the `MixturePrevision`.
  - `ProductMeasure.getproperty` shield (`src/ontology.jl:372-378`): change from `return getproperty(getfield(m, :prevision), s)` to reconstruct Measure wrappers on read, wrapped in `FrozenVectorView`. Specifically, `:factors` branch returns `FrozenVectorView([wrap_in_measure(p) for p in getfield(m, :prevision).factors])` — new fresh vector each call, wrapped to throw on mutation. Need a helper `wrap_in_measure(p::Prevision) → Measure` that pattern-matches on Prevision subtype and produces the canonical Measure wrapper.
  - `MixtureMeasure.getproperty` shield (`src/ontology.jl:411-417`): same reconstruction treatment for `:components`. `:log_weights` is `Vector{Float64}` on both sides; shield now also wraps the return in `FrozenVectorView` so that skin's `push!(state.belief.log_weights, ...)` at `apps/skin/server.jl:614` fails loudly in parallel with the `push!(state.belief.components, ...)` at line 611. Consistent mutation-hazard treatment for both fields.
  - `TaggedBetaMeasure.getproperty` shield (`src/ontology.jl:215-221`): `:beta` branch wraps `BetaPrevision` into fresh `BetaMeasure` on read. Single-element return, no `FrozenVectorView` needed.
  - New helper: `wrap_in_measure(p::Prevision) → Measure` dispatching on concrete Prevision subtype. Minimum coverage: `BetaPrevision → BetaMeasure`, `TaggedBetaPrevision → TaggedBetaMeasure`, `GaussianPrevision → GaussianMeasure`, `DirichletPrevision → DirichletMeasure`, `GammaPrevision → GammaMeasure`, `NormalGammaPrevision → NormalGammaMeasure`, `ProductPrevision → ProductMeasure`, `MixturePrevision → MixtureMeasure`, `CategoricalPrevision → CategoricalMeasure`. Needs the Measure's `space` — stored on the Prevision where canonical, or reconstructed where derivable.
- Internal `src/ontology.jl` readers that index into `m.factors[i]` / `m.components[i]` (e.g. `src/ontology.jl:1097` `copy(m.logw)`, `src/ontology.jl:1556-1640` mixture-condition path, `src/ontology.jl:1828-1838` threshold-prune path) — these read through the shield and receive reconstructed `Vector{Measure}` / `Vector{Float64}`. Semantically unchanged (values are the same), but now each read allocates a fresh Vector. No behaviour change at the invariance-target level.

Deletes: none.

Renames: none.

**Commit phasing.** Code PR lands as four commits:

1. **Phase 1** — add `push_component!` / `replace_component!` API surface at the Prevision level. New functions only; no callers. Full test suite green.
2. **Phase 2** — add `FrozenVectorView{T}` type in `ontology.jl`: read-pass-through wrapper that throws on mutation. Methods cover the read surface used by internal `src/` readers and by tests (`getindex`, `length`, `iterate`, `eachindex`, `firstindex`, `lastindex`, `size`, `axes`); mutation methods (`push!`, `setindex!`, `append!`, `pop!`, `resize!`, `empty!`) throw with a message pointing at `push_component!` / `replace_component!`. Unit-tested for read-pass-through behaviour but not wired into any shield yet. Full test suite green.
3. **Phase 3** — add `wrap_in_measure(p::Prevision) → Measure` helper covering all nine concrete Prevision subtypes. Full test suite green.
4. **Phase 4** — tighten the three struct field types (`TaggedBetaPrevision.beta`, `ProductPrevision.factors`, `MixturePrevision.components`); update shield returns to use `wrap_in_measure` and wrap the returned Vectors in `FrozenVectorView`; update constructors to unwrap `.prevision` from Measure arguments. Full test suite green; `--verify` passes. This is the phase where the executable guard becomes active.

The helpers (API surface + `FrozenVectorView` + `wrap_in_measure`) all land before the type-tightening so Phase 4 is a minimal, focused diff. Each phase passes the test suite independently.

## 3. Behaviour preserved

Move 0 fixture at `test/fixtures/posture-3-capture/` (6118 site×value tuples at branch-point `5c6a94e`) is the invariance target. Move 2 asserts:

- Every Stratum-1/2/3 tolerance-shape assertion → same value within tolerance.
- Every Exact-shape assertion → bit-exact. This includes assertions that read `m.factors[i].alpha` or `m.components[i].beta` via the shield — they still return the same value because the reconstructed BetaMeasure's `.alpha` goes through the BetaMeasure shield to `BetaPrevision.alpha`, which is the stored field unchanged.
- Every Directional-shape → inequality preserved.
- Every Structural-shape → expression evaluates to `true`. Notable: structural assertions that use `isa` on mixture components (e.g. `m.components[1] isa TaggedBetaMeasure` in some test sites) **still hold** because the reconstructed wrapper is the expected `TaggedBetaMeasure`.
- Every `bad2_*` corpus file → still triggers its expected slug.
- Zero new Failing assertions.

**Specific concern — `.space` access on elements.** Tests that do `m.factors[i].space` expect Measure behaviour. The reconstructed `BetaMeasure(p.beta, p.space)` wrapper has `.space`; the underlying `BetaPrevision` does not. Shield reconstruction preserves `.space` access.

**Specific concern — identity tests.** Any test that does `m.components[1] === m2.components[1]` breaks under shield reconstruction (fresh wrapper each read). Grep: `grep -rE "\.(components|factors|beta)\[.*\] ===" test/` returns 0 hits. No consumer relies on identity.

Expected fixture diff: zero. Any regression halts Move 2 before the code PR merges.

## 4. Worked end-to-end example

One representative call through the modified surface.

**Before (current master, post-Move-1):**

```julia
m = MixtureMeasure(Measure[BetaMeasure(2.0, 3.0), BetaMeasure(5.0, 1.0)], [0.0, 0.0])
components_vec = m.components   # Vector{Measure}, by reference to internal Vector{Measure}
first_alpha = m.components[1].alpha   # 2.0
push!(m.components, BetaMeasure(1.0, 1.0))   # works: push! into internal Vector{Measure}
```

Dispatch:
- `MixtureMeasure(...)` → `src/ontology.jl:403-405` → constructor wraps `Vector{Measure}(components)` into `MixturePrevision`.
- `m.components` → shield at `src/ontology.jl:411-417` → returns `getfield(m, :prevision).components` by reference (stored `Vector{Measure}`).
- `m.components[1].alpha` → `Vector{Measure}` → `BetaMeasure` → shield → `BetaPrevision.alpha` → 2.0.
- `push!(m.components, ...)` → shield returns the same stored vector → push! writes there; subsequent `m.components` sees the new element.

**After (Move 2 tip):**

```julia
m = MixtureMeasure(Measure[BetaMeasure(2.0, 3.0), BetaMeasure(5.0, 1.0)], [0.0, 0.0])
components_vec = m.components   # fresh Vector{Measure}, reconstructed from Vector{Prevision}
first_alpha = m.components[1].alpha   # 2.0 (unchanged)
push!(m.components, BetaMeasure(1.0, 1.0))   # SILENT BREAKAGE — push! into fresh vector, not stored
```

Dispatch:
- `MixtureMeasure(...)` → constructor extracts `.prevision` from each component, stores as `Vector{Prevision}`. Internal type tightened.
- `m.components` → shield reconstructs `[wrap_in_measure(p) for p in prevision.components]` → fresh `Vector{Measure}` allocated each call.
- `m.components[1].alpha` → reconstructed BetaMeasure → shield → BetaPrevision.alpha → 2.0 (same value).
- `push!(m.components, ...)` → into fresh vector, lost when reference goes out of scope. **Silently broken for `apps/skin/server.jl:611-614`.**

The Move 7 migration target for this pattern:

```julia
# Move 7 replaces apps/skin/server.jl:611-614 from:
push!(state.belief.components, TaggedBetaMeasure(Interval(0.0, 1.0), tag, BetaMeasure()))
push!(state.belief.log_weights, lw)
# To:
push_component!(state.belief.prevision, TaggedBetaPrevision(tag, BetaPrevision(1.0, 1.0)), lw)
```

`push_component!` lands in Move 2 as surface-ready API; skin migrates to it in Move 7.

## 5. Open design questions

### 5.1 Shield reconstruction: from silent breakage to loud breakage

Prompt 3 framed this as "per-access allocation" — that's only half the concern. The load-bearing issue is the **shared-reference contract breakage** for `push!`-through-shield patterns, and the mitigation is a **`FrozenVectorView` executable guard**, not a §6 prose flag.

**The actual breakage.** Before Move 2, `m.components` returns the internal `Vector{Measure}` by reference. `push!(m.components, ...)` at `apps/skin/server.jl:611-614` writes to that vector; subsequent reads see the new element. After Move 2, the internal vector is `Vector{Prevision}`, and the shield reconstructs a fresh `Vector{Measure}` on every read via `wrap_in_measure(p)`. Without a guard, `push!`-through-shield writes to an ephemeral vector that nothing subsequently reads — **silent state corruption**.

**The bound — honestly named.** Per `docs/posture-4/decision-log.md` §Decision 3, the credence repo has zero external users. The bound on the silent-corruption risk isn't "non-CI-tested, non-test-exercised" (that's true but thin); the load-bearing bound is **this repo has one developer, and that developer has now been notified of the specific hazard via this design doc and its PR body.** The risk isn't "no one runs skin during Moves 2–6"; it's "no one runs skin between Moves 2 and 7 without knowing about this specific breakage." Mitigation works through a named, documented hazard that the one-developer cohort sees at merge time.

Even that mitigation is too weak on its own. A design-doc §6 prose flag is skippable over a five-move window; human memory of "I saw a flag about push! somewhere" doesn't reliably survive 2–3 months of reviewer-driven cadence. The mitigation must be executable.

**The mechanism — `FrozenVectorView{T}`.** The Measure-level shields wrap their reconstructed `Vector{Measure}` / `Vector{Float64}` returns in `FrozenVectorView`:

```julia
struct FrozenVectorView{T}
    inner::Vector{T}
end

Base.getindex(v::FrozenVectorView, i::Int) = v.inner[i]
Base.length(v::FrozenVectorView) = length(v.inner)
# ... iterate, eachindex, firstindex, lastindex, size, axes ...

function Base.push!(v::FrozenVectorView, ::Any...)
    error("push! not supported on shield-reconstructed FrozenVectorView; " *
          "use push_component!(::MixturePrevision, ...) at the Prevision level. " *
          "See docs/posture-4/move-2-design.md §5.1.")
end
# ... same for setindex!, append!, pop!, resize!, empty! ...
```

Post-Move-2, `push!(state.belief.components, ...)` at skin's line 611 fails **loudly** on the first invocation with a pointer to the fix. Not silently. This is the load-bearing mitigation; §6's risk statement is commentary.

**Three options (restated under the guard mechanism):**

- **Option A (my prior):** Full master-plan Move 2 scope, with `FrozenVectorView` converting the silent-corruption hazard to a loud-runtime-error hazard. Skin's push!-through-shield pattern fails loudly at first invocation on a Move-2+ tip; `push_component!` lands as the migration target; Move 7 migrates skin. The executable guard does the discipline work; the author-notification bound provides the residual backstop.

- **Option B:** Full Move 2 + migrate `apps/skin/server.jl:611-614` now. Scope-expansion beyond `src/` boundary. Accelerates Move 7's skin work by five moves but does it the wrong way — the skin rewrite in Move 7 is a coordinated redesign with the JSON-RPC wire format, not a piecewise mutation-API swap.

- **Option C:** Partial Move 2 — tighten `TaggedBetaPrevision.beta::BetaPrevision` only; defer `Vector{Prevision}` element-type on `MixturePrevision` / `ProductPrevision` to Move 5 / 7 concurrent with shield retirement. Avoids the reconstruction / `FrozenVectorView` question entirely. Functional, but bundles two logically separable concerns (element-type-tightening + shield-retirement) into Move 5/7, **degrading the provenance clarity that the partitioned master plan exists to produce.**

**Why A over C.** Option C's functional fallback is real — if `FrozenVectorView` ergonomics turn out to not compose cleanly with some downstream accessor pattern, reverting to C with a one-paragraph master-plan amendment is defensible. But A preserves the principled partition: Move 2 does element-type, Move 5 retires shields, Move 7 migrates skin. One concern per move is the virtue the master plan claims, and paying for it with a flagged-and-guarded breakage in a non-CI path is the right trade. Option C is the fallback if `FrozenVectorView` doesn't compose; it is not the default.

**Why A over B.** B brings skin work forward by accident of convenience, not by design. Move 7 plans skin's rewrite as a coordinated redesign (wire format, push/replace APIs, mutation semantics). A piecemeal migration in Move 2 couples `src/` to `apps/skin/` prematurely and risks a second migration at Move 7 when the wire format redesigns.

### 5.2 `ExchangeablePrevision.decompose` return type tightening

Prompt 3 raises this as Q2. Currently `decompose(p::ExchangeablePrevision)` returns a `MixturePrevision` whose `components` field is a `Vector` (untyped). Post-Move-2, with `MixturePrevision.components::Vector{Prevision}`, the return type's `components` field is typed.

**Question:** should `decompose`'s return type be parametrically tightened to `MixturePrevision{T}` where `T <: Prevision` names the concrete component type (e.g., `MixturePrevision{BetaPrevision}` for Beta-component mixtures), or does Julia's existing `MixturePrevision` shape suffice?

**My prior: existing shape suffices.** Adding a type parameter to `MixturePrevision` is a breaking change that ripples into every `MixturePrevision` call site — not just `decompose`'s return type but every construction, every dispatch, every existing specialisation like `condition(p::MixturePrevision, e::TagSet)`. The parameter would be set to `Prevision` (the abstract supertype) in most call sites anyway, reducing to what we already have. The work is substantial and the gain is cosmetic — `decompose`'s return shape doesn't need the extra compile-time guarantee because its consumers (future `expect`/`condition` methods on decomposed mixtures) work uniformly on the `Prevision` abstract interface.

A future move that introduces a specialisation demanding `MixturePrevision{BetaPrevision}` specifically would justify the parameterisation then. Until then, the existing shape is sufficient.

### 5.3 Internal `src/` readers: `mean(components[i])` migration timing

Prompt 3 raises this as Q3. Currently `src/ontology.jl:1097` does `new_logw = copy(m.logw)` and various mixture-condition paths read `m.components[i].alpha` via shield. Post-Move-2, `m.components[i]` is a reconstructed `BetaMeasure`; `.alpha` goes through the BetaMeasure shield to `BetaPrevision.alpha` — works but bypasses the `expect-through-accessor` discipline we're preparing for Move 5.

**Question:** migrate the internal `src/` reads to `mean(components[i])` / `expect(components[i], Identity())` now (in Move 2), or leave them as-is and fold into Move 5's stdlib introduction?

**My prior: leave as-is.** Move 5 introduces the `expect-through-accessor` lint slug specifically to catch these call sites and drive them to the stdlib. Migrating ahead of the lint means doing the migration without the machinery that verifies completeness. Move 5's stdlib lands concurrent with the lint; the migration is mechanical and can be done once, thoroughly, at that point.

Migrating now would be a preemptive cleanup that (1) doesn't have the lint as a safety net, (2) adds diff noise to Move 2 that distracts from the typed-fields change, (3) would need to be re-audited at Move 5 anyway when the stdlib lands. Better to keep Move 2 focused on its declared scope and let Move 5's lint catch these sites en bloc.

## 6. Risk + mitigation

**Risk (formerly silent, now loud):** `apps/skin/server.jl:611-614` `push!`-through-shield pattern.

**Mitigation (primary, executable):** `FrozenVectorView` wrapping in Phase 4 — the shield's reconstructed return throws on `push!` / `setindex!` / `append!` / `pop!` / `resize!` / `empty!` with a message pointing at `push_component!` / `replace_component!` as the migration target. Skin's line 611 fails loudly at first invocation post-Move-2 rather than silently corrupting state. See §5.1.

**Mitigation (secondary, notification):** the one-developer cohort has been notified via this design doc and its PR body. Not load-bearing on its own (see §5.1 on why prose flags alone are insufficient), but operates as a backstop — the developer reading the stack trace from `FrozenVectorView.push!` has context about why the guard exists.

**Mitigation (migration target):** `push_component!` / `replace_component!` land as surface-ready API in Phase 1 so Move 7's skin rewrite has a disciplined target rather than inventing one mid-migration.

**Risk (medium):** `wrap_in_measure(p::Prevision)` helper misses a concrete Prevision subtype, producing a MethodError at runtime from the shield reconstruction.

**Mitigation:** exhaustive dispatch in Phase 2 — one method per concrete Prevision subtype, landing before the type-tightening in Phase 3. Any subtype missed surfaces as a MethodError in Phase 3's test suite, not as silent corruption. The nine subtypes to cover are enumerated in §2's Modifies list.

**Risk (low):** the `.space` access on reconstructed Measure wrappers requires the Measure to carry `.space` — reconstructing `BetaMeasure(p.beta, space)` needs `space` to come from somewhere. The current Measure subtypes store `space` separately from their contained Prevision.

**Mitigation:** the Prevision holds the information to reconstruct the space where possible. `BetaPrevision` → canonical `Interval(0.0, 1.0)`. `GaussianPrevision` → canonical `Euclidean(1)`. `GammaPrevision` → canonical `PositiveReals()`. For `CategoricalPrevision` / `DirichletPrevision`, the space is atom-dependent and isn't stored on the Prevision — but these types have a different wrapping Measure (`CategoricalMeasure{T}.space::Finite{T}`) and the reconstruction needs the `Finite` in scope. The `wrap_in_measure` helper for these specific cases takes an optional `space` argument, or dispatches through the wrapping Measure context. Resolve at helper-authoring time in Phase 2.

**Risk (low):** Allocation cost of per-read reconstruction in hot paths.

**Mitigation:** profile if Move-0 verify shows any regression beyond Stratum-3 tolerance. The 6118-site capture passes at bit-exact Stratum-1 for most reads (values pass through); allocation cost is a performance concern not a correctness concern. Measure the diff run-time of `--verify` pre- and post-Phase-3; accept if within 2x of baseline (the allocation overhead on mixture-access-heavy tests is an expected transient cost).

**Risk (review-process):** §5.1 Option A commits to silent breakage in a specific code path; if the reviewer objects, the fallback is Option C (partial Move 2, defer element-type tightening to Move 5/7).

**Mitigation:** Option C requires amending the master plan to narrow Move 2's scope. The amendment is a one-paragraph PR before Move 2's code PR lands. Cheap to pivot if §5.1 review rejects Option A.

## 7. Verification cadence

End-of-PR verification (after each phase):

```bash
# All 13 test files pass after each phase
julia test/test_core.jl
julia test/test_prevision_unit.jl
julia test/test_prevision_conjugate.jl
julia test/test_prevision_mixture.jl
julia test/test_prevision_particle.jl
julia test/test_host.jl
julia test/test_flat_mixture.jl
julia test/test_events.jl
julia test/test_persistence.jl
julia test/test_grid_world.jl
julia test/test_email_agent.jl
julia test/test_rss.jl
julia test/test_program_space.jl

# Move 0 invariance check — the authoritative gate
julia --project=scripts scripts/capture-invariance.jl --verify
# Expected: ✓ Verified: manifests identical (modulo timestamp)
```

CI runs `credence_router` + `credence_agents` pytests. These don't exercise the `apps/skin/` push!-through-shield path, so silent breakage there won't surface in CI. That's both the bound and the limit of the risk per §5.1 Option A.

Full `scripts/capture-invariance.jl --verify` takes ~5 minutes locally.

## 8. de Finettian discipline self-audit

1. **Is every numerical query in this move routed through `expect`?** N/A — Move 2 doesn't introduce new numerical queries. Existing queries that flow through the typed-container changes (e.g., `mean(m.components[i])`) produce the same values via the same dispatch paths.

2. **Does this move hold a Prevision inside a Measure, or a Measure inside a Prevision, for any reason?** The **opposite** — Move 2 *retires* the Measure-inside-Prevision pattern that Posture 3 Move 3 accepted as pragmatic impurity (documented at `src/prevision.jl:349-351` as "allocates nothing on access; a future cleanup can replace with Previsions + view reconstruction if perf justifies"). Post-Move-2, the Prevision fields hold Previsions; Measures wrap them at the Measure level via reconstruction. The Prevision-inside-Measure pattern (`CategoricalMeasure.prevision::Prevision`) persists per `master-plan.md` §Move 5 — dated-deprecation note: retires in Move 5.

3. **Does this move introduce an opaque closure where a declared structure would fit?** No. All new surfaces (`wrap_in_measure`, `push_component!`, `replace_component!`) take declared typed arguments. No closures.

4. **Does this move add a `getproperty` override on any Prevision subtype?** No. The three `getproperty` overrides touched are all Measure-level (`TaggedBetaMeasure`, `ProductMeasure`, `MixtureMeasure`), scheduled for retirement in Move 5.

---

## Reviewer checklist

- [ ] §0 Final-state alignment is a paragraph and explicitly names the silent-breakage transient state (§5.1).
- [ ] §5 contains three non-trivial open questions with stated priors; §5.1 in particular is argued honestly (not by appeal to master-plan authority) with both sides of the silent-breakage tradeoff laid out.
- [ ] §8 self-audit: (1) N/A on new queries; (2) retires Measure-inside-Prevision — this is the opposite of the self-audit's failure mode; (3) no new closures; (4) no new Prevision-level `getproperty`.
- [ ] File-path:line citations current (surveyed at master SHA `534e8ed` post-PR-#47).
- [ ] §5.1 Option A's silent-breakage risk is acceptable per reviewer judgement, OR Option C fallback is chosen and Move 2's scope narrows accordingly.
