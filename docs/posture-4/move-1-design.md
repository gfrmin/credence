# Move 1 — `src/` internal cleanup: field-name unification, parametric `ConditionalPrevision`, shield retirement inside `src/`

## 0. Final-state alignment

Move 1 converges the current tip one step toward `master-plan.md` §"Final-state architecture" by retiring three of the "internal representation invariants" items the master plan declares: (a) **unified `log_weights` field name** across the five carriers that hold normalised log-mass vectors, (b) **typed `ConditionalPrevision{E <: Event}`** replacing the opaque-closure `event_predicate::Function`, (c) the **`CategoricalMeasure.getproperty` shield's stored-subtype branch** retiring inside `src/`. All three are under the `src/` boundary and leave every caller outside `src/` (apps, tests, skin, Python, BDSL stdlib) operating against the same externally-visible surface — the `CategoricalMeasure.logw` name remains readable post-Move-1 via a simplified shield that forwards unconditionally to the underlying Prevision's `log_weights`. Measure subtypes remain alive (they retire in Move 5); the Posture 4 tip's final shape is reached incrementally, not in one step.

## 1. Purpose

Unify the log-weight field name so that every carrier holds its data under `log_weights`, converting the existing shield's carrier-specific dispatch branch into a no-op forward. Retire the opaque-closure `event_predicate` representation of `ConditionalPrevision` in favour of a declared `event::E` field parameterised on the Event hierarchy — closing the last Invariant-2 discipline hole in the Prevision module. Leave `_dispatch_path` in place with a one-paragraph docstring update reframing it under Posture 4 conventions.

## 2. Files touched

Creates: none.

Modifies:
- `src/prevision.jl`
  - `CategoricalPrevision` struct definition (currently `src/prevision.jl:279-298`): rename `logw` → `log_weights`. Update constructor to match. Update the 3-line normalisation arithmetic to reference `log_weights`.
  - `ConditionalPrevision` struct (currently `src/prevision.jl:543-552`): replace `event_predicate::Function` with `event::E` where `E <: Event`. Parameterise `ConditionalPrevision{E <: Event}`. Update the constructor's inner `new(...)` call to match. Docstring (`src/prevision.jl:517-542`) rewritten to describe the typed-Event representation; retire the "stored as a predicate rather than an `Event` so prevision.jl avoids circular dependency with ontology.jl's Event hierarchy" clause (see §5.1 for the resolution).
  - `_dispatch_path` docstring (`src/prevision.jl:667-678`): no code change; update the comment to note that the hook stays as-is for Posture 4 (per §5.3 resolution).
  - Export list update: `ConditionalPrevision` export stays; no new symbol exported (the `{E}` parameter is implicit).
- `src/ontology.jl`
  - `CategoricalMeasure.getproperty` shield (`src/ontology.jl:135-148`): retire the `if p isa CategoricalPrevision ... else ... end` branch. The shield's `:logw` path becomes a one-line forward: `getfield(m, :prevision).log_weights`. The shield continues to exist (external callers still read `m.logw`) but its dispatch logic collapses.
  - `CategoricalMeasure` constructor and comments (`src/ontology.jl:85-120`): update comments that reference "p.logw" / "CategoricalPrevision has `logw`" to the unified `log_weights` naming. The actual constructor takes `logw::Vector{Float64}` as the parameter name (public constructor signature, not a struct field) — rename to `log_weights` here too for consistency.
  - `weights(m::CategoricalMeasure)` (`src/ontology.jl:152-160`): body reads `m.logw` via shield; shield's return value is unchanged, so code works unmodified. Reference comment (currently "shield resolves .logw → qp.log_weights") updated to reflect the collapsed branch.
  - Other internal `m.logw` reads via shield: `src/ontology.jl:1097` (`new_logw = copy(m.logw)`) and `src/ontology.jl:1300` (comment). The first is a runtime read through the shield, unchanged. The comment gets updated for consistency.
  - `condition(p::MixturePrevision, e::TagSet)` and related event-form specialisations (`src/ontology.jl:1651-1665`): **no change** — these continue to return concrete `MixturePrevision` shapes, not `ConditionalPrevision`. The typed-Event `ConditionalPrevision{E}` is used only by the generic fallback path, which is not exercised by any current call site (no `ConditionalPrevision` constructor is called anywhere — verified via grep). Move 1's typed rewrite is **prospective type-system work for the Posture 4 freeze**, not a rewiring of existing behaviour.

Deletes: none in this move. The Measure subtypes and their shields retire en bloc in Move 5.

Renames: none as file-level renames; the field-name rename `logw` → `log_weights` is described above.

**Commit phasing (for the code PR that follows this design doc).** Per the repo convention that renames land before semantic changes (see Move 6 Phase 1 precedent), the code PR lands as two commits:

1. **Phase 1 — rename.** `CategoricalPrevision.logw` → `log_weights`: struct field rename + the one internal normalisation arithmetic reference + comment/docstring updates + the corresponding `src/ontology.jl` comment updates. The `getproperty` shield branch still exists at this point (it's still dispatching), but its `p.logw` read becomes `p.log_weights`. Full test suite green.
2. **Phase 2 — shield collapse + `ConditionalPrevision{E}`.** Retire the shield's carrier-dispatch branch (collapse to one-line forward). Rewrite `ConditionalPrevision` to `ConditionalPrevision{E <: Event}` with `event::E`. `_dispatch_path` docstring touch-up. Full test suite green.

The phasing is internal to the code PR and does not require a separate review cycle; the design-doc review covers both phases.

## 3. Behaviour preserved

The Move 0 fixture at `test/fixtures/posture-3-capture/` (6118 unique site×value tuples, pinned at master branch-point `5c6a94e`) is the invariance target. Move 1 asserts:

- Every captured **Stratum-1 / Stratum-2 / Stratum-3** assertion at the Posture 4-Move-1 tip produces the same value to within its declared tolerance.
- Every captured **Directional** assertion's inequality is preserved (values may drift within the inequality's slack; drift is forensic, not oracular, per §3 of the design doc).
- Every captured **Structural** assertion still evaluates to `true`, and every membership-subtype assertion's `S_new` matches `S_old` under canonical ordering.
- Every captured **`bad2_*` corpus file** still triggers its expected slug under `credence_lint.py` pass-two. (The `bad2_*` corpus is untouched by Move 1 — the lint's slug logic depends on assertion patterns in code, not on Prevision field names — but the invariance is asserted for discipline-completeness.)
- Zero new **Failing** assertions (no previously-passing assertion regresses into the failing-shape bucket).

All three modifications are type-level renames / representation shifts that preserve runtime semantics. No new conjugate pair, no new dispatch path, no new kernel. `expect`, `condition`, `push`, `density` retain identical behaviour on identical inputs.

**Specific expectations per modification:**

- **`logw` → `log_weights` rename.** `CategoricalMeasure.logw` reads via the shield return the same `Vector{Float64}` by reference (shared-reference contract preserved — the shield's collapsed-branch forward still returns `getfield(m, :prevision).log_weights` directly, not a copy). Posterior `weights(m)` computations that read `m.logw` produce bit-exact output.
- **Shield branch collapse.** External callers reading `m.logw` see the same value. The shield's computational path shortens from "dispatch on p's type, read the right field" to "read `p.log_weights` directly." Stratum-1 equality holds.
- **`ConditionalPrevision{E <: Event}` rewrite.** No current call site constructs a `ConditionalPrevision` (verified: `grep -r 'ConditionalPrevision(' src/ test/ apps/` returns only the struct definition and constructor). The rewrite is structural type-system work; runtime behaviour is unchanged because no runtime currently exercises this type. The `bad2_*` corpus inventory records the type's existence (via `credence_lint.py`'s structural checks on `src/prevision.jl`) and is unaffected by the parameterisation.

**Expected fixture diff:** zero. If any captured assertion diverges at its declared tolerance, Move 1 halts and the cause is diagnosed before the code PR merges.

## 4. Worked end-to-end example

One representative call through the modified surface, naming which module owns each step.

**Before (current master):**

```julia
using Credence
m = CategoricalMeasure(Finite([:a, :b, :c]))  # constructor; logw = [0.0, 0.0, 0.0] uniform
weights_vec = weights(m)  # = [1/3, 1/3, 1/3]
logw_vec = m.logw          # = [−log 3, −log 3, −log 3] by reference through shield
```

Dispatch trace:
- `CategoricalMeasure(Finite([:a,:b,:c]))` → `src/ontology.jl:122` default-weights constructor → calls `CategoricalMeasure{Symbol}(space, fill(0.0, 3))` → `src/ontology.jl:103` inner constructor → wraps `CategoricalPrevision([0.0, 0.0, 0.0])` → `src/prevision.jl:282` CategoricalPrevision constructor normalises via `log_total = max_lw + log(sum(exp.(logw .- max_lw)))` and stores `logw .- log_total` into the `logw` field.
- `weights(m)` → `src/ontology.jl:152` → reads `max_lw = maximum(m.logw)` → shield at `src/ontology.jl:135` fires, branches on `p isa CategoricalPrevision` (true) → returns `p.logw` by reference → `max_lw` computed.
- `m.logw` at call site → same shield path, returns `p.logw` by reference.

**After (Move 1 tip):**

```julia
using Credence
m = CategoricalMeasure(Finite([:a, :b, :c]))  # constructor; log_weights = [0.0, 0.0, 0.0] uniform
weights_vec = weights(m)  # = [1/3, 1/3, 1/3]
logw_vec = m.logw          # still readable via shield; returns p.log_weights by reference
```

Dispatch trace (the parts that change are italicised):

- `CategoricalMeasure(...)` → unchanged through to the `CategoricalPrevision` constructor, which now *stores `log_weights .- log_total` into the `log_weights` field* (not `logw`).
- `weights(m)` → reads `m.logw` via shield → shield at `src/ontology.jl:135` fires, *branch-dispatch retired*, returns `getfield(m, :prevision).log_weights` directly — for every Prevision subtype the `CategoricalMeasure` may wrap (CategoricalPrevision, ParticlePrevision, QuadraturePrevision, EnumerationPrevision — all five carriers share the unified name, so one forward suffices).
- `m.logw` at call site → same shield, same reference return. External callers reading `m.logw` observe zero change.

**Type-check trace for the `ConditionalPrevision{E}` rewrite:**

Pre-Move-1 attempted construction (hypothetical — no current call site exists):
```julia
# Would have been allowed by the untyped `event_predicate::Function`:
cp = ConditionalPrevision(base_prevision, s -> s.tag == 1, 0.5)  # opaque closure
```

Post-Move-1 the equivalent requires a declared Event:
```julia
e = TagSet(Set([1]))
cp = ConditionalPrevision(base_prevision, e, 0.5)  # typed: ConditionalPrevision{TagSet}
```

And `expect(cp, f)` becomes `expect(base_prevision, LinearCombination([(1.0, f), ...], Indicator(e))) / mass` — all declared TestFunctions, no opaque closures. Move 1 does not land this consumer path (no caller today); it lands the *type* so when a caller is added in Move 2+ the discipline is already in place.

## 5. Open design questions

### 5.1 Where does `ConditionalPrevision{E <: Event}` live?

The struct currently lives in `src/prevision.jl` with the comment "Stored as a predicate rather than an `Event` so prevision.jl avoids circular dependency with ontology.jl's Event hierarchy." (`src/prevision.jl:527-529`). Retiring the opaque closure in favour of `event::E` means `E <: Event` must be in scope at the struct-definition site.

**Two options:**

- **Option A (preserve current module boundary, relax the `E` constraint at the struct level):** Define `ConditionalPrevision{E}` with `E` *unconstrained* at the struct level — the `E <: Event` bound is carried by a constructor method that lives wherever `Event` is in scope. This matches the current discipline of `Indicator{E}` at `src/prevision.jl:184` (which is parameterised by `E` without declaring the `<: Event` bound at the struct level; the bound is enforced at call sites via the constructor's signature).
- **Option B (module restructure — move `ConditionalPrevision` to an Event-aware module):** Either split `prevision.jl` into two files with one depending on `Event` (introducing a small conjugate.jl-style or events.jl-style module), or move `ConditionalPrevision` to `src/ontology.jl` where `Event` is already defined. This enforces the constraint at the struct level (`struct ConditionalPrevision{E <: Event} <: Prevision`) but introduces module-boundary churn.

**My prior: Option A.**

The discipline Option B claims to buy is illusory in this repo. `Indicator{E}` has run through Posture 3 Moves 7–8 with the same unconstrained-`E`-at-struct-level / enforced-at-constructor pattern without a single issue — because the constructor is the only entry point anyone ever uses. `ConditionalPrevision{String}(...)` type-checking at the struct level is a theoretical hole that no working Julia code ever steps into. Option B would purchase the closure of a hole that has never been exploited, at the price of a cross-cutting module restructure that Move 5 has legitimate reasons to own. Don't front-load Move 5 for type-system purity that `Indicator{E}` has already established the repo doesn't need.

The master plan's §"New module structure" splits `ontology.jl` into six files at Move 5, including a dedicated `events.jl` that both `previsions.jl` and `kernels.jl` can depend on. If the struct-level `<: Event` bound is ever wanted, Move 5's split is when it lands — naturally, not as a concession purchased mid-refactor. Until then, the consistency with `Indicator{E}` is the load-bearing argument, not a consolation prize.

### 5.2 Does `CategoricalMeasure.logw` retire entirely, or does the shield remain as a one-line forward?

The shield currently has a dispatch branch on the stored Prevision's subtype (`src/ontology.jl:135-148`). Move 1 collapses the branch; the question is whether the shield's `:logw` handling retires entirely or remains as a one-line `getproperty(m, :logw) = m.prevision.log_weights`.

**Two options:**

- **Option A (retire entirely):** Delete the `:logw` branch from the shield. External readers that still say `m.logw` get a `MethodError` / property-not-found error. Per `master-plan.md` §"Move 5", `CategoricalMeasure` retires entirely in Move 5, so preserving `.logw` only delays the deletion by four moves.
- **Option B (retain as one-line forward):** Keep `if s === :logw return getfield(m, :prevision).log_weights end` as a single-line shield. External callers keep working. The shield collapses from "branch on stored subtype" (current) to "unconditional forward" (Move 1 tip) to "gone" (Move 5).

**My prior: Option B for Move 1; retirement in Move 5 with the rest of `CategoricalMeasure`.**

The shield's purpose (Posture 3 Decision 2, per `docs/posture-3/decision-log.md` §Decision 2) is to hold compatibility with existing apps/tests during the migration. Posture 4 retires that compatibility, but it retires it *in Move 5*, not in Move 1. Breaking the shield early — in the move that's scoped to "`src/` internal cleanup, no caller outside `src/` is touched" — contradicts the master-plan scope. `apps/skin/server.jl`, `apps/python/*`, and `test/*` still read `m.logw` in some paths (verified: `grep -r '\.logw' apps/skin/ apps/python/ test/` returns hits); breaking them in Move 1 requires a simultaneous fix elsewhere, which is what Move 1's scope discipline rules out.

A counter-argument would be: if Move 5 is going to delete the shield anyway, why not let Move 1's breakage be the test that we've found every caller? The answer is that the Move 0 capture gives us a more disciplined surface for this question — Move 4 rewrites every test against Prevision constructors, Move 5 deletes Measure, and the Move 0 capture verifier catches any assertion that still references `.logw` through the rewrite. The assertive discovery happens at Move 4 when tests are systematically migrated, not at Move 1 where it would be an unhandled-exception surprise.

### 5.3 `_dispatch_path` observability hook: stay or migrate?

`src/prevision.jl:679` defines `_dispatch_path(p::Prevision, k) = maybe_conjugate(p, k) === nothing ? :particle : :conjugate` as a test-only observability hook, underscore-prefixed per the repo conventions. Posture 4's cleaner type system might motivate replacing this with a typed observability pattern (e.g., a `DispatchPath` enum returned via multiple dispatch on `(p, k)` pairs).

**My prior: stay as-is.** Two reasons.

First, the hook works. The return type is already well-defined (`Symbol`, with two known values `:particle` and `:conjugate`). The call sites (Stratum-2 test assertions that pin dispatch-path decisions, per the current docstring) consume the return value as-is. Replacing the hook with an enum adds structural type-work without adding discriminatory power — both `Symbol` and `@enum DispatchPath` discriminate equally for the test's purposes.

Second, the underscore-prefix convention (repo CLAUDE.md's scope-boundary rule) explicitly sanctions test-only surface as not-production-API. Posture 4's discipline tightening is for *production* surfaces: the prevision-as-primitive story, the event-primary condition form, the Invariant-2 typed structures. Tightening a test-observability hook that already respects the underscore convention is discipline-creep — it would pull `_dispatch_path` out of test scope into production scope, and then we'd need to justify its existence on production-API grounds rather than test-observability grounds.

The one real question `_dispatch_path` raises is its naming once the Prevision hierarchy becomes primary (post-Move 5). `_dispatch_path` was named when Measure was the primary surface and "dispatch" referred to condition's Measure-method dispatch table. With Prevision primary, the same name still reads correctly (dispatch happens at the Prevision-subtype-pair level too); no rename warranted. Docstring gets a one-sentence update for Posture 4 framing; code unchanged.

## 6. Risk + mitigation

**Risk (low):** `logw → log_weights` rename missed an internal read.

**Mitigation:** The grep at design-doc time shows 16 hits for `\.logw` in `src/` — all inside the shield / around the shield's documentation / inside `weights(m::CategoricalMeasure)`. The rename touches `src/prevision.jl`'s single `.logw` internal read (in the CategoricalPrevision constructor) plus the three `m.logw` reads in `src/ontology.jl` (which go through the shield unchanged). The shield retains `m.logw` as a public surface for external callers. Full test suite must pass post-rename; Move 0 fixture diff must be zero.

**Risk (low):** `ConditionalPrevision{E}` parameter-shape change breaks a hidden dependency.

**Mitigation:** Zero current call sites exist (`grep -r 'ConditionalPrevision(' src/ test/ apps/` returns only the definition and constructor). The rewrite is prospective type-system work with no runtime consumer today; risk is limited to future consumers that won't land until Move 2+.

**Risk (medium):** The shield-branch collapse introduces a silent behaviour change for a Prevision subtype other than `CategoricalPrevision`.

**Mitigation:** The pre-Move-1 shield at `src/ontology.jl:135-148` has two branches — one reading `p.logw` (for `CategoricalPrevision`), one reading `p.log_weights` (for `ParticlePrevision` / `QuadraturePrevision` / `EnumerationPrevision`). The post-Move-1 shield reads `p.log_weights` unconditionally. For `CategoricalPrevision` specifically, this is only correct *after* its field is renamed from `logw` to `log_weights` (Phase 1 of the code PR). The phase ordering in §2 — rename first, shield collapse second — ensures that when the shield collapses, `p.log_weights` exists on every Prevision subtype `CategoricalMeasure` may wrap. Full test suite after each phase asserts this.

**Risk (review-process):** the three open questions in §5 are answered definitively; a reviewer might want more back-and-forth.

**Mitigation:** Per Posture 3's cadence, the review cycle is where I get pushback. Each §5 question states its prior and the reasoning; reviewers who disagree can argue against the specific reason. If §5.1 (module location) is contested, the practical consequence is one of {keep current file; move to ontology.jl; split into sub-module}; each is bounded. If §5.2 (retire vs. forward) is contested, the consequence is {forward now; retire now}; the retire-now path would require a simultaneous breaking-change discovery across apps/tests that exceeds Move 1's scope. §5.3 (observability hook) is the least-contested; it's here because the master plan's §"Move 1" listed it, not because it's genuinely undecided.

## 7. Verification cadence

End-of-PR verification for the code PR that follows this design doc:

```bash
# (Phase 1 — rename) verify after first commit
julia test/test_core.jl
julia test/test_flat_mixture.jl
julia test/test_host.jl
julia test/test_prevision_unit.jl
julia test/test_prevision_conjugate.jl
julia test/test_prevision_mixture.jl
julia test/test_prevision_particle.jl
julia test/test_program_space.jl
julia test/test_grid_world.jl
julia test/test_email_agent.jl
julia test/test_rss.jl
julia test/test_events.jl
julia test/test_persistence.jl

# (Phase 2 — shield collapse + ConditionalPrevision{E}) verify after second commit
julia test/...  (same suite)

# Move 0 fixture diff verification — the authoritative invariance check
julia --project=scripts scripts/capture-invariance.jl --verify
# Expected: ✓ Verified: manifests identical (modulo timestamp)
```

CI (the `publish-image.yml` pipeline) runs Python+Julia integration tests automatically on the PR. No `scripts/capture-invariance.jl` invocation lives in CI today; the local verify-run is the gate.

Expect the CI run to take ~7 minutes (matching prior Posture 4 PRs).

## 8. de Finettian discipline self-audit

1. **Is every numerical query in this move routed through `expect`?** N/A — Move 1 introduces no new numerical queries. The `weights(m)` and `m.logw` reads that change in representation are existing consumer paths; the rename / shield-collapse preserves their return values bit-exactly (Stratum 1 per the Move 0 capture). No new query, no new pathway.

2. **Does this move hold a Prevision inside a Measure, or a Measure inside a Prevision, for any reason?** `CategoricalMeasure.prevision::Prevision` continues to hold a Prevision inside a Measure — inherited from Posture 3, scheduled for retirement in Move 5. The shield at `src/ontology.jl:135-148` is *the* visible artefact of this residency, and Move 1 simplifies (does not retire) it. A dated-deprecation note pointing at the move that removes it: **Move 5 retires `CategoricalMeasure` and every `getproperty` shield it carries, per `master-plan.md` §"Move 5" and §"Types deleted".**

3. **Does this move introduce an opaque closure where a declared structure would fit?** The opposite — Move 1 *retires* an opaque closure. The pre-Move-1 `ConditionalPrevision.event_predicate::Function` is an opaque closure representation; the post-Move-1 `ConditionalPrevision{E <: Event}.event::E` carries the event as a declared Event subtype. This is Invariant-2 enforcement at the struct level.

4. **Does this move add a `getproperty` override on any Prevision subtype?** No. The only `getproperty` override touched is `CategoricalMeasure.getproperty` — a Measure-level override (scheduled for retirement in Move 5). No Prevision-level override is added.

---

## Reviewer checklist

- [ ] §0 Final-state alignment is a paragraph (not a sentence) and names any transient state explicitly (`CategoricalMeasure` + its shield survive Move 1 and retire in Move 5).
- [ ] §5 contains three non-trivial open questions, each with a stated prior and reasoning.
- [ ] §8 returns "yes" / acceptable-with-justification on all four: (1) N/A — no new numerical queries; (2) Prevision-inside-Measure retained with Move-5 deprecation note; (3) closure *retired*, not added; (4) no new Prevision-level `getproperty` override.
- [ ] File-path:line citations are current (surveyed at master SHA `5c6a94e` at design-doc drafting time). Reviewer should verify the citations against the SHA the code PR opens from.
- [ ] Move 1 as described does not require Move 2 to retract or rework it. The `logw → log_weights` rename is complete at Move 1; Move 2+ builds on the unified name. The `ConditionalPrevision{E}` rewrite is prospective (no current caller) and stands as Move-1-complete with no retraction expected.
