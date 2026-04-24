# Move 4 — Test suite migrated to Prevision constructors

## 0. Final-state alignment

Move 4 converges the current tip toward `master-plan.md` §"Types deleted" (Measure and every concrete subtype go in Move 5) by **removing every `*Measure(...)` constructor call site from `test/test_*.jl`** and replacing it with the corresponding Prevision-level construction. The Measure surface still exists through Move 4's tip (deletion is Move 5's scope) — tests that need to pass a Measure-shaped object to a Measure-dispatching function (e.g., pre-Move-5 `condition(m::MixtureMeasure, k, obs)`) can still obtain one by wrapping: `wrap_in_measure(p)` via the helper Move 2 Phase 3 landed. The transient state Move 4 leaves: tests construct via Prevision but some assertions still read `.alpha` / `.beta` / `.space` through the Measure shields on those wrappers. Move 5 retires both the shields and the need for them, via the Prevision-primary `condition` rewrite that makes all post-condition outputs concentrated Previsions over the ambient space.

## 1. Purpose

Systematically rewrite every Measure construction site in `test/` as Prevision construction. Scope is mechanical but volumetric — roughly 300 constructor call sites across 13 files. Fold the `test-oracle` pragma-policy decision (Q4) into this move so the 42 pragma'd test sites in `apps/skin/test_skin.py` and `test/test_prevision_*.jl` aren't re-litigated per-site during the rewrite. Test file renames happen where they improve clarity, not for mechanical tidiness alone.

## 2. Files touched

Modifies:
- `test/test_core.jl` (~76 Measure construction sites)
- `test/test_email_agent.jl` (~9 sites)
- `test/test_events.jl` (~12 sites)
- `test/test_flat_mixture.jl` (~37 sites)
- `test/test_grid_world.jl` (~8 sites)
- `test/test_host.jl` (~35 sites)
- `test/test_persistence.jl` (~7 sites — post-Move-3 already v3, mostly Measure constructors in round-trip test setup)
- `test/test_prevision_conjugate.jl` (~11 sites) — file is Prevision-first but has some Measure constructors in fixture setup
- `test/test_prevision_mixture.jl` (~13 sites)
- `test/test_prevision_particle.jl` (~6 sites)
- `test/test_prevision_unit.jl` (~66 sites) — heavily Measure-keyed in its current form
- `test/test_program_space.jl` (~11 sites)
- `test/test_rss.jl` (~4 sites)

No new test files. No deletions.

**Renames considered — §5.1 resolves which if any land.** Candidate renames:
- `test_flat_mixture.jl` → `test_mixture_prevision.jl` (content overlaps with `test_prevision_mixture.jl`; consolidation vs rename).

**Not modified by Move 4:** the `apps/skin/test_skin.py` pytest file (Python-side; Move 7's skin rewrite owns its test migration). The `apps/julia/*/host.jl` files (Move 6's apps migration). The Move-0 capture fixture (`test/fixtures/posture-3-capture/`) stays pinned and untouched.

**Commit phasing — four sub-PRs.** The diff is too large for one PR to be reviewable; splitting into four thematically-focused sub-PRs:

1. **Sub-PR 4a: Prevision-first test files** (`test_prevision_conjugate.jl`, `test_prevision_mixture.jl`, `test_prevision_particle.jl`, `test_prevision_unit.jl`). ~96 sites. Already closest to target surface; easiest to start with. Establishes the migration pattern for subsequent sub-PRs.
2. **Sub-PR 4b: Core DSL and event tests** (`test_core.jl`, `test_events.jl`, `test_flat_mixture.jl`, `test_host.jl`, `test_program_space.jl`). ~167 sites. The bulk; largest diff.
3. **Sub-PR 4c: Domain-app tests** (`test_email_agent.jl`, `test_grid_world.jl`, `test_rss.jl`). ~21 sites. Domain-specific; smallest of the three code sub-PRs.
4. **Sub-PR 4d: Persistence test + fixture reconstruction** (`test_persistence.jl` — 7 sites; fixture capture scripts adjusted if they hold stray Measure constructors).

Each sub-PR runs the full test suite locally + `--verify` before merging. The four merge under the same Move 4 design doc (this file); sub-PR bodies reference this design doc by name.

Test file renames (§5.1 resolution) land as a single rename-only commit within whichever sub-PR owns the renamed file. Per the repo's rename-before-refactor convention, renames go first within that sub-PR — not bundled with the content rewrite.

## 3. Behaviour preserved

Move 0 fixture at `test/fixtures/posture-3-capture/` (6118 site×value tuples at branch-point `5c6a94e`) is the invariance target.

**Expected divergences:** none in captured assertion values. The rewrite changes how test fixtures are constructed — `BetaMeasure(Interval(0,1), 2.0, 3.0)` → `BetaPrevision(2.0, 3.0)` plus `wrap_in_measure(p)` where the consumer needs Measure shape — but the resulting prevision-level values are identical. `expect(p, f)` on the prevision returns the same number as `expect(m, f)` on the wrapped Measure. Stratum-1 assertions (`==` and `atol=1e-14`) hold bit-exact. Directional and Structural assertions hold by inequality / predicate preservation.

**Expected non-divergences confirmed explicitly (per §3 symmetry discipline):**
- `test/fixtures/posture-3-capture/` untouched with identical bytes. Not recaptured.
- `test/fixtures/particle_canonical_v1.jls` untouched.
- `test/fixtures/agent_state_v3.jls` and `email_agent_state_v3.jls` (Move 3) untouched — Move 4 doesn't re-capture v3 fixtures; they were captured at the Move-3 tip via Measure constructors but the serialised bytes don't change when tests are rewritten because the serialised bytes depend on the actual object's struct layout, not on how it's constructed. `save_state` / `load_state` round-trip in the updated `test_persistence.jl` still produces the same loaded shape.

**Per-sub-PR verification gate.** Each of 4a / 4b / 4c / 4d asserts:
- All 13 test files pass locally.
- `scripts/capture-invariance.jl --verify` produces `✓ Verified: manifests identical (modulo timestamp)`.
- No captured assertion's value diverges from the Move 0 capture at its stratum tolerance (this is a conceptual check — `--verify` tests intra-run stability; divergence against the Move 0 fixtures would need a dedicated comparator tool, out of scope for Move 4; in practice, divergence-from-fixture is caught by the test-file assertions themselves because Move 0's capture records the same assertion's pre-refactor value and the post-refactor value should match, with the test-file's own assertions failing loudly if they don't).

## 4. Worked end-to-end example

The `TaggedBetaMeasure(Interval(0,1), 1, BetaMeasure(2.0, 3.0))` pattern is the canonical mechanical rewrite.

**Before (pre-Move-4):**

```julia
# In test/test_flat_mixture.jl or similar
c1 = TaggedBetaMeasure(Interval(0.0, 1.0), 1, BetaMeasure(2.0, 3.0))
c2 = TaggedBetaMeasure(Interval(0.0, 1.0), 2, BetaMeasure(5.0, 1.0))
m = MixtureMeasure(Interval(0.0, 1.0), Measure[c1, c2], [log(1.0), log(1.0)])

# Assertion against Measure-level accessor
@assert m.components[1].beta.alpha == 2.0
```

**After (Move 4 tip):**

```julia
# Prevision-level construction
c1_p = TaggedBetaPrevision(1, BetaPrevision(2.0, 3.0))
c2_p = TaggedBetaPrevision(2, BetaPrevision(5.0, 1.0))

# For assertions that need Measure shape (post-Move-2 MixturePrevision
# still holds Measures in .components; that tightens in Move 5). Pending
# Move 5, construct the Measure wrapper when needed:
c1 = wrap_in_measure(c1_p)   # TaggedBetaMeasure
c2 = wrap_in_measure(c2_p)
m = MixtureMeasure(Interval(0.0, 1.0), Measure[c1, c2], [log(1.0), log(1.0)])

# Same Measure-level assertion works via the shield
@assert m.components[1].beta.alpha == 2.0   # unchanged; shield reads through
```

**Simpler case — single BetaPrevision for unit tests:**

```julia
# Before
m = BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0)
@assert mean(m) == 2.0 / 5.0

# After
p = BetaPrevision(2.0, 3.0)
@assert mean(wrap_in_measure(p)) == 2.0 / 5.0
# Or, where the test is Prevision-native (as it will be post-Move-5):
@assert expect(p, Identity()) == 2.0 / 5.0
```

**The migration pattern:**
- `BetaMeasure(space, α, β)` → `BetaPrevision(α, β)`; wrap via `wrap_in_measure(p)` if a Measure shape is needed downstream.
- `TaggedBetaMeasure(space, tag, beta)` → `TaggedBetaPrevision(tag, BetaPrevision(...))` or reuse the existing outer constructor that accepts a `BetaMeasure`.
- `CategoricalMeasure(space, log_weights)` → `CategoricalPrevision(log_weights)` (space context carried separately through the test; see §5.2 resolution).
- `GaussianMeasure(Euclidean(1), μ, σ)` → `GaussianPrevision(μ, σ)`.
- `GammaMeasure(PositiveReals(), α, β)` → `GammaPrevision(α, β)`.
- `ProductMeasure(factors)` → `ProductMeasure` remains (MixturePrevision/ProductPrevision element-type tightening is deferred to Move 5 per the Move 2 pivot); but constructed from `wrap_in_measure(p)` of each factor's Prevision.
- `MixtureMeasure(space, components, log_weights)` → similarly remains at Measure level until Move 5.

## 5. Open design questions

### 5.1 Test file renames

Candidate consolidations:

- `test_flat_mixture.jl` + `test_prevision_mixture.jl` → one file. Content overlap: both test MixturePrevision / flat-mixture semantics, from complementary angles (the first is pre-Prevision-primary; the second was added in Posture 3 Move 5 against the Prevision surface directly). Post-Move-4, when both files construct via Prevision, the split is vestigial.

**Prior: defer consolidation to Move 5.** Move 5 introduces the `expect-through-accessor` lint slug and retires the Measure-surface wrappers; that's the right moment for a structural reshuffle of test files that became redundant under the unified surface. Consolidating in Move 4 while the Measure shields are still live means the consolidated file carries two distinct testing idioms concurrently, which reads worse than keeping them separate until the unification is complete.

Counter-argument: deferring Move 4's tidying to Move 5 expands Move 5's scope, which is already "the largest diff of the branch" per master-plan §Move 5. One more cleanup item may not be load-bearing but adds to Move 5's cognitive load. The counter-counter-argument: the consolidation isn't that big (two files → one); it fits naturally into Move 5's "split `ontology.jl` + introduce stdlib.jl" structural rework.

Prior stands: defer.

### 5.2 Constructor surface — where does the atom type live post-Measure-deletion?

Currently `CategoricalMeasure{T}(space::Finite{T}, log_weights)` carries the atom type `T` at the Measure level. `CategoricalPrevision(log_weights)` carries only log_weights — no atom type.

**The question:** post-Move-5 Measure deletion, when a consumer wants to compute `probability(p, Indicator(event))` for an event over atoms of type `T`, where does `T` live?

Three candidate answers:

- **Option A (Prior):** `T` lives on the `Indicator{E}`-side. `probability(p, Indicator(TagSet(Set([1,2]))))` — the Event carries the atom type indirectly through its `tags::Set{Int}` field. The Prevision doesn't know the atom type; it only knows log_weights. `expect(p::CategoricalPrevision, Indicator(e))` dispatches on the Event subtype to get the right indicator computation; it indexes into log_weights by position, with the Event's data determining which positions to sum.
- **Option B:** Add `space::Finite{T}` to CategoricalPrevision as a field. The Prevision becomes atom-type-aware. Simpler dispatch; but carries observational information (the atom labels) on the Prevision, which violates the Prevision-primary principle.
- **Option C:** The atom type moves to a separate `CategoricalSpace{T}` object that travels with the CategoricalPrevision via a container type or a pairing convention.

**Prior: Option A.** The Prevision-primary principle resolves this cleanly — atom labels are observational content (they're what appears in the sample space when drawing; they're what the host sees). They belong on the Indicator / Event side, not on the Prevision. Move 5's `condition(p, e::Event)` rewrite is where the dispatch lands; Move 4 just prepares tests to construct `CategoricalPrevision(log_weights)` without the atom type and defers how the atom type flows through the Event / Indicator types to Move 5.

**This is the first application of Prevision-primary outside its original surface.** Move 2's §5.1 codified the principle in `condition`'s output position — concentrated Previsions over the ambient space rather than reduced-space Measures. Move 4 extends the principle to the dispatch-input position: atom identity flows *into* `condition` via Event / Indicator, and atom identity is observational vocabulary, so it lives on the observational (Event) side even after Measure retires. The principle is now a documented precedent applicable at every "should this live on Prevision or on Measure?" boundary later moves surface. Move 5 will apply it again when `expect`'s dispatch on functional families is rewritten.

**What Option B rejects, named explicitly.** Option B would place the observational vocabulary — atom identity — on the prevision layer. This is the inverse of where it belongs: atom identity is precisely the observational content Measure was the historical home for. The Prevision-primary principle says that content stays on the observational side even after Measure retires, carried by whatever types (Finite, Event) remain observational. Landing atom type on Prevision would be a local win (cleaner CategoricalPrevision dispatch) at the cost of a principle-wide regression.

What Move 4 does: when a test needs both a CategoricalPrevision and atom-type context (e.g., to construct an Event indicator), it passes both explicitly — the Prevision in one variable, the atom type / Finite space in another. Move 5 may introduce sugar to bind these together at the Indicator-construction site.

### 5.3 Move 0 capture invariance — value vs expression

Prompt 7 raised the concern: "the Julia value constructed is different even when the semantic content is the same. The invariance check is behavioural — `expect(p, f)` returns the same number — not syntactic." My Stage-2 patch fold-in added the resolution: the Move 0 capture script captures **values**, not expressions, and the invariance check compares captured-value to re-computed-value across the refactor.

**Confirmation against the actual `scripts/capture-invariance.jl`:** the script's `_record_passing` path for Exact shape stores `(lhs_value, rhs_value)` directly (not the LHS / RHS expression strings). For Tolerance shape, it stores `(lhs_value, rhs_value, atol, rtol)` — values, not expressions. The captured `expr_source` is recorded alongside for debugging but not consulted during verification. Move 4's test rewrites change how the LHS/RHS values are *constructed* but not *what they equal*; the capture sees the same values, verify passes.

**Prior: the capture protocol is value-based (already verified in the code); no action needed in Move 4 beyond relying on this.**

### 5.4 Test-oracle pragma policy (42 sites)

Per Prompt 0 task 4 fold-in (Stage 2 patch PR #42): PR #40's pass-two taint analysis left 42 call sites pragma'd with `# credence-lint: allow — precedent:test-oracle — <reason>`. Predominantly in `apps/skin/test_skin.py` (Python-side, Move 7's scope) and `test/test_prevision_*.jl` (Julia-side, Move 4's scope). These sites read structural fields (`m.alpha`, `p.log_weights[i]`) as part of test oracles against known closed-form posteriors.

**Move 4 rewrites every test.** State the policy explicitly so it's not re-litigated per-site during the rewrite.

- **Option A (Prior):** oracle sites remain oracles against structural fields. The `test-oracle` pragma persists after the Measure → Prevision rename. The oracle's purpose is to catch drift in the *representation* — if `BetaPrevision.alpha` stopped being the α parameter (a bug in how Move 2 Phase 4 wired the outer constructor), the oracle catches it. A pure-`expect` oracle would miss the representation-level bug.
- **Option B:** migrate to oracles against `expect` output. Stricter — oracle becomes representation-independent — but tests become sensitive to representation-irrelevant implementation changes (e.g., a future refactor that changes `expect(::BetaPrevision, ::Identity)`'s dispatch path without changing its value causes the oracle to hit a different code path that may or may not produce the bit-exact same value).

**Prior: A.** The `test-oracle` pragma is the repo's sanctioned form of "this site reads a structural field and that's correct here." Migrating to pure-`expect` oracles is a different testing discipline that belongs in a separate move (if it happens at all); it's orthogonal to Move 4's mechanical constructor rewrite.

Concretely: when Move 4 rewrites `BetaMeasure(2, 3).alpha == 2.0` → `BetaPrevision(2.0, 3.0).alpha == 2.0`, the pragma carries forward unchanged. The oracle-against-structural-field remains; only the constructor syntax changes.

**Invariant on the pragma count.** The 42 pragma sites stay at 42 across sub-PRs 4a–4d. No new pragmas added (the mechanical rewrite doesn't introduce new Prevision-primary discipline violations); no existing pragmas retired (the rewrite doesn't turn a structural oracle into a pure-`expect` one — that would be Option B scope creep). A sub-PR landing with 41 or 43 pragma sites is a **flag**, not unremarked drift — investigate whether a site genuinely stopped needing a pragma (rare; verify), whether one accidentally appeared (bug; fix), or whether the count boundary needs documenting as a legitimate exception (amend this §5.4 in the affected sub-PR).

## 6. Risk + mitigation

**Risk (medium — volumetric):** A Measure construction site missed by the rewrite compiles and runs but doesn't exercise the Prevision-primary surface Move 5 depends on.

**Mitigation:** `grep -cE 'Measure\(' test/test_*.jl` at each sub-PR's tip. Target post-Move-4: 0 (modulo comments, docstrings, and legitimate `wrap_in_measure` return sites which are `Measure`-typed but not Measure-constructed). Each sub-PR's PR body reports the grep count before and after.

**Risk (medium — per-component space):** A test constructs `CategoricalMeasure(Finite([...]), log_weights)` and migrates to `CategoricalPrevision(log_weights)`; subsequent `wrap_in_measure(p)` call needs the `Finite` space, which the Prevision doesn't carry. §5.2's Option A means the test has to hold the `Finite` space separately.

**Mitigation:** The migration pattern in §4 calls this out. When the test needs `wrap_in_measure(p)` for a Measure-requiring downstream consumer, the `Finite` space is threaded through explicitly. For categorical cases specifically, the wrap is `CategoricalMeasure(space, p)` using the existing 2-arg constructor, not the generic `wrap_in_measure(p)`.

**Risk (medium — `test-oracle` pragma count drift):** The 42 pragma sites include some in `apps/skin/test_skin.py` that Move 4 doesn't touch. After Move 4, the Julia-side pragma count may change if the rewrite reveals sites that no longer need the pragma (the Measure-surface read becomes a Prevision-surface read that doesn't trigger the lint). The drift is a signal not a failure — it means Move 4 moved test code closer to the invariant that the lint encodes.

**Mitigation:** report pragma-count delta in each sub-PR's PR body. Drift is expected and acceptable; unexplained drift is investigated.

**Risk (review-process):** four sub-PRs under one design doc is unusual cadence.

**Mitigation:** sub-PR bodies cross-reference this design doc by name. Each sub-PR merges autonomously once CI greens + no pixel6 pushback per the repo's merge-authority convention. Reviewer objections to any sub-PR re-open this design doc for amendment; the remaining sub-PRs pause pending resolution.

## 7. Verification cadence

Per sub-PR:

```bash
# Full test suite (no test file is yet known to be Prevision-surface-clean)
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

# Measure constructor count in modified files (should monotonically decrease)
for f in <files touched by this sub-PR>; do
    echo "$f: $(grep -cE 'Measure\(' $f)"
done
```

Per-sub-PR CI via `.github/workflows/publish-image.yml`. Cumulative verification at the end of sub-PR 4d: all 13 test files constructed fully through Prevision-primary surface (modulo `wrap_in_measure` wraps for Measure-dispatch-required consumers); Move 5 proceeds from this tip.

## 8. de Finettian discipline self-audit

1. **Is every numerical query in this move routed through `expect`?** N/A — Move 4 introduces no new numerical queries. It rewrites how existing queries are constructed; the `expect(p, f)` / `expect(m, f)` dispatch is the same surface.

2. **Does this move hold a Prevision inside a Measure, or a Measure inside a Prevision, for any reason?** The opposite — Move 4 *moves* tests to construct Previsions directly rather than going through Measure constructors that internally wrap a Prevision. The `wrap_in_measure(p)` helper used in this move is a read-time reconstruction, not a persistent Prevision-inside-Measure field. Prevision-inside-Measure fields (`CategoricalMeasure.prevision`, etc.) retire entirely in Move 5 when Measure deletes.

3. **Does this move introduce an opaque closure where a declared structure would fit?** No. Constructor rewrites are pure type-name substitutions + argument reshaping; no closure captures introduced.

4. **Does this move add a `getproperty` override on any Prevision subtype?** No. Move 4 rewrites construction sites; shield definitions (all Measure-level) are not touched. Move 5 retires them with Measure.

---

## Reviewer checklist

- [ ] §0 Final-state alignment is a paragraph and names the transient `wrap_in_measure` bridge explicitly.
- [ ] §5 contains four non-trivial open questions with stated priors: renames defer to Move 5; atom type lives on Indicator/Event per Prevision-primary; capture is value-based (already verified); test-oracle pragmas persist.
- [ ] §8 self-audit: (1) N/A no new queries; (2) retires Measure-inside-Prevision (opposite); (3) no new closures; (4) no new `getproperty` override.
- [ ] File-path:line citations current (surveyed at master SHA `6e51d93`, post-PR-#53).
- [ ] Sub-PR phasing (4a / 4b / 4c / 4d) plausibly keeps each sub-PR diff reviewable.
