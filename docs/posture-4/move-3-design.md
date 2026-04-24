# Move 3 — Persistence v3-only

## 0. Final-state alignment

Move 3 converges the current tip toward `master-plan.md` §"Types deleted" in two respects: it deletes **persistence v1 and v2 loader paths** (named in "Types deleted" as transitional artefacts that retire with the Measure type system) and it retires the **`MigrationError` path** associated with v1 files that Julia's struct-layout-dependent `Serialization` can't deserialise across the Posture 3 Move 3 struct change. The `:__schema_version` marker stays (see §5.2 for the argument); the stored value bumps to `3`. Persistence v1 fixtures (`test/fixtures/agent_state_v1.jls`, `test/fixtures/email_agent_state_v1.jls`) are deleted from git via the same PR that recaptures v3 fixtures from the post-Move-2 tip. No transient state carries forward; Move 4's test migration reads the new v3 fixtures against the Prevision-primary test surface that Move 4 establishes.

## 1. Purpose

Retire the v1 and v2 persistence paths in favour of a v3-only surface pinned at the post-Move-2 SHA. Delete the v1 fixtures and the `MigrationError` handling they exercise. Recapture v3 fixtures from current master (post-Move-2 tip `2937c1d` or its tip at the Move 3 code PR opening). Update `test/test_persistence.jl` to assert v3 round-trip only. The `__schema_version = 3` marker is retained on-disk for future schema bumps (Move 9 production-state persistence is the nearest expected consumer).

## 2. Files touched

Modifies:
- `src/persistence.jl` (~136 lines currently):
  - `SCHEMA_VERSION = 2` → `SCHEMA_VERSION = 3` (`src/persistence.jl:45`).
  - Module docstring (lines 1–39) rewritten: remove the v1 / v2 discussion; state v3 as the only supported version; note that future schema bumps (Move 9) follow the same `__schema_version` dispatch pattern.
  - `load_state` (lines 60–130): delete the v1 `MigrationError` fallback branch (catches `TypeError` / `MethodError`). The `load_state` body reduces to: open + deserialize + check `:__schema_version == 3` + return. Unknown-version files raise `MigrationError` with a short message naming the expected version.
  - `MigrationError` (lines 48–58): retained — future-proofs against accidental-unknown-version files. Short docstring.
- `test/test_persistence.jl` (~121 lines currently):
  - Delete the "v1 fixture migration path" section (lines ~79–121). Two `let` blocks testing `agent_state_v1.jls` and `email_agent_state_v1.jls` fixtures → gone.
  - Retain the v3 round-trip section (the current "v2 round-trip" block renamed + values bumped): save → load → assert `__schema_version == 3`; assert structural fields preserved.
  - Add a small "unknown-version `MigrationError` path" section: write a file with `:__schema_version => 99`, assert `load_state` raises `MigrationError` with a message naming v3 as expected. Keeps the `MigrationError` branch live in test coverage.

Deletes:
- `test/fixtures/agent_state_v1.jls` (`git rm`).
- `test/fixtures/email_agent_state_v1.jls` (`git rm`).

Creates:
- `test/fixtures/agent_state_v3.jls` — recaptured from post-Move-2 tip via a construction script colocated with the fixture (see §4 worked example). Schema version = 3; structure matches the current `BetaMeasure` / `TaggedBetaMeasure` struct layout (post-Move-2 `TaggedBetaPrevision.beta::BetaPrevision`).
- `test/fixtures/email_agent_state_v3.jls` — recaptured analogously.
- Updated entries in `test/fixtures/README.md`: remove v1 fixture sections; add v3 fixture sections with SHA pin + construction script + expected load values.

Renames: none.

Out of scope:
- `test/fixtures/particle_canonical_v1.jls` stays (separate Move 6 canonical-particle fixture; not persistence schema).
- `test/fixtures/posture-3-capture/` stays (Move 0 invariance target).

**Commit phasing.** Single-commit PR. The three changes are tightly coupled (deleting v1 fixtures requires updating the test that references them; bumping SCHEMA_VERSION requires recapturing fixtures at the new version) and reviewing them in isolation doesn't add information. Full test suite green; `scripts/capture-invariance.jl --verify` passes.

## 3. Behaviour preserved

Move 0 fixture at `test/fixtures/posture-3-capture/` (6118 site×value tuples at branch-point `5c6a94e`) is the invariance target. Move 3 expects:

- **Every captured assertion in test files other than `test_persistence.jl`** passes unchanged. Persistence changes are isolated to the persistence module + test; no other test file reads or writes state files.
- **`test_persistence.jl`'s captured assertions** are expected to diverge — the v1 fixture test sites go away (deletion) and the v3 round-trip test sites replace the v2 round-trip. This is an **intended divergence**, explicitly named here, matching the pattern from Move 2's `test_shared_reference_contract` retirement (which Option C narrowed out of Move 2's actual code scope but the discipline stands: intentional assertion retirements must be named in the move's design doc).
- The Move 0 fixture captures `test_persistence.jl`'s assertions. Post-Move-3, those captured sites either don't run (deleted v1 tests) or hit at different values (v3 schema version instead of v2). The `--verify` check — which runs the capture twice on the current tree and diffs — continues to pass because both runs see the post-Move-3 state.
- **The invariance-against-Move-0-fixtures check would fail** for `test_persistence.jl` sites if such a check existed. It doesn't — `--verify` is intra-run stability, not inter-tree comparison. Move 3 halting-on-divergence is therefore not triggered by the persistence deletions, consistent with the design.

For the other 12 test files, zero divergence is expected. Persistence is a boundary surface that tests-that-aren't-persistence don't touch.

**Non-divergence confirmed (symmetric to the named divergences above).** The two fixture families explicitly out of scope in §2 remain untouched and unregenerated under Move 3: `test/fixtures/particle_canonical_v1.jls` (Move 6 canonical-particle fixture) stays pinned at its original SHA with identical bytes; `test/fixtures/posture-3-capture/` (the Move 0 invariance target — 6118 site×value tuples at branch-point `5c6a94e`) stays pinned and identical. Move 3 does not recapture, touch, or reference either. The named-divergence discipline is symmetrical: what-the-move-does-not-touch is recorded alongside what-it-changes, so a reviewer reads the assertion changes against the scope exclusions without cross-referencing.

## 4. Worked end-to-end example

The v3 fixture recapture, end-to-end.

**Construction script** (saved as a comment block in `test/fixtures/README.md`'s v3 section, not as a live file):

```julia
# Recapture script for agent_state_v3.jls — run once from master at SHA <post-Move-2-tip>.
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
    :__schema_version => 3,
    :belief => m,
    :note => "agent_state_v3 fixture; captured post-Move-2; 3 TaggedBetaMeasure components with posterior after 2 pos + 1 neg FiringByTag(1,2) observations. Post-Move-2: TaggedBetaPrevision.beta is BetaPrevision (not BetaMeasure).",
)
open(io -> serialize(io, state), "test/fixtures/agent_state_v3.jls", "w")
```

**Load trace in the round-trip test:**

```julia
let path = tempname()
    # v3 save via public save_state API
    rel_beliefs = MixtureMeasure(...)  # as in the current v2 test
    cov_beliefs = MixtureMeasure(...)
    cat_belief  = CategoricalMeasure(...)

    save_state(path;
               rel_beliefs = rel_beliefs,
               cov_beliefs = cov_beliefs,
               cat_belief  = cat_belief,
               total_score = 1.5,
               total_cost  = 0.25)

    loaded = load_state(path)
    # v3 round-trip assertions
    @assert loaded[:__schema_version] == 3
    @assert loaded[:rel_beliefs].components[1].factors[1].alpha == 2.0  # Beta(2,3) α
    # ... etc
    rm(path)
end
```

Dispatch trace (unchanged by Move 3 — the load path just stops branching on v1/v2):
- `save_state(...)` → `src/persistence.jl:50-62` → writes Dict with `:__schema_version => 3` via `serialize`.
- `load_state(path)` → `src/persistence.jl:65-95` → `open(deserialize, ...)` → checks `:__schema_version == 3` → returns Dict.
- Accessing `loaded[:rel_beliefs].components[1].factors[1].alpha` → via the `MixtureMeasure` and `ProductMeasure` shields (unchanged by Move 3) → BetaPrevision.alpha field read.

**Unknown-version error path:**

```julia
let path = tempname()
    # Write a file with __schema_version = 99 (bogus future version)
    bogus = Dict(:__schema_version => 99, :rel_beliefs => nothing,
                 :cov_beliefs => nothing, :cat_belief => nothing,
                 :total_score => 0.0, :total_cost => 0.0)
    open(io -> serialize(io, bogus), path, "w")
    # load_state raises MigrationError
    try
        load_state(path)
        @assert false "should have thrown"
    catch e
        @assert e isa Credence.Persistence.MigrationError
        @assert occursin("expected 3", sprint(showerror, e))
    end
    rm(path)
end
```

## 5. Open design questions

### 5.1 Fixture regeneration discipline

The Posture 3 precedent (Move 3 code PR, per `test/fixtures/README.md` §Rules) is: fixtures are captured once and never regenerated to fix loading bugs. Posture 4's v3 fixtures are regenerated because they're new; future Move 9 body-work state is another first capture. The question: does the "never regenerate" rule apply to v3 as soon as Move 3 lands, or only after Move 9's production-state fixtures are also captured?

**Options:**

- **A (my prior):** "never regenerate" applies **post-Move-9**. Moves 3–8 may invalidate and recapture v3 fixtures if a refactor changes the serialised shape of the belief state. This matters because Move 5 retires Measure; Move 6 migrates apps; Move 7 rewrites skin; any of these might change the struct layout of what `save_state` writes. Requiring "never regenerate" at Move 3 would either lock in an obsolete layout or force every subsequent move to carry a v3 → v4 migration that isn't load-bearing for anything.
- **B:** "never regenerate" applies immediately at Move 3. Any Move 4–8 struct-layout change triggers a v3 → v4 schema bump with migration, matching the Posture 3 Move 3 v1 → v2 pattern.

**Argument for A.** The Posture 3 "never regenerate" rule was introduced because fixture regeneration can silently mask migration test bugs — if `agent_state_v1.jls` is regenerated from a v2-aware codebase, the v1 → v2 migration test passes on the regenerated file but doesn't actually exercise the v1 load codepath. The rule exists to prevent that failure mode. The failure mode is not in play during Moves 3–8: there is no external v2 user state to migrate; the fixtures exist to audit the save/load round-trip at the current tip. If Move 5 changes the struct layout and the v3 fixture regenerates at Move 5's tip to match, the round-trip test still checks what it was designed to check (write a state → read it back → structural fields preserved). No silent masking. The "never regenerate" rule applies to fixtures whose purpose is to test migration from an old layout to a new one; Posture 4's intermediate-move fixtures have no such old-layout consumer.

**Argument against A.** Relaxing the rule during intermediate moves could creep — someone mid-Move-5 might regenerate the v3 fixture "because the struct layout changed" without noticing that the regeneration is actually hiding a save/load bug. Mitigation: the move's design doc must explicitly call out fixture regeneration and justify it (matches the §5 question-naming discipline). Plus: Moves 3–8 are under the reviewer-driven cadence; regeneration would appear in the diff and the reviewer would ask why.

**Argument against B.** Forces every intermediate move that changes the belief-state struct layout to design a v3 → v4 migration, then v4 → v5, then v5 → v6 — for schema versions no one will ever write. Each migration is a one-off discipline-exercise with no downstream beneficiary. The Posture 3 v1 → v2 migration was justified by "v1 state might exist on user machines" — here, no such state exists.

**Prior: A.** Move 9 is when the production state schema stabilises — before then, the schema is a moving target and regeneration is the honest response to that. Post-Move-9, the "never regenerate" rule kicks in because the schema becomes real.

**Named condition on intermediate-move regeneration.** The rule's failure mode is absent from Moves 3–8 (no migration test is in play), but the *discipline* the rule encodes — leaving forensic evidence of why a fixture moved — is still worth preserving in weakened form. Without it, the weakening reads as "regenerate freely during Moves 3–8", which is the short step from "regenerate when necessary" that the original rule exists to foreclose. Regeneration during Moves 3–8 is permitted **only when accompanied by**: (a) a design-doc note in the move that triggered regeneration identifying the specific refactor that invalidated the prior fixture, and (b) a one-line entry in `test/fixtures/README.md` recording the pre-regeneration SHA and the post-regeneration SHA. Reviewers reject intermediate-move PRs that regenerate a fixture without both.

### 5.2 `__schema_version` marker: retain or retire?

With v1 and v2 gone, `__schema_version` is marking a single version. Does the field stay (for forward compatibility when Move 9 production persistence bumps to v4) or retire (YAGNI)?

**Prior: retain**, with `SCHEMA_VERSION = 3`.

**Argument.** Retiring the marker now saves five lines of code (the version-check branch in `load_state`). Retaining it gives Move 9's production persistence a migration surface: when production state introduces fields (connection registries, program caches) that v3 doesn't have, the bump v3 → v4 works through the same `__schema_version` dispatch the Posture 3 Move 3 v1 → v2 migration established. Retiring now and re-introducing in Move 9 would be a code churn with no offsetting benefit.

The `MigrationError` type also stays (same rationale — it's the load-path's vocabulary for "version I don't understand"), though its docstring simplifies: the current docstring discusses the v1 → v2 struct-layout-incompatibility that no longer applies.

YAGNI would apply if `__schema_version` were a speculative feature. It isn't — Move 9 is a planned consumer. "YAGNI unless you actually need it within the current branch" is the correct read; Move 9 is within the branch.

## 6. Risk + mitigation

**Risk (low):** `agent_state_v3.jls` captured from a SHA that doesn't match the Move 3 code PR's base. The v3 fixture's serialised bytes encode the struct layout at capture time; if the PR is rebased onto a newer master with a different layout, the fixture might no longer roundtrip.

**Mitigation:** capture the v3 fixture as the first commit of the Move 3 code PR, with the capture script's pinned SHA recorded in `test/fixtures/README.md`. If the PR is rebased, re-capture on the new base and note the SHA update. The base SHA is documented; divergences between documented SHA and actual serialised layout are reviewable.

**Risk (low):** The v1 fixtures are referenced somewhere outside `test/test_persistence.jl` and the references break silently on deletion.

**Mitigation:** `grep -rn 'agent_state_v1\|email_agent_state_v1' .` at PR opening, resolve every hit. Expected hits: the two test blocks in `test_persistence.jl` and the README entries. Any third hit is investigated.

**Risk (medium):** The v3 fixture's content post-Move-2 struct layout doesn't survive Move 5's Measure deletion. Persistence tests fail at Move 5.

**Mitigation:** Per §5.1 Option A, Move 5 may re-capture v3 fixtures at Move 5's tip. The Move 5 design doc tracks this explicitly if the struct layout changes affect what's serialisable.

**Risk (review-process):** §5.1's "never regenerate" relaxation is contested; reviewer prefers Option B.

**Mitigation:** Option B is a more restrictive path but implementable — each intermediate move would add a v3 → v4, v4 → v5, etc. migration. The Move 3 code PR's structure is the same under either option; only the Move 4–8 discipline changes. Adopt via design-doc amendment if reviewer disagrees.

## 7. Verification cadence

```bash
# All 13 test files pass post-Move-3
julia test/test_core.jl
julia test/test_prevision_unit.jl
julia test/test_prevision_conjugate.jl
julia test/test_prevision_mixture.jl
julia test/test_prevision_particle.jl
julia test/test_host.jl
julia test/test_flat_mixture.jl
julia test/test_events.jl
julia test/test_persistence.jl    # the updated v3-only test
julia test/test_grid_world.jl
julia test/test_email_agent.jl
julia test/test_rss.jl
julia test/test_program_space.jl

# Move 0 invariance check
julia --project=scripts scripts/capture-invariance.jl --verify
# Expected: ✓ Verified: manifests identical (modulo timestamp)

# No stale v1 fixture references
grep -rn 'agent_state_v1\|email_agent_state_v1' . --exclude-dir=.git
# Expected: empty, modulo the README's historical note section.

# v3 fixture round-trip smoke
julia -e '
push!(LOAD_PATH, "src"); using Credence
loaded = Credence.Persistence.load_state("test/fixtures/agent_state_v3.jls")
@assert loaded[:__schema_version] == 3
println("v3 fixture loads cleanly at schema version 3")
'
```

## 8. de Finettian discipline self-audit

1. **Is every numerical query in this move routed through `expect`?** N/A — Move 3 introduces no new numerical queries. Persistence is structural round-trip plus schema-version gate; no prevision-level arithmetic.

2. **Does this move hold a Prevision inside a Measure, or a Measure inside a Prevision, for any reason?** Measure-inside-Prevision persists in the `MixturePrevision.components::Vector` / `ProductPrevision.factors::Vector` fields that Move 2 left untightened per the Option C pivot (see `docs/posture-4/move-2-design.md` §5.1.1). Move 3 doesn't touch these fields; they retire in Move 5 concurrent with `condition`'s Prevision-primary rewrite. Dated-deprecation note per the self-audit: **Move 5** tightens both.

3. **Does this move introduce an opaque closure where a declared structure would fit?** No. The persistence layer uses `Serialization.serialize` / `deserialize`; the `Dict` payload is declared structure. `save_state` and `load_state` take explicit keyword arguments (no closure captures).

4. **Does this move add a `getproperty` override on any Prevision subtype?** No.

---

## Reviewer checklist

- [ ] §0 Final-state alignment is a paragraph, not a sentence, and names the `particle_canonical_v1.jls` and `posture-3-capture/` fixtures as explicitly out of scope.
- [ ] §5 contains two non-trivial open questions with stated priors (regeneration discipline; `__schema_version` retention).
- [ ] §8 self-audit: (1) N/A no new queries; (2) Measure-inside-Prevision retained, dated-deprecation Move 5; (3) no closures added; (4) no new Prevision-level `getproperty`.
- [ ] File-path:line citations current (surveyed at master SHA `cd65ab4`, post-PR-#51).
- [ ] Move 3 as described does not require Move 4 to retract or rework it — v3 fixtures are self-contained; `__schema_version` stays at 3 until Move 9 bumps to v4.
