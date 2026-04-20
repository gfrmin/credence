# Test fixtures — provenance protocol

This directory holds frozen reference state used by tests that verify schema-version migrations and other point-in-time invariants. Every fixture is **commit-pinned**: it is captured from a named SHA, that SHA is recorded in this README, and the fixture is never regenerated to fix a loading bug. If a future-discovered bug affects how the fixture is loaded, the fix goes in the load code; the fixture stays as-is.

This protocol exists because fixture regeneration silently invalidates migration tests. If `test/fixtures/agent_state_v1.jls` is regenerated from a v2-aware codebase to "fix" a load failure, the test passes — and the migration codepath the test was supposed to verify is no longer exercised. Real users with v1 state on disk discover the corruption in production. The protocol prevents this by making regeneration a procedural violation rather than a one-line script.

## Planned fixtures

The following fixtures are planned but **not yet captured**. They are captured immediately before Move 3's code PR opens, after `de-finetti/posture-2-events` fully merges to master. Posture 2's gate-7 (`946a30f`) touches Measure-adjacent code; capturing fixtures before that merge would freeze a pre-Posture-2 v1 shape that doesn't match what real users will have on disk after Posture 2 ships.

### `agent_state_v1.jls` — planned

**Source SHA:** *to be filled in immediately before Move 3 code PR opens*
**Capture date:** *TBD*
**Represents:** an `AgentState` produced by current code — `MixtureMeasure` of `TaggedBetaMeasure` components with explicit `logw` field, plus the `rel_beliefs`, `cov_beliefs`, `cat_belief` triple persisted by `src/persistence.jl:save_state` at v1 schema.
**Capture protocol:**
1. Check out the source SHA on a clean clone (no Posture 3 changes).
2. Run a minimal script that constructs a representative `AgentState`, applies a few `condition` calls to put non-trivial weight structure on it, and calls `save_state(path, state)`.
3. Copy the resulting `.jls` to `test/fixtures/agent_state_v1.jls`.
4. Record the SHA, the construction script (verbatim, in this README), and the expected loaded values (means, weights, alpha/beta of each component) for the test to assert against.
**Invalidation conditions:** the v1 schema's struct layout changes upstream (would require either a v0 fixture for that older shape too, or extending the load code to handle both). A change to the v2 schema does *not* invalidate this fixture — that's the whole point.

### `email_agent_state_v1.jls` — planned

**Source SHA:** *to be filled in immediately before Move 3 code PR opens*
**Capture date:** *TBD*
**Represents:** the email-agent shape — `MixtureMeasure` of `ProductMeasure` of `BetaMeasure`, as persisted by the email-agent host (`apps/julia/email_agent/host.jl`).
**Capture protocol:** as above, but constructed from the email-agent's `initial_reliability_state` + a few synthetic observations to get realistic factor structure.
**Invalidation conditions:** as above.

## Once captured

Each fixture's entry above is updated with:
1. The exact source SHA (40 hex chars).
2. The capture date.
3. The verbatim Julia construction script used to produce it.
4. The expected loaded values (means, weights, parameters) the test asserts against.

The verbatim construction script is part of provenance, not just a curiosity — if a v1 representation bug is discovered later, the script lets us reason about what state the fixture *should* represent without regenerating it.

## Loading these fixtures

`test/test_persistence.jl` (created in Move 3) loads each fixture in v2 code and asserts the resulting object's weights/parameters/structure match the recorded expected values. The test file documents which fixture covers which load codepath; if a new load codepath is added later, a new fixture covers it (do not extend an existing fixture's expectations).

## Rules

- Fixtures are **read-only** in the test suite. Tests load them and assert; tests never write back.
- Fixtures are **never regenerated** to fix loading bugs. The fix goes in load code.
- A new schema version (v3, v4, …) gets a new set of fixtures captured at the SHA that introduced that version. Old fixtures remain to verify backward-compat load.
- Fixture binary blobs are checked into git. They should be small (KB range, not MB); if a fixture grows large, that signals it's capturing too much — split it.
