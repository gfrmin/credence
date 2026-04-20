# Skin smoke-test fixtures

Separate from `test/fixtures/` because skin-side serialisation uses
Julia `Serialization.serialize` base64-encoded through the JSON-RPC
boundary (not the `test/fixtures/` Dict-of-Measure Julia serialisation).
Pinning wire-format invariance needs a fixture at the RPC boundary.

## `beta_v1.b64`

**Source SHA:** `bf74f985821c37b89fa4321e74b07092d1f63b65` (master tip at
Move 3 code PR opening; pre-Move-3 struct layout)
**Capture date:** 2026-04-20
**Represents:** `BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0)` serialised
via `Serialization.serialize` into an `IOBuffer` and `base64encode`'d —
the exact format `handle_snapshot_state` in `apps/skin/server.jl`
produces. The Measure inside has the pre-Move-3 struct layout
(`BetaMeasure(space, alpha, beta)` with three direct fields).

**Expected values (the capture construction):** `alpha=2.0, beta=3.0,
mean=0.4`.

**Invalidation conditions:** the pre-Move-3 struct layout for
`BetaMeasure` changes upstream of Move 3 (would require a v0 fixture
for that older shape). A post-Move-3 change to the `BetaMeasure` struct
does *not* invalidate this fixture — the point is to verify v1→v2
behaviour.

## Loading in the smoke test

`apps/skin/test_skin.py` includes `test_v1_snapshot_restore` which:
- Reads the base64 blob from this file.
- Passes it to `skin.restore_state(blob)` over the JSON-RPC wire.
- Asserts the call fails (the Julia side raises due to struct
  mismatch; the JSON-RPC error surfaces as a SkinError).

The blob is NOT expected to round-trip cleanly into a post-Move-3
Measure. The test documents the behavioural contract: v1 snapshot blobs
fail loudly rather than silently corrupting state. Users holding v1
snapshots reinitialise; the skin server is not a persistence-migration
layer.

## Rules

Same as `test/fixtures/`: fixtures are read-only in the test suite,
never regenerated to fix load bugs (fix the load code instead),
commit-pinned with SHA recorded, new schema versions get new fixtures
while old ones stay to verify backward-compat behaviour.
