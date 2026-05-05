# Pass 2 — open notes

Cross-step Pass-1 observations that warrant Pass-2 attention. Each
note is a breadcrumb, not a commitment; Pass 2 may resolve, defer, or
explicitly reject any item, but should not encounter the issue without
prior knowledge.

## Body / extension

### `lastEventIdByTool` Map grows unboundedly on long sessions

Surface: `apps/credence-pi/extension/src/index.ts` declaration site
of `lastEventIdByTool`. Every distinct tool name a session has called
becomes a permanent entry in the Map; entries are never deleted.

Risk: bounded growth on long-running sessions. Pass 1 sessions are
short enough that this is theoretical, but a Pass 2 deployment with
multi-hour sessions and many distinct tool names could accumulate
thousands of entries.

Possible Pass-2 fixes: LRU cap, per-event-id cleanup tied to
`tool_execution_end`, or expiry-on-stale (e.g. drop entries older than
last user message). The right answer depends on Pass-2 traffic patterns
not yet observed.

Inline marker at the declaration site reads `PASS-2-NOTE: …` so a
future grep finds this note.

### `tool-completed` JSON round-trip not exercised by replay test

Surface: `apps/credence-pi/tests/julia/test_observation_log.jl` covers
`user-responded` round-trip + replay correctness. `test_server.jl`
covers the no-signal-emitted invariant for `tool-completed`. Neither
exercises the JSON round-trip on `tool-completed.outcome` —
specifically that `result_summary: null`, `error: null`, and absent
fields all preserve correctly through serialise → log → read → BDSL
dispatch.

Risk: Pass 1's BDSL doesn't condition on `tool-completed`, so the
round-trip is currently academic. The moment Pass 2 starts using the
field — secondary-signal observation model is the obvious place — a
silent serialiser bug would manifest as Pass-2-only test failures and
look like a Pass-2 regression rather than a Pass-1 latent.

Pass-2 action: add the round-trip oracle to whichever test file owns
the secondary-signal observation, before wiring up the conditioning.

### `tests/typescript/` lives outside the extension's working directory

Surface: file layout. `extension/`'s `package.json` test script reaches
up with `node --import tsx --test ../tests/typescript/*.test.ts`. The
CI workflow consequently sets
`working-directory: apps/credence-pi/extension` and runs `npm test`
from there.

Comparison: `tests/julia/` is co-located beside `daemon/`. The TS
asymmetry means a developer browsing `extension/` cannot find the
tests without knowing about the sibling `tests/typescript/` directory.

Pass-2 cleanup: move `tests/typescript/*.test.ts` under
`extension/tests/` (Node convention), update the test-script glob,
update CI's `working-directory`. Mechanical change; no semantic
impact. Out of Pass-1 scope.

## Brain / daemon

### Local `julia --project=. test_*.jl` requires `Pkg.instantiate()`

Surface: `apps/credence-pi/daemon/observation_log.jl` and
`apps/credence-pi/tests/julia/test_observation_log.jl` import JSON3.
Running the test directly with `julia --project=. test_*.jl` from a
fresh worktree fails with `JSON3 not installed` until
`Pkg.instantiate()` runs. CI handles this in the
`Instantiate Credence Julia project` step.

Possible Pass-2 fix: add a one-line hint to `daemon/README.md`
("first-run: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`")
and/or to the top-of-repo README. Local-dev-only concern; CI is
already correct.

### `tool-completed` is collected but not learned-from

Surface: `decide.bdsl` has no path that conditions on
`tool-completed`. The observation log accumulates `tool-completed`
events for forward compatibility (Pass 2's secondary-signal observation
model). This is intentional in Pass 1.

Pass-2 action: when the secondary-signal kernel is introduced,
`tool-completed` becomes informative. Wire schema doesn't need to
change.

### `project_id` is opaque in Pass 1

Surface: SPEC.md's "features" section explicitly defers
`project_id` (and `parent_tool_call_name` beyond the core seven) to
Pass 2. Pass 1 records it on every event but doesn't condition on it.

Pass-2 action: declare an appropriate space (likely a growing
categorical) when the structure-learning machinery is in place.
Spec already names this; no Pass-1 change needed.

## Architecture / spec

### `precedent:test-oracle` widening was a real spec gap

Surface: SPEC.md "Lint pragma policy". Pass 1 widened the precedent
to admit both value equality (`eu_ask == 0.0`) and structural
parameter equality (`p.alpha == 3.0 && p.beta == 2.0`).

Pass 2 may surface a third equality form — joint distribution
equality on factors, perhaps, or graph-isomorphism on CEG
posteriors. The same one-precedent-with-multiple-equality-forms
framing extends naturally; resist adding sibling precedents
unless the underlying invariant ("tests of the reasoner need an
oracle stronger than production code") genuinely differs.

### Effector signal action-space guard

Surface: `daemon/server.jl`'s `action_to_signal` raises on actions
outside the manifest. Pass 1's BDSL only emits manifest actions, so
the guard is currently belt-and-braces. Pass 2 may add `substitute`
(or other effectors); the guard already enforces the manifest as the
authoritative wire vocabulary.

No Pass-2 action — flagging that the design is correct under
extension.

## Out of Pass-1 scope (recap from SPEC.md, for completeness)

The list under SPEC.md § "Out of scope (Pass 1)" and § "Decisions
deferred to Pass 2" is canonical; Pass 2 should treat that list as
authoritative for major scope decisions. The notes above are the
narrower implementation-shape breadcrumbs that surfaced during
Pass 1 execution and might otherwise be lost.
