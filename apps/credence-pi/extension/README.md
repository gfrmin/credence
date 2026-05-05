# credence-pi extension

The body. A pi extension that hooks `tool_call` and `tool_execution_end`,
emits sensor events to the daemon, and dispatches effector signals back
into pi via the registered effector implementations.

## Responsibilities

- Verify at startup that every effector declared in
  `bdsl/capabilities.bdsl` has a TypeScript implementation registered,
  and that every feature declared in `bdsl/features.bdsl` has an
  extractor registered. Either omission is a fatal startup error.
- On `tool_call`: extract features, post a `tool-proposed` sensor
  event, hold the hook open until either an effector signal arrives,
  the post fails, or the per-hook timeout fires. Failing open is the
  fallback — pi never blocks on the daemon.
- On `tool_execution_end`: post a `tool-completed` sensor event.
  Pass 1's BDSL doesn't condition on it; the log accumulates it for
  Pass 2.
- Keep an SSE connection to the daemon's `/signals` endpoint open;
  reconnect with exponential backoff if it drops.

The body never decides anything. "Yes means proceed" is a brain choice
expressed in `decide.bdsl`'s `followup-after-response`; the body waits
for the brain's signal.

## Build

    cd apps/credence-pi/extension
    npm install
    npm run build       # tsc --noEmit (type-check only; tsx runs the source directly)

## Test

    npm test            # node --import tsx --test ../tests/typescript/*.test.ts

The test suite covers manifest parsing, feature extraction edges, the
SSE client (POST timeout, reconnection), per-effector dispatch, and
the full hook flow with mocked pi and daemon.

## Layout

    src/
      index.ts          extension factory; hook registration
      manifest.ts       parses capabilities.bdsl and features.bdsl
      client.ts         SSE consumer + POST sensor
      features/         per-feature extractor; index.ts is the table
      effectors.ts      effector dispatch table
      effectors/        per-effector implementation (ask, proceed, block)
      types.ts          shared types
