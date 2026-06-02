# credence-pi Pass 2 — Move 1 design: OpenClaw-plugin body (MVP-0, deploy-first)

> Per `docs/posture-3/DESIGN-DOC-TEMPLATE.md`. This is the FORCED deploy-first move (the
> observation dataset is empty). It ships the installable product that begins accumulating data
> **and** the cost signal. Brain (daemon + BDSL) is reused; the new code is a body + a small
> daemon cost-ingestion branch.

## Purpose

Make credence-pi a real, installable **OpenClaw plugin** that governs tool calls on the user's own
OpenClaw sessions, talking to the existing credence-pi Julia daemon, and deploy it to accumulate
(a) governance observations and (b) the per-turn dollar-cost signal. The MVP brain stays the Pass-1
single-Beta `ask`/`proceed`/`block` (`voi`-gated) — deliberately simple to ship fast; the
cost-denominated halting utility is Move 2.

## OQ1 — RESOLVED (binding reality)

OpenClaw forked pi's coding-agent in-tree and runs its gateway agent with `noExtensions: true`; a pi
`ExtensionFactory` is inert inside OpenClaw. The supported interception point is an **OpenClaw plugin
`before_tool_call` hook** (`~/git/openclaw/src/plugins/hook-types.ts:516-564`). So MVP-0's body is an
OpenClaw plugin (this move); the existing pi-shaped `apps/credence-pi/extension/` stays as the
**secondary pi-direct body** (untouched here). Two bodies, one brain.

## Architecture — reuse the async wire, don't change it

credence-pi's daemon wire is **POST `/sensor` (ack-only) + GET `/signals` (SSE, correlated by
`event_id`)** — unchanged. `before_tool_call` is async, so the plugin:

1. on `before_tool_call`: build a `tool-proposed` sensor event (features + `toolCallId`), POST to
   `/sensor`, and **await** the correlated effector signal off the SSE stream (with a bounded
   timeout = the hook's `timeoutMs`); map the signal to the hook result.
2. on `after_tool_call`: POST `tool-completed` (`result`/`error`/`durationMs` — all provided by
   OpenClaw here).
3. on `llm_output`: compute per-turn USD via `calculateCost(model, usage)` (from
   `openclaw/plugin-sdk/llm`) and POST a new `turn-cost` sensor event.
4. `ask` → `requireApproval`: the plugin returns `{requireApproval:{…, onResolution}}`; OpenClaw's
   native UI **enforces** the decision; `onResolution(decision)` POSTs `user-responded` so the brain
   **conditions** (learns). The daemon's `followup-after-response` proceed/block signal is redundant
   for OpenClaw (OpenClaw already enforced) and is harmlessly dropped (no awaiter) — see OQ-c.

This reuses `extension/src/client.ts` (SSE+POST, fail-open, reconnect) **verbatim** and the awaiter /
event_id correlation pattern from `extension/src/index.ts` (adapted off the pi hooks onto the
OpenClaw hooks).

### Decision mapping (daemon effector signal → `before_tool_call` result)

| daemon signal `effector` | `before_tool_call` result |
|---|---|
| `proceed` | `undefined` (allow) |
| `block` | `{ block: true, blockReason: <reason> }` |
| `ask` | `{ requireApproval: { title, description, severity:"warning", timeoutMs, timeoutBehavior:"deny", allowedDecisions:["allow-once","allow-always","deny"], onResolution } }` |
| (timeout / daemon unreachable) | `undefined` (fail-open) |

`onResolution(decision)` maps `allow-once`/`allow-always`→`yes`, `deny`→`no`,
`timeout`/`cancelled`→`timeout`, and POSTs `user-responded`.

### Cost reconstruction (the money signal)

Per-turn USD is not handed to plugins. From `llm_output` `{model, usage:{input,output,cacheRead,
cacheWrite,total}}`, the plugin calls `calculateCost(model, usage)` (SDK-exported;
`Model.cost` is $/M tokens) to get USD, and POSTs `turn-cost`. **Requires
`plugins.entries.<id>.hooks.allowConversationAccess: true`** for `llm_output` — documented in the
install steps. If `model` can't be resolved or cost is 0 (custom/OAuth providers), emit token counts
with `usd: null` (Move 2's surface handles the token×price fallback).

### Feature extraction (the 5 declared features, OpenClaw-shaped)

`before_tool_call` event/ctx + a plugin-maintained buffer keyed by `runId`/`sessionKey`
(`api.runContext` or an in-memory ring):

| feature | source |
|---|---|
| `tool-name` | `event.toolName` (bucketed to the declared finite set) |
| `working-directory-relative` | `workspaceDir` captured from an agent-context hook (`before_agent_start`/`llm_output`) or `api.config`, vs the tool's path args |
| `parent-tool-call-name` | last tool in the per-run history buffer |
| `recent-repetition-count` | count of same-tool entries in the recent buffer (the prior-art `recentHistory` pattern) |
| `time-since-last-user-message` | `Date.now()` minus a timestamp stamped on `message_received`/`session_start` |

Correlation key for `tool-completed` is the stable **`toolCallId`** (OpenClaw runs tools in
parallel) — not toolName.

## Files

New body `apps/credence-pi/openclaw-plugin/` (distinct from the legacy sidecar `apps/openclaw-plugin/`):
- `package.json` — `@openclaw/plugin-sdk` dep, `"openclaw"` block (`runtimeExtensions: ["./dist/index.js"]`, `compat`), build emit (tsc → dist).
- `openclaw.plugin.json` — manifest (`id: "credence-pi"`, `activation.onStartup:true`, strict `configSchema`: `daemonUrl`, `hookTimeoutMs`).
- `tsconfig.json`.
- `src/index.ts` — `definePluginEntry({ id, name, register })`; wires the four hooks; awaiter/correlation core (adapted from the pi extension); decision mapping.
- `src/daemon-client.ts` — re-export/adapt `extension/src/client.ts` (SSE+POST). Prefer importing the shared client; if cross-package import is awkward, vendor a copy with a shared-source note.
- `src/features.ts` — buffer-based extractors producing the declared kebab-case values.
- `src/cost.ts` — `llm_output` → `calculateCost` → `turn-cost` payload.
- `README.md` — install/run (`openclaw plugins install -l`, `allowConversationAccess:true`, gateway restart, daemon start).
- `tests/` — mapping + features + cost reconstruction + fail-open (Node test runner, mirroring the pi extension's tests).

Brain (small):
- `apps/credence-pi/daemon/server.jl` — add `elseif event_type == "turn-cost"` (log-only for Move 1; `append_event!` already logs-first, so this just makes intent explicit and suppresses the unknown-type `@warn`). No signal, no posterior change.
- `apps/credence-pi/tests/julia/test_server.jl` (or test_observation_log.jl) — assert `turn-cost` is logged and emits **no** signal.
- `apps/credence-pi/SPEC.md` — document the `turn-cost` sensor event in the wire schema.

## Behaviour preserved

Daemon wire unchanged; Pass-1 brain (`decide-action`/`observe-response`/`followup-after-response`)
unchanged; fail-open everywhere (daemon down ⇒ tool proceeds, one warning); **zero production-side
lint pragmas**; the pi extension body untouched.

## Worked end-to-end example

OpenClaw agent proposes `bash rm -rf build/` → `before_tool_call` fires → plugin extracts features,
POSTs `tool-proposed` (event_id=evt_x, toolCallId=tc_1) → daemon logs it, runs `decide-action` on
the Beta(2,2) posterior; `voi(ask) > EU(proceed)=EU(block)=0` at cold-start → emits `ask` signal
(in_response_to=evt_x) → plugin's SSE awaiter resolves → returns `{requireApproval{…, onResolution}}`
→ OpenClaw shows the approval dialog → user picks `deny` → OpenClaw blocks the tool; `onResolution`
POSTs `user-responded`(no) → daemon conditions posterior toward refuse. Tool didn't run; brain
learned. (Daemon down at step 2 ⇒ plugin fails open, tool proceeds, one warning.)

## Open design questions

- **OQ-a (turn-cost schema):** fields = `{event_type:"turn-cost", event_id, session_id, timestamp, usd:number|null, total_tokens, input_tokens, output_tokens, cache_read, cache_write, model}`. Confirm this is what Move 2's surface needs; keep it generic.
- **OQ-b (shared client):** import `extension/src/client.ts` across the two body packages vs vendor a copy. Lean: shared import if the workspace allows; else vendored copy with a `// shared-source:` note and a test that both stay in sync.
- **OQ-c (redundant followup):** the daemon emits a proceed/block followup on `user-responded` that OpenClaw doesn't need (it enforced via requireApproval). Harmless drop in Move 1; a body-type flag could suppress it later. Acceptable now.
- **OQ-d (cwd reliability):** `workspaceDir` isn't on the tool ctx; capture from an agent-context hook or `api.config`. Confirm the most reliable source at implementation.

## Risk + mitigation

- **Scope/version:** the plugin imports `@openclaw/plugin-sdk` types only (NOT pi packages), so the `@mariozechner`↔`@earendil-works` split doesn't bite here. Pin `compat.minGatewayVersion` to current.
- **`allowConversationAccess`:** without it, `llm_output` is blocked for non-bundled plugins ⇒ no cost signal. Documented as a required config step; degrade gracefully (governance still works; cost just absent) if unset.
- **Hook latency:** the daemon round-trip is in the agent's critical path. Set the hook `timeoutMs` (e.g. 2–5s) so a slow/dead daemon fails open fast.
- **Parallel tools:** correlate by `toolCallId`, never toolName.
- **Prior-art drift:** `after_tool_call.userApproval` no longer exists — use `requireApproval.onResolution`. Replace `api:any` with typed SDK + `definePluginEntry`.

## Verification

- `cd apps/credence-pi/openclaw-plugin && npm install && npm run build && npm test` (TS body).
- `julia apps/credence-pi/tests/julia/test_server.jl` (turn-cost logged, no signal).
- `python3 tools/credence-lint/credence_lint.py --repo-root . check apps/credence-pi/` → zero violations.
- **Manual deploy gate:** `openclaw plugins install -l apps/credence-pi/openclaw-plugin`, set `allowConversationAccess:true`, start the daemon, restart the gateway, run a real session, confirm `~/.credence-pi/observations.jsonl` accumulates `tool-proposed`/`tool-completed`/`turn-cost` and that an `ask` surfaces an approval dialog.
