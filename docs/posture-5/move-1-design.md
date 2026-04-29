# Move 1 — OpenClaw integration prototype

## 1. Strategic context

Posture 5's master plan (amended 2026-04-28) targets a Bayesian governance
sidecar for agentic harnesses, with OpenClaw as the first integration. The
routing-only positioning was invalidated by Move 0's cache finding and Move 2's
routing-collapse benchmark. The MVP is a working OpenClaw plugin that
demonstrates Bayesian governance of the agent loop.

Move 1 is the architectural deep-dive and integration proof. It answers two
questions: (a) does OpenClaw's plugin surface actually support the interventions
the master plan assumes? (b) does the sidecar IPC path work at acceptable
latency? Move 2 builds the production plugin; Move 3 evaluates it. Move 1's job
is to de-risk both by surfacing architectural surprises before substantial code
commits.

## 2. Scope

1. Document OpenClaw's integration surface (hook system, persistence layer, tool
   catalog, model provider configuration, runtime architecture).
2. Design the sidecar integration architecture (IPC protocol, latency budget,
   persistence strategy, intervention vocabulary, failure modes).
3. Build a minimal prototype demonstrating one end-to-end intervention
   (loop-detection veto on `before_tool_call`) via sidecar IPC.

## 3. Out of scope

- Production-grade plugin (Move 2).
- Full Credence brain wired to sidecar (Move 2).
- Four-intervention vocabulary beyond prototype's veto-on-loop (Move 2).
- Targeted demonstration evaluation (Move 3).
- Multi-harness adapters (post-MVP roadmap).
- Publication track (post-MVP roadmap).
- Changes to Credence's existing Julia substrate.
- Changes to credence-router.

## 4. Definitions

**Sidecar**: A separate process (Julia HTTP server) that the TypeScript plugin
calls via HTTP. The sidecar runs Credence's decision-theoretic machinery; the
plugin is a thin integration shim in OpenClaw's process.

**Intervention**: A governance action the plugin takes on a tool call. Four
types defined in the master plan: veto-and-downgrade, veto-and-halt,
route-to-cheaper-model, escalate-to-user-confirmation. The prototype
demonstrates only veto-and-halt.

## 5. Design decisions

### §5.1 OpenClaw integration surface

OpenClaw (v2026.4.27) is a TypeScript/Node.js pnpm monorepo built on
`@mariozechner/pi-coding-agent`. It runs as a single Node.js process with a
command lane queue serialising agent turns. Plugins run in-process — no
subprocess boundary between plugin code and the agent loop.

#### Hook system

OpenClaw has a typed plugin hook system with three hook categories:

| Category | Execution | Composition | Examples |
|---|---|---|---|
| **Modifying** | Sequential by priority | Results merged per-hook | `before_tool_call`, `before_prompt_build` |
| **Claiming** | Sequential by priority | First `handled: true` wins | `before_agent_reply` |
| **Void** | Parallel, fire-and-forget | None | `after_tool_call`, `before_compaction`, `after_compaction`, `agent_end` |

All typed hooks are async-capable. Handlers return `Promise<Result | void> | Result | void`.

**`before_tool_call`** — the primary governance hook:

```typescript
before_tool_call: (
  event: { toolName: string; params: Record<string, unknown>;
           runId?: string; toolCallId?: string },
  ctx: PluginHookToolContext,
) => Promise<PluginHookBeforeToolCallResult | void>;

type PluginHookBeforeToolCallResult = {
  params?: Record<string, unknown>;    // modified params
  block?: boolean;                     // if true, blocks execution
  blockReason?: string;                // shown to agent
  requireApproval?: {
    title: string;
    description: string;
    severity?: "info" | "warning" | "critical";
    timeoutMs?: number;
    timeoutBehavior?: "allow" | "deny";
    onResolution?: (decision: PluginApprovalResolution) => Promise<void> | void;
  };
};
```

This hook natively supports the two critical interventions:
- **Veto**: return `{ block: true, blockReason: "..." }`.
- **Escalate**: return `{ requireApproval: { ... } }`.

No default timeout on `before_tool_call`. The plugin can set its own via
registration options. Multiple plugins compose sequentially in priority order;
once any plugin returns `block: true`, subsequent handlers are skipped.

**`after_tool_call`** — the observation hook:

```typescript
after_tool_call: (
  event: { toolName: string; params: Record<string, unknown>;
           result?: unknown; error?: string; durationMs?: number },
  ctx: PluginHookToolContext,
) => Promise<void> | void;
```

Void hook, runs in parallel. The plugin observes tool outcomes here and sends
them to the sidecar asynchronously for posterior update. No latency impact on the
agent loop since it's fire-and-forget.

**`before_compaction` / `after_compaction`** — checkpoint hooks:

Both are void/parallel. `before_compaction` receives `messageCount`,
`tokenCount`, `sessionFile`; `after_compaction` adds `compactedCount`. The
plugin uses `before_compaction` to trigger a sidecar checkpoint (persist
posterior to disk before context shrinks).

**`before_prompt_build`** — context injection:

Modifying hook, sequential. Returns optional `systemPrompt`, `prependContext`,
`appendContext`. The plugin can inject governance-relevant context (e.g.,
"the following tools have been vetoed in this session: ...") into the agent's
system prompt. Not used in the prototype; relevant for Move 2.

**`agent_end`** — session cleanup:

Void hook, 30-second timeout. The plugin triggers a final sidecar checkpoint
and session summary here.

**Plugin registration**: Plugins register via `api.registerHook()` or
`api.hooks?.on()` in their `register()` function. Hooks have optional
`priority` (higher = runs first) and `timeoutMs` overrides.

**Plugin loading**: Plugins are discovered from bundled, installed (npm/git/
local), and workspace (`.openclaw/plugins/`) sources. Each plugin declares a
manifest (package.json or manifest.json) with hook directories.

#### Persistence layer

OpenClaw uses a three-tier memory system:

| Tier | Storage | Plugin access |
|---|---|---|
| **Hot** | Session transcript (in-memory) | Read via hook context |
| **Warm** | Daily memory files (`memory/{YYYY-MM-DD}.md`) | Read only |
| **Cold** | `MEMORY.md` (promoted via dreaming system) | Read only |

**Critical constraint: plugins cannot write to MEMORY.md or SQLite.** Both are
system-controlled. MEMORY.md is written only by the dreaming system's
short-term promotion logic. SQLite (with sqlite-vec for embeddings) stores
chunk embeddings and is accessed via read-only connections from plugin code.

**Compaction vulnerability**: Compaction strips `toolResult.details` and
internal runtime context messages. This is the Issue #1084 pattern — confirm
instructions stored in tool result details get dropped during compaction. Any
state the Credence plugin needs to persist across compactions cannot rely on
OpenClaw's persistence layer.

**Implication for design**: The posterior must be managed by the sidecar, not
stored in OpenClaw's persistence. The sidecar owns its own state file, reads
and writes it directly, and the TypeScript plugin is stateless (or holds only
a cached copy for latency). The `before_compaction` hook triggers a sidecar
checkpoint; `agent_end` triggers a final persist. Session restart loads from
the sidecar's state file.

#### Tool catalog

Core tools are statically defined. Plugins register tool factories at load
time but cannot modify the core catalog at runtime. MCP tools are
materialised via the `bundle-mcp` tool during runs. The `before_tool_call`
hook can modify params or block execution but cannot change the tool catalog
itself.

For the governance plugin, the tool catalog is read-only input: the plugin
observes which tools are called and can block individual calls, but cannot
add or remove tools from the catalog. This is sufficient for the four
interventions in the master plan. Veto-and-downgrade (replacing a tool call
with a cheaper alternative) works by blocking the original call and letting
the agent re-plan — the blockReason tells the agent why and suggests the
alternative.

#### Model provider configuration

OpenClaw supports OpenAI-compatible proxy endpoints via config:

```yaml
models:
  providers:
    credence-proxy:
      baseUrl: "http://localhost:8080/v1"
      apiKey: "${CREDENCE_API_KEY}"
      api: "openai-completions"
      models:
        - id: "claude-sonnet-4-6"
          name: "Sonnet via Credence"
          reasoning: true
          input: ["text", "image"]
          contextWindow: 200000
          maxTokens: 8192
          cost: { input: 0.003, output: 0.015 }
```

This means credence-router can be slotted in as a model provider transparently.
The request shape is OpenAI-compatible regardless of which model OpenClaw routes
to the proxy. The `PluginHookBeforeModelResolveEvent` hook also allows plugins
to influence model selection at runtime.

Route-to-cheaper-model intervention can work two ways:
1. **Plugin-side**: `before_tool_call` returns modified params that trigger a
   cheaper model for the next LLM call (indirect, via prompt injection in
   `before_prompt_build`).
2. **Proxy-side**: credence-router receives all requests and routes by cost/
   quality posterior (existing functionality, just needs to be configured as
   the provider).

Option 2 is simpler and reuses existing code. Move 2 should pursue it.

#### Runtime architecture

- Single Node.js process. Plugin hooks execute in the same event loop.
- Command lane queue serialises agent turns (configurable concurrency).
- Subprocess execution only for tool calls (bash commands) and external CLI
  agents, not for the agent loop itself.
- No explicit latency budget documented for `before_tool_call`, but since
  it's synchronous and sequential, latency directly adds to every tool call.

### §5.2 Integration architecture

#### IPC mechanism: HTTP sidecar

The TypeScript plugin calls a Julia HTTP server via localhost HTTP. The sidecar
exposes two endpoints:

| Endpoint | Called from | Purpose |
|---|---|---|
| `POST /evaluate` | `before_tool_call` | Evaluate candidate tool call, return intervention decision |
| `POST /observe` | `after_tool_call` | Report tool outcome for posterior update |

Request/response bodies are JSON. The sidecar is a long-running process started
before or alongside OpenClaw.

**Why HTTP over alternatives:**
- HTTP is the simplest cross-language IPC that both TypeScript and Julia speak
  natively (Julia's `HTTP.jl`, Node's `fetch`).
- Unix domain sockets would reduce latency (~0.1ms vs ~0.5ms per call) but add
  platform-specific complexity for no meaningful gain at this scale.
- stdio/pipe would require managing a child process from the plugin, adding
  lifecycle complexity.
- gRPC/protobuf would add a schema layer with no benefit at the prototype stage.

#### Latency budget

**Target: <100ms per `before_tool_call` round-trip.**

Budget breakdown:
- HTTP round-trip over localhost: ~1ms
- JSON serialization/deserialization: ~1ms
- Julia-side EU calculation: ~5–50ms (depending on posterior size)
- Safety margin: ~48ms

This is acceptable because tool calls themselves take 100ms–10s (bash commands,
file operations, LLM calls). A 100ms governance check on a tool call that takes
1s is 10% overhead; on an LLM call that takes 5s, it's 2%. The latency of
`before_tool_call` is invisible relative to the tool execution itself.

**If the budget is exceeded**: The prototype measures actual round-trip latency
and reports it. If Julia cold-start or GC pauses push latency above 200ms, the
design conversation for Move 2 absorbs the finding. Mitigation options include:
pre-warming the Julia process, pinning the posterior in memory (no per-call disk
reads), or moving the hot path to a compiled Julia system image.

#### Persistence strategy

The sidecar owns all persistent state. The TypeScript plugin is stateless.

| Event | Action |
|---|---|
| **Sidecar start** | Load posterior from `~/.credence/openclaw/{session-id}/state.json`, or initialise fresh priors |
| **`POST /evaluate`** | Read in-memory posterior, compute EU, return decision. No disk I/O. |
| **`POST /observe`** | Update in-memory posterior via `condition`. Optionally flush to disk (configurable cadence). |
| **`before_compaction` hook** | Plugin sends `POST /checkpoint` to sidecar. Sidecar flushes posterior to disk. |
| **`agent_end` hook** | Plugin sends `POST /checkpoint` to sidecar. Final persist. |
| **Sidecar shutdown** | Flush posterior to disk. |

**State file format**: JSON. Contains the posterior parameters (for conjugate
models: alpha/beta per model × category × tool-class), the session ID, and a
schema version for forward compatibility.

**Why not OpenClaw's persistence**: Plugins cannot write to MEMORY.md or SQLite
(§5.1). Even if they could, the compaction vulnerability (Issue #1084) means
state stored in OpenClaw's context would be dropped during compaction. The
sidecar's own filesystem is the only reliable persistence path.

**Cross-session state**: v0.1 uses per-session posteriors only. Each OpenClaw
session starts with fresh priors or loads from the most recent checkpoint for
that session. Cross-session belief sharing (learning from session A to inform
session B) is deferred to post-MVP. The state directory structure
(`~/.credence/openclaw/{session-id}/`) supports this future extension without
schema changes.

#### Intervention vocabulary mapping

The four master-plan interventions map to OpenClaw's hook return values:

| Intervention | Hook | Return value |
|---|---|---|
| **Veto-and-halt** | `before_tool_call` | `{ block: true, blockReason: "Credence: EU of continuing is below idle threshold. The last N tool calls have not improved the outcome. Halting to prevent runaway loop." }` |
| **Veto-and-downgrade** | `before_tool_call` | `{ block: true, blockReason: "Credence: EU favours [alternative tool/approach]. Consider [specific suggestion]." }` |
| **Route-to-cheaper-model** | Model provider config | credence-router configured as OpenAI-compatible provider; routing happens at the proxy level, transparent to the plugin |
| **Escalate-to-user-confirmation** | `before_tool_call` | `{ requireApproval: { title: "Credence governance check", description: "...", severity: "warning", timeoutMs: 60000, timeoutBehavior: "deny" } }` |

All four interventions are natively supported by OpenClaw's hook system. No
workarounds required. This is the strongest finding from Task 1.

#### Failure modes

**Sidecar unavailable**: The plugin fails open. If the sidecar doesn't respond
within 200ms (configurable), the plugin returns `void` from `before_tool_call`
(no intervention, no block). OpenClaw runs without governance. A log message
records the failure. The user sees no protection but also no disruption. This
is the correct default for the prototype; a production deployment conversation
would revisit whether fail-open is acceptable for high-stakes tasks.

**Sidecar crash mid-session**: Same as unavailable. The plugin catches the
connection error and fails open. The posterior for that session is lost unless
a prior checkpoint was written. The `before_compaction` checkpoint strategy
limits data loss to at most the observations since the last compaction.

**Plugin exception**: OpenClaw's hook runner catches plugin exceptions and logs
them. The tool call proceeds as if the plugin wasn't registered. The plugin
should catch its own exceptions to provide better error messages, but the
safety net exists.

**Latency spike**: If the sidecar's response time exceeds the plugin's timeout,
the plugin treats it as unavailable (fail open). The agent proceeds; a log
message records the spike. Persistent spikes would degrade to no-governance
mode, which is detectable via monitoring.

#### Multi-instance state management

v0.1 supports per-session posteriors only. Multiple OpenClaw instances (e.g.,
different terminal sessions, different devices) each have their own sidecar
process and their own posterior. They do not share state.

Cross-session belief sharing is deferred to post-MVP. When it's needed, the
architecture supports it: the sidecar's state directory is a filesystem path;
a shared store (SQLite, Redis, or even a shared filesystem) could replace the
per-session JSON file without changing the IPC protocol.

### §5.3 Prototype design

#### Plugin scope

A minimal OpenClaw plugin (`apps/openclaw-plugin/`) that:
1. Registers on `before_tool_call` and `after_tool_call`.
2. On `before_tool_call`: sends the candidate tool call and recent tool history
   to the sidecar via `POST /evaluate`.
3. If the sidecar returns `{ action: "block" }`, returns
   `{ block: true, blockReason: "..." }` to OpenClaw.
4. On `after_tool_call`: sends the tool outcome to the sidecar via
   `POST /observe` (fire-and-forget, no latency impact).

The only intervention demonstrated is **veto-on-loop**: if the sidecar detects
that the same tool with the same arguments has been called more than N times in
the recent history, it returns a block decision.

#### Sidecar scope

A minimal Julia HTTP server (`apps/credence-governance-sidecar/`) that:
1. Listens on `localhost:3100` (configurable).
2. `POST /evaluate`: receives `{ toolName, params, recentHistory }`, checks for
   repetition in `recentHistory`, returns `{ action: "proceed" }` or
   `{ action: "block", reason: "..." }`.
3. `POST /observe`: receives `{ toolName, params, result, error, durationMs }`,
   appends to in-memory history. No posterior update in the prototype (the
   repetition counter is the "brain" for the prototype; Move 2 wires up the
   actual Credence brain).

The prototype sidecar is deliberately simple. The point is to prove the IPC
path works, not to demonstrate Bayesian governance. Move 2 replaces the
repetition counter with the actual brain.

#### Test scenario

A toy OpenClaw task that, without the plugin, enters a loop:
1. Configure OpenClaw with a task that triggers repeated identical tool calls
   (e.g., a prompt that causes the agent to run the same shell command
   repeatedly looking for a file that doesn't exist).
2. Run without the plugin. Record the number of tool calls, token usage, and
   wall-clock time until the agent gives up or hits a limit.
3. Run with the plugin enabled (N=3 repetition threshold). Record when the
   plugin vetoes, and the resulting tool call count, token usage, and
   wall-clock time.

Document before/after numbers in `docs/posture-5/move-1-prototype-demo.md`.

## 6. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Julia cold-start latency exceeds budget | Medium | `before_tool_call` adds >200ms per call | Pre-warm sidecar; system image; measure in prototype |
| OpenClaw plugin loading rejects third-party plugins | Low | Users can't install the plugin | Test installation path in prototype; report if friction |
| `block: true` produces error states OpenClaw doesn't handle gracefully | Low | Blocked tool calls confuse the agent | Test in prototype; document agent behaviour on block |
| Sidecar process management adds user friction | Medium | Users must start/stop a separate process | Document clearly; consider auto-start in Move 2 |
| OpenClaw's hook API changes between versions | Medium | Plugin breaks on upgrade | Pin to tested OpenClaw version; document compatibility |

## 7. Open design questions

1. **Should the plugin auto-start the sidecar?** The prototype requires the user
   to start the sidecar manually. Move 2 could auto-start it as a child process
   from the plugin's `register()` function. Trade-off: convenience vs. lifecycle
   complexity (graceful shutdown, port conflicts, process orphaning).

2. **Should `blockReason` include a suggested alternative?** When the plugin
   vetoes a tool call, it can include a suggestion in the block reason (e.g.,
   "try reading the file instead of executing it"). The agent may or may not
   follow the suggestion. Empirical evidence from the prototype will inform
   whether suggestions improve agent behaviour or just add noise.

3. **Is `POST /observe` fire-and-forget acceptable?** The prototype sends
   observations asynchronously from `after_tool_call` (void hook, no latency
   impact). If the sidecar is unavailable, the observation is lost. For the
   prototype this is fine; Move 2 may need a small queue or retry.

4. **Should the plugin register on `before_prompt_build`?** Injecting governance
   context into the system prompt (e.g., "these tools have been vetoed") could
   help the agent avoid re-requesting vetoed tools. Not in the prototype; worth
   testing in Move 2.

## 8. Acceptance criteria

### PR 1 (this design doc)

- This file exists at `docs/posture-5/move-1-design.md` and follows Posture 4
  structural conventions.
- OpenClaw integration surface documented with specific file paths and type
  signatures from the OpenClaw codebase.
- Integration architecture specifies IPC mechanism, latency budget, persistence
  strategy, intervention vocabulary mapping, failure modes, multi-instance
  state management.
- Prototype scope named explicitly.
- Risks surfaced with mitigations.
- Superseded routing-era design docs relocated to `docs/posture-5/superseded/`.

### PR 2 (prototype)

- `apps/openclaw-plugin/` contains a working OpenClaw plugin registering on
  `before_tool_call` and `after_tool_call`, calling the sidecar.
- `apps/credence-governance-sidecar/` contains a working Julia HTTP server
  exposing `/evaluate` and `/observe`.
- A documented test scenario demonstrates loop detection: without the plugin
  the agent loops; with the plugin the loop is vetoed after N calls.
- `docs/posture-5/move-1-prototype-demo.md` documents before/after numbers.
- Both directories have `README.md` files explaining local setup.
- CI passes; existing functionality unchanged.

## 9. Move 2 acceptance criteria preview

What Move 2 must deliver beyond the prototype:

- All four interventions (veto-and-halt, veto-and-downgrade,
  route-to-cheaper-model, escalate-to-user-confirmation) implemented and
  tested.
- Actual Credence brain wired to sidecar (posterior over model × tool ×
  task-category, updated via `condition`, decisions via `expect`/`optimise`).
- Persistence across compaction boundaries (checkpoint on `before_compaction`,
  load on session start).
- credence-router configured as OpenClaw model provider for
  route-to-cheaper-model intervention.
- Installable plugin with documented setup instructions.
- Targeting the three documented failure modes: Issue #34574 (exec repetition),
  Issue #1084 (compaction wipes confirm instruction), Issue #65550 (dreaming
  loops).
