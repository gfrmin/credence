// openclaw-types.ts — minimal local declarations of the OpenClaw plugin
// API surface this body consumes.
//
// Sourced from OpenClaw (HEAD 36a596aa9f, 2026-06-02):
//   - src/plugins/types.ts          (OpenClawPluginApi, api.on)
//   - src/plugins/hook-types.ts     (before_tool_call / after_tool_call /
//                                     llm_output events + results)
//
// We declare ONLY what we consume so the plugin builds standalone with no
// @openclaw/openclaw dependency — mirroring the dependency-free pattern of
// apps/openclaw-plugin/ (which used `api: any`). At runtime OpenClaw calls
// register(api) with the real, fully-typed api; these declarations are our
// compile-time view. If the host surface drifts, the runtime still works;
// only this file needs occasional resync. Keep it narrow.

export type PluginApprovalResolution =
  | "allow-once"
  | "allow-always"
  | "deny"
  | "timeout"
  | "cancelled";

export interface RequireApprovalPayload {
  title: string;
  description: string;
  severity?: "info" | "warning" | "critical";
  timeoutMs?: number;
  timeoutBehavior?: "allow" | "deny";
  allowedDecisions?: Array<"allow-once" | "allow-always" | "deny">;
  pluginId?: string;
  onResolution?: (decision: PluginApprovalResolution) => Promise<void> | void;
}

export interface BeforeToolCallEvent {
  toolName: string;
  params: Record<string, unknown>;
  toolKind?: string;
  toolInputKind?: string;
  runId?: string;
  toolCallId?: string;
  derivedPaths?: readonly string[];
}

export interface BeforeToolCallResult {
  params?: Record<string, unknown>;
  block?: boolean;
  blockReason?: string;
  requireApproval?: RequireApprovalPayload;
}

export interface AfterToolCallEvent {
  toolName: string;
  params: Record<string, unknown>;
  runId?: string;
  toolCallId?: string;
  result?: unknown;
  error?: string;
  durationMs?: number;
}

export interface LlmUsage {
  input?: number;
  output?: number;
  cacheRead?: number;
  cacheWrite?: number;
  total?: number;
}

export interface LlmOutputEvent {
  runId?: string;
  sessionId?: string;
  provider?: string;
  model?: string;
  resolvedRef?: string;
  usage?: LlmUsage;
}

export interface ToolContext {
  agentId?: string;
  sessionKey?: string;
  sessionId?: string;
  runId?: string;
  toolName?: string;
  toolCallId?: string;
  channelId?: string;
  workspaceDir?: string;
}

export interface PluginLogger {
  info?: (msg: string) => void;
  warn?: (msg: string) => void;
  error?: (msg: string) => void;
}

export type HookHandler<E, R = void> = (
  event: E,
  ctx: ToolContext,
) => Promise<R | void> | R | void;

export interface HookOpts {
  priority?: number;
  timeoutMs?: number;
}

export interface OpenClawPluginApi {
  id?: string;
  logger?: PluginLogger;
  // This plugin's resolved config (validated against openclaw.plugin.json).
  pluginConfig?: Record<string, unknown>;

  // Cleanup hooks run on reset/delete/reload paths
  // (OpenClaw src/plugins/host-hooks.ts PluginRuntimeLifecycleRegistration).
  // Optional so the plugin still loads on hosts that predate it.
  lifecycle?: {
    registerRuntimeLifecycle: (registration: {
      id: string;
      description?: string;
      cleanup?: (ctx: {
        reason?: string;
        sessionKey?: string;
        runId?: string;
      }) => void | Promise<void>;
    }) => void;
  };

  on(
    hook: "before_tool_call",
    handler: HookHandler<BeforeToolCallEvent, BeforeToolCallResult>,
    opts?: HookOpts,
  ): void;
  on(
    hook: "after_tool_call",
    handler: HookHandler<AfterToolCallEvent, void>,
    opts?: HookOpts,
  ): void;
  on(
    hook: "llm_output",
    handler: HookHandler<LlmOutputEvent, void>,
    opts?: HookOpts,
  ): void;
  // Catch-all for hooks we register opportunistically (e.g. agent_end);
  // typed loosely on purpose.
  on(
    hook: string,
    handler: (...args: unknown[]) => unknown,
    opts?: HookOpts,
  ): void;
}

// The default export shape OpenClaw loads. apps/openclaw-plugin/ exported a
// bare object literal of this shape (works via compat); we do the same to
// stay dependency-free (no `definePluginEntry` import from the host).
export interface PluginEntry {
  id: string;
  name: string;
  description?: string;
  register: (api: OpenClawPluginApi) => void | Promise<void>;
}
