// openclaw-types.ts — minimal local declarations of the OpenClaw plugin
// API surface this body consumes.
//
// Sourced from OpenClaw (HEAD 7019da8c7b, 2026-06-17):
//   - src/plugins/types.ts                         (OpenClawPluginApi, api.on, api.config)
//   - src/plugins/hook-types.ts                    (before/after_tool_call, llm_output)
//   - src/plugins/hook-before-agent-start.types.ts (before_model_resolve event/result)
//   - src/config/types.models.ts                   (models.providers[*].models[*].cost)
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

// before_model_resolve: runs before model resolution with the prompt only (no session
// messages). Returning a provider/model override switches the model for this turn — the
// seam credence-pi uses to route by EU-max. Needs hooks.allowConversationAccess:true.
export interface BeforeModelResolveEvent {
  prompt?: string;
  /** Attachment metadata (current OpenClaw): kind + mimeType, for future file-aware routing. */
  attachments?: Array<{ kind?: string; mimeType?: string }>;
}

export interface BeforeModelResolveResult {
  providerOverride?: string;
  modelOverride?: string;
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

// Narrow structural view of the OpenClawConfig slice we read to discover the user's model
// roster (api.config). We declare ONLY these fields so the plugin stays dependency-free and
// resilient to config churn; everything else on the real config is ignored.
export interface OpenClawModelDefinitionSlice {
  id: string;
  name?: string;
  input?: string[]; // supported modalities (text/image/…) — for future file-aware routing
  cost?: { input?: number; output?: number }; // USD per million tokens
}
export interface OpenClawConfigSlice {
  models?: {
    providers?: Record<string, { models?: OpenClawModelDefinitionSlice[] }>;
  };
  agents?: {
    // Allow-list of model refs ("provider/id") the agent may use, if configured.
    defaults?: { models?: Record<string, unknown> };
  };
}

export interface OpenClawPluginApi {
  id?: string;
  logger?: PluginLogger;
  // This plugin's resolved config (validated against openclaw.plugin.json).
  pluginConfig?: Record<string, unknown>;

  // The host's resolved OpenClaw config (api-builder.ts attaches it). We read ONLY
  // `models.providers` (the model catalog + pricing) and `agents.defaults.models` (the
  // allow-list) to discover the routing roster — see OpenClawConfigSlice.
  config?: OpenClawConfigSlice;

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
  on(
    hook: "before_model_resolve",
    handler: HookHandler<BeforeModelResolveEvent, BeforeModelResolveResult>,
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
