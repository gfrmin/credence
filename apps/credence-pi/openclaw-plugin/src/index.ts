// index.ts — credence-pi OpenClaw-plugin body.
//
// Governs the pi/OpenClaw agent loop by intercepting tool calls and
// routing the decision to the credence-pi Julia daemon (the opaque
// brain). Reuses credence-pi's async wire UNCHANGED: POST /sensor (a
// tool-proposed sensor event) then await the correlated effector signal
// off the SSE /signals stream; map it to OpenClaw's before_tool_call
// result. Also logs tool outcomes and reconstructed per-turn cost so the
// observation log accumulates the data the dollars-saved surface (Move 2)
// needs.
//
// Discipline (matches the pi-extension body):
//   - The BRAIN decides; the body only translates. ask -> requireApproval
//     (OpenClaw enforces the user's choice natively); the body posts
//     user-responded via onResolution so the brain learns, but does not
//     itself decide proceed/block on the reply.
//   - Fail-open everywhere: daemon unreachable / slow ⇒ the tool proceeds,
//     with one warning per outage.
//
// The orchestration lives in `createGovernor`, separated from `register`
// so it can be unit-tested with an injected DaemonClient (see
// tests/index.test.ts).

import { randomUUID } from "node:crypto";

import {
  createDaemonClient,
  type DaemonClient,
  type SignalEnvelope,
  type Logger,
} from "./daemon-client.js";
import { FeatureTracker } from "./features.js";
import { SafetyTracker } from "./safety.js";
import { buildPriceTable, computeTurnCost, type PriceTable } from "./cost.js";
import type {
  PluginEntry,
  BeforeToolCallEvent,
  BeforeToolCallResult,
  AfterToolCallEvent,
  LlmOutputEvent,
  ToolContext,
  RequireApprovalPayload,
} from "./openclaw-types.js";

const DEFAULT_DAEMON_URL = "http://127.0.0.1:8787";
const DEFAULT_HOOK_TIMEOUT_MS = 3_000;
// How long OpenClaw waits for the human on an `ask` (requireApproval).
// Distinct from the daemon-decision timeout above.
const DEFAULT_APPROVAL_TIMEOUT_MS = 120_000;

function newEventId(): string {
  return `evt_${randomUUID().slice(0, 12)}`;
}

export function mapSignal(
  sig: SignalEnvelope | undefined,
  originatingEventId: string,
  client: DaemonClient,
  approvalTimeoutMs: number,
): BeforeToolCallResult | undefined {
  if (!sig) return undefined; // timeout / fail-open ⇒ proceed
  const p = sig.parameters ?? {};
  switch (sig.effector) {
    case "proceed":
      return undefined;
    case "block":
      return {
        block: true,
        blockReason: `credence-pi: ${
          typeof p.reason === "string"
            ? p.reason
            : "tool call vetoed by expected-utility calculation"
        }`,
      };
    case "ask": {
      const description =
        typeof p.text === "string" ? p.text : "Confirm this tool call?";
      const requireApproval: RequireApprovalPayload = {
        title: "credence-pi governance",
        description,
        severity: "warning",
        timeoutMs: approvalTimeoutMs,
        timeoutBehavior: "deny",
        allowedDecisions: ["allow-once", "allow-always", "deny"],
        onResolution: async (decision) => {
          const response =
            decision === "allow-once" || decision === "allow-always"
              ? "yes"
              : decision === "deny"
                ? "no"
                : "timeout";
          // Fire-and-forget: the brain conditions on the reply; OpenClaw
          // has already enforced the decision. The daemon's follow-up
          // proceed/block signal is unneeded here and harmlessly dropped
          // (no awaiter).
          await client.postSensor({
            event_type: "user-responded",
            event_id: newEventId(),
            in_response_to: originatingEventId,
            timestamp: new Date().toISOString(),
            response,
          });
        },
      };
      return { requireApproval };
    }
    default:
      return undefined; // unknown effector ⇒ fail open
  }
}

export interface GovernorOpts {
  hookTimeoutMs: number;
  approvalTimeoutMs: number;
  priceTable: PriceTable;
  redactToolInputs: boolean;
  /** Observe-only: the brain still decides and the daemon still logs the
   *  decision, but the body NEVER enforces (always proceeds). Lets operators
   *  measure what governance WOULD do on real usage without affecting runs —
   *  the basis for counterfactual telemetry. Default false (enforcing). */
  shadowMode: boolean;
  log: Logger;
}

export interface Governor {
  beforeToolCall: (
    event: BeforeToolCallEvent,
    ctx: ToolContext,
  ) => Promise<BeforeToolCallResult | undefined>;
  afterToolCall: (event: AfterToolCallEvent, ctx: ToolContext) => Promise<void>;
  llmOutput: (event: LlmOutputEvent, ctx: ToolContext) => Promise<void>;
  cleanup: () => void;
  /** Test/inspection accessor: in-flight tool_call awaiters. */
  pendingCount: () => number;
}

// The governance orchestration over an injected DaemonClient. register()
// wires this to the OpenClaw hook API; tests drive it with a fake client.
export function createGovernor(
  client: DaemonClient,
  opts: GovernorOpts,
): Governor {
  const { hookTimeoutMs, approvalTimeoutMs, priceTable, redactToolInputs,
    shadowMode, log } = opts;
  const tracker = new FeatureTracker();
  // Safety (multi-outcome): accumulates taint from tool results and emits the taint-flow
  // features the harm posterior conditions on. The daemon ignores them unless harm
  // governance is enabled (harm-cost>0 + the shipped harm posterior), so emitting them is
  // safe with any daemon version.
  const safety = new SafetyTracker();

  // event_id -> resolver for the awaited effector signal. The single SSE
  // consumer dispatches signals here by in_response_to. Unmatched signals
  // (e.g. an ask-followup after the hook already resolved) find no resolver
  // and are dropped.
  const awaiters = new Map<string, (sig: SignalEnvelope | undefined) => void>();

  const sse = client.connectSignalsStream((sig) => {
    const resolve = awaiters.get(sig.in_response_to);
    if (resolve) resolve(sig);
  });

  let warnedDown = false;
  let announcedUp = false;
  let down = false;

  async function beforeToolCall(
    event: BeforeToolCallEvent,
    ctx: ToolContext,
  ): Promise<BeforeToolCallResult | undefined> {
    const eventId = newEventId();
    const t0 = Date.now();
    const signalPromise = new Promise<SignalEnvelope | undefined>((resolve) => {
      const timer = setTimeout(() => {
        awaiters.delete(eventId);
        resolve(undefined);
      }, hookTimeoutMs);
      awaiters.set(eventId, (sig) => {
        clearTimeout(timer);
        awaiters.delete(eventId);
        resolve(sig);
      });
    });

    const features = tracker.extractAndRecord(event, ctx);
    // Add the safety (taint-flow) features against the session's causal taint state. The
    // brain reads only its declared subsets, so this is additive and backward-compatible.
    Object.assign(features, safety.extractSafety(event.toolName, event.params, ctx));
    const post = await client.postSensor({
      event_type: "tool-proposed",
      event_id: eventId,
      session_id: ctx.sessionId ?? ctx.sessionKey ?? "",
      timestamp: new Date().toISOString(),
      features,
      // Tool inputs can carry secrets (commands, tokens). Operators may
      // redact them; the brain does not condition on input (Move 1), only
      // the daemon's ask-text preview uses it.
      proposed_call: {
        tool_name: event.toolName,
        input: redactToolInputs ? null : event.params,
      },
    });

    if (!post.ok) {
      const r = awaiters.get(eventId);
      if (r) r(undefined); // clean up timer + awaiter
      if (!warnedDown) {
        log(
          `credence-pi: daemon unreachable at the configured URL; proceeding without governance`,
        );
        warnedDown = true;
      }
      down = true;
      announcedUp = false;
      return undefined; // fail open
    }
    if (down && !announcedUp) {
      log("credence-pi: daemon reachable again; governance resumed");
      announcedUp = true;
      down = false;
      warnedDown = false; // re-arm the unreachable warning for the next outage
    }

    const sig = await signalPromise;

    // Governance round-trip overhead (sensor POST + signal wait). Recorded for
    // the honest *time* accounting (governance adds latency). Fire-and-forget;
    // the daemon logs unknown event_types before warning, so it lands in the
    // observation log without affecting the decision.
    const latencyMs = Date.now() - t0;
    void client.postSensor({
      event_type: "governance-latency",
      event_id: newEventId(),
      in_response_to: eventId,
      timestamp: new Date().toISOString(),
      latency_ms: latencyMs,
    });

    // Observe-only: the brain decided and the daemon logged it; the body does
    // not enforce. This is what lets operators measure counterfactual
    // governance on real usage without changing any run.
    if (shadowMode) {
      if (sig && (sig.effector === "block" || sig.effector === "ask")) {
        log(
          `credence-pi[shadow]: would ${sig.effector} \`${event.toolName}\` — proceeding (shadow mode)`,
        );
      }
      return undefined;
    }

    return mapSignal(sig, eventId, client, approvalTimeoutMs);
  }

  async function afterToolCall(
    event: AfterToolCallEvent,
    ctx: ToolContext,
  ): Promise<void> {
    // Taint SOURCE: the tool result is content from the outside world. Accumulate its
    // distinctive tokens + imperative verbs so a later sink carrying them is flagged
    // (causal — only results seen BEFORE a call can taint it).
    safety.observeResult(event.result, ctx);
    // Correlate by the stable toolCallId (tools run in parallel).
    await client.postSensor({
      event_type: "tool-completed",
      event_id: newEventId(),
      in_response_to: event.toolCallId ?? "",
      timestamp: new Date().toISOString(),
      outcome: {
        success: event.error == null,
        duration_ms: event.durationMs ?? null,
        result_summary: null,
        error: event.error ?? null,
      },
    });
  }

  async function llmOutput(
    event: LlmOutputEvent,
    ctx: ToolContext,
  ): Promise<void> {
    const tc = computeTurnCost(event, priceTable);
    await client.postSensor({
      event_type: "turn-cost",
      event_id: newEventId(),
      session_id: ctx.sessionId ?? event.sessionId ?? "",
      timestamp: new Date().toISOString(),
      usd: tc.usd,
      total_tokens: tc.total_tokens,
      input_tokens: tc.input_tokens,
      output_tokens: tc.output_tokens,
      cache_read: tc.cache_read,
      cache_write: tc.cache_write,
      model: tc.model,
    });
  }

  function cleanup(): void {
    sse.close();
    for (const resolve of awaiters.values()) resolve(undefined);
    awaiters.clear();
  }

  return {
    beforeToolCall,
    afterToolCall,
    llmOutput,
    cleanup,
    pendingCount: () => awaiters.size,
  };
}

const plugin: PluginEntry = {
  id: "credence-pi",
  name: "credence-pi governance",
  description:
    "Bayesian in-loop governance for the pi/OpenClaw agent — intercepts tool calls (allow/block/ask) via the credence-pi brain and logs outcomes + per-turn cost.",

  register(api) {
    const cfg = api.pluginConfig ?? {};
    const daemonUrl =
      typeof cfg.daemonUrl === "string" ? cfg.daemonUrl : DEFAULT_DAEMON_URL;
    const hookTimeoutMs =
      typeof cfg.hookTimeoutMs === "number"
        ? cfg.hookTimeoutMs
        : DEFAULT_HOOK_TIMEOUT_MS;
    const approvalTimeoutMs =
      typeof cfg.approvalTimeoutMs === "number"
        ? cfg.approvalTimeoutMs
        : DEFAULT_APPROVAL_TIMEOUT_MS;
    const silent = cfg.silent === true;
    const redactToolInputs = cfg.redactToolInputs === true;
    const shadowMode = cfg.shadowMode === true;
    const priceTable = buildPriceTable(cfg.pricing);

    const log: Logger = (m, e) => {
      if (silent) return;
      const msg = e === undefined ? m : `${m} ${String(e)}`;
      api.logger?.warn?.(msg);
    };

    const client = createDaemonClient({
      baseUrl: daemonUrl,
      timeoutMs: hookTimeoutMs,
      logger: log,
    });
    const gov = createGovernor(client, {
      hookTimeoutMs,
      approvalTimeoutMs,
      priceTable,
      redactToolInputs,
      shadowMode,
      log,
    });
    if (shadowMode) log("credence-pi: SHADOW MODE — observing only, never enforcing");

    api.on("before_tool_call", gov.beforeToolCall, {
      priority: 100,
      timeoutMs: hookTimeoutMs + 1_000,
    });
    api.on("after_tool_call", gov.afterToolCall);

    // Per-turn cost REQUIRES plugins.entries.credence-pi.hooks
    // .allowConversationAccess: true. Wrapped so a blocked registration
    // never breaks governance — cost is just absent.
    try {
      api.on("llm_output", gov.llmOutput);
    } catch (err) {
      log(
        "credence-pi: llm_output hook unavailable (set hooks.allowConversationAccess:true for the cost signal)",
        err,
      );
    }

    // Close the SSE stream + drain awaiters on reset/delete/reload so a
    // hot-reload does not accumulate daemon connections. Optional-chained
    // for hosts predating the lifecycle API.
    try {
      api.lifecycle?.registerRuntimeLifecycle?.({
        id: "credence-pi-governor",
        description:
          "Close the credence-pi daemon SSE stream and drain pending tool-call awaiters.",
        cleanup: () => gov.cleanup(),
      });
    } catch (err) {
      log("credence-pi: could not register lifecycle cleanup", err);
    }
  },
};

export default plugin;
