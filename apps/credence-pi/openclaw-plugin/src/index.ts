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

import { randomUUID } from "node:crypto";

import {
  createDaemonClient,
  type DaemonClient,
  type SignalEnvelope,
  type Logger,
} from "./daemon-client.js";
import { FeatureTracker } from "./features.js";
import { buildPriceTable, computeTurnCost } from "./cost.js";
import type {
  PluginEntry,
  BeforeToolCallResult,
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
    const tracker = new FeatureTracker();

    // event_id -> resolver for the awaited effector signal. The single SSE
    // consumer dispatches signals here by in_response_to. Unmatched
    // signals (e.g. an ask-followup after the hook already resolved) find
    // no resolver and are dropped.
    const awaiters = new Map<string, (sig: SignalEnvelope | undefined) => void>();

    const sse = client.connectSignalsStream((sig) => {
      const resolve = awaiters.get(sig.in_response_to);
      if (resolve) resolve(sig);
    });
    void sse; // SSE runs for the plugin's lifetime; OpenClaw owns shutdown.

    let warnedDown = false;
    let announcedUp = false;
    let down = false;

    // 1. Tool-call interception (no allowConversationAccess needed).
    api.on(
      "before_tool_call",
      async (event, ctx): Promise<BeforeToolCallResult | void> => {
        const eventId = newEventId();
        const signalPromise = new Promise<SignalEnvelope | undefined>(
          (resolve) => {
            const timer = setTimeout(() => {
              awaiters.delete(eventId);
              resolve(undefined);
            }, hookTimeoutMs);
            awaiters.set(eventId, (sig) => {
              clearTimeout(timer);
              awaiters.delete(eventId);
              resolve(sig);
            });
          },
        );

        const features = tracker.extractAndRecord(event, ctx);
        const post = await client.postSensor({
          event_type: "tool-proposed",
          event_id: eventId,
          session_id: ctx.sessionId ?? ctx.sessionKey ?? "",
          timestamp: new Date().toISOString(),
          features,
          proposed_call: { tool_name: event.toolName, input: event.params },
        });

        if (!post.ok) {
          const r = awaiters.get(eventId);
          if (r) r(undefined); // clean up timer + awaiter
          if (!warnedDown) {
            log(
              `credence-pi: daemon unreachable at ${daemonUrl}; proceeding without governance`,
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
        }

        const sig = await signalPromise;
        return mapSignal(sig, eventId, client, approvalTimeoutMs);
      },
      { priority: 100, timeoutMs: hookTimeoutMs + 1_000 },
    );

    // 2. Tool outcome (no allowConversationAccess needed). Correlate by
    //    the stable toolCallId (tools run in parallel).
    api.on("after_tool_call", async (event, _ctx) => {
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
    });

    // 3. Per-turn cost (REQUIRES plugins.entries.credence-pi.hooks
    //    .allowConversationAccess: true). Wrapped so a blocked
    //    registration never breaks governance — cost is just absent.
    try {
      api.on("llm_output", async (event, ctx) => {
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
      });
    } catch (err) {
      log(
        "credence-pi: llm_output hook unavailable (set hooks.allowConversationAccess:true for the cost signal)",
        err,
      );
    }
  },
};

export default plugin;
