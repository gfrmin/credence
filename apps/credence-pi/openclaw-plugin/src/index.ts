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
import { extractRoutingFeatures } from "./routing-features.js";
import { buildRoster, type RoutingModel } from "./roster.js";
import { buildPriceTable, computeTurnCost, type PriceTable } from "./cost.js";
import { resolveProfile, type Profile } from "./profile.js";
import type {
  PluginEntry,
  BeforeToolCallEvent,
  BeforeToolCallResult,
  AfterToolCallEvent,
  LlmOutputEvent,
  BeforeModelResolveEvent,
  BeforeModelResolveResult,
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

// Map a `route` effector signal to OpenClaw's model override. A missing/timed-out/
// non-route signal ⇒ undefined ⇒ OpenClaw keeps its configured model (fail open, exactly
// like the daemon being down). In shadow mode we log the would-route and still defer, so
// operators can see routing decisions on real traffic without changing the model used.
export function mapRouteSignal(
  sig: SignalEnvelope | undefined,
  shadowMode: boolean,
  log: Logger,
): BeforeModelResolveResult | undefined {
  if (!sig || sig.effector !== "route") return undefined;
  const p = sig.parameters ?? {};
  const model = typeof p.model === "string" ? p.model : undefined;
  const provider = typeof p.provider === "string" ? p.provider : undefined;
  if (!model) return undefined;
  if (shadowMode) {
    log(
      `credence-pi[shadow]: would route to ${provider ?? "?"}/${model} — keeping OpenClaw's model (shadow mode)`,
    );
    return undefined;
  }
  const result: BeforeModelResolveResult = { modelOverride: model };
  if (provider) result.providerOverride = provider;
  return result;
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
  /** The user's model roster (discovered from api.config), sent in each route-request so the
   *  brain routes over the models OpenClaw actually has. Empty ⇒ routing is not registered. */
  roster: RoutingModel[];
  /** The user's utility profile (cost/time/quality/risk trade-offs), sent in each sensor event
   *  so the daemon applies the user's preferences per-request (no restart). The brain does the
   *  EU-max; this is preference DATA the body ships. */
  profile: Profile;
  log: Logger;
}

export interface Governor {
  beforeToolCall: (
    event: BeforeToolCallEvent,
    ctx: ToolContext,
  ) => Promise<BeforeToolCallResult | undefined>;
  beforeModelResolve: (
    event: BeforeModelResolveEvent,
    ctx: ToolContext,
  ) => Promise<BeforeModelResolveResult | undefined>;
  afterToolCall: (event: AfterToolCallEvent, ctx: ToolContext) => Promise<void>;
  llmOutput: (event: LlmOutputEvent, ctx: ToolContext) => Promise<void>;
  cleanup: () => void;
  /** Test/inspection accessor: in-flight sensor awaiters (tool-call + route). */
  pendingCount: () => number;
}

// The governance orchestration over an injected DaemonClient. register()
// wires this to the OpenClaw hook API; tests drive it with a fake client.
export function createGovernor(
  client: DaemonClient,
  opts: GovernorOpts,
): Governor {
  const { hookTimeoutMs, approvalTimeoutMs, priceTable, redactToolInputs,
    shadowMode, roster, profile, log } = opts;
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

  // Register an awaiter for the correlated effector signal (dispatched by the single SSE
  // consumer on `in_response_to == eventId`) and a fail-open timeout. Shared by
  // beforeToolCall (tool-proposed → proceed/block/ask) and beforeModelResolve
  // (route-request → route): same wire, same correlation, same timeout discipline.
  function awaitSignal(eventId: string): Promise<SignalEnvelope | undefined> {
    return new Promise<SignalEnvelope | undefined>((resolve) => {
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
  }

  async function beforeToolCall(
    event: BeforeToolCallEvent,
    ctx: ToolContext,
  ): Promise<BeforeToolCallResult | undefined> {
    const eventId = newEventId();
    const t0 = Date.now();
    const signalPromise = awaitSignal(eventId);

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
      // The user's utility weights (λ/q/harm/w_time read here) — preference, not learning.
      profile,
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

  // before_model_resolve: route this turn to the EU-max model. Same wire as
  // beforeToolCall — post a route-request, await the correlated `route` signal — but the
  // signal carries {model, provider} the brain chose, mapped to OpenClaw's override.
  // Fail open everywhere (daemon down / slow / routing unconfigured ⇒ undefined ⇒
  // OpenClaw keeps its configured model). The brain decides; the body only translates.
  async function beforeModelResolve(
    event: BeforeModelResolveEvent,
    ctx: ToolContext,
  ): Promise<BeforeModelResolveResult | undefined> {
    const eventId = newEventId();
    const signalPromise = awaitSignal(eventId);

    const features = extractRoutingFeatures(event);
    const post = await client.postSensor({
      event_type: "route-request",
      event_id: eventId,
      session_id: ctx.sessionId ?? ctx.sessionKey ?? "",
      timestamp: new Date().toISOString(),
      features,
      // The user's live model roster (roster-aware routing): the brain routes over THESE
      // models + their costs, not a hardcoded list. Always ≥2 here (register gates on it).
      models: roster,
      // The user's utility weights (reward + w_time read here) — the time/money trade-off.
      profile,
    });

    if (!post.ok) {
      awaiters.get(eventId)?.(undefined); // clean up timer + awaiter
      if (!warnedDown) {
        log("credence-pi: daemon unreachable; routing skipped (OpenClaw keeps its model)");
        warnedDown = true;
      }
      down = true;
      announcedUp = false;
      return undefined; // fail open
    }
    if (down && !announcedUp) {
      log("credence-pi: daemon reachable again; routing resumed");
      announcedUp = true;
      down = false;
      warnedDown = false;
    }

    const sig = await signalPromise;
    return mapRouteSignal(sig, shadowMode, log);
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
      // session_id lets the daemon credit this outcome to the turn's routed model
      // (online routing learning). The governance path correlates by toolCallId, not this.
      session_id: ctx.sessionId ?? ctx.sessionKey ?? "",
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
    beforeModelResolve,
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
    const routingEnabled = cfg.routing !== false; // model routing is ON by default
    const priceTable = buildPriceTable(cfg.pricing);
    // The user's utility profile: a named preset (default `balanced`) + an optional per-dial
    // override, resolved to the full weight vector sent with each sensor event. This is how a
    // user expresses their cost/time/quality/risk trade-off — no daemon restart, no Julia edit.
    const profile = resolveProfile(
      typeof cfg.profile === "string" ? cfg.profile : undefined,
      typeof cfg.profileOverride === "object" && cfg.profileOverride !== null
        ? (cfg.profileOverride as Record<string, unknown>)
        : undefined,
    );

    const log: Logger = (m, e) => {
      if (silent) return;
      const msg = e === undefined ? m : `${m} ${String(e)}`;
      api.logger?.warn?.(msg);
    };

    // Roster-aware routing: discover the user's models (+ costs) from the host config. Guarded —
    // a missing/odd config yields an empty roster ⇒ routing stays inactive (never mis-routes to
    // a hardcoded default). The body reads config (IO); the brain stays config-agnostic.
    let roster: RoutingModel[] = [];
    try {
      roster = buildRoster(api.config, priceTable);
    } catch (err) {
      log("credence-pi: could not read the model roster from config; routing inactive", err);
    }
    // Route only when there are ≥2 priceable models to choose between; else leave OpenClaw's.
    const routingActive = routingEnabled && roster.length >= 2;

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
      roster,
      profile,
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

    // Model routing — ON by default, roster-aware. The body discovered the user's models from
    // config; route only when ≥2 are priceable (else OpenClaw keeps its model — a no-op, so
    // single-model installs and undiscoverable rosters are unaffected). Set `routing: false` to
    // disable. Fails open: daemon down / slow ⇒ OpenClaw's model.
    if (routingActive) {
      try {
        api.on("before_model_resolve", gov.beforeModelResolve, {
          priority: 100,
          timeoutMs: hookTimeoutMs + 1_000,
        });
        log(
          `credence-pi: model routing ACTIVE over ${roster.length} models ` +
            `(${roster.map((m) => m.model).join(", ")})`,
        );
      } catch (err) {
        log(
          "credence-pi: before_model_resolve hook unavailable on this host; routing disabled",
          err,
        );
      }
    } else if (routingEnabled) {
      log(
        `credence-pi: routing enabled but ${roster.length} priceable model(s) discovered ` +
          `(need ≥2); routing inactive — keeping OpenClaw's model`,
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
