// index.ts — credence-pi extension factory.
//
// Wires the body's three concerns together:
//
//   1. Startup verification. Parses capabilities.bdsl and
//      features.bdsl, asserts effector implementations and feature
//      extractors are registered for every declared item, and exits
//      with a hard error otherwise.
//
//   2. tool_call hook. Builds a tool-proposed sensor event (with
//      bucketed features), opens an awaiter keyed by event_id,
//      posts the event to the daemon, and returns a Promise the
//      hook blocks on. The awaiter resolves on:
//
//          (a) effector signal arriving via SSE       — brain decided
//          (b) per-hook timeout (default 30s)         — fail-open
//          (c) postSensor returning {ok: false}       — fail-open
//
//      All three paths remove the entry from the correlation table.
//      Memory leak class bug if any path is missed; index.test.ts
//      exercises each scenario and asserts pendingAwaiterCount() ==
//      0 afterwards.
//
//   3. tool_execution_end hook. Emits a tool-completed sensor event;
//      Pass 1's BDSL doesn't condition on it but the log accumulates
//      it for Pass 2.
//
// Effector dispatch keeps the brain in charge of decisions. When the
// brain emits an `ask` signal, the body invokes `ctx.ui.confirm`,
// then posts a user-responded sensor event, but does NOT resolve the
// hook on the user's reply — the brain's follow-up signal does that.
// The body never short-circuits "yes means proceed".

import { randomUUID } from "node:crypto";

import {
  readCapabilities, readFeatures,
  verifyEffectors, verifyFeatures,
} from "./manifest.js";

import {
  createDaemonClient, type DaemonClient, type SignalEnvelope, type Logger,
} from "./client.js";

import {
  effectors as defaultEffectors,
  type EffectorImpl, type EffectorContext, type HookReturn, type HookAwaiter,
} from "./effectors.js";

import {
  extractors as defaultExtractors, extractFeatures,
  type Extractor,
} from "./features/index.js";

import type { Message, Session, ToolCallEvent } from "./types.js";

// ── pi-shaped surface (minimal subset the body uses) ──────────────────

export interface PiToolCallEvent { toolName: string; input: unknown; }
export interface PiToolEndEvent {
  toolName?: string;
  isError?: boolean;
  durationMs?: number;
  error?: string | null;
}

export interface PiContext {
  session: { sessionId: string };
  ui?: { confirm?: (text: string) => Promise<boolean> };
  cwd?: string;
  projectRoot?: string;
  agent?: { state?: { messages?: Message[] } };
}

export type ToolCallHandler =
  (event: PiToolCallEvent, ctx: PiContext) => Promise<HookReturn> | HookReturn;
export type ToolEndHandler =
  (event: PiToolEndEvent, ctx: PiContext) => Promise<void> | void;

export interface PiAPI {
  on(event: "tool_call", handler: ToolCallHandler): void;
  on(event: "tool_execution_end", handler: ToolEndHandler): void;
}

// ── Configuration & lifecycle ─────────────────────────────────────────

export interface ExtensionDeps {
  capabilitiesPath: string;
  featuresPath: string;
  hookTimeoutMs?: number;
  newEventId?: () => string;
  client?: DaemonClient;
  daemonUrl?: string;
  effectorRegistry?: Record<string, EffectorImpl>;
  extractorRegistry?: Record<string, Extractor>;
  logger?: Logger;
}

export interface InstalledExtension {
  uninstall(): Promise<void>;
  // Test accessor: number of in-flight tool_call awaiters. Asserted
  // to be 0 after each scenario in index.test.ts.
  pendingAwaiterCount(): number;
}

const DEFAULT_HOOK_TIMEOUT_MS = 30_000;
const DEFAULT_DAEMON_URL = "http://127.0.0.1:8787";

export function installCredencePiExtension(
  pi: PiAPI, deps: ExtensionDeps,
): InstalledExtension {
  const log: Logger = deps.logger ??
    ((m, e) => e === undefined ? console.warn(m) : console.warn(m, e));
  const hookTimeoutMs = deps.hookTimeoutMs ?? DEFAULT_HOOK_TIMEOUT_MS;
  const newEventId = deps.newEventId ?? (() => `evt_${randomUUID().slice(0, 12)}`);
  const effectorRegistry = deps.effectorRegistry ?? defaultEffectors;
  const extractorRegistry = deps.extractorRegistry ?? defaultExtractors;

  // 1. Verify manifests at startup. Throws on missing items; the
  //    factory's caller decides whether to exit the host.
  verifyEffectors(readCapabilities(deps.capabilitiesPath), effectorRegistry);
  verifyFeatures(readFeatures(deps.featuresPath),         extractorRegistry);

  // 2. Daemon client.
  const client = deps.client ?? createDaemonClient({
    baseUrl: deps.daemonUrl ?? DEFAULT_DAEMON_URL,
    logger: log,
  });

  // 3. Correlation tables. `awaiters` keys by tool-proposed
  //    event_id; `slims` carries per-hook UI/post-back closures so
  //    the SSE dispatcher can reach pi's ctx.ui without round-
  //    tripping through the awaiter shape.
  type Entry = { awaiter: HookAwaiter; timer: ReturnType<typeof setTimeout> };
  type Slim = {
    confirm: ((t: string) => Promise<boolean>) | undefined;
    postUserResponded: (response: "yes" | "no" | "timeout") => Promise<void>;
  };
  const awaiters = new Map<string, Entry>();
  const slims    = new Map<string, Slim>();
  // PASS-2-NOTE: bounded growth on long-running sessions; consider LRU or per-event-id cleanup if Pass 2 traffic patterns warrant. See apps/credence-pi/PASS-2-NOTES.md.
  const lastEventIdByTool = new Map<string, string>();

  function registerAwaiter(
    eventId: string,
    hookResolve: (r: HookReturn) => void,
  ): Entry {
    let pending = true;
    const finish = (r: HookReturn) => {
      if (!pending) return;
      pending = false;
      const entry = awaiters.get(eventId);
      awaiters.delete(eventId);
      slims.delete(eventId);
      if (entry) clearTimeout(entry.timer);
      hookResolve(r);
    };
    const timer = setTimeout(() => {
      log(`credence-pi: hook timeout (${hookTimeoutMs}ms) for ${eventId}; failing open`);
      finish(undefined);
    }, hookTimeoutMs);
    const awaiter: HookAwaiter = {
      resolve: finish,
      get pending() { return pending; },
    };
    return { awaiter, timer };
  }

  function buildEffectorContext(originatingEventId: string): EffectorContext {
    const s = slims.get(originatingEventId);
    return {
      originatingEventId,
      newEventId,
      warn: log,
      confirm: s?.confirm,
      postUserResponded: s?.postUserResponded ?? (async () => {}),
    };
  }

  // 4. SSE consumer dispatches each signal to the named effector,
  //    looking up the awaiter and per-hook context lazily.
  const sseConn = client.connectSignalsStream((sig: SignalEnvelope) => {
    const entry = awaiters.get(sig.in_response_to);
    if (!entry) return;                   // signal raced past timeout
    const impl = effectorRegistry[sig.effector];
    if (!impl) {
      log(`credence-pi: signal references unknown effector "${sig.effector}"; dropping`);
      return;
    }
    const ctx = buildEffectorContext(sig.in_response_to);
    Promise.resolve()
      .then(() => impl(sig.parameters, entry.awaiter, ctx))
      .catch(err => log(`credence-pi: effector ${sig.effector} threw`, err));
  });

  // 5. Hooks.
  pi.on("tool_call", (event, piCtx) => new Promise<HookReturn>((hookResolve) => {
    const eventId = newEventId();
    const entry = registerAwaiter(eventId, hookResolve);
    awaiters.set(eventId, entry);
    lastEventIdByTool.set(event.toolName, eventId);

    slims.set(eventId, {
      confirm: piCtx.ui?.confirm?.bind(piCtx.ui),
      postUserResponded: async (response) => {
        const result = await client.postSensor({
          event_type:     "user-responded",
          event_id:       newEventId(),
          in_response_to: eventId,
          timestamp:      new Date().toISOString(),
          response,
        });
        if (!result.ok) {
          log(`credence-pi: user-responded post failed for ${eventId}; relying on hook timeout`);
        }
      },
    });

    const session: Session = {
      cwd:         piCtx.cwd ?? "",
      projectRoot: piCtx.projectRoot ?? "",
      messages:    piCtx.agent?.state?.messages ?? [],
    };
    const toolCallEvent: ToolCallEvent = { toolName: event.toolName, input: event.input };
    const features = extractFeatures(toolCallEvent, session);
    const sensorEvent = {
      event_type:    "tool-proposed",
      event_id:      eventId,
      session_id:    piCtx.session.sessionId,
      timestamp:     new Date().toISOString(),
      features,
      proposed_call: { tool_name: event.toolName, input: event.input },
    };

    client.postSensor(sensorEvent).then(result => {
      if (!result.ok && entry.awaiter.pending) {
        entry.awaiter.resolve(undefined);
      }
    });
  }));

  pi.on("tool_execution_end", async (event, _piCtx) => {
    const inResponseTo = event.toolName != null
      ? lastEventIdByTool.get(event.toolName) ?? ""
      : "";
    await client.postSensor({
      event_type:     "tool-completed",
      event_id:       newEventId(),
      in_response_to: inResponseTo,
      timestamp:      new Date().toISOString(),
      outcome: {
        success:        event.isError !== true,
        duration_ms:    event.durationMs ?? null,
        result_summary: null,
        error:          event.error ?? null,
      },
    });
  });

  return {
    uninstall: async () => {
      sseConn.close();
      await sseConn.done;
      // Resolve any in-flight awaiters fail-open so the hook Promises
      // don't leak.
      for (const [, entry] of awaiters) {
        clearTimeout(entry.timer);
        entry.awaiter.resolve(undefined);
      }
      awaiters.clear();
      slims.clear();
      lastEventIdByTool.clear();
    },
    pendingAwaiterCount: () => awaiters.size,
  };
}
