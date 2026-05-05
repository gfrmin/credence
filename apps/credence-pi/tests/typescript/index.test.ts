// index.test.ts — step 7 of credence-pi: full extension factory flow
// + correlation-table cleanup invariants. The two load-bearing
// claims here:
//
//   1. Cleanup on (a) signal, (b) timeout, (c) daemon-unreachable.
//      The pendingAwaiterCount() must return 0 after each scenario;
//      a leaked entry is a memory-leak class bug. Tested
//      deterministically using a stub DaemonClient so signals can be
//      injected on demand.
//
//   2. ask flow integration. The body must NOT short-circuit
//      "yes means proceed". The test runs a full tool_call cycle
//      with a mock ctx.ui.confirm that returns true, asserts the
//      ask effector ran AND posted user-responded AND that the hook
//      Promise resolved on the followup signal — not on the user's
//      answer alone.

import { test } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  installCredencePiExtension,
  type PiAPI, type PiContext,
} from "../../extension/src/index.js";
import type {
  DaemonClient, SignalEnvelope, SignalsConnection,
} from "../../extension/src/client.js";
import type { HookReturn } from "../../extension/src/effectors.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BDSL_DIR = path.resolve(HERE, "..", "..", "bdsl");
const CAPABILITIES_PATH = path.join(BDSL_DIR, "capabilities.bdsl");
const FEATURES_PATH     = path.join(BDSL_DIR, "features.bdsl");

// ── Test harness ────────────────────────────────────────────────────

interface Posted { type: string; payload: Record<string, unknown>; }

interface Stub {
  client: DaemonClient;
  posts: Posted[];
  push: (sig: SignalEnvelope) => void;
  setPostResult: (ok: boolean) => void;
}

function stubClient(): Stub {
  const posts: Posted[] = [];
  let onSignal: ((sig: SignalEnvelope) => void) | null = null;
  let postOk = true;
  let connClosed = false;
  let connDoneResolve: () => void = () => {};
  const conn: SignalsConnection = {
    close: () => { connClosed = true; connDoneResolve(); },
    done: new Promise<void>((r) => { connDoneResolve = r; }),
  };
  return {
    posts,
    push: (sig) => { onSignal?.(sig); },
    setPostResult: (ok) => { postOk = ok; },
    client: {
      postSensor: async (event) => {
        const e = event as Record<string, unknown>;
        posts.push({ type: String(e["event_type"] ?? "?"), payload: e });
        return { ok: postOk };
      },
      connectSignalsStream: (cb) => {
        if (connClosed) return conn;
        onSignal = cb;
        return conn;
      },
    },
  };
}

interface FakePi {
  pi: PiAPI;
  invokeToolCall: (event: { toolName: string; input: unknown },
                   ctx: PiContext) => Promise<HookReturn>;
  invokeToolEnd: (event: { toolName?: string; isError?: boolean;
                            durationMs?: number; error?: string | null },
                   ctx: PiContext) => Promise<void>;
}

function fakePi(): FakePi {
  let toolCallHandler: any = null;
  let toolEndHandler: any = null;
  return {
    pi: {
      on(event, handler) {
        if (event === "tool_call") toolCallHandler = handler;
        else if (event === "tool_execution_end") toolEndHandler = handler;
      },
    },
    invokeToolCall: (event, ctx) => Promise.resolve(toolCallHandler!(event, ctx)),
    invokeToolEnd: (event, ctx) => Promise.resolve(toolEndHandler!(event, ctx)),
  };
}

const baseCtx = (): PiContext => ({
  session: { sessionId: "sess_test" },
  cwd: "/proj/src",
  projectRoot: "/proj",
  agent: { state: { messages: [] } },
});

let eventIdCounter = 0;
const seqEventId = () => `evt_${++eventIdCounter}`;

// ── 1. Startup verification ─────────────────────────────────────────

test("install: throws when an effector implementation is missing", () => {
  const fp = fakePi();
  const stub = stubClient();
  assert.throws(
    () => installCredencePiExtension(fp.pi, {
      capabilitiesPath: CAPABILITIES_PATH,
      featuresPath:     FEATURES_PATH,
      client:           stub.client,
      effectorRegistry: {
        ask:     () => {},
        proceed: () => {},
        // block deliberately omitted
      } as Record<string, () => void>,
    }),
    { message: /block/ },
  );
});

test("install: throws when a feature extractor is missing", () => {
  const fp = fakePi();
  const stub = stubClient();
  assert.throws(
    () => installCredencePiExtension(fp.pi, {
      capabilitiesPath: CAPABILITIES_PATH,
      featuresPath:     FEATURES_PATH,
      client:           stub.client,
      extractorRegistry: {
        "tool-name":                    () => "bash",
        "working-directory-relative":   () => "subdirectory",
        // parent-tool-call-name deliberately omitted
        "recent-repetition-count":      () => "rep-0",
        "time-since-last-user-message": () => "lt-30s",
      },
    }),
    { message: /parent-tool-call-name/ },
  );
});

test("install: returns a working InstalledExtension under the happy path", async () => {
  const fp = fakePi();
  const stub = stubClient();
  const ext = installCredencePiExtension(fp.pi, {
    capabilitiesPath: CAPABILITIES_PATH,
    featuresPath:     FEATURES_PATH,
    client:           stub.client,
    newEventId:       seqEventId,
    logger:           () => {},
  });
  assert.equal(ext.pendingAwaiterCount(), 0);
  await ext.uninstall();
});

// ── 2. tool_call: signal-resolved cleanup ───────────────────────────

test("tool_call: proceed signal resolves hook with undefined and clears awaiter", async () => {
  const fp = fakePi();
  const stub = stubClient();
  const ext = installCredencePiExtension(fp.pi, {
    capabilitiesPath: CAPABILITIES_PATH,
    featuresPath:     FEATURES_PATH,
    client:           stub.client,
    newEventId:       seqEventId,
    hookTimeoutMs:    1000,
    logger:           () => {},
  });
  try {
    const hook = fp.invokeToolCall(
      { toolName: "bash", input: { command: "ls" } },
      baseCtx(),
    );
    // After registration, exactly one awaiter is pending.
    await new Promise(r => setTimeout(r, 10));
    assert.equal(ext.pendingAwaiterCount(), 1);

    // Find the event_id from the posted tool-proposed event.
    const proposed = stub.posts.find(p => p.type === "tool-proposed");
    assert.ok(proposed);
    const eventId = String(proposed!.payload["event_id"]);

    // Push the brain's signal.
    stub.push({
      signal_type: "effector",
      signal_id: "sig_1",
      in_response_to: eventId,
      effector: "proceed",
      parameters: {},
    });

    const result = await hook;
    assert.equal(result, undefined);
    assert.equal(ext.pendingAwaiterCount(), 0,
      "awaiter must be removed after signal-driven resolution");
  } finally {
    await ext.uninstall();
  }
});

test("tool_call: block signal resolves hook with {block, reason} and clears awaiter", async () => {
  const fp = fakePi();
  const stub = stubClient();
  const ext = installCredencePiExtension(fp.pi, {
    capabilitiesPath: CAPABILITIES_PATH,
    featuresPath:     FEATURES_PATH,
    client:           stub.client,
    newEventId:       seqEventId,
    hookTimeoutMs:    1000,
    logger:           () => {},
  });
  try {
    const hook = fp.invokeToolCall(
      { toolName: "bash", input: { command: "rm -rf /" } },
      baseCtx(),
    );
    await new Promise(r => setTimeout(r, 10));
    const eventId = String(stub.posts.find(p => p.type === "tool-proposed")!.payload["event_id"]);
    stub.push({
      signal_type: "effector", signal_id: "sig", in_response_to: eventId,
      effector: "block", parameters: { reason: "no" },
    });
    const result = await hook;
    assert.deepEqual(result, { block: true, reason: "no" });
    assert.equal(ext.pendingAwaiterCount(), 0);
  } finally {
    await ext.uninstall();
  }
});

// ── 3. tool_call: timeout-resolved cleanup ──────────────────────────

test("tool_call: per-hook timeout resolves fail-open and clears awaiter", async () => {
  const fp = fakePi();
  const stub = stubClient();
  const ext = installCredencePiExtension(fp.pi, {
    capabilitiesPath: CAPABILITIES_PATH,
    featuresPath:     FEATURES_PATH,
    client:           stub.client,
    newEventId:       seqEventId,
    hookTimeoutMs:    100,
    logger:           () => {},
  });
  try {
    const hook = fp.invokeToolCall(
      { toolName: "bash", input: null },
      baseCtx(),
    );
    const result = await hook;
    assert.equal(result, undefined);
    assert.equal(ext.pendingAwaiterCount(), 0,
      "awaiter must be removed after timeout fail-open");
  } finally {
    await ext.uninstall();
  }
});

// ── 4. tool_call: daemon-unreachable cleanup ────────────────────────

test("tool_call: postSensor {ok: false} resolves fail-open and clears awaiter", async () => {
  const fp = fakePi();
  const stub = stubClient();
  stub.setPostResult(false);
  const ext = installCredencePiExtension(fp.pi, {
    capabilitiesPath: CAPABILITIES_PATH,
    featuresPath:     FEATURES_PATH,
    client:           stub.client,
    newEventId:       seqEventId,
    hookTimeoutMs:    5000,                  // long, to prove early fail-open
    logger:           () => {},
  });
  try {
    const t0 = Date.now();
    const hook = fp.invokeToolCall(
      { toolName: "bash", input: null },
      baseCtx(),
    );
    const result = await hook;
    const elapsed = Date.now() - t0;
    assert.equal(result, undefined);
    assert.ok(elapsed < 1000,
      `expected fast fail-open on postSensor !ok; took ${elapsed}ms`);
    assert.equal(ext.pendingAwaiterCount(), 0,
      "awaiter must be removed after daemon-unreachable fail-open");
  } finally {
    await ext.uninstall();
  }
});

// ── 5. ask flow: body waits for the followup, does NOT short-circuit ─

test("ask flow: confirm yes posts user-responded then hook resolves on followup signal", async () => {
  const fp = fakePi();
  const stub = stubClient();
  let confirmCalls = 0;
  const ext = installCredencePiExtension(fp.pi, {
    capabilitiesPath: CAPABILITIES_PATH,
    featuresPath:     FEATURES_PATH,
    client:           stub.client,
    newEventId:       seqEventId,
    hookTimeoutMs:    5000,
    logger:           () => {},
  });
  try {
    const ctx: PiContext = {
      ...baseCtx(),
      ui: { confirm: async (_text) => { confirmCalls++; return true; } },
    };
    const hook = fp.invokeToolCall(
      { toolName: "bash", input: { command: "ls" } },
      ctx,
    );
    await new Promise(r => setTimeout(r, 10));

    // (a) ask signal arrives — body must invoke confirm and post
    //     user-responded but NOT yet resolve the hook.
    const proposed = stub.posts.find(p => p.type === "tool-proposed")!;
    const originatingEventId = String(proposed.payload["event_id"]);
    stub.push({
      signal_type: "effector", signal_id: "sig_ask",
      in_response_to: originatingEventId,
      effector: "ask",
      parameters: { text: "Allow `bash` to run `ls`?" },
    });

    // Wait for ask's async body (confirm + postUserResponded) to flush.
    for (let i = 0; i < 50; i++) {
      if (stub.posts.some(p => p.type === "user-responded")) break;
      await new Promise(r => setTimeout(r, 20));
    }
    assert.equal(confirmCalls, 1, "ctx.ui.confirm must be invoked exactly once");
    const userResponded = stub.posts.find(p => p.type === "user-responded");
    assert.ok(userResponded, "user-responded must be posted after confirm resolves");
    assert.equal(userResponded!.payload["response"], "yes");
    assert.equal(userResponded!.payload["in_response_to"], originatingEventId);

    // (b) Hook is STILL pending — the brain's followup signal hasn't
    //     arrived yet. This is the architectural commitment: the
    //     body cannot short-circuit "yes means proceed".
    assert.equal(ext.pendingAwaiterCount(), 1,
      "hook must still be pending; brain decides the followup");

    // (c) Followup signal arrives — now the hook resolves.
    stub.push({
      signal_type: "effector", signal_id: "sig_followup",
      in_response_to: originatingEventId,
      effector: "proceed",
      parameters: {},
    });
    const result = await hook;
    assert.equal(result, undefined);
    assert.equal(ext.pendingAwaiterCount(), 0);
  } finally {
    await ext.uninstall();
  }
});

// ── 6. tool_execution_end ──────────────────────────────────────────

test("tool_execution_end: posts tool-completed correlated to the most recent tool-proposed", async () => {
  const fp = fakePi();
  const stub = stubClient();
  const ext = installCredencePiExtension(fp.pi, {
    capabilitiesPath: CAPABILITIES_PATH,
    featuresPath:     FEATURES_PATH,
    client:           stub.client,
    newEventId:       seqEventId,
    hookTimeoutMs:    1000,
    logger:           () => {},
  });
  try {
    const hook = fp.invokeToolCall(
      { toolName: "bash", input: null },
      baseCtx(),
    );
    await new Promise(r => setTimeout(r, 10));
    const proposed = stub.posts.find(p => p.type === "tool-proposed")!;
    const originatingEventId = String(proposed.payload["event_id"]);
    stub.push({
      signal_type: "effector", signal_id: "sig", in_response_to: originatingEventId,
      effector: "proceed", parameters: {},
    });
    await hook;

    await fp.invokeToolEnd(
      { toolName: "bash", isError: false, durationMs: 17 },
      baseCtx(),
    );
    const completed = stub.posts.find(p => p.type === "tool-completed");
    assert.ok(completed);
    assert.equal(completed!.payload["in_response_to"], originatingEventId);
    const outcome = completed!.payload["outcome"] as Record<string, unknown>;
    assert.equal(outcome["success"], true);
    assert.equal(outcome["duration_ms"], 17);
  } finally {
    await ext.uninstall();
  }
});

// ── 7. End-to-end feature dict shape ───────────────────────────────

test("tool_call posts a tool-proposed event with kebab-case feature dict", async () => {
  const fp = fakePi();
  const stub = stubClient();
  const ext = installCredencePiExtension(fp.pi, {
    capabilitiesPath: CAPABILITIES_PATH,
    featuresPath:     FEATURES_PATH,
    client:           stub.client,
    newEventId:       seqEventId,
    hookTimeoutMs:    100,
    logger:           () => {},
  });
  try {
    fp.invokeToolCall({ toolName: "bash", input: null }, baseCtx());
    await new Promise(r => setTimeout(r, 30));
    const proposed = stub.posts.find(p => p.type === "tool-proposed")!;
    const features = proposed.payload["features"] as Record<string, string>;
    assert.deepEqual(Object.keys(features).sort(), [
      "parent-tool-call-name",
      "recent-repetition-count",
      "time-since-last-user-message",
      "tool-name",
      "working-directory-relative",
    ]);
    assert.equal(features["tool-name"], "bash");
  } finally {
    await ext.uninstall();
  }
});
