import { test } from "node:test";
import assert from "node:assert/strict";

import { createGovernor } from "../src/index.js";
import { buildPriceTable } from "../src/cost.js";
import { resolveProfile } from "../src/profile.js";
import type { DaemonClient, SignalEnvelope } from "../src/daemon-client.js";
import type { BeforeToolCallEvent, ToolContext } from "../src/openclaw-types.js";

// Let queued microtasks + the post round-trip settle.
const flush = () => new Promise((r) => setTimeout(r, 5));

type Posted = Record<string, unknown>;

function harness(postOk = true, shadowMode = false) {
  const posted: Posted[] = [];
  let onSignal: ((s: SignalEnvelope) => void) | undefined;
  let closed = false;
  const client: DaemonClient = {
    postSensor: async (e) => {
      posted.push(e as Posted);
      return { ok: postOk };
    },
    connectSignalsStream: (cb) => {
      onSignal = cb;
      return {
        close: () => {
          closed = true;
        },
        done: Promise.resolve(),
      };
    },
  };
  const gov = createGovernor(client, {
    hookTimeoutMs: 50,
    approvalTimeoutMs: 5_000,
    priceTable: buildPriceTable(undefined),
    redactToolInputs: false,
    shadowMode,
    roster: [],
    profile: resolveProfile("balanced", undefined),
    log: () => {},
  });
  const find = (t: string) => posted.find((e) => e.event_type === t);
  return {
    gov,
    posted,
    find,
    closed: () => closed,
    signal: (effector: string, eventId: string, parameters: Record<string, unknown> = {}) =>
      onSignal?.({
        signal_type: "effector",
        signal_id: "s",
        in_response_to: eventId,
        effector,
        parameters,
      }),
    lastProposedId: () =>
      [...posted].reverse().find((e) => e.event_type === "tool-proposed")
        ?.event_id as string,
    lastRouteId: () =>
      [...posted].reverse().find((e) => e.event_type === "route-request")
        ?.event_id as string,
  };
}

const ev = (
  toolName = "bash",
  over: Partial<BeforeToolCallEvent> = {},
): BeforeToolCallEvent => ({ toolName, params: { command: "x" }, ...over });
const ctx: ToolContext = { runId: "r1", sessionId: "s1" };

test("before_tool_call: proceed → allow (undefined), no awaiter leak", async () => {
  const h = harness();
  const p = h.gov.beforeToolCall(ev(), ctx);
  await flush();
  assert.equal(h.gov.pendingCount(), 1);
  h.signal("proceed", h.lastProposedId());
  assert.equal(await p, undefined);
  assert.equal(h.gov.pendingCount(), 0);
  h.gov.cleanup();
});

test("before_tool_call: block → {block, blockReason}", async () => {
  const h = harness();
  const p = h.gov.beforeToolCall(ev(), ctx);
  await flush();
  h.signal("block", h.lastProposedId(), { reason: "runaway loop" });
  const r = await p;
  assert.equal(r?.block, true);
  assert.match(r?.blockReason ?? "", /runaway loop/);
  h.gov.cleanup();
});

test("before_tool_call: ask → requireApproval; onResolution posts user-responded", async () => {
  const h = harness();
  const p = h.gov.beforeToolCall(ev(), ctx);
  await flush();
  const eid = h.lastProposedId();
  h.signal("ask", eid, { text: "Allow?" });
  const r = await p;
  assert.ok(r?.requireApproval);
  await r.requireApproval.onResolution?.("deny");
  const ur = h.find("user-responded");
  assert.equal(ur?.response, "no");
  assert.equal(ur?.in_response_to, eid);
  h.gov.cleanup();
});

test("before_tool_call: daemon timeout → fail-open, awaiter cleaned up", async () => {
  const h = harness();
  const r = await h.gov.beforeToolCall(ev(), ctx); // no signal; times out at 50ms
  assert.equal(r, undefined);
  assert.equal(h.gov.pendingCount(), 0);
  h.gov.cleanup();
});

test("before_tool_call: postSensor failure → fail-open immediately, no leak", async () => {
  const h = harness(false);
  const r = await h.gov.beforeToolCall(ev(), ctx);
  assert.equal(r, undefined);
  assert.equal(h.gov.pendingCount(), 0);
  h.gov.cleanup();
});

test("redactToolInputs: tool input omitted from the sensor event", async () => {
  const posted: Posted[] = [];
  const client: DaemonClient = {
    postSensor: async (e) => {
      posted.push(e as Posted);
      return { ok: true };
    },
    connectSignalsStream: () => ({ close: () => {}, done: Promise.resolve() }),
  };
  const gov = createGovernor(client, {
    hookTimeoutMs: 20,
    approvalTimeoutMs: 5_000,
    priceTable: buildPriceTable(undefined),
    redactToolInputs: true,
    shadowMode: false,
    roster: [],
    profile: resolveProfile("balanced", undefined),
    log: () => {},
  });
  await gov.beforeToolCall(ev("bash", { params: { command: "secret-token-123" } }), ctx);
  const tp = posted.find((e) => e.event_type === "tool-proposed") as
    | { proposed_call: { tool_name: string; input: unknown } }
    | undefined;
  assert.equal(tp?.proposed_call.input, null);
  assert.equal(tp?.proposed_call.tool_name, "bash");
  gov.cleanup();
});

test("shadowMode: block signal → proceeds (undefined), but still sensed + latency logged", async () => {
  const h = harness(true, true); // postOk, shadowMode
  const p = h.gov.beforeToolCall(ev(), ctx);
  await flush();
  h.signal("block", h.lastProposedId());
  // Brain said block, but shadow mode never enforces → tool proceeds.
  assert.equal(await p, undefined);
  // Governance still ran: the proposal was sensed (daemon logs the decision)…
  assert.ok(h.find("tool-proposed"), "tool-proposed must still be posted in shadow mode");
  // …and the round-trip latency was recorded.
  assert.ok(h.find("governance-latency"), "governance-latency must be posted");
  h.gov.cleanup();
});

test("shadowMode: block → posts counterfactual-decision (effector/tool/in_response_to) and proceeds", async () => {
  const h = harness(true, true); // postOk, shadowMode
  const p = h.gov.beforeToolCall(ev("bash"), ctx);
  await flush();
  const eid = h.lastProposedId();
  h.signal("block", eid);
  assert.equal(await p, undefined); // shadow never enforces
  const cf = h.find("counterfactual-decision") as
    | { effector: string; tool_name: string; in_response_to: string }
    | undefined;
  assert.ok(cf, "a block in shadow mode must record a counterfactual-decision");
  assert.equal(cf?.effector, "block");
  assert.equal(cf?.tool_name, "bash");
  assert.equal(cf?.in_response_to, eid);
  h.gov.cleanup();
});

test("shadowMode: ask → counterfactual-decision with effector=ask", async () => {
  const h = harness(true, true);
  const p = h.gov.beforeToolCall(ev("bash"), ctx);
  await flush();
  h.signal("ask", h.lastProposedId(), { text: "Allow?" });
  assert.equal(await p, undefined);
  const cf = h.find("counterfactual-decision") as { effector: string } | undefined;
  assert.equal(cf?.effector, "ask");
  h.gov.cleanup();
});

test("shadowMode: proceed → no counterfactual-decision (nothing would have been enforced)", async () => {
  const h = harness(true, true);
  const p = h.gov.beforeToolCall(ev("bash"), ctx);
  await flush();
  h.signal("proceed", h.lastProposedId());
  assert.equal(await p, undefined);
  assert.equal(h.find("counterfactual-decision"), undefined);
  h.gov.cleanup();
});

test("after_tool_call: posts tool-completed correlated by toolCallId", async () => {
  const h = harness();
  await h.gov.afterToolCall({ toolName: "bash", toolCallId: "tc1", params: {}, durationMs: 12 }, ctx);
  const tc = h.find("tool-completed") as
    | { in_response_to: string; session_id: string; outcome: { success: boolean; duration_ms: number } }
    | undefined;
  assert.equal(tc?.in_response_to, "tc1");
  // session_id lets the daemon credit this outcome to the turn's routed model (online routing).
  assert.equal(tc?.session_id, "s1");
  assert.equal(tc?.outcome.success, true);
  assert.equal(tc?.outcome.duration_ms, 12);
  h.gov.cleanup();
});

test("llm_output: posts turn-cost with reconstructed USD", async () => {
  const h = harness();
  await h.gov.llmOutput(
    { model: "claude-opus-4-8", usage: { input: 1_000_000, output: 1_000_000 } },
    ctx,
  );
  const t = h.find("turn-cost") as { usd: number; total_tokens: number } | undefined;
  // opus $5 in + $25 out per Mtok (verified 2026-06): 1M·5 + 1M·25 = $30 (was $90 at the
  // retired 15/75 pricing — corrected in cost.ts).
  assert.equal(t?.usd, 30);
  assert.equal(t?.total_tokens, 2_000_000);
  h.gov.cleanup();
});

test("before_model_resolve: route signal → model + provider override; feature sensed", async () => {
  const h = harness();
  const p = h.gov.beforeModelResolve({ prompt: "a short question" }, ctx);
  await flush();
  assert.equal(h.gov.pendingCount(), 1);
  h.signal("route", h.lastRouteId(), {
    model: "claude-haiku-4-5",
    provider: "anthropic",
    name: "haiku",
  });
  const r = await p;
  assert.equal(r?.modelOverride, "claude-haiku-4-5");
  assert.equal(r?.providerOverride, "anthropic");
  assert.equal(h.gov.pendingCount(), 0);
  const rr = h.find("route-request") as { features: Record<string, string> } | undefined;
  assert.equal(rr?.features["prompt-length"], "short");
  h.gov.cleanup();
});

test("before_model_resolve: long prompt → prompt-length=long feature", async () => {
  const h = harness();
  const p = h.gov.beforeModelResolve({ prompt: "x".repeat(1001) }, ctx);
  await flush();
  h.signal("route", h.lastRouteId(), { model: "claude-opus-4-8", provider: "anthropic" });
  await p;
  const rr = h.find("route-request") as { features: Record<string, string> } | undefined;
  assert.equal(rr?.features["prompt-length"], "long");
  h.gov.cleanup();
});

test("before_model_resolve: timeout → undefined (OpenClaw default), no leak", async () => {
  const h = harness();
  const r = await h.gov.beforeModelResolve({ prompt: "x" }, ctx); // no signal; 50ms timeout
  assert.equal(r, undefined);
  assert.equal(h.gov.pendingCount(), 0);
  h.gov.cleanup();
});

test("before_model_resolve: daemon down → undefined immediately, no leak", async () => {
  const h = harness(false);
  const r = await h.gov.beforeModelResolve({ prompt: "x" }, ctx);
  assert.equal(r, undefined);
  assert.equal(h.gov.pendingCount(), 0);
  h.gov.cleanup();
});

test("before_model_resolve: shadow mode → undefined (keeps model) but route-request sensed", async () => {
  const h = harness(true, true); // postOk, shadowMode
  const p = h.gov.beforeModelResolve({ prompt: "x" }, ctx);
  await flush();
  h.signal("route", h.lastRouteId(), {
    model: "claude-opus-4-8",
    provider: "anthropic",
    name: "opus",
  });
  assert.equal(await p, undefined); // brain routed, but shadow never overrides
  assert.ok(h.find("route-request"), "route-request must still be posted in shadow mode");
  h.gov.cleanup();
});

test("before_model_resolve: route signal with no model → undefined (fail open)", async () => {
  const h = harness();
  const p = h.gov.beforeModelResolve({ prompt: "x" }, ctx);
  await flush();
  h.signal("route", h.lastRouteId(), {}); // route effector but empty params
  assert.equal(await p, undefined);
  assert.equal(h.gov.pendingCount(), 0);
  h.gov.cleanup();
});

test("cleanup: closes the SSE stream", () => {
  const h = harness();
  assert.equal(h.closed(), false);
  h.gov.cleanup();
  assert.equal(h.closed(), true);
});
