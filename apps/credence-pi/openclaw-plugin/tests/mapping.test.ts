import { test } from "node:test";
import assert from "node:assert/strict";

import { mapSignal } from "../src/index.js";
import type { DaemonClient, SignalEnvelope } from "../src/daemon-client.js";

function fakeClient(posted: Record<string, unknown>[]): DaemonClient {
  return {
    postSensor: async (e) => {
      posted.push(e as Record<string, unknown>);
      return { ok: true };
    },
    connectSignalsStream: () => ({ close: () => {}, done: Promise.resolve() }),
  };
}

const sig = (
  effector: string,
  parameters: Record<string, unknown> = {},
): SignalEnvelope => ({
  signal_type: "effector",
  signal_id: "s1",
  in_response_to: "evt_1",
  effector,
  parameters,
});

test("proceed signal → undefined (allow)", () => {
  assert.equal(mapSignal(sig("proceed"), "evt_1", fakeClient([]), 1000), undefined);
});

test("no signal (daemon timeout) → undefined (fail-open)", () => {
  assert.equal(mapSignal(undefined, "evt_1", fakeClient([]), 1000), undefined);
});

test("unknown effector → undefined (fail-open)", () => {
  assert.equal(mapSignal(sig("substitute"), "evt_1", fakeClient([]), 1000), undefined);
});

test("block signal → {block:true, blockReason carrying the reason}", () => {
  const r = mapSignal(sig("block", { reason: "runaway loop" }), "evt_1", fakeClient([]), 1000);
  assert.equal(r?.block, true);
  assert.match(r?.blockReason ?? "", /runaway loop/);
});

test("ask signal → requireApproval, and onResolution posts mapped user-responded", async () => {
  const posted: Record<string, unknown>[] = [];
  const r = mapSignal(sig("ask", { text: "Allow bash rm -rf?" }), "evt_1", fakeClient(posted), 5000);
  assert.ok(r?.requireApproval, "ask must yield requireApproval");
  assert.equal(r.requireApproval.description, "Allow bash rm -rf?");
  assert.equal(r.requireApproval.timeoutBehavior, "deny");
  assert.equal(r.requireApproval.timeoutMs, 5000);
  assert.deepEqual(r.requireApproval.allowedDecisions, ["allow-once", "allow-always", "deny"]);

  await r.requireApproval.onResolution?.("allow-once");
  await r.requireApproval.onResolution?.("allow-always");
  await r.requireApproval.onResolution?.("deny");
  await r.requireApproval.onResolution?.("timeout");

  assert.equal(posted.length, 4);
  assert.equal(posted[0].event_type, "user-responded");
  assert.equal(posted[0].in_response_to, "evt_1");
  assert.deepEqual(
    posted.map((p) => p.response),
    ["yes", "yes", "no", "timeout"],
  );
});
