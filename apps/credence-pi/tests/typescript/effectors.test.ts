// effectors.test.ts — step 7 of credence-pi: per-effector dispatch
// correctness. Each effector is a pure unit so tests construct an
// awaiter, invoke the effector, and assert the resolution shape.
//
// The `ask` test exercises the architectural commitment: the body
// must NOT short-circuit on a positive `ctx.ui.confirm` answer. The
// effector posts a user-responded sensor event but leaves the
// awaiter pending; the brain's follow-up signal is what resolves
// the originating hook. The test asserts both halves: posting
// happens, awaiter stays pending.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  effectors,
  type EffectorContext,
  type EffectorImpl,
  type HookAwaiter,
  type HookReturn,
} from "../../extension/src/effectors.js";

interface AwaiterCapture {
  awaiter: HookAwaiter;
  resolved: HookReturn | "<unresolved>";
}

function makeAwaiter(): AwaiterCapture {
  let pending = true;
  let resolved: HookReturn | "<unresolved>" = "<unresolved>";
  const awaiter: HookAwaiter = {
    resolve: (r) => { if (!pending) return; pending = false; resolved = r; },
    get pending() { return pending; },
  };
  return {
    awaiter,
    get resolved() { return resolved; },
  } as AwaiterCapture;
}

interface CtxCapture {
  ctx: EffectorContext;
  posted: Array<"yes" | "no" | "timeout">;
  warnings: string[];
}

function makeCtx(opts: {
  confirm?: (text: string) => Promise<boolean>;
  postUserResponded?: (response: "yes" | "no" | "timeout") => Promise<void>;
} = {}): CtxCapture {
  const posted: Array<"yes" | "no" | "timeout"> = [];
  const warnings: string[] = [];
  return {
    ctx: {
      originatingEventId: "evt_x",
      newEventId: () => "evt_synth",
      warn: (msg) => { warnings.push(msg); },
      confirm: opts.confirm,
      postUserResponded: opts.postUserResponded
        ?? (async (r) => { posted.push(r); }),
    },
    posted,
    warnings,
  };
}

// ── proceed ────────────────────────────────────────────────────────

test("proceed: resolves awaiter with undefined", () => {
  const a = makeAwaiter();
  const c = makeCtx();
  effectors.proceed!({}, a.awaiter, c.ctx);
  assert.equal(a.resolved, undefined);
  assert.equal(a.awaiter.pending, false);
});

test("proceed: idempotent if awaiter already resolved", () => {
  const a = makeAwaiter();
  const c = makeCtx();
  effectors.proceed!({}, a.awaiter, c.ctx);
  effectors.proceed!({}, a.awaiter, c.ctx);
  assert.equal(a.resolved, undefined);
});

// ── block ──────────────────────────────────────────────────────────

test("block: resolves awaiter with the brain-supplied reason", () => {
  const a = makeAwaiter();
  const c = makeCtx();
  effectors.block!({ reason: "no shell commands in /tmp" }, a.awaiter, c.ctx);
  assert.deepEqual(a.resolved, { block: true, reason: "no shell commands in /tmp" });
});

test("block: falls back to a default reason when parameters omit it", () => {
  const a = makeAwaiter();
  const c = makeCtx();
  effectors.block!({}, a.awaiter, c.ctx);
  const r = a.resolved as { block: true; reason: string };
  assert.equal(r.block, true);
  assert.ok(typeof r.reason === "string" && r.reason.length > 0);
});

test("block: ignores non-string reason", () => {
  const a = makeAwaiter();
  const c = makeCtx();
  effectors.block!({ reason: 42 as unknown }, a.awaiter, c.ctx);
  const r = a.resolved as { block: true; reason: string };
  assert.equal(typeof r.reason, "string");
  assert.notEqual(r.reason, "42");
});

// ── ask ────────────────────────────────────────────────────────────

test("ask: invokes confirm, posts user-responded `yes`, leaves awaiter pending", async () => {
  const a = makeAwaiter();
  const c = makeCtx({ confirm: async () => true });
  await effectors.ask!({ text: "Allow `bash`?" }, a.awaiter, c.ctx);
  assert.deepEqual(c.posted, ["yes"]);
  assert.equal(a.awaiter.pending, true,
    "ask MUST NOT resolve the hook on a yes answer; the brain's followup signal does that");
  assert.equal(a.resolved, "<unresolved>");
});

test("ask: maps a `false` confirm answer to `no` and still leaves awaiter pending", async () => {
  const a = makeAwaiter();
  const c = makeCtx({ confirm: async () => false });
  await effectors.ask!({ text: "Allow `bash`?" }, a.awaiter, c.ctx);
  assert.deepEqual(c.posted, ["no"]);
  assert.equal(a.awaiter.pending, true);
});

test("ask: missing UI → posts `timeout` and leaves awaiter pending", async () => {
  const a = makeAwaiter();
  const c = makeCtx({ confirm: undefined });
  await effectors.ask!({ text: "Allow?" }, a.awaiter, c.ctx);
  assert.deepEqual(c.posted, ["timeout"]);
  assert.equal(a.awaiter.pending, true);
});

test("ask: confirm rejection logs a warning and reports timeout", async () => {
  const a = makeAwaiter();
  const c = makeCtx({ confirm: async () => { throw new Error("ui dismissed"); } });
  await effectors.ask!({ text: "Allow?" }, a.awaiter, c.ctx);
  assert.deepEqual(c.posted, ["timeout"]);
  assert.ok(c.warnings.some(w => /confirm rejected/i.test(w)));
  assert.equal(a.awaiter.pending, true);
});

test("ask: postUserResponded throwing is logged but not propagated", async () => {
  const a = makeAwaiter();
  const c = makeCtx({
    confirm: async () => true,
    postUserResponded: async () => { throw new Error("daemon down"); },
  });
  await effectors.ask!({ text: "Allow?" }, a.awaiter, c.ctx);
  assert.ok(c.warnings.some(w => /user-responded post failed/i.test(w)));
  assert.equal(a.awaiter.pending, true,
    "even if the user-responded post fails, awaiter stays pending until hook timeout");
});

test("ask: defensive — already-resolved awaiter exits early without posting", async () => {
  const a = makeAwaiter();
  a.awaiter.resolve(undefined);              // simulate timeout race
  const c = makeCtx({ confirm: async () => true });
  await effectors.ask!({ text: "Allow?" }, a.awaiter, c.ctx);
  assert.deepEqual(c.posted, [],
    "ask must short-circuit if the awaiter has already been resolved");
});

// ── registry ──────────────────────────────────────────────────────

test("effectors registry exposes the three kebab-case effector keys", () => {
  assert.deepEqual(Object.keys(effectors).sort(), ["ask", "block", "proceed"]);
  for (const impl of Object.values(effectors)) {
    assert.equal(typeof impl, "function");
  }
});

// Trivial type-level check that EffectorImpl is what we say it is.
const _typeCheck: EffectorImpl = effectors.proceed!;
void _typeCheck;
