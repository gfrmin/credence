import { test } from "node:test";
import assert from "node:assert/strict";

import { FeatureTracker } from "../src/features.js";
import type { BeforeToolCallEvent, ToolContext } from "../src/openclaw-types.js";

const ctx = (over: Partial<ToolContext> = {}): ToolContext => ({
  runId: "r1",
  sessionId: "s1",
  ...over,
});
const ev = (
  toolName: string,
  over: Partial<BeforeToolCallEvent> = {},
): BeforeToolCallEvent => ({ toolName, params: {}, ...over });

test("tool-name buckets known tools verbatim and unknowns to other", () => {
  const t = new FeatureTracker();
  assert.equal(t.extractAndRecord(ev("bash"), ctx(), 1000)["tool-name"], "bash");
  // OpenClaw's primary shell tool is `exec` (distinct from `bash`); it must
  // bucket to its own first-class category, not collapse into `other`. A
  // failure here is the "dead category" trap: features.bdsl declares `exec`
  // but the body never produces it. (See bdsl/features.bdsl tool-name-space.)
  assert.equal(t.extractAndRecord(ev("exec"), ctx(), 1000)["tool-name"], "exec");
  assert.equal(t.extractAndRecord(ev("Exec"), ctx(), 1000)["tool-name"], "exec");
  // `process` and `apply_patch` are OpenClaw's other native agent tools — also
  // first-class categories, not "other".
  assert.equal(t.extractAndRecord(ev("process"), ctx(), 1000)["tool-name"], "process");
  assert.equal(t.extractAndRecord(ev("apply_patch"), ctx(), 1000)["tool-name"], "apply_patch");
  assert.equal(t.extractAndRecord(ev("WeirdTool"), ctx(), 1000)["tool-name"], "other");
});

test("parent-tool-call-name and recent-repetition-count come from per-run history", () => {
  const t = new FeatureTracker();
  const c = ctx({ runId: "rep" });
  let f = t.extractAndRecord(ev("bash"), c, 1000);
  assert.equal(f["parent-tool-call-name"], "none");
  assert.equal(f["recent-repetition-count"], "rep-0");
  f = t.extractAndRecord(ev("bash"), c, 1001);
  assert.equal(f["parent-tool-call-name"], "bash");
  assert.equal(f["recent-repetition-count"], "rep-1");
  f = t.extractAndRecord(ev("bash"), c, 1002);
  assert.equal(f["recent-repetition-count"], "rep-2");
  f = t.extractAndRecord(ev("bash"), c, 1003);
  assert.equal(f["recent-repetition-count"], "rep-3plus");
});

test("recent-identical-call-count: same args loop vs same-tool different args", () => {
  const t = new FeatureTracker();
  const c = ctx({ runId: "ident" });
  // The SAME exec command repeated → a loop: ident climbs with rep.
  const loop = (n: number) =>
    t.extractAndRecord(ev("exec", { params: { command: "ls" } }), c, n);
  assert.equal(loop(1)["recent-identical-call-count"], "ident-0");
  assert.equal(loop(2)["recent-identical-call-count"], "ident-1");
  assert.equal(loop(3)["recent-identical-call-count"], "ident-2");
  assert.equal(loop(4)["recent-identical-call-count"], "ident-3plus");

  // Same TOOL, DIFFERENT args → not a loop: rep climbs but ident stays low.
  const t2 = new FeatureTracker();
  const c2 = ctx({ runId: "distinct" });
  t2.extractAndRecord(ev("exec", { params: { command: "ls a" } }), c2, 1);
  const f = t2.extractAndRecord(ev("exec", { params: { command: "ls b" } }), c2, 2);
  assert.equal(f["recent-repetition-count"], "rep-1"); // same tool
  assert.equal(f["recent-identical-call-count"], "ident-0"); // different args
});

test("recent-identical-call-count: argument key order doesn't matter", () => {
  const t = new FeatureTracker();
  const c = ctx({ runId: "order" });
  t.extractAndRecord(ev("exec", { params: { a: 1, b: 2 } }), c, 1);
  const f = t.extractAndRecord(ev("exec", { params: { b: 2, a: 1 } }), c, 2);
  assert.equal(f["recent-identical-call-count"], "ident-1");
});

test("recent-identical-call-count: long commands sharing a prefix do NOT collide", () => {
  // Two heredocs writing DIFFERENT scripts but identical in their first ~600
  // chars. A truncated fingerprint would mis-count them as a repeat; the full
  // content hash must keep them distinct (ident-0).
  const t = new FeatureTracker();
  const c = ctx({ runId: "collide" });
  const head = "cat << 'EOF' > f.py\n" + "x = 1  # padding ".repeat(40);
  t.extractAndRecord(ev("exec", { params: { command: head + "\nprint('A')" } }), c, 1);
  const f = t.extractAndRecord(ev("exec", { params: { command: head + "\nprint('B')" } }), c, 2);
  assert.equal(f["recent-identical-call-count"], "ident-0");
});

test("recent-identical-call-count: counted session-wide, not just last 5 calls", () => {
  const t = new FeatureTracker();
  const c = ctx({ runId: "wide" });
  t.extractAndRecord(ev("exec", { params: { command: "make" } }), c, 1);
  // 8 distinct calls in between (would fall outside a 5-call window)
  for (let i = 0; i < 8; i++) {
    t.extractAndRecord(ev("exec", { params: { command: `step-${i}` } }), c, 2 + i);
  }
  const f = t.extractAndRecord(ev("exec", { params: { command: "make" } }), c, 100);
  assert.equal(f["recent-identical-call-count"], "ident-1"); // the distant repeat is seen
});

test("recent-identical-call-count: non-informative args are never a loop", () => {
  const t = new FeatureTracker();
  const c = ctx({ runId: "noninfo" });
  // A lone field echoing the tool name (corpus collapse) carries no evidence.
  t.extractAndRecord(ev("exec", { params: { command: "exec" } }), c, 1);
  const a = t.extractAndRecord(ev("exec", { params: { command: "exec" } }), c, 2);
  assert.equal(a["recent-identical-call-count"], "ident-0");
  // Empty args likewise.
  const t2 = new FeatureTracker();
  const c2 = ctx({ runId: "empty" });
  t2.extractAndRecord(ev("read", { params: {} }), c2, 1);
  const b = t2.extractAndRecord(ev("read", { params: {} }), c2, 2);
  assert.equal(b["recent-identical-call-count"], "ident-0");
});

test("repetition is per-run-isolated", () => {
  const t = new FeatureTracker();
  t.extractAndRecord(ev("bash"), ctx({ runId: "a" }), 1);
  const f = t.extractAndRecord(ev("bash"), ctx({ runId: "b" }), 1);
  assert.equal(f["recent-repetition-count"], "rep-0");
  assert.equal(f["parent-tool-call-name"], "none");
});

test("time-since-last-user-message bucketing (with injected clock)", () => {
  const t = new FeatureTracker();
  const c = ctx({ runId: "time" });
  assert.equal(
    t.extractAndRecord(ev("bash"), c, 10_000)["time-since-last-user-message"],
    "gt-10m",
  );
  t.markUserMessage(c, 10_000);
  const at = (ms: number) =>
    t.extractAndRecord(ev("bash"), c, 10_000 + ms)["time-since-last-user-message"];
  assert.equal(at(10_000), "lt-30s");
  assert.equal(at(60_000), "lt-2m");
  assert.equal(at(300_000), "lt-10m");
  assert.equal(at(1_200_000), "gt-10m");
});

test("working-directory-relative classifies against workspaceDir", () => {
  const t = new FeatureTracker();
  const c = ctx({ runId: "wd", workspaceDir: "/home/u/proj" });
  assert.equal(
    t.extractAndRecord(ev("bash"), c, 1)["working-directory-relative"],
    "no-path",
  );
  assert.equal(
    t.extractAndRecord(ev("edit", { derivedPaths: ["/home/u/proj/src/a.ts"] }), c, 2)[
      "working-directory-relative"
    ],
    "subdirectory",
  );
  assert.equal(
    t.extractAndRecord(ev("read", { derivedPaths: ["/etc/passwd"] }), c, 3)[
      "working-directory-relative"
    ],
    "outside-project",
  );
  assert.equal(
    t.extractAndRecord(ev("ls", { derivedPaths: ["/home/u/proj"] }), c, 4)[
      "working-directory-relative"
    ],
    "project-root",
  );
});
