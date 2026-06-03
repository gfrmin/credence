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
