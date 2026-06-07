// features.test.ts — step 6 of credence-pi: feature extractor unit
// tests + the load-bearing subset assertion that every extractor's
// declared POSSIBLE_OUTPUTS is a subset of the corresponding
// features.bdsl space members. Anything outside the declared space
// would be a wire/brain mismatch — a startup-time bug surfaced as a
// test-time bug.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  parseSExprs,
  readFeatures,
  type SExpr,
  type FeatureDecl,
} from "../../extension/src/manifest.js";

import {
  extractors,
  extractFeatures,
} from "../../extension/src/features/index.js";
import { extractToolName, POSSIBLE_OUTPUTS as toolNameOutputs }
  from "../../extension/src/features/tool_name.js";
import { extractWorkingDirectoryRelative,
         POSSIBLE_OUTPUTS as wdOutputs }
  from "../../extension/src/features/working_directory.js";
import { extractParentToolCallName,
         POSSIBLE_OUTPUTS as parentOutputs }
  from "../../extension/src/features/parent_tool.js";
import { extractRecentRepetitionCount,
         POSSIBLE_OUTPUTS as repOutputs }
  from "../../extension/src/features/repetition.js";
import { extractRecentIdenticalCallCount,
         POSSIBLE_OUTPUTS as identOutputs }
  from "../../extension/src/features/identical_call.js";
import { extractTimeSinceLastUserMessage,
         POSSIBLE_OUTPUTS as timeOutputs }
  from "../../extension/src/features/time_since_user.js";

import type { Session, Message } from "../../extension/src/types.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FEATURES_PATH = path.resolve(HERE, "..", "..", "bdsl", "features.bdsl");
const FEATURES_SRC = readFileSync(FEATURES_PATH, "utf-8");

// Walk a parsed features.bdsl tree to find `(define <spaceName>
// (space :finite m1 m2 ...))` and return [m1, m2, ...]. The tokeniser
// preserves identifiers verbatim so member symbols are returned as
// kebab-case strings ("rep-3plus", "lt-30s", …).
function spaceMembers(spaceName: string): string[] {
  const tree = parseSExprs(FEATURES_SRC);
  for (const expr of tree) {
    if (!Array.isArray(expr) || expr[0] !== "define") continue;
    if (expr[1] !== spaceName) continue;
    const body = expr[2];
    if (!Array.isArray(body) || body[0] !== "space" || body[1] !== ":finite") {
      throw new Error(`define ${spaceName}: expected (space :finite …)`);
    }
    return body.slice(2).map((m: SExpr) => {
      if (typeof m !== "string") throw new Error(`non-symbol space member in ${spaceName}`);
      return m;
    });
  }
  throw new Error(`space ${spaceName} not found in features.bdsl`);
}

function spaceForFeature(featureName: string): string {
  const decls: FeatureDecl[] = readFeatures(FEATURES_PATH);
  const decl = decls.find(d => d.name === featureName);
  if (!decl) throw new Error(`feature ${featureName} not declared in features.bdsl`);
  return decl.spaceName;
}

const blankSession = (): Session => ({ cwd: "", projectRoot: "", messages: [] });

// ── 1. Subset assertion (the load-bearing claim) ───────────────────

test("every extractor's POSSIBLE_OUTPUTS is a subset of its space's members", () => {
  const cases: Array<[string, readonly string[]]> = [
    ["tool-name",                    toolNameOutputs],
    ["working-directory-relative",   wdOutputs],
    ["parent-tool-call-name",        parentOutputs],
    ["recent-repetition-count",      repOutputs],
    ["recent-identical-call-count",  identOutputs],
    ["time-since-last-user-message", timeOutputs],
  ];
  for (const [featureName, outputs] of cases) {
    const members = spaceMembers(spaceForFeature(featureName));
    const memberSet = new Set(members);
    for (const v of outputs) {
      assert.ok(
        memberSet.has(v),
        `extractor for ${featureName} can return ${JSON.stringify(v)}, ` +
          `which is not in space members ${JSON.stringify(members)}`,
      );
    }
  }
});

// ── 2. Per-extractor bucketing edges ───────────────────────────────

test("extractToolName: known lowercased; unknown → other", () => {
  const s = blankSession();
  assert.equal(extractToolName({ toolName: "Bash", input: null }, s), "bash");
  assert.equal(extractToolName({ toolName: "READ", input: null }, s), "read");
  assert.equal(extractToolName({ toolName: "ls",   input: null }, s), "ls");
  // `exec` is OpenClaw's primary shell tool — a first-class category, not "other".
  assert.equal(extractToolName({ toolName: "exec", input: null }, s), "exec");
  assert.equal(extractToolName({ toolName: "Exec", input: null }, s), "exec");
  assert.equal(extractToolName({ toolName: "process", input: null }, s), "process");
  assert.equal(extractToolName({ toolName: "apply_patch", input: null }, s), "apply_patch");
  assert.equal(extractToolName({ toolName: "ContextCompactor", input: null }, s),
               "other");
  assert.equal(extractToolName({ toolName: "",     input: null }, s), "other");
});

test("extractWorkingDirectoryRelative: project root / subdir / outside / no-path", () => {
  const ev = { toolName: "ls", input: null };
  const root = "/home/dev/proj";
  assert.equal(extractWorkingDirectoryRelative(ev,
    { cwd: root, projectRoot: root, messages: [] }), "project-root");
  assert.equal(extractWorkingDirectoryRelative(ev,
    { cwd: root + "/", projectRoot: root, messages: [] }), "project-root");
  assert.equal(extractWorkingDirectoryRelative(ev,
    { cwd: `${root}/src`, projectRoot: root, messages: [] }), "subdirectory");
  assert.equal(extractWorkingDirectoryRelative(ev,
    { cwd: "/tmp", projectRoot: root, messages: [] }), "outside-project");
  assert.equal(extractWorkingDirectoryRelative(ev,
    { cwd: "", projectRoot: root, messages: [] }), "no-path");
  assert.equal(extractWorkingDirectoryRelative(ev,
    { cwd: root, projectRoot: "", messages: [] }), "no-path");
});

test("extractParentToolCallName: none / known / other", () => {
  const ev = { toolName: "edit", input: null };
  assert.equal(extractParentToolCallName(ev, blankSession()), "none");

  const sess = (msgs: Message[]): Session => ({
    cwd: "", projectRoot: "", messages: msgs,
  });
  assert.equal(extractParentToolCallName(ev, sess([
    { role: "user" },
    { role: "tool_call", toolName: "Read" },
    { role: "tool_result" },
  ])), "read");
  assert.equal(extractParentToolCallName(ev, sess([
    { role: "tool_call", toolName: "ContextCompactor" },
  ])), "other");
  // Most-recent tool_call wins, even if other roles intervene.
  assert.equal(extractParentToolCallName(ev, sess([
    { role: "tool_call", toolName: "bash" },
    { role: "tool_call", toolName: "grep" },
    { role: "user" },
  ])), "grep");
});

test("extractRecentRepetitionCount: 0/1/2/3+ buckets", () => {
  const ev = { toolName: "bash", input: null };
  const sess = (toolNames: string[]): Session => ({
    cwd: "", projectRoot: "",
    messages: toolNames.map(t => ({ role: "tool_call", toolName: t })),
  });
  assert.equal(extractRecentRepetitionCount(ev, sess([])),                     "rep-0");
  assert.equal(extractRecentRepetitionCount(ev, sess(["read"])),               "rep-0");
  assert.equal(extractRecentRepetitionCount(ev, sess(["bash"])),               "rep-1");
  assert.equal(extractRecentRepetitionCount(ev, sess(["bash", "bash"])),       "rep-2");
  assert.equal(extractRecentRepetitionCount(ev,
    sess(["bash", "read", "bash", "bash"])),                                   "rep-3plus");
  assert.equal(extractRecentRepetitionCount(ev,
    sess(["bash", "bash", "bash", "bash", "bash", "bash"])),                   "rep-3plus");
  // Window: only the LAST 5 tool_calls in the LAST 20 messages count.
  const tail5 = ["read", "read", "read", "read", "read"];
  const head50 = Array.from({ length: 50 }, () => "bash");
  assert.equal(extractRecentRepetitionCount(ev, sess([...head50, ...tail5])),  "rep-0");
});

test("extractTimeSinceLastUserMessage: 30s / 2m / 10m / >10m / no-user", () => {
  const ev = { toolName: "ls", input: null };
  const now = Date.parse("2026-05-04T12:00:00.000Z");
  const sess = (offsetSec: number, hasTimestamp = true): Session => ({
    cwd: "", projectRoot: "",
    messages: [{
      role: "user",
      timestamp: hasTimestamp
        ? new Date(now - offsetSec * 1000).toISOString()
        : undefined,
    }],
  });
  assert.equal(extractTimeSinceLastUserMessage(ev, sess(5),    now), "lt-30s");
  assert.equal(extractTimeSinceLastUserMessage(ev, sess(60),   now), "lt-2m");
  assert.equal(extractTimeSinceLastUserMessage(ev, sess(300),  now), "lt-10m");
  assert.equal(extractTimeSinceLastUserMessage(ev, sess(3600), now), "gt-10m");
  // No user message anywhere → most-elapsed bucket.
  assert.equal(extractTimeSinceLastUserMessage(ev, blankSession(), now), "gt-10m");
  // User message without a parseable timestamp → most-elapsed bucket.
  assert.equal(extractTimeSinceLastUserMessage(ev, sess(0, false), now), "gt-10m");
});

// ── 3. Dispatch + extractFeatures ──────────────────────────────────

test("extractors registry has the kebab-case keys matching features.bdsl", () => {
  const decls = readFeatures(FEATURES_PATH);
  const declared = new Set(decls.map(d => d.name));
  const registered = new Set(Object.keys(extractors));
  assert.deepEqual(registered, declared);
});

test("extractFeatures returns a kebab-keyed dict with one value per feature", () => {
  const features = extractFeatures(
    { toolName: "bash", input: { command: "ls" } },
    {
      cwd: "/proj/src",
      projectRoot: "/proj",
      messages: [{ role: "user", timestamp: new Date().toISOString() }],
    },
  );
  assert.deepEqual(Object.keys(features).sort(), [
    "parent-tool-call-name",
    "recent-identical-call-count",
    "recent-repetition-count",
    "time-since-last-user-message",
    "tool-name",
    "working-directory-relative",
  ]);
  assert.equal(features["tool-name"], "bash");
  assert.equal(features["working-directory-relative"], "subdirectory");
  assert.equal(features["parent-tool-call-name"], "none");
});
