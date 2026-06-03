// features.ts — the body's sensory periphery for the OpenClaw plugin.
//
// Produces the brain's declared kebab-case feature vocabulary
// (apps/credence-pi/bdsl/features.bdsl) from the before_tool_call event +
// a small per-run history buffer. In Move 1 the brain does NOT condition
// on features at decision time (global Beta), so these are logged for the
// dollars-saved surface and future feature-conditioned learning; the
// loop-relevant ones (tool-name, parent, repetition) are exact, while
// working-directory-relative and time-since-last-user-message are
// best-effort (OpenClaw does not put cwd or message timestamps on the
// tool ctx — see move-1-design.md OQ-d).

import type { BeforeToolCallEvent, ToolContext } from "./openclaw-types.js";

export type Features = Record<string, string>;

const KNOWN_TOOLS = new Set([
  "read",
  "write",
  "edit",
  "bash",
  "grep",
  "find",
  "ls",
]);

const HISTORY_CAP = 50;
const REPETITION_WINDOW = 5;

interface Entry {
  tool: string;
  ts: number;
}

interface RunState {
  history: Entry[];
  lastUserTs?: number;
}

function bucketTool(name: string | undefined): string {
  if (!name) return "other";
  const t = name.toLowerCase();
  return KNOWN_TOOLS.has(t) ? t : "other";
}

function runKey(ctx: ToolContext): string {
  return ctx.runId ?? ctx.sessionKey ?? ctx.sessionId ?? "default";
}

function timeSinceUserBucket(elapsedMs: number | undefined): string {
  if (elapsedMs === undefined) return "gt-10m";
  const s = elapsedMs / 1000;
  if (s < 30) return "lt-30s";
  if (s < 120) return "lt-2m";
  if (s < 600) return "lt-10m";
  return "gt-10m";
}

function workingDirRelative(
  ctx: ToolContext,
  event: BeforeToolCallEvent,
): string {
  const root = ctx.workspaceDir;
  const paths = event.derivedPaths ?? [];
  if (paths.length === 0) {
    // No path-bearing tool (e.g. a bare bash with no file args we can see).
    return "no-path";
  }
  if (!root) return "no-path";
  const norm = (p: string) => p.replace(/\/+$/, "");
  const r = norm(root);
  let sawOutside = false;
  let sawRootExact = false;
  let sawSub = false;
  for (const raw of paths) {
    const p = norm(raw);
    if (p === r) {
      sawRootExact = true;
    } else if (p.startsWith(r + "/")) {
      sawSub = true;
    } else {
      sawOutside = true;
    }
  }
  if (sawOutside) return "outside-project";
  if (sawRootExact && !sawSub) return "project-root";
  return "subdirectory";
}

export class FeatureTracker {
  private runs = new Map<string, RunState>();

  /** Stamp the most recent user message for a run (best-effort; wired
   *  from a message hook if one is available). */
  markUserMessage(ctx: ToolContext, ts: number): void {
    const k = runKey(ctx);
    const st = this.runs.get(k) ?? { history: [] };
    st.lastUserTs = ts;
    this.runs.set(k, st);
  }

  /** Compute features from the run's history BEFORE this call, then record
   *  the current call. `now` is injectable for deterministic tests. */
  extractAndRecord(
    event: BeforeToolCallEvent,
    ctx: ToolContext,
    now: number = Date.now(),
  ): Features {
    const k = runKey(ctx);
    const st = this.runs.get(k) ?? { history: [] };
    const tool = bucketTool(event.toolName);

    const parent =
      st.history.length > 0 ? st.history[st.history.length - 1].tool : "none";

    const recent = st.history.slice(-REPETITION_WINDOW);
    const reps = recent.filter((e) => e.tool === tool).length;
    const repBucket =
      reps === 0 ? "rep-0" : reps === 1 ? "rep-1" : reps === 2 ? "rep-2" : "rep-3plus";

    const elapsed =
      st.lastUserTs === undefined ? undefined : now - st.lastUserTs;

    const features: Features = {
      "tool-name": tool,
      "working-directory-relative": workingDirRelative(ctx, event),
      "parent-tool-call-name": parent,
      "recent-repetition-count": repBucket,
      "time-since-last-user-message": timeSinceUserBucket(elapsed),
    };

    // Record current call.
    st.history.push({ tool, ts: now });
    if (st.history.length > HISTORY_CAP) st.history.shift();
    this.runs.set(k, st);

    return features;
  }

  /** Drop a run's buffer (call on agent_end to bound memory). */
  clearRun(ctx: ToolContext): void {
    this.runs.delete(runKey(ctx));
  }

  /** Test/inspection accessor. */
  runCount(): number {
    return this.runs.size;
  }
}
