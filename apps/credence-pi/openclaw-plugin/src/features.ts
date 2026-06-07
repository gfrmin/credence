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
  "exec",
  "process",
  "apply_patch",
  "grep",
  "find",
  "ls",
]);

const HISTORY_CAP = 50;
const REPETITION_WINDOW = 5;

interface Entry {
  tool: string;
  /** Stable fingerprint of the tool's arguments — lets us count repeats of the
   *  SAME call (a loop), distinct from repeats of the same tool with different
   *  args (legitimate). This is the signal tool-name repetition alone misses. */
  argFp: string;
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

// Stable fingerprint of a tool's params: JSON with keys sorted at every level,
// so {a:1,b:2} and {b:2,a:1} match. Truncated — we only need equality, not the
// content. Unhashable shapes fall back to a constant (treated as "same").
function fingerprintArgs(params: unknown): string {
  const norm = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(norm);
    if (v && typeof v === "object") {
      const out: Record<string, unknown> = {};
      for (const k of Object.keys(v as Record<string, unknown>).sort()) {
        out[k] = norm((v as Record<string, unknown>)[k]);
      }
      return out;
    }
    return v;
  };
  try {
    return JSON.stringify(norm(params)).slice(0, 512);
  } catch {
    return "∅";
  }
}

// rep-style 0/1/2/3plus bucketing shared by tool- and argument-repetition.
function countBucket(n: number, prefix: string): string {
  return n === 0
    ? `${prefix}-0`
    : n === 1
      ? `${prefix}-1`
      : n === 2
        ? `${prefix}-2`
        : `${prefix}-3plus`;
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

    const argFp = fingerprintArgs(event.params);
    const recent = st.history.slice(-REPETITION_WINDOW);
    const reps = recent.filter((e) => e.tool === tool).length;
    // Argument-level repetition: how many recent calls were the SAME (tool,args)
    // — i.e. this exact call repeated. This is the loop signal; the eval showed
    // tool-level repetition alone cannot isolate it (a cell mixes a re-run of one
    // command with distinct commands of the same tool).
    const identical = recent.filter(
      (e) => e.tool === tool && e.argFp === argFp,
    ).length;

    const elapsed =
      st.lastUserTs === undefined ? undefined : now - st.lastUserTs;

    const features: Features = {
      "tool-name": tool,
      "working-directory-relative": workingDirRelative(ctx, event),
      "parent-tool-call-name": parent,
      "recent-repetition-count": countBucket(reps, "rep"),
      "recent-identical-call-count": countBucket(identical, "ident"),
      "time-since-last-user-message": timeSinceUserBucket(elapsed),
    };

    // Record current call.
    st.history.push({ tool, argFp, ts: now });
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
