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

const HISTORY_CAP = 200;
const REPETITION_WINDOW = 5;
// Argument-repetition (loop) is counted session-wide, not just over the last few
// calls: re-running an identical call is wasteful whether the duplicate was 2 or
// 40 steps ago. (rep, the tool-level count, stays tight at REPETITION_WINDOW.)
const IDENTICAL_WINDOW = HISTORY_CAP;

interface Entry {
  tool: string;
  /** Stable fingerprint of the tool's arguments — lets us count repeats of the
   *  SAME call (a loop), distinct from repeats of the same tool with different
   *  args (legitimate). This is the signal tool-name repetition alone misses. */
  argFp: string;
  /** Whether the args carry genuine distinguishing information. A call with no
   *  args, or whose only arg echoes the tool name, gives no evidence of a loop
   *  (we can't tell two such calls apart), so it is never counted as a repeat. */
  informative: boolean;
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

// Two cheap, stable string hashes (djb2 + sdbm) combined into one token. Used to
// fingerprint args by content WITHOUT truncation — a prefix-truncated fingerprint
// collides on long inputs that share a head (e.g. two heredocs writing different
// scripts), which would mis-count distinct calls as repeats. Hashing the full
// normalised JSON discriminates them; the 2×32-bit space makes collision
// negligible at our scale.
function _hash2(s: string): string {
  let d = 5381, m = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    d = ((d * 33) ^ c) >>> 0;
    m = (c + (m << 6) + (m << 16) - m) >>> 0;
  }
  return d.toString(36) + ":" + m.toString(36);
}

// Stable fingerprint of a tool's params: JSON with keys sorted at every level
// (so {a:1,b:2} and {b:2,a:1} match), hashed over the FULL content. Unhashable
// shapes fall back to a constant (treated as "same").
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
    return _hash2(JSON.stringify(norm(params)));
  } catch {
    return "∅";
  }
}

// Do the args carry distinguishing information? No, if there are none, or the
// only field's value just echoes the tool name (e.g. {command:"read"} — some
// transcripts collapse a tool's args to its name). Two such calls are
// indistinguishable, so neither is evidence of a loop.
function argsInformative(params: unknown, tool: string): boolean {
  if (!params || typeof params !== "object") return false;
  const entries = Object.entries(params as Record<string, unknown>).filter(
    ([, v]) => v != null && v !== "",
  );
  if (entries.length === 0) return false;
  if (entries.length === 1) {
    const v = entries[0][1];
    if (typeof v === "string" && v.trim().toLowerCase() === tool) return false;
  }
  return true;
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
    const informative = argsInformative(event.params, tool);
    const reps = st.history.slice(-REPETITION_WINDOW).filter((e) => e.tool === tool).length;
    // Argument-level repetition: how many calls (session-wide, capped at
    // IDENTICAL_WINDOW) were the SAME (tool, args) — i.e. this exact call
    // repeated. The loop signal tool-level repetition cannot isolate. Counted
    // only for informative args (a call with no distinguishing args is no
    // evidence of a loop), and over a wide window because a re-run is wasteful
    // however far back the original was.
    const identical = informative
      ? st.history
          .slice(-IDENTICAL_WINDOW)
          .filter((e) => e.tool === tool && e.informative && e.argFp === argFp).length
      : 0;

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
    st.history.push({ tool, argFp, informative, ts: now });
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
