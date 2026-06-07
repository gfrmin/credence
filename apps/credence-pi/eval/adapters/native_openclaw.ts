// native_openclaw.ts — adapt a native OpenClaw session transcript
// (~/.openclaw/agents/main/sessions/*.jsonl and the qa/scenarios replay
// fixtures) into the harness's NormalizedEvent stream.
//
// Format (Anthropic-style, confirmed against real sessions + qa fixtures):
//   {"type":"session","cwd":"…",…}                              (workspace dir)
//   {"type":"message","message":{"role":"user","content":"…"},"timestamp":…}
//   {"type":"message","message":{"role":"assistant",
//      "content":[{"type":"tool_use","name":"read","input":{…}}, …],
//      "usage":{"totalTokens":N,"cost":{"total":U,…}}}, "timestamp":…}
//   {"type":"message","message":{"role":"tool","toolName":"read",
//      "content":"…","isError":bool}}
//
// A tool call is a tool_use block inside an assistant message. Its turn cost is
// that assistant message's usage. Its failure is the matching role:"tool"
// result's error flag. Feature bucketing is delegated to the body's
// FeatureTracker — the ONE source of truth — so the replay sees exactly what a
// live body would emit.

import { FeatureTracker } from "../../openclaw-plugin/src/features.js";
import type {
  BeforeToolCallEvent,
  ToolContext,
} from "../../openclaw-plugin/src/openclaw-types.js";
import type { NormalizedEvent } from "../types.js";

// Native transcripts always carry tool-name, parent, and repetition (they are
// reconstructable from the call sequence). working-directory-relative needs the
// session cwd + path-bearing args; time-since-last-user-message needs
// timestamps. Both are usually present in real sessions, sometimes absent in
// fixtures — recorded per event by featuresAvailable().
const ALWAYS_AVAILABLE = [
  "tool-name",
  "parent-tool-call-name",
  "recent-repetition-count",
];

interface RawRecord {
  type?: string;
  cwd?: string;
  timestamp?: string | number;
  message?: {
    role?: string;
    content?: unknown;
    toolName?: string;
    isError?: boolean;
    error?: unknown;
    usage?: {
      totalTokens?: number;
      total?: number;
      cost?: { total?: number };
    };
  };
}

interface ToolUseBlock {
  type: string;
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
}

function toEpochMs(ts: string | number | undefined): number | null {
  if (ts == null) return null;
  if (typeof ts === "number") return ts;
  const ms = Date.parse(ts);
  return Number.isNaN(ms) ? null : ms;
}

// Pull path-like strings out of a tool input so the body's
// working-directory-relative bucketer has something to chew on. OpenClaw
// derives this host-side; we approximate from common path-bearing arg names
// plus any value that looks like a filesystem path.
function derivePaths(input: Record<string, unknown> | undefined): string[] {
  if (!input) return [];
  const out: string[] = [];
  const PATH_KEYS = /^(path|file|filename|filepath|dir|directory|cwd|target)$/i;
  for (const [k, v] of Object.entries(input)) {
    if (typeof v !== "string" || v.length === 0) continue;
    if (PATH_KEYS.test(k) || /[/\\]/.test(v)) out.push(v);
  }
  return out;
}

function summariseInput(input: unknown): string {
  const s = JSON.stringify(input ?? null);
  return s.length > 120 ? s.slice(0, 120) + "…" : s;
}

function isToolUse(b: unknown): b is ToolUseBlock {
  return (
    typeof b === "object" &&
    b !== null &&
    (b as ToolUseBlock).type === "tool_use"
  );
}

/**
 * Parse one native session transcript (array of parsed JSONL records) into the
 * ordered NormalizedEvent stream. `sessionId` labels the source.
 */
export function adaptNative(
  records: RawRecord[],
  sessionId: string,
  meta: Record<string, unknown> = {},
): NormalizedEvent[] {
  const tracker = new FeatureTracker();
  const ctx: ToolContext = {
    runId: sessionId,
    sessionId,
    workspaceDir: undefined,
  };

  // First pass: workspace dir (so working-directory-relative resolves). The
  // session record (or the first cwd-bearing record) sets it.
  for (const r of records) {
    if (r.cwd) {
      ctx.workspaceDir = r.cwd;
      break;
    }
  }

  // Build an index of tool results by call order so we can flag failures. In
  // this format, each role:"tool" message is the result of the most recent
  // unmatched tool_use, in order.
  const events: NormalizedEvent[] = [];
  let idx = 0;
  const pendingFailures: boolean[] = [];

  // Pre-scan tool results in document order (their failure flags line up with
  // tool_use blocks in order).
  for (const r of records) {
    if (r.type === "message" && r.message?.role === "tool") {
      const errored =
        r.message.isError === true ||
        (r.message.error != null && r.message.error !== false);
      pendingFailures.push(errored);
    }
  }

  for (const r of records) {
    // A record is a message iff it carries `.message`. Real sessions also tag
    // it `type:"message"`; the qa fixtures omit `type` entirely — key on
    // `.message` so both formats parse. Non-message records (session/cwd,
    // compaction, model_change) are skipped here.
    if (!r.message) continue;
    const role = r.message.role;
    const tsMs = toEpochMs(r.timestamp);

    if (role === "user") {
      if (tsMs != null) tracker.markUserMessage(ctx, tsMs);
      continue;
    }

    if (role !== "assistant") continue;

    const content = r.message.content;
    if (!Array.isArray(content)) continue; // plain text assistant turn: no tools

    // Turn-level usage applies to every tool call emitted in this turn.
    const usage = r.message.usage;
    const turnTokens = usage?.totalTokens ?? usage?.total ?? null;
    const turnCost = usage?.cost?.total ?? null;

    for (const block of content) {
      if (!isToolUse(block)) continue;
      const toolName = String(block.name ?? "");
      const input = block.input ?? {};
      const event: BeforeToolCallEvent = {
        toolName,
        params: input,
        runId: sessionId,
        derivedPaths: derivePaths(input),
      };
      const now = tsMs ?? undefined;
      const features = tracker.extractAndRecord(event, ctx, now);

      const featuresAvailable = [...ALWAYS_AVAILABLE];
      if (ctx.workspaceDir && (event.derivedPaths?.length ?? 0) > 0) {
        featuresAvailable.push("working-directory-relative");
      }
      if (tsMs != null) featuresAvailable.push("time-since-last-user-message");

      events.push({
        session_id: sessionId,
        idx,
        ts: tsMs,
        tool_name: toolName,
        input_summary: summariseInput(input),
        features,
        features_available: featuresAvailable,
        failed: pendingFailures[idx] ?? false,
        turn_tokens: turnTokens,
        turn_cost_usd: turnCost,
        label: null,
        meta,
      });
      idx += 1;
    }
  }

  return events;
}
