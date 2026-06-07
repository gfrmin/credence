// atbench.ts — adapt AI45Research/ATBench-Claw into the harness's
// NormalizedEvent stream.
//
// ATBench-Claw (HF, Apache-2.0): a JSON array of 500 OpenClaw trajectories,
// each { trajectory:{events:[...]}, labels:{is_safe, risk_source, failure_mode,
// harm_type, defense_type}, reason }. `events` is the native OpenClaw message
// timeline, but tool calls are `{type:"toolCall", name, arguments}` blocks
// inside assistant `content` (not the `tool_use`/`input` shape the native
// session adapter uses), so it gets its own adapter.
//
// The independent label is `is_safe` (per trajectory): 296 unsafe / 204 safe.
// It is NOT derived from repetition, so correlating the brain's block/ask
// decisions with it removes the self-consistency caveat of the loop label.
//
// Bucketing is delegated to the body's FeatureTracker — same as every adapter.

import { FeatureTracker } from "../../openclaw-plugin/src/features.js";
import type {
  BeforeToolCallEvent,
  ToolContext,
} from "../../openclaw-plugin/src/openclaw-types.js";
import type { NormalizedEvent } from "../types.js";

const ALWAYS_AVAILABLE = [
  "tool-name",
  "parent-tool-call-name",
  "recent-repetition-count",
];

interface ToolCallBlock {
  type?: string;
  id?: string;
  name?: string;
  arguments?: Record<string, unknown>;
}

interface Event {
  type?: string;
  message?: { role?: string; content?: unknown; toolName?: string };
}

interface Labels {
  is_safe?: boolean;
  risk_source?: string;
  failure_mode?: string;
  harm_type?: string;
  defense_type?: string | null;
}

export interface AtbenchRecord {
  trajectory?: { events?: Event[]; modelId?: string; provider?: string };
  labels?: Labels;
  reason?: string;
}

function derivePaths(args: Record<string, unknown> | undefined): string[] {
  if (!args) return [];
  const out: string[] = [];
  const PATH_KEYS = /^(path|file|filename|filepath|dir|directory|cwd|target)$/i;
  for (const [k, v] of Object.entries(args)) {
    if (typeof v !== "string" || !v) continue;
    if (PATH_KEYS.test(k) || /[/\\]/.test(v)) out.push(v);
    // exec-style: pull path-like tokens out of a command string
    if (/^(command|cmd|script)$/i.test(k)) {
      for (const tok of v.split(/\s+/)) {
        if (/[/]/.test(tok) && !tok.startsWith("-")) out.push(tok);
      }
    }
  }
  return out;
}

function isToolCall(b: unknown): b is ToolCallBlock {
  return typeof b === "object" && b !== null && (b as ToolCallBlock).type === "toolCall";
}

/** Adapt one ATBench-Claw record into its ordered NormalizedEvent stream. The
 *  per-trajectory `is_safe` label is stamped on every event as "safe"/"unsafe". */
export function adaptAtbench(rec: AtbenchRecord, sessionId: string): NormalizedEvent[] {
  const tracker = new FeatureTracker();
  const ctx: ToolContext = { runId: sessionId, sessionId };
  const events = rec.trajectory?.events ?? [];
  const isSafe = rec.labels?.is_safe;
  const label = isSafe === true ? "safe" : isSafe === false ? "unsafe" : null;

  const out: NormalizedEvent[] = [];
  let idx = 0;
  for (const ev of events) {
    const msg = ev.message;
    if (!msg) continue;
    if (msg.role === "user") continue; // no timestamps; nothing to mark
    if (msg.role !== "assistant") continue;
    const content = msg.content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (!isToolCall(block)) continue;
      const toolName = String(block.name ?? "");
      if (!toolName) continue;
      const args = block.arguments ?? {};
      const event: BeforeToolCallEvent = {
        toolName,
        params: args,
        runId: sessionId,
        derivedPaths: derivePaths(args),
      };
      const features = tracker.extractAndRecord(event, ctx);
      const featuresAvailable = [...ALWAYS_AVAILABLE];
      if ((event.derivedPaths?.length ?? 0) > 0) {
        featuresAvailable.push("working-directory-relative(rootless)");
      }
      out.push({
        session_id: sessionId,
        idx,
        ts: null,
        tool_name: toolName,
        input_summary: JSON.stringify(args).slice(0, 120),
        features,
        features_available: featuresAvailable,
        failed: false, // ATBench carries no per-step tool error flag
        turn_tokens: null,
        turn_cost_usd: null,
        label,
        meta: {
          corpus: "atbench-claw",
          model: rec.trajectory?.modelId || null,
          is_safe: isSafe,
          risk_source: rec.labels?.risk_source ?? null,
          failure_mode: rec.labels?.failure_mode ?? null,
          harm_type: rec.labels?.harm_type ?? null,
          defense_type: rec.labels?.defense_type ?? null,
        },
      });
      idx += 1;
    }
  }
  return out;
}
