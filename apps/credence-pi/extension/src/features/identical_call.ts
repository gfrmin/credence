// identical_call.ts — bucket how many of this run's prior tool_calls were the
// SAME (tool, arguments) as this one: the argument-level loop signal (re-running
// the *same* call), distinct from calling the same tool with different args.
// Tool-level repetition (repetition.ts) cannot tell these apart; the offline
// eval (eval/results/FINDINGS.md) showed that gap makes loops invisible.
//
// Mirrors the openclaw-plugin FeatureTracker exactly: counted SESSION-WIDE (a
// re-run is wasteful however far back the original was), only for INFORMATIVE
// args (no args, or a lone field echoing the tool name, is no evidence of a
// loop), with a COLLISION-FREE content hash (a truncated fingerprint would
// mis-count long inputs that share a head, e.g. two heredocs).
//
// Output space: ident-0 | ident-1 | ident-2 | ident-3plus.

import type { ToolCallEvent, Session } from "../types.js";
import { KNOWN_TOOLS } from "./tool_name.js";

export const POSSIBLE_OUTPUTS = [
  "ident-0",
  "ident-1",
  "ident-2",
  "ident-3plus",
] as const;

const bucket = (name: string): string =>
  KNOWN_TOOLS.has(name.toLowerCase()) ? name.toLowerCase() : "other";

// Two cheap stable hashes (djb2 + sdbm) combined — fingerprint by full content,
// never truncated, so distinct long inputs sharing a prefix don't collide.
function hash2(s: string): string {
  let d = 5381, m = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    d = ((d * 33) ^ c) >>> 0;
    m = (c + (m << 6) + (m << 16) - m) >>> 0;
  }
  return d.toString(36) + ":" + m.toString(36);
}

function fingerprint(input: unknown): string {
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
    return hash2(JSON.stringify(norm(input)));
  } catch {
    return "∅";
  }
}

// Do the args carry distinguishing information? No, if there are none or the
// only field's value just echoes the tool name (some transcripts collapse a
// tool's args to its name). Two such calls are indistinguishable.
function informative(input: unknown, tool: string): boolean {
  if (!input || typeof input !== "object") return false;
  const entries = Object.entries(input as Record<string, unknown>).filter(
    ([, v]) => v != null && v !== "",
  );
  if (entries.length === 0) return false;
  if (entries.length === 1) {
    const v = entries[0][1];
    if (typeof v === "string" && v.trim().toLowerCase() === tool) return false;
  }
  return true;
}

export function extractRecentIdenticalCallCount(
  event: ToolCallEvent,
  session: Session,
): string {
  const tool = bucket(event.toolName);
  if (!informative(event.input, tool)) return "ident-0";
  const fp = fingerprint(event.input);
  const matches = session.messages.filter(
    (m) =>
      m.role === "tool_call" &&
      bucket(m.toolName ?? "") === tool &&
      informative(m.input, tool) &&
      fingerprint(m.input) === fp,
  ).length;
  if (matches === 0) return "ident-0";
  if (matches === 1) return "ident-1";
  if (matches === 2) return "ident-2";
  return "ident-3plus";
}
