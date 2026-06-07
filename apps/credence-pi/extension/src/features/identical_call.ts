// identical_call.ts — bucket how many of the recent tool_calls were the SAME
// (tool, arguments) as this one. This is the argument-level loop signal:
// re-running the *same* call, distinct from calling the same tool with different
// arguments. Tool-level repetition (repetition.ts) cannot tell these apart; the
// offline eval (eval/results/FINDINGS.md) showed that gap makes loops invisible.
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

const RECENT_MESSAGE_WINDOW = 20;
const RECENT_TOOL_CALL_WINDOW = 5;

const bucket = (name: string): string =>
  KNOWN_TOOLS.has(name.toLowerCase()) ? name.toLowerCase() : "other";

// Stable fingerprint of arguments: JSON with keys sorted at every level, so key
// order doesn't matter. Mirrors the openclaw-plugin body. Equality is all we
// need; unhashable shapes collapse to a constant.
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
    return JSON.stringify(norm(input)).slice(0, 512);
  } catch {
    return "∅";
  }
}

export function extractRecentIdenticalCallCount(
  event: ToolCallEvent,
  session: Session,
): string {
  const tool = bucket(event.toolName);
  const fp = fingerprint(event.input);
  const recentToolCalls = session.messages
    .slice(-RECENT_MESSAGE_WINDOW)
    .filter((m) => m.role === "tool_call")
    .slice(-RECENT_TOOL_CALL_WINDOW);
  const matches = recentToolCalls.filter(
    (m) => bucket(m.toolName ?? "") === tool && fingerprint(m.input) === fp,
  ).length;
  if (matches === 0) return "ident-0";
  if (matches === 1) return "ident-1";
  if (matches === 2) return "ident-2";
  return "ident-3plus";
}
