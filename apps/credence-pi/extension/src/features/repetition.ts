// repetition.ts — bucket how many times the bucketed tool name has
// appeared among the last five tool_calls of the last twenty messages.
// Output space: rep-0 | rep-1 | rep-2 | rep-3plus.
//
// Bucketing is consistent with tool_name.ts: unknown tools collapse to
// "other" before counting, so repeated calls to a non-KNOWN tool also
// register as repetitions.

import type { ToolCallEvent, Session } from "../types.js";
import { KNOWN_TOOLS } from "./tool_name.js";

export const POSSIBLE_OUTPUTS = ["rep-0", "rep-1", "rep-2", "rep-3plus"] as const;

const RECENT_MESSAGE_WINDOW = 20;
const RECENT_TOOL_CALL_WINDOW = 5;

const bucket = (name: string): string =>
  KNOWN_TOOLS.has(name.toLowerCase()) ? name.toLowerCase() : "other";

export function extractRecentRepetitionCount(
  event: ToolCallEvent,
  session: Session,
): string {
  const target = bucket(event.toolName);
  const recentMessages = session.messages.slice(-RECENT_MESSAGE_WINDOW);
  const recentToolCalls = recentMessages
    .filter(m => m.role === "tool_call")
    .slice(-RECENT_TOOL_CALL_WINDOW);
  const matches = recentToolCalls
    .filter(m => bucket(m.toolName ?? "") === target).length;
  if (matches === 0) return "rep-0";
  if (matches === 1) return "rep-1";
  if (matches === 2) return "rep-2";
  return "rep-3plus";
}
