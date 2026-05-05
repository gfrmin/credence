// parent_tool.ts — classify the most recent prior tool_call into the
// brain's `parent-tool-name-space`: read | write | edit | bash | grep |
// find | ls | other | none. "none" when the session has no preceding
// tool_call (i.e., the proposed call is the first). The current
// (proposed) call is NOT yet in session.messages — it's only present in
// the event — so we just look at the tail of the message list.

import type { ToolCallEvent, Session } from "../types.js";
import { KNOWN_TOOLS } from "./tool_name.js";

export const POSSIBLE_OUTPUTS = [
  "read", "write", "edit", "bash", "grep", "find", "ls", "other", "none",
] as const;

export function extractParentToolCallName(
  _event: ToolCallEvent,
  session: Session,
): string {
  for (let i = session.messages.length - 1; i >= 0; i--) {
    const m = session.messages[i]!;
    if (m.role !== "tool_call") continue;
    const t = (m.toolName ?? "").toLowerCase();
    return KNOWN_TOOLS.has(t) ? t : "other";
  }
  return "none";
}
