// tool_name.ts — bucket the proposed tool's name into the brain's
// `tool-name-space`: read | write | edit | bash | grep | find | ls |
// other. Tool name is lowercased before classification; anything not
// in KNOWN_TOOLS becomes "other".

import type { ToolCallEvent, Session } from "../types.js";

export const KNOWN_TOOLS: ReadonlySet<string> =
  new Set(["read", "write", "edit", "bash", "grep", "find", "ls"]);

export const POSSIBLE_OUTPUTS = [
  "read", "write", "edit", "bash", "grep", "find", "ls", "other",
] as const;

export function extractToolName(event: ToolCallEvent, _session: Session): string {
  const t = event.toolName.toLowerCase();
  return KNOWN_TOOLS.has(t) ? t : "other";
}
