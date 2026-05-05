// time_since_user.ts — classify the elapsed time since the last user
// message into the brain's `time-since-user-space`: lt-30s | lt-2m |
// lt-10m | gt-10m. Sessions with no user message, or a user message
// without a parseable timestamp, bucket to gt-10m (the most-elapsed
// bucket) — the brain's prior should treat those situations as if the
// user has been away.

import type { ToolCallEvent, Session } from "../types.js";

export const POSSIBLE_OUTPUTS = ["lt-30s", "lt-2m", "lt-10m", "gt-10m"] as const;

export function extractTimeSinceLastUserMessage(
  _event: ToolCallEvent,
  session: Session,
  // `now` is injectable for deterministic tests.
  now: number = Date.now(),
): string {
  for (let i = session.messages.length - 1; i >= 0; i--) {
    const m = session.messages[i]!;
    if (m.role !== "user") continue;
    if (!m.timestamp) return "gt-10m";
    const t = Date.parse(m.timestamp);
    if (Number.isNaN(t)) return "gt-10m";
    const elapsedSec = (now - t) / 1000;
    if (elapsedSec < 30) return "lt-30s";
    if (elapsedSec < 120) return "lt-2m";
    if (elapsedSec < 600) return "lt-10m";
    return "gt-10m";
  }
  return "gt-10m";
}
