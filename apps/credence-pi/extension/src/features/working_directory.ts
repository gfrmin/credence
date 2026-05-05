// working_directory.ts — classify session.cwd against session.projectRoot
// into the brain's `wd-relative-space`: project-root | subdirectory |
// outside-project | no-path. Exact-equality of cwd and projectRoot maps
// to project-root; cwd inside the project tree is subdirectory; cwd
// outside (or on a different drive) is outside-project; missing or
// empty cwd is no-path. The trailing-separator handling tolerates both
// "/proj" and "/proj/" project root forms.

import type { ToolCallEvent, Session } from "../types.js";

export const POSSIBLE_OUTPUTS = [
  "project-root", "subdirectory", "outside-project", "no-path",
] as const;

export function extractWorkingDirectoryRelative(
  _event: ToolCallEvent,
  session: Session,
): string {
  if (!session.cwd || !session.projectRoot) return "no-path";
  const root = session.projectRoot.replace(/\/+$/, "");
  const cwd = session.cwd.replace(/\/+$/, "");
  if (cwd === root) return "project-root";
  if (cwd.startsWith(root + "/")) return "subdirectory";
  return "outside-project";
}
