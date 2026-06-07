// features/index.ts — extractor dispatch and feature-dict assembly.
// Registry keys are kebab-case BDSL feature names verbatim (matches
// features.bdsl); manifest.ts's verifyFeatures will check missing
// extractors against this set at startup.

import type { ToolCallEvent, Session, FeatureDict } from "../types.js";
import { extractToolName }                  from "./tool_name.js";
import { extractWorkingDirectoryRelative }  from "./working_directory.js";
import { extractParentToolCallName }        from "./parent_tool.js";
import { extractRecentRepetitionCount }     from "./repetition.js";
import { extractRecentIdenticalCallCount }  from "./identical_call.js";
import { extractTimeSinceLastUserMessage }  from "./time_since_user.js";

export type Extractor = (event: ToolCallEvent, session: Session) => string;

export const extractors: Record<string, Extractor> = {
  "tool-name":                    extractToolName,
  "working-directory-relative":   extractWorkingDirectoryRelative,
  "parent-tool-call-name":        extractParentToolCallName,
  "recent-repetition-count":      extractRecentRepetitionCount,
  "recent-identical-call-count":  extractRecentIdenticalCallCount,
  "time-since-last-user-message": extractTimeSinceLastUserMessage,
};

export function extractFeatures(event: ToolCallEvent, session: Session): FeatureDict {
  const out: FeatureDict = {};
  for (const [name, fn] of Object.entries(extractors)) {
    out[name] = fn(event, session);
  }
  return out;
}
