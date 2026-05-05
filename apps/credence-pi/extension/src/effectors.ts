// effectors.ts — dispatch table. Keys are kebab-case strings matching
// the BDSL declarations in capabilities.bdsl verbatim, so manifest.ts's
// verifyEffectors check at startup keys against this set.

import type { EffectorImpl } from "./effectors/types.js";
import { ask }     from "./effectors/ask.js";
import { proceed } from "./effectors/proceed.js";
import { block }   from "./effectors/block.js";

export const effectors: Record<string, EffectorImpl> = {
  "ask":     ask,
  "proceed": proceed,
  "block":   block,
};

export type { EffectorImpl, HookReturn, HookAwaiter, EffectorContext }
  from "./effectors/types.js";
