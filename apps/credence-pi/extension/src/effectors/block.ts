// block.ts — refuse the proposed tool_call. Resolves the originating
// hook with `{block: true, reason}`; pi propagates the reason to the
// agent. Reason defaults to a generic refusal string when the
// brain's signal omits parameters.reason (Pass 1's BDSL emits a
// constant reason; Pass 2 may template per event).

import type { EffectorImpl } from "./types.js";

const DEFAULT_REASON = "Refused based on prior approval observations.";

export const block: EffectorImpl = (parameters, awaiter, _ctx) => {
  const raw = parameters["reason"];
  const reason = typeof raw === "string" && raw.length > 0 ? raw : DEFAULT_REASON;
  awaiter.resolve({ block: true, reason });
};
