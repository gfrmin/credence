// proceed.ts — let the proposed tool_call execute as-is.
// Resolves the originating hook with `undefined`, which pi treats
// as "no veto, proceed".

import type { EffectorImpl } from "./types.js";

export const proceed: EffectorImpl = (_parameters, awaiter, _ctx) => {
  awaiter.resolve(undefined);
};
