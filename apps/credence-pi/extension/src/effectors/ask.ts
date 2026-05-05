// ask.ts — show the user a yes/no confirmation, then post a
// `user-responded` sensor event back to the daemon. Crucially, this
// effector does NOT resolve the hook on the user's answer — it waits
// for the brain's follow-up signal (proceed or block), which arrives
// back on the same SSE stream addressed to the originating tool-
// proposed event_id.
//
// The body must NOT short-circuit "yes means proceed" here; the
// brain owns that decision. In Pass 1 the brain's followup is
// deterministic (yes→proceed, no→block) but architecturally it is
// one of several actions the brain could pick.
//
// If `ctx.confirm` is unavailable (no UI), or the user-responded
// post throws, the effector posts/treats it as a timeout response
// and lets the brain decide. The originating awaiter eventually
// resolves via either the followup signal or the per-hook timeout
// (see index.ts).

import type { EffectorImpl } from "./types.js";

export const ask: EffectorImpl = async (parameters, awaiter, ctx) => {
  if (!awaiter.pending) return;          // already resolved (timeout race)

  const text = typeof parameters["text"] === "string"
    ? (parameters["text"] as string)
    : "Allow this tool call?";

  let response: "yes" | "no" | "timeout";
  if (typeof ctx.confirm !== "function") {
    response = "timeout";
  } else {
    try {
      const answer = await ctx.confirm(text);
      response = answer === true ? "yes" : answer === false ? "no" : "timeout";
    } catch (err) {
      ctx.warn("credence-pi: ctx.ui.confirm rejected; reporting timeout", err);
      response = "timeout";
    }
  }

  if (!awaiter.pending) return;          // resolved while we awaited the user

  try {
    await ctx.postUserResponded(response);
  } catch (err) {
    // Even if the user-responded post fails, we keep the awaiter
    // open. The per-hook timeout in index.ts will fail-open if no
    // followup signal arrives.
    ctx.warn("credence-pi: user-responded post failed; relying on hook timeout", err);
  }
  // The awaiter remains pending; the followup signal will resolve it.
};
