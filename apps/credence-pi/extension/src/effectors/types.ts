// effectors/types.ts — shared types for the body's effector
// implementations and the dispatch path that invokes them.
//
// `EffectorImpl` is the contract every effector file (ask, proceed,
// block) implements. The dispatcher in `index.ts` looks up an
// effector by name in the registry, then calls it with the signal's
// `parameters`, the awaiter that owns the originating tool_call
// hook's resolution, and a small EffectorContext.
//
// Conceptually: effector = "behaviour the body has registered to
// run when the brain selects this tentacle". The brain decides
// which effector; the body decides what the effector does.

export type HookReturn = undefined | { block: true; reason: string };

// Awaiter wraps the in-flight tool_call hook's resolve/reject. The
// extension factory (`index.ts`) constructs awaiters that also
// remove themselves from the correlation table on resolution. From
// the effector's point of view, calling resolve() ends the hook.
export interface HookAwaiter {
  // Resolve the originating tool_call hook with the given return.
  // Idempotent: subsequent calls are no-ops.
  resolve(result: HookReturn): void;
  // True iff the awaiter has not yet been resolved (i.e. the hook
  // is still waiting). Used by `ask` to decide whether to keep
  // waiting after posting the user-responded follow-up.
  readonly pending: boolean;
}

export interface EffectorContext {
  // The event_id of the original tool-proposed event whose awaiter
  // this effector resolves (or, for `ask`, eventually resolves
  // through a follow-up signal).
  readonly originatingEventId: string;
  // ctx.ui.confirm equivalent. Resolves with true/false; throws or
  // rejects if the UI is unavailable. The `ask` effector treats
  // rejection / undefined as "timeout".
  confirm: ((text: string) => Promise<boolean>) | undefined;
  // Generates a fresh event_id for the user-responded sensor event
  // posted by `ask`. Mirrors the format used elsewhere in the body.
  newEventId(): string;
  // Posts a user-responded sensor event keyed against
  // `originatingEventId`. The brain's followup signal arrives back
  // on the same SSE stream, addressed to `originatingEventId`.
  postUserResponded(response: "yes" | "no" | "timeout"): Promise<void>;
  // Logger.
  warn(msg: string, err?: unknown): void;
}

export type EffectorImpl = (
  parameters: Record<string, unknown>,
  awaiter: HookAwaiter,
  ctx: EffectorContext,
) => Promise<void> | void;
