// effectors.ts — the body's tentacles. Each enacts ONE terminal brain decision as
// a pi-mono tool_call outcome over the governed `answer` call:
//
//   answer   → ALLOW (return undefined): rewrite event.input in place to the brain's
//              decided value + credence, so the answer tool emits the BRAIN's answer,
//              not the LLM's draft (move-4-design §4 step 4).
//   ask-user → BLOCK with a reason that tells the agent to put the question to the owner.
//   abstain  → BLOCK with the withheld-leader reason (the abstain-show-withheld contract).
//   gather   → not a terminal (the daemon's gather is enacted inside decideWithGather); a
//              safe fallback block if one ever reaches here.
//
// The registry keys are the manifest effector names (capabilities.bdsl); the daemon's
// action→effector mapping (report|hedge → answer, …) already happened in daemon.ts.

import type { EffectorRegistry } from "@credence/brain-body";

/** A pi-mono tool_call result: undefined = allow, {block} = stop + steer. */
export type ToolCallResultLike = { block: true; reason: string } | undefined;

/** The mutable tool_call event the body governs (its `input` is the live args object the
 *  tool will execute with — agent-loop.ts passes the same reference, so a mutation here
 *  reaches execute). */
export interface GovernedEvent {
	input: Record<string, unknown>;
}

export type Effector = (params: Record<string, unknown>, event: GovernedEvent) => ToolCallResultLike;

const answer: Effector = (params, event) => {
	// Rewrite the answer in place to the brain's decision; execute emits this.
	event.input.value = params.value;
	event.input.credence = params.credence;
	event.input.hedged = params.hedged ?? false;
	return undefined; // allow
};

const askUser: Effector = (params) => ({
	block: true,
	reason: `Put this question to the owner (the oracle): ${String(params.text ?? "")}`,
});

const abstain: Effector = (params) => ({
	block: true,
	reason: `Do not answer — ${String(params.reason ?? "the evidence does not settle it")}.`,
});

const gather: Effector = (params) => ({
	// Defensive: the recency gather is enacted inside decideWithGather and should never
	// surface as a terminal. If a future probe does, steer rather than silently answer.
	block: true,
	reason: `Gather more evidence first: ${String(params.probe ?? "probe")} on "${String(params.target ?? "")}".`,
});

export const effectors: EffectorRegistry<Effector> = {
	answer,
	"ask-user": askUser,
	abstain,
	gather,
};
