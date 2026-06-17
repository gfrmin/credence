// governor.ts — the transport-agnostic govern skeleton. Every credence
// answer-brain body, on the call it governs, does the same three things:
//
//   1. compose the brain request from its accumulated evidence + features
//      (`buildRequest`, body-owned);
//   2. ask the brain and receive the chosen effector (`askBrain`, the INJECTED
//      transport — this is the one seam that differs per body: answer-brain's
//      synchronous POST /decide vs credence-pi's async SSE correlation);
//   3. enact the effector (`dispatch`, built from the dispatch.ts harness).
//
// And — the discipline that makes a governor safe to put in front of a tool —
// it FAILS OPEN: if the brain is unreachable, errors, or exceeds the timeout,
// `govern` does not throw and does not hang; it returns the body-supplied
// `failOpen` value. (What "open" means is the body's call: credence-pi proceeds;
// an answer body may instead step aside to an abstain so an *ungoverned*
// confident answer never escapes. The lib only guarantees it returns that value
// rather than propagating the failure.)
//
// No posterior or EU is computed here — `govern` transports the brain's decision
// (move-4-design §2 single-reasoner invariant).

import type { EffectorSignal } from "./wire.js";

export interface GovernSpec<TReq, TResult> {
	/** Compose the brain request from accumulated evidence (+ features). */
	buildRequest: () => TReq | Promise<TReq>;
	/** The injected transport: send the request, get the chosen effector. */
	askBrain: (req: TReq) => Promise<EffectorSignal>;
	/** Enact the effector (look it up in the registry, run it → host result). */
	dispatch: (signal: EffectorSignal) => TResult | Promise<TResult>;
	/** Returned on brain-unreachable / error / timeout — the body chooses it. */
	failOpen: TResult;
	/** Backstop timeout around `askBrain` (ms). The transport may also abort. */
	timeoutMs: number;
	/** Logged once on a fail-open. */
	onError?: (msg: string, err?: unknown) => void;
}

/**
 * Govern one call. Returns the dispatched effector's result, or — on any
 * failure reaching/decoding the brain decision — the `failOpen` value.
 */
export async function govern<TReq, TResult>(spec: GovernSpec<TReq, TResult>): Promise<TResult> {
	const { buildRequest, askBrain, dispatch, failOpen, timeoutMs, onError } = spec;
	try {
		const req = await buildRequest();
		const signal = await withTimeout(askBrain(req), timeoutMs);
		return await dispatch(signal);
	} catch (err) {
		onError?.("brain-body: govern failed open (brain unavailable / errored)", err);
		return failOpen;
	}
}

/**
 * Race a promise against a timeout. The underlying promise is not cancelled
 * (the lib is transport-agnostic and cannot know how) — a transport that holds
 * a socket should wire its own AbortController; this is the backstop that keeps
 * a hung brain from hanging the governed call.
 */
export function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
	return new Promise<T>((resolve, reject) => {
		const timer = setTimeout(
			() => reject(new Error(`brain-body: askBrain timed out after ${ms}ms`)),
			ms,
		);
		p.then(
			(v) => {
				clearTimeout(timer);
				resolve(v);
			},
			(e: unknown) => {
				clearTimeout(timer);
				reject(e instanceof Error ? e : new Error(String(e)));
			},
		);
	});
}
