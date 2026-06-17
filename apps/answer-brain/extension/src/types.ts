// types.ts — the shapes the answer-brain body holds and moves. The brain↔body
// wire types (EffectorSignal, Manifest) come from @credence/brain-body; these are
// the answer-brain-specific ones: the two backends' responses and the per-question
// evidence accumulator the body grows and resends (the daemon + bridge are both
// stateless — move-3/move-4-design).

/** One abstract observation — the string-blind tuple the bridge's /extract emits and
 *  the daemon's /decide consumes verbatim (the parity boundary). The body forwards
 *  these; it never reads the fields. */
export interface AbstractObs {
	reports: number;
	group: number;
	authority: number;
	subject_factor: number;
	time_factor: number;
}

export interface Hit {
	artifact_cache_key: string;
	[k: string]: unknown;
}

export interface RouteResult {
	construct: string;
	time_indexed: boolean;
}

export interface ExtractResult {
	candidates: string[];
	observations: AbstractObs[];
	rho: number;
	indeterminate: number;
	era_split: boolean;
}

/** The daemon's POST /decide response (apps/answer-brain/daemon/server.jl). */
export interface DecideResponse {
	effector: string; // report | hedge | ask_clarify | abstain | gather
	report_index: number | null;
	value: string | null;
	probe: string | null; // the gather probe (e.g. "recency"), or null
	target: string | null; // the candidate to test, or null
	credences: number[];
	p_none: number;
	eu: number;
}

/**
 * The per-question evidence the body accumulates across its tool calls and resends
 * each refinement (the daemon + bridge hold no per-question state). One of these per
 * pi session; replaced wholesale on a new question (statelessness by construction).
 */
export interface Evidence {
	question: string;
	hits: Hit[];
	hitKeys: string[];
	docDate: Record<string, string | null>;
	route: RouteResult | null;
	ownerScoped: boolean;
	candidates: string[];
	observations: AbstractObs[];
	rho: number;
	eraSplit: boolean;
	appliedProbes: string[]; // grows [] → ["recency"]; guarantees termination
	extracted: boolean; // has /extract run at least once?
	lastCredences: number[]; // from the last /decide — feeds the display features
	lastPNone: number;
}

export function freshEvidence(question: string): Evidence {
	return {
		question,
		hits: [],
		hitKeys: [],
		docDate: {},
		route: null,
		ownerScoped: false,
		candidates: [],
		observations: [],
		rho: 0,
		eraSplit: false,
		appliedProbes: [],
		extracted: false,
		lastCredences: [],
		lastPNone: 1,
	};
}
