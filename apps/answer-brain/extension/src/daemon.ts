// daemon.ts — the body's transport to the brain (POST /decide) AND the injected
// `askBrain` the lib's `govern` calls. `decideWithGather` posts the accumulated
// evidence, and when the daemon steers `gather(recency)` it enacts it INTERNALLY —
// re-extract with recency applied, re-decide — until a terminal effector (the §4
// worked example, steps 2-4). The brain decides every step; the body only enacts
// and re-asks. `applied_probes` (resent) makes recency fire at most once → it
// terminates. No posterior/EU is computed here (single-reasoner invariant); the
// daemon-action→manifest-effector mapping and the human-facing param composition
// (the withheld leader, the ask text) are the body's render job.

import type { EffectorSignal } from "@credence/brain-body";

import type { BridgeClient } from "./bridge.js";
import type { AbstractObs, DecideResponse, Evidence } from "./types.js";

export interface DecideRequest {
	candidates: string[];
	observations: AbstractObs[];
	rho: number;
	u_bar: Record<string, number>;
	era_split: boolean;
	owner_scoped: boolean; // daemon v0 ignores it (reserved for the subject refinement)
	applied_probes: string[];
}

export interface DecideOptions {
	daemonUrl: string;
	timeoutMs: number;
	fetchImpl?: typeof fetch;
	onStep?: (msg: string) => void;
}

const MAX_GATHER_STEPS = 4; // recency fires at most once; this is a runaway backstop

function composeDecide(ev: Evidence, uBar: Record<string, number>): DecideRequest {
	return {
		candidates: ev.candidates,
		observations: ev.observations,
		rho: ev.rho,
		u_bar: uBar,
		era_split: ev.eraSplit,
		owner_scoped: ev.ownerScoped,
		applied_probes: ev.appliedProbes,
	};
}

async function postDecide(req: DecideRequest, opts: DecideOptions): Promise<DecideResponse> {
	const fetchImpl = opts.fetchImpl ?? fetch;
	const ctrl = new AbortController();
	const timer = setTimeout(() => ctrl.abort(), opts.timeoutMs);
	try {
		const resp = await fetchImpl(`${opts.daemonUrl.replace(/\/+$/, "")}/decide`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(req),
			signal: ctrl.signal,
		});
		if (!resp.ok) {
			throw new Error(`daemon POST /decide → ${resp.status}`);
		}
		return (await resp.json()) as DecideResponse;
	} finally {
		clearTimeout(timer);
	}
}

function leaderIndex(credences: number[]): number {
	let best = 0;
	for (let i = 1; i < credences.length; i++) {
		if ((credences[i] ?? 0) > (credences[best] ?? 0)) best = i;
	}
	return best;
}

function pct(credences: number[], index: number | null): string {
	const i = index ?? (credences.length ? leaderIndex(credences) : 0);
	return `${((credences[i] ?? 0) * 100).toFixed(0)}%`;
}

// The abstain-show-withheld contract: name the leader the brain declined to assert,
// without asserting it (move-3/foundations).
function abstainReason(ev: Evidence, resp: DecideResponse): string {
	if (!ev.candidates.length || !resp.credences.length) {
		return "no grounded candidate to assert";
	}
	const i = leaderIndex(resp.credences);
	const leader = ev.candidates[i] ?? "(unknown)";
	return `evidence too uncertain to assert — leading candidate "${leader}" at ${pct(resp.credences, i)} (below the report bar)`;
}

function askText(ev: Evidence): string {
	const what = ev.route?.construct ?? "the value";
	return `I could not settle ${what} from the documents. Can you tell me?`;
}

// A terminal /decide response → the manifest effector + the body-composed params.
// (report|hedge → answer; ask_clarify → ask-user; abstain → abstain.)
function terminalSignal(resp: DecideResponse, ev: Evidence): EffectorSignal {
	const credence = pct(resp.credences, resp.report_index);
	switch (resp.effector) {
		case "report":
			return { effector: "answer", parameters: { value: resp.value ?? "", credence, hedged: false } };
		case "hedge":
			return {
				effector: "answer",
				parameters: { value: ev.candidates.join(" · "), credence, hedged: true },
			};
		case "ask_clarify":
			return { effector: "ask-user", parameters: { text: askText(ev) } };
		case "abstain":
			return { effector: "abstain", parameters: { reason: abstainReason(ev, resp) } };
		default:
			return {
				effector: "abstain",
				parameters: { reason: `unexpected terminal effector "${resp.effector}"` },
			};
	}
}

// Terminal brain actions; `gather` is a steer the body enacts internally, never logged.
const TERMINAL_ACTIONS = new Set(["report", "hedge", "ask_clarify", "abstain"]);

// The verdict-emission seam (move-4 successor): post the terminal decision the governor
// enacted to the bridge's /log_decision, so the owner's one-bit verdict folds into u_wrong
// through life-agent's EXISTING reaction loop. BEST-EFFORT: a logging failure is non-fatal —
// it must never change the decision — so errors are swallowed (noted via onStep). The bridge
// owns the write and sorts the candidate-order credences leader-first.
async function logTerminalDecision(
	ev: Evidence,
	resp: DecideResponse,
	bridge: BridgeClient,
	opts: DecideOptions,
): Promise<void> {
	if (!TERMINAL_ACTIONS.has(resp.effector)) return; // a non-terminal slipped through; nothing to log
	try {
		const { decision_id } = await bridge.logDecision(ev.question, ev.hitKeys, {
			effector: resp.effector,
			credences: resp.credences,
			p_none: resp.p_none,
			eu: resp.eu,
			candidates: ev.candidates,
			n_obs: ev.observations.length,
		});
		opts.onStep?.(`decision logged ${decision_id} — react with: /react ${decision_id} g|b`);
	} catch (e) {
		opts.onStep?.(`decision logging failed (non-fatal, answer unaffected): ${String(e)}`);
	}
}

/**
 * The injected `askBrain`: decide, enacting any `gather(recency)` steer internally,
 * and return the terminal effector signal. Mutates `ev` (the accumulator) in place —
 * appliedProbes, the re-extracted candidates/observations, and the last credences for
 * the display features. Throws on a daemon/bridge failure → `govern` fail-opens.
 */
export async function decideWithGather(
	ev: Evidence,
	uBar: Record<string, number>,
	bridge: BridgeClient,
	opts: DecideOptions,
): Promise<EffectorSignal> {
	for (let step = 0; step < MAX_GATHER_STEPS; step++) {
		const resp = await postDecide(composeDecide(ev, uBar), opts);
		ev.lastCredences = resp.credences;
		ev.lastPNone = resp.p_none;

		const steerRecency =
			resp.effector === "gather" &&
			resp.probe === "recency" &&
			!ev.appliedProbes.includes("recency");
		if (!steerRecency) {
			const signal = terminalSignal(resp, ev);
			// Record the enacted decision for the reaction loop — best-effort and FIRE-AND-FORGET:
			// NOT awaited, so a slow or hung bridge logger can never add latency to the govern path
			// (an await here could, worst case, eat the govern timeout → fail-closed → silently
			// withhold a good answer). logTerminalDecision swallows all its own errors, so the
			// un-awaited promise never rejects.
			void logTerminalDecision(ev, resp, bridge, opts);
			return signal;
		}

		opts.onStep?.(
			`brain steers gather(recency) on "${resp.target ?? "?"}" — re-extracting with recency applied`,
		);
		ev.appliedProbes = [...ev.appliedProbes, "recency"];
		const re = await bridge.extract(ev.question, ev.hits, { doc_date: ev.docDate }, true);
		ev.candidates = re.candidates;
		ev.observations = re.observations;
		ev.rho = re.rho;
		ev.eraSplit = re.era_split;
	}
	return { effector: "abstain", parameters: { reason: "gather did not converge" } };
}
