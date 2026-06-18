// tools.ts — wire the body: register the tools the LLM proposes (retrieve →
// extract → answer), accumulate the evidence across them, and GOVERN the answer
// tool. The govern loop, manifest harness, and dispatch come from
// @credence/brain-body; the transport (decideWithGather) and effectors are injected.
//
// The LLM drives the surface (pi-mono is reactive — the plan's key finding); the
// brain governs the one call that asserts. retrieve/extract are bridge proxies that
// grow the accumulator; answer is intercepted on tool_call, its evidence posted to
// /decide, and the returned effector enacted (allow+rewrite / block+steer).

import { dispatchEffector, govern } from "@credence/brain-body";
import { Type } from "typebox";

import type { BridgeClient } from "./bridge.js";
import { decideWithGather } from "./daemon.js";
import { effectors } from "./effectors.js";
import type { PiLike, PiToolCallEvent, PiToolResult, ToolCallOutcome } from "./pi.js";
import { type Evidence, freshEvidence } from "./types.js";

export interface InstallDeps {
	bridge: BridgeClient;
	daemonUrl: string;
	/** Injected for tests (fakes /decide); production uses the global fetch. */
	fetchImpl?: typeof fetch;
	log?: (msg: string) => void;
	retrieveK?: number;
	/** Fired best-effort when a terminal decision is logged — the app binds the owner's
	 *  in-session verdict to this decisionId (POST /log_reaction). */
	onDecision?: (decision: { decisionId: string; effector: string }) => void;
}

const ANSWER_TOOL = "answer";
const GOVERN_TIMEOUT_MS = 120_000; // backstop around the whole govern loop

// demo-grade owner-scope proxy: the gate uses ask.owner_question; here a lexical
// "my X" test. owner_scoped is daemon-ignored in v0, so this is non-load-bearing.
function ownerScopedHeuristic(q: string): boolean {
	return /\b(my|mine|i|me|i'm|im|myself|our)\b/i.test(q);
}

function text(t: string): PiToolResult {
	return { content: [{ type: "text", text: t }], details: {} };
}

function dedupeKeys(hits: { artifact_cache_key: string }[]): string[] {
	return [...new Set(hits.map((h) => h.artifact_cache_key))];
}

export function installAnswerBrain(pi: PiLike, deps: InstallDeps): void {
	const log = deps.log ?? (() => {});
	const k = deps.retrieveK ?? 20;
	const bridge = deps.bridge;
	const perCallTimeout = deps.fetchImpl ? 5_000 : 60_000;

	// One evidence accumulator per question; replaced wholesale on a new retrieve
	// (statelessness across questions — the daemon/bridge hold none).
	let evidence: Evidence = freshEvidence("");
	let uBar: Record<string, number> | null = null;
	const getUBar = async (): Promise<Record<string, number>> => {
		if (uBar === null) uBar = await bridge.utility();
		return uBar;
	};

	pi.registerTool({
		name: "retrieve_documents",
		label: "Retrieve documents",
		description:
			"Retrieve documents from the personal corpus for a point-fact question and route it. " +
			"Call this FIRST, passing the user's question verbatim.",
		parameters: Type.Object({
			question: Type.String({ description: "the user's question, verbatim" }),
		}),
		async execute(_id, params) {
			const question = String(params.question ?? "");
			evidence = freshEvidence(question);
			evidence.ownerScoped = ownerScopedHeuristic(question);
			const route = await bridge.route(question);
			evidence.route = route;
			if (route === null) {
				return text(
					`"${question}" is not a single point-fact lookup. Answer it from general reading; ` +
						`the answer tool will not gather typed evidence for it.`,
				);
			}
			const hits = await bridge.retrieve(question, "", k);
			evidence.hits = hits;
			evidence.hitKeys = dedupeKeys(hits);
			evidence.docDate = await bridge.probeRecency(evidence.hitKeys);
			return text(
				`Routed as "${route.construct}" (time-indexed: ${route.time_indexed}). ` +
					`Retrieved ${hits.length} document(s). Now call extract_candidates.`,
			);
		},
	});

	pi.registerTool({
		name: "extract_candidates",
		label: "Extract candidates",
		description:
			"Extract the candidate values for the question from the retrieved documents. " +
			"Call after retrieve_documents and before answer.",
		parameters: Type.Object({}),
		async execute() {
			if (!evidence.route) return text("No typed lookup in progress — call retrieve_documents first.");
			if (!evidence.hits.length) return text("No documents retrieved — call retrieve_documents first.");
			const timeIndexed = evidence.route.time_indexed;
			const re = await bridge.extract(
				evidence.question,
				evidence.hits,
				{ doc_date: evidence.docDate },
				timeIndexed,
			);
			evidence.candidates = re.candidates;
			evidence.observations = re.observations;
			evidence.rho = re.rho;
			evidence.eraSplit = re.era_split;
			// mirror gather.py: recency is pre-applied iff the router time-indexed the value.
			evidence.appliedProbes = timeIndexed ? ["recency"] : [];
			evidence.extracted = true;
			return text(
				`Extracted ${re.candidates.length} candidate value(s)` +
					(re.era_split ? " — their documents span eras, so recency may discriminate" : "") +
					`. Now call answer with your best value.`,
			);
		},
	});

	pi.registerTool({
		name: ANSWER_TOOL,
		label: "Answer",
		description:
			"Answer the question. The answer brain governs this call: it may rewrite your value to the " +
			"evidence-backed one, send you to gather more, or withhold if the evidence is too uncertain.",
		parameters: Type.Object({
			value: Type.String({ description: "your best answer to the question" }),
			credence: Type.Optional(Type.String()),
			hedged: Type.Optional(Type.Boolean()),
		}),
		async execute(_id, params) {
			// Reached only when the governor ALLOWED (report/hedge), or for a narrative
			// question with no typed governance. `value` is the brain's by then (rewritten).
			const value = String(params.value ?? "");
			const credence = params.credence ? ` (credence ${String(params.credence)})` : "";
			const hedged = params.hedged ? " [candidate set]" : "";
			return text(`${value}${credence}${hedged}`);
		},
	});

	pi.on("tool_call", (event: PiToolCallEvent) => {
		if (event.toolName !== ANSWER_TOOL) return undefined;
		// Narrative path / nothing extracted ⇒ no typed candidates to govern ⇒ let the
		// LLM's answer through (the daemon would have nothing to decide over).
		if (!evidence.extracted || !evidence.candidates.length) return undefined;

		return govern<{ ev: Evidence; u: Record<string, number> }, ToolCallOutcome>({
			buildRequest: async () => ({ ev: evidence, u: await getUBar() }),
			askBrain: ({ ev, u }) =>
				decideWithGather(ev, u, bridge, {
					daemonUrl: deps.daemonUrl,
					timeoutMs: perCallTimeout,
					fetchImpl: deps.fetchImpl,
					onStep: log,
					onDecision: deps.onDecision,
				}),
			dispatch: (signal) => dispatchEffector(effectors, signal, (impl, p) => impl(p, event)),
			// Fail CLOSED for an answer agent: a brain-unreachable failure must NOT leak an
			// ungoverned (possibly confident-wrong) answer — withhold and say why.
			failOpen: { block: true, reason: "The answer brain is unavailable; not answering ungoverned." },
			timeoutMs: GOVERN_TIMEOUT_MS,
			onError: (m, e) => log(`${m}${e === undefined ? "" : `: ${String(e)}`}`),
		});
	});
}
