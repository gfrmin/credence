// Transport-free body tests: a fake pi + a fake bridge + a fake daemon (scripted
// /decide). The §1b proof obligation — gather-then-report on the mobile class, an
// abstain that withholds its leader, statelessness across questions — plus
// fail-closed on an unreachable daemon and manifest verify.

import { test } from "node:test";
import assert from "node:assert/strict";

import type { BridgeClient } from "../src/bridge.js";
import { createAnswerBrainExtension } from "../src/index.js";
import type { PiLike, PiToolCallEvent, PiToolDefinition, ToolCallHandler, ToolCallOutcome } from "../src/pi.js";
import type { DecideResponse, ExtractResult, Hit, LoggedDecision, RouteResult } from "../src/types.js";

const MANIFEST = {
	effectors: [
		{ name: "answer", parameters: [{ name: "value", type: "string" }] },
		{ name: "ask-user", parameters: [{ name: "text", type: "string" }] },
		{ name: "abstain", parameters: [{ name: "reason", type: "string" }] },
		{ name: "gather", parameters: [{ name: "target", type: "string" }] },
	],
	features: [
		{ name: "posterior-dispersion", spaceName: "dispersion-space" },
		{ name: "leader-credence-band", spaceName: "leader-band-space" },
		{ name: "candidates-era-split", spaceName: "era-split-space" },
		{ name: "owner-scoped", spaceName: "owner-scoped-space" },
	],
};

interface Harness {
	pi: PiLike;
	tool(name: string): PiToolDefinition;
	govern(event: PiToolCallEvent): Promise<ToolCallOutcome>;
}

function fakePi(): Harness {
	const tools = new Map<string, PiToolDefinition>();
	let handler: ToolCallHandler | undefined;
	const pi: PiLike = {
		registerTool: (t) => void tools.set(t.name, t),
		on: (_e, h) => void (handler = h),
	};
	return {
		pi,
		tool: (n) => {
			const t = tools.get(n);
			if (!t) throw new Error(`no tool ${n}`);
			return t;
		},
		govern: async (event) => {
			if (!handler) throw new Error("no tool_call handler registered");
			return handler(event);
		},
	};
}

function extractResult(over: Partial<ExtractResult> = {}): ExtractResult {
	return {
		candidates: ["Vstale", "Vcur"],
		observations: [{ reports: 1, group: 1, authority: 0.9, subject_factor: 1, time_factor: 1 }],
		rho: 0.7,
		indeterminate: 0,
		era_split: true,
		...over,
	};
}

function decide(over: Partial<DecideResponse>): DecideResponse {
	return {
		effector: "abstain",
		report_index: null,
		value: null,
		probe: null,
		target: null,
		credences: [],
		p_none: 1,
		eu: 0,
		...over,
	};
}

interface FakeBridgeOpts {
	route?: RouteResult | null;
	hits?: Hit[];
	extractBaseline?: ExtractResult;
	extractRecency?: ExtractResult;
	logThrows?: boolean; // /log_decision rejects — proves logging is non-fatal to the decision
}

interface LogCall {
	question: string;
	retrievalKeys: string[];
	decision: LoggedDecision;
}

function fakeBridge(opts: FakeBridgeOpts = {}): {
	bridge: BridgeClient;
	extractCalls: boolean[];
	logged: LogCall[];
} {
	const extractCalls: boolean[] = [];
	const logged: LogCall[] = [];
	const bridge: BridgeClient = {
		route: async () => (opts.route === undefined ? { construct: "mobile number", time_indexed: false } : opts.route),
		retrieve: async () => opts.hits ?? [{ artifact_cache_key: "d1" }, { artifact_cache_key: "d2" }],
		probeRecency: async () => ({ d1: "2026-01-01", d2: "2015-01-01" }),
		extract: async (_q, _h, _cov, timeIndexed) => {
			extractCalls.push(timeIndexed);
			return timeIndexed && opts.extractRecency
				? opts.extractRecency
				: (opts.extractBaseline ?? extractResult());
		},
		utility: async () => ({ u_correct: 1, u_wrong: -4, u_hedged: -0.5, u_abstain: 0, lambda_int: 1 }),
		logDecision: async (question, retrievalKeys, decision) => {
			if (opts.logThrows) throw new Error("bridge /log_decision → 500");
			logged.push({ question, retrievalKeys, decision });
			return { decision_id: `ab-test-${logged.length}` };
		},
	};
	return { bridge, extractCalls, logged };
}

// GET /manifest → MANIFEST; POST /decide → the next scripted response.
function fakeFetch(decideQueue: DecideResponse[], manifest: unknown = MANIFEST): typeof fetch {
	const queue = [...decideQueue];
	return (async (url: string) => {
		const u = String(url);
		if (u.endsWith("/manifest")) return { ok: true, status: 200, json: async () => manifest };
		if (u.endsWith("/decide")) {
			const r = queue.shift();
			if (!r) throw new Error("fakeFetch: /decide queue exhausted");
			return { ok: true, status: 200, json: async () => r };
		}
		throw new Error(`fakeFetch: unexpected url ${u}`);
	}) as unknown as typeof fetch;
}

test("gather-then-report: the daemon steers recency, the body re-extracts and reports the current value", async () => {
	const fb = fakeBridge({ extractBaseline: extractResult(), extractRecency: extractResult() });
	const ff = fakeFetch([
		decide({ effector: "gather", probe: "recency", target: "Vstale", credences: [0.55, 0.3], p_none: 0.15 }),
		decide({ effector: "report", value: "Vcur", report_index: 1, credences: [0.1, 0.85], p_none: 0.05 }),
	]);
	const h = fakePi();
	await createAnswerBrainExtension(h.pi, { bridge: fb.bridge, fetchImpl: ff, log: () => {} });

	await h.tool("retrieve_documents").execute("1", { question: "what is my mobile number?" });
	await h.tool("extract_candidates").execute("2", {});
	const event: PiToolCallEvent = { toolName: "answer", input: { value: "Vstale" } };
	const outcome = await h.govern(event);

	assert.equal(outcome, undefined, "allowed (report)");
	assert.equal(event.input.value, "Vcur", "rewritten to the brain's current value");
	assert.deepEqual(fb.extractCalls, [false, true], "baseline extract, then recency-applied re-extract");
});

test("abstain withholds and names the leader (abstain-show-withheld)", async () => {
	const fb = fakeBridge({ extractBaseline: extractResult({ era_split: false }) });
	const ff = fakeFetch([decide({ effector: "abstain", credences: [0.3, 0.25], p_none: 0.45 })]);
	const h = fakePi();
	await createAnswerBrainExtension(h.pi, { bridge: fb.bridge, fetchImpl: ff, log: () => {} });

	await h.tool("retrieve_documents").execute("1", { question: "what is my X?" });
	await h.tool("extract_candidates").execute("2", {});
	const event: PiToolCallEvent = { toolName: "answer", input: { value: "Vstale" } };
	const outcome = await h.govern(event);

	assert.ok(outcome?.block, "blocked");
	assert.match(outcome.reason, /Vstale/, "names the withheld leader");
	assert.equal(event.input.value, "Vstale", "input not rewritten (no assertion)");
});

test("statelessness: a new narrative question does not reuse the prior question's candidates", async () => {
	let routeCall = 0;
	const bridge: BridgeClient = {
		route: async () => (++routeCall === 1 ? { construct: "c", time_indexed: false } : null),
		retrieve: async () => [{ artifact_cache_key: "d1" }],
		probeRecency: async () => ({ d1: "2026-01-01" }),
		extract: async () => extractResult({ candidates: ["A"], era_split: false }),
		utility: async () => ({ u_correct: 1, u_wrong: -4, u_hedged: -0.5, u_abstain: 0, lambda_int: 1 }),
		logDecision: async () => ({ decision_id: "ab-test" }),
	};
	// Only ONE /decide is scripted (q1). If q2's evidence leaked, q2 would decide → throw.
	const ff = fakeFetch([decide({ effector: "report", value: "A", report_index: 0, credences: [0.9], p_none: 0.1 })]);
	const h = fakePi();
	await createAnswerBrainExtension(h.pi, { bridge, fetchImpl: ff, log: () => {} });

	await h.tool("retrieve_documents").execute("1", { question: "what is my id?" });
	await h.tool("extract_candidates").execute("2", {});
	const e1: PiToolCallEvent = { toolName: "answer", input: { value: "x" } };
	assert.equal(await h.govern(e1), undefined);
	assert.equal(e1.input.value, "A");

	// q2 routes narrative → retrieve resets the accumulator → governor must not reuse q1's candidates.
	await h.tool("retrieve_documents").execute("3", { question: "tell me about my week" });
	const e2: PiToolCallEvent = { toolName: "answer", input: { value: "my own answer" } };
	assert.equal(await h.govern(e2), undefined, "not governed (evidence reset, nothing extracted)");
	assert.equal(e2.input.value, "my own answer", "left untouched");
});

test("fails CLOSED (withholds) when the daemon is unreachable", async () => {
	const fb = fakeBridge({ extractBaseline: extractResult({ era_split: false }) });
	const ff = ((url: string) => {
		if (String(url).endsWith("/manifest")) return Promise.resolve({ ok: true, status: 200, json: async () => MANIFEST });
		return Promise.reject(new Error("ECONNREFUSED"));
	}) as unknown as typeof fetch;
	const h = fakePi();
	await createAnswerBrainExtension(h.pi, { bridge: fb.bridge, fetchImpl: ff, log: () => {} });

	await h.tool("retrieve_documents").execute("1", { question: "my id?" });
	await h.tool("extract_candidates").execute("2", {});
	const event: PiToolCallEvent = { toolName: "answer", input: { value: "draft" } };
	const outcome = await h.govern(event);

	assert.ok(outcome?.block, "fail-closed → blocked");
	assert.match(outcome.reason, /unavailable/i);
	assert.equal(event.input.value, "draft", "no ungoverned answer emitted");
});

test("createAnswerBrainExtension fails closed when the manifest declares an unimplemented effector", async () => {
	const bad = { effectors: [...MANIFEST.effectors, { name: "teleport", parameters: [] }], features: MANIFEST.features };
	const ff = fakeFetch([], bad);
	const h = fakePi();
	await assert.rejects(
		() => createAnswerBrainExtension(h.pi, { bridge: fakeBridge().bridge, fetchImpl: ff, log: () => {} }),
		/no impl for declared effector\(s\): teleport/,
	);
});

test("registers retrieve / extract / answer tools", async () => {
	const h = fakePi();
	await createAnswerBrainExtension(h.pi, { bridge: fakeBridge().bridge, fetchImpl: fakeFetch([]), log: () => {} });
	assert.ok(h.tool("retrieve_documents"));
	assert.ok(h.tool("extract_candidates"));
	assert.ok(h.tool("answer"));
});

// --- the verdict-emission seam: the body posts the terminal decision to /log_decision -----

async function runOnce(
	fb: ReturnType<typeof fakeBridge>,
	decideQueue: DecideResponse[],
	question = "what is my mobile number?",
): Promise<ToolCallOutcome> {
	const h = fakePi();
	await createAnswerBrainExtension(h.pi, { bridge: fb.bridge, fetchImpl: fakeFetch(decideQueue), log: () => {} });
	await h.tool("retrieve_documents").execute("1", { question });
	await h.tool("extract_candidates").execute("2", {});
	return h.govern({ toolName: "answer", input: { value: "draft" } });
}

test("logs the terminal report decision to the bridge for the reaction loop", async () => {
	const fb = fakeBridge({ extractBaseline: extractResult({ era_split: false }) });
	await runOnce(fb, [decide({ effector: "report", value: "Vcur", report_index: 1, credences: [0.1, 0.85], p_none: 0.05 })]);
	assert.equal(fb.logged.length, 1, "one decision logged");
	const { question, retrievalKeys, decision } = fb.logged[0];
	assert.equal(question, "what is my mobile number?");
	assert.deepEqual(retrievalKeys, ["d1", "d2"], "the hit keys the decision was grounded on");
	assert.equal(decision.effector, "report");
	assert.deepEqual(decision.credences, [0.1, 0.85], "credences forwarded in candidate order (bridge sorts)");
	assert.deepEqual(decision.candidates, ["Vstale", "Vcur"]);
	assert.equal(decision.p_none, 0.05);
});

test("logs the abstain decision (the row that folds into u_wrong)", async () => {
	const fb = fakeBridge({ extractBaseline: extractResult({ era_split: false }) });
	const outcome = await runOnce(fb, [decide({ effector: "abstain", credences: [0.3, 0.25], p_none: 0.45 })]);
	assert.ok(outcome?.block, "abstain blocks");
	assert.equal(fb.logged.length, 1);
	assert.equal(fb.logged[0].decision.effector, "abstain");
});

test("gather-then-report logs ONCE — the terminal report, not the intermediate steer", async () => {
	const fb = fakeBridge({ extractBaseline: extractResult(), extractRecency: extractResult() });
	await runOnce(fb, [
		decide({ effector: "gather", probe: "recency", target: "Vstale", credences: [0.55, 0.3], p_none: 0.15 }),
		decide({ effector: "report", value: "Vcur", report_index: 1, credences: [0.1, 0.85], p_none: 0.05 }),
	]);
	assert.equal(fb.logged.length, 1, "the gather steer is not a logged decision");
	assert.equal(fb.logged[0].decision.effector, "report");
});

test("a logging failure is non-fatal: the report is still allowed (answer unaffected)", async () => {
	const fb = fakeBridge({ extractBaseline: extractResult({ era_split: false }), logThrows: true });
	const h = fakePi();
	await createAnswerBrainExtension(h.pi, {
		bridge: fb.bridge,
		fetchImpl: fakeFetch([decide({ effector: "report", value: "Vcur", report_index: 1, credences: [0.1, 0.85], p_none: 0.05 })]),
		log: () => {},
	});
	await h.tool("retrieve_documents").execute("1", { question: "my mobile?" });
	await h.tool("extract_candidates").execute("2", {});
	const event: PiToolCallEvent = { toolName: "answer", input: { value: "draft" } };
	const outcome = await h.govern(event);
	assert.equal(outcome, undefined, "report still ALLOWED despite the logging failure");
	assert.equal(event.input.value, "Vcur", "value still rewritten to the brain's");
});

test("does not log when the daemon is unreachable (no decision was made)", async () => {
	const fb = fakeBridge({ extractBaseline: extractResult({ era_split: false }) });
	const ff = ((url: string) => {
		if (String(url).endsWith("/manifest")) return Promise.resolve({ ok: true, status: 200, json: async () => MANIFEST });
		return Promise.reject(new Error("ECONNREFUSED"));
	}) as unknown as typeof fetch;
	const h = fakePi();
	await createAnswerBrainExtension(h.pi, { bridge: fb.bridge, fetchImpl: ff, log: () => {} });
	await h.tool("retrieve_documents").execute("1", { question: "my id?" });
	await h.tool("extract_candidates").execute("2", {});
	const outcome = await h.govern({ toolName: "answer", input: { value: "draft" } });
	assert.ok(outcome?.block, "fail-closed");
	assert.equal(fb.logged.length, 0, "nothing logged — only real terminal decisions are recorded");
});

test("a hung logger does not block or delay the decision (fire-and-forget)", async () => {
	const fb = fakeBridge({ extractBaseline: extractResult({ era_split: false }) });
	// A logger that NEVER resolves: were it awaited, the govern loop would hang to its timeout and
	// fail-closed — silently withholding a good answer. Fire-and-forget must return the decision
	// without waiting on the logger. (With `await` instead of `void`, this test would time out.)
	fb.bridge.logDecision = () => new Promise<{ decision_id: string }>(() => {});
	const outcome = await runOnce(fb, [
		decide({ effector: "report", value: "Vcur", report_index: 1, credences: [0.1, 0.85], p_none: 0.05 }),
	]);
	assert.equal(outcome, undefined, "report allowed immediately, not waiting on the logger");
});

// --- onDecision: the hook the app binds the in-session verdict to -------------------------

const flush = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

test("onDecision fires with the logged decision_id and effector (the app's react hook)", async () => {
	const fb = fakeBridge({ extractBaseline: extractResult({ era_split: false }) });
	const seen: { decisionId: string; effector: string }[] = [];
	const h = fakePi();
	await createAnswerBrainExtension(h.pi, {
		bridge: fb.bridge,
		fetchImpl: fakeFetch([decide({ effector: "report", value: "Vcur", report_index: 1, credences: [0.1, 0.85], p_none: 0.05 })]),
		log: () => {},
		onDecision: (d) => seen.push(d),
	});
	await h.tool("retrieve_documents").execute("1", { question: "my mobile?" });
	await h.tool("extract_candidates").execute("2", {});
	await h.govern({ toolName: "answer", input: { value: "draft" } });
	await flush(); // onDecision fires after the fire-and-forget log resolves
	assert.equal(seen.length, 1);
	assert.equal(seen[0].effector, "report");
	assert.ok(seen[0].decisionId.startsWith("ab-test-"), "carries the bridge's decision_id");
});

test("onDecision does not fire when no decision is logged (daemon unreachable)", async () => {
	const fb = fakeBridge({ extractBaseline: extractResult({ era_split: false }) });
	const seen: unknown[] = [];
	const ff = ((url: string) => {
		if (String(url).endsWith("/manifest")) return Promise.resolve({ ok: true, status: 200, json: async () => MANIFEST });
		return Promise.reject(new Error("ECONNREFUSED"));
	}) as unknown as typeof fetch;
	const h = fakePi();
	await createAnswerBrainExtension(h.pi, { bridge: fb.bridge, fetchImpl: ff, log: () => {}, onDecision: (d) => seen.push(d) });
	await h.tool("retrieve_documents").execute("1", { question: "my id?" });
	await h.tool("extract_candidates").execute("2", {});
	await h.govern({ toolName: "answer", input: { value: "draft" } });
	await flush();
	assert.equal(seen.length, 0, "no decision logged → no react hook");
});
