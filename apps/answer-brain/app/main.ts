/**
 * answer-brain app — a pi-mono agent driven by a local Ollama model, answering the owner's
 * point-fact questions through the answer-brain body and capturing a one-bit good/bad verdict
 * that folds into u_wrong (Stage B). THIS IS THE DOGFOOD DRIVER, NOT THE GATE: the gate
 * (life-agent/scripts/answer_brain_gate.py) certifies the decision math deterministically; here
 * a real (nondeterministic) LLM drives the tools and the brain governs the answer, end-to-end.
 *
 * Use `bin/answer-brain` (life-agent), which starts the prereqs for you:
 *   1. the daemon (Julia, :8799)   2. the bridge (:8798)   3. Ollama (model pulled)
 *
 * Run:  npx tsx main.ts                 # REPL: ask, react [g]ood/[b]ad, repeat
 *       npx tsx main.ts "my mobile?"    # one question, react, exit
 */

import * as readline from "node:readline/promises";

import { getModel as _getModel } from "@earendil-works/pi-ai";
import {
	AuthStorage,
	createAgentSession,
	DefaultResourceLoader,
	type ExtensionAPI,
	getAgentDir,
	ModelRegistry,
	SessionManager,
} from "@earendil-works/pi-coding-agent";

import { createAnswerBrainExtension, type PiLike } from "answer-brain-extension";

void _getModel; // (kept for reference; we register a local provider below)

const DAEMON_URL = process.env.ANSWER_BRAIN_DAEMON_URL ?? "http://127.0.0.1:8799";
const BRIDGE_URL = process.env.LIFE_AGENT_BRIDGE_URL ?? "http://127.0.0.1:8798";
const MODEL_ID = process.env.ANSWER_BRAIN_MODEL ?? "qwen2.5:7b-instruct";
const OLLAMA_URL = process.env.OLLAMA_BASE_URL ?? "http://localhost:11434/v1";

const SYSTEM_PROMPT = `You are the owner's personal answer agent. For ANY question that asks for a
specific fact about the owner or their documents (a number, ID, date, name, address, amount, status),
you MUST use the tools — never answer from memory:
  1. retrieve_documents(question)  — pass the user's question verbatim
  2. extract_candidates()
  3. answer(value)                 — your best value
The answer brain governs the answer tool: it may rewrite your value to the evidence-backed one, send
you back to gather more evidence, or withhold if the evidence is too uncertain. Trust its decision and
relay it. After the answer tool returns, state the final answer to the owner in one sentence.`;

// Adapt the real ExtensionAPI to the body's minimal PiLike. The casts bridge the
// (wider) real tool/handler types to the body's subset; structurally compatible at runtime.
function adaptPi(pi: ExtensionAPI): PiLike {
	return {
		registerTool: (tool) => pi.registerTool(tool as never),
		on: (event, handler) => pi.on(event, handler as never),
	};
}

// The decision the brain just logged for the current question (set by the extension's
// onDecision hook), or null on a narrative/unlogged answer. The verdict prompt binds to it.
let lastDecision: { decisionId: string; effector: string } | null = null;
// Resolves the current question's "decision logged" wait — set per question in ask(), called by
// the onDecision hook so the verdict prompt binds deterministically, not on a timed guess.
let onDecisionLogged: (() => void) | null = null;

// Post the owner's one-bit verdict to the bridge (/log_reaction) and echo the fold fate.
// Fail-open: a verdict that fails to record must never break the loop.
async function postReaction(decisionId: string, valence: "good" | "bad"): Promise<void> {
	try {
		const resp = await fetch(`${BRIDGE_URL}/log_reaction`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ decision_id: decisionId, valence }),
		});
		if (!resp.ok) {
			console.error(`  verdict not recorded (HTTP ${resp.status})`);
			return;
		}
		const { folds, chosen_action } = (await resp.json()) as { folds: boolean; chosen_action: string };
		const fate = folds ? "folds into u_wrong on the next gate run" : "recorded (only abstain verdicts fold)";
		console.error(`  → ${valence.toUpperCase()} on ${chosen_action} — ${fate}`);
	} catch (e) {
		console.error(`  verdict not recorded: ${String(e)}`);
	}
}

// One bit, no free text (the owner's prose is the loop's one expensive resource). Enter skips.
async function captureVerdict(rl: readline.Interface): Promise<void> {
	if (!lastDecision) return; // narrative path, or the decision was not logged — nothing to grade
	const ans = (await rl.question("  [g]ood / [b]ad / Enter=skip > ")).trim().toLowerCase();
	const valence = ans === "g" ? "good" : ans === "b" ? "bad" : null;
	if (valence) await postReaction(lastDecision.decisionId, valence);
}

// Ask one question end-to-end: reset the per-question decision, stream the answer, then give the
// fire-and-forget decision log a beat to land so the verdict prompt can bind to it.
async function ask(session: { prompt: (q: string) => Promise<unknown> }, question: string): Promise<void> {
	lastDecision = null;
	// Await the fire-and-forget decision log deterministically (bounded) so a verdictable decision
	// is never dropped on a timing race — the onDecision hook resolves `logged`. Times out at 2s
	// (a loopback append is sub-ms; a narrative/unlogged answer never resolves it → skip).
	const logged = new Promise<void>((resolve) => {
		onDecisionLogged = resolve;
	});
	console.error(`\nQ: ${question}\n`);
	await session.prompt(question);
	console.log();
	await Promise.race([logged, new Promise((r) => setTimeout(r, 2000))]);
	onDecisionLogged = null;
}

async function main(): Promise<void> {
	const authStorage = AuthStorage.create();
	const modelRegistry = ModelRegistry.create(authStorage);
	// Register the local Ollama model (OpenAI-compatible). The apiKey is a dummy Ollama ignores.
	modelRegistry.registerProvider("ollama", {
		baseUrl: OLLAMA_URL,
		apiKey: "ollama",
		api: "openai-completions",
		models: [
			{
				id: MODEL_ID,
				name: `${MODEL_ID} (Ollama)`,
				reasoning: false,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 32768,
				maxTokens: 4096,
			},
		],
	});
	const model = modelRegistry.find("ollama", MODEL_ID);
	if (!model) throw new Error(`model ollama/${MODEL_ID} not registered`);

	const resourceLoader = new DefaultResourceLoader({
		cwd: process.cwd(),
		agentDir: getAgentDir(),
		systemPromptOverride: () => SYSTEM_PROMPT,
		appendSystemPromptOverride: () => [],
		extensionFactories: [
			async (pi) => {
				await createAnswerBrainExtension(adaptPi(pi), {
					daemonUrl: DAEMON_URL,
					bridgeUrl: BRIDGE_URL,
					log: (m) => console.error(`  [answer-brain] ${m}`),
					onDecision: (d) => {
						lastDecision = d;
						onDecisionLogged?.();
						onDecisionLogged = null;
					},
				});
			},
		],
	});
	await resourceLoader.reload();

	const { session } = await createAgentSession({
		model,
		authStorage,
		modelRegistry,
		resourceLoader,
		sessionManager: SessionManager.inMemory(),
		// Leave only the answer-brain tools active (no built-in read/bash/edit/write).
		noTools: "builtin",
		thinkingLevel: "off",
	});

	session.subscribe((event) => {
		if (event.type === "message_update" && event.assistantMessageEvent.type === "text_delta") {
			process.stdout.write(event.assistantMessageEvent.delta);
		}
	});

	const argvQuestion = process.argv.slice(2).join(" ").trim();
	const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
	try {
		if (argvQuestion) {
			await ask(session, argvQuestion);
			await captureVerdict(rl);
		} else {
			console.error("answer-brain — ask a point-fact question about yourself; 'quit' to exit.");
			for (;;) {
				const q = (await rl.question("\nask> ")).trim();
				if (q === "" || q === "quit" || q === "exit") break;
				await ask(session, q);
				await captureVerdict(rl);
			}
		}
	} finally {
		rl.close();
		session.dispose();
	}
}

main().catch((err: unknown) => {
	console.error(err);
	process.exit(1);
});
