/**
 * answer-brain demo app — a minimal pi-mono agent driven by a local Ollama model,
 * answering the owner's point-fact questions through the answer-brain body
 * (move-4-design §2E). THIS IS THE DEMO, NOT THE GATE: the gate
 * (life-agent/scripts/answer_brain_gate.py) certifies the decision math
 * deterministically; here a real LLM drives the tools and the brain governs the
 * answer, to *show* the govern+steer loop end-to-end. Its result is reported, not
 * assumed (the LLM is nondeterministic).
 *
 * Prereqs (all on-machine; the corpus is the owner's real PII, so no cloud model):
 *   1. the daemon:  julia --project=$HOME/git/credence $HOME/git/credence/apps/answer-brain/daemon/main.jl
 *   2. the bridge:  (from a checkout with bridge `era_split`) python -m life_agent.bridge.server
 *   3. Ollama with the model pulled (default qwen2.5:7b-instruct).
 *
 * Run:  npx tsx main.ts "what is my mobile phone number?"
 */

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

async function main(): Promise<void> {
	const question = process.argv.slice(2).join(" ").trim() || "What is my mobile phone number?";

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

	console.error(`Q: ${question}\n`);
	try {
		session.subscribe((event) => {
			if (event.type === "message_update" && event.assistantMessageEvent.type === "text_delta") {
				process.stdout.write(event.assistantMessageEvent.delta);
			}
		});
		await session.prompt(question);
		console.log();
	} finally {
		session.dispose();
	}
}

main().catch((err: unknown) => {
	console.error(err);
	process.exit(1);
});
