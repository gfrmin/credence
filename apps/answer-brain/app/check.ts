/**
 * check.ts — a deterministic wiring check, no Ollama, no bridge needed. Loads the
 * answer-brain extension against the LIVE daemon (GET /manifest), and asserts the
 * body verifies the manifest + registers its tools + the governor. This is the
 * runnable, model-free half of the demo: it proves the app↔daemon plumbing and the
 * fetch-manifest-and-verify path against the real daemon.
 *
 * Run (daemon up on :8799):  npx tsx check.ts
 */

import {
	createAnswerBrainExtension,
	type PiLike,
	type PiToolDefinition,
	type ToolCallHandler,
} from "answer-brain-extension";

const DAEMON_URL = process.env.ANSWER_BRAIN_DAEMON_URL ?? "http://127.0.0.1:8799";
const EXPECTED_TOOLS = ["answer", "extract_candidates", "retrieve_documents"];

async function main(): Promise<void> {
	const tools = new Map<string, PiToolDefinition>();
	let handler: ToolCallHandler | undefined;
	const pi: PiLike = {
		registerTool: (t) => void tools.set(t.name, t),
		on: (_e, h) => void (handler = h),
	};

	await createAnswerBrainExtension(pi, {
		daemonUrl: DAEMON_URL,
		log: (m) => console.log(`  [answer-brain] ${m}`),
	});

	const names = [...tools.keys()].sort();
	const toolsOk = EXPECTED_TOOLS.every((n) => tools.has(n));
	const governorOk = handler !== undefined;
	console.log(`daemon:    ${DAEMON_URL}`);
	console.log(`tools:     ${names.join(", ")}`);
	console.log(`governor:  ${governorOk ? "registered (tool_call)" : "MISSING"}`);
	const ok = toolsOk && governorOk;
	console.log(ok ? "\nOK — extension loaded + manifest verified against the live daemon." : "\nFAIL");
	process.exit(ok ? 0 : 1);
}

main().catch((err: unknown) => {
	console.error(`check failed: ${String(err)}`);
	console.error("(is the daemon up?  julia --project=$HOME/git/credence apps/answer-brain/daemon/main.jl)");
	process.exit(1);
});
