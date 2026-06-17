// index.ts — the answer-brain extension factory. Verifies the body against the
// daemon-served manifest (fail closed), then installs the tools + governor. Stays
// typed on `PiLike` so the extension typechecks + tests without pi-mono; the demo
// app (apps/answer-brain/app) adapts the real ExtensionAPI to it.

import { fetchManifest, verify } from "@credence/brain-body";

import { type BridgeClient, createBridgeClient } from "./bridge.js";
import { extractors } from "./features.js";
import { effectors } from "./effectors.js";
import type { PiLike } from "./pi.js";
import { installAnswerBrain } from "./tools.js";

export interface ExtensionOptions {
	/** Daemon base URL (GET /manifest, POST /decide). Default $ANSWER_BRAIN_DAEMON_URL or :8799. */
	daemonUrl?: string;
	/** Bridge base URL. Default $LIFE_AGENT_BRIDGE_URL or :8798. */
	bridgeUrl?: string;
	/** Injected fetch (tests fake /manifest + /decide). */
	fetchImpl?: typeof fetch;
	/** Injected bridge (tests fake it); built from bridgeUrl otherwise. */
	bridge?: BridgeClient;
	log?: (msg: string) => void;
	retrieveK?: number;
}

const DEFAULT_DAEMON_URL = "http://127.0.0.1:8799";
const DEFAULT_BRIDGE_URL = "http://127.0.0.1:8798";

/**
 * Verify the body against the daemon's manifest (the body has no parser — it fetches
 * + checks), then register the tools + governor on `pi`. Throws if the manifest is
 * unreachable or a declared effector/feature lacks an impl (fail closed at startup).
 */
export async function createAnswerBrainExtension(pi: PiLike, opts: ExtensionOptions = {}): Promise<void> {
	const env = (k: string): string | undefined => globalThis.process?.env?.[k];
	const daemonUrl = opts.daemonUrl ?? env("ANSWER_BRAIN_DAEMON_URL") ?? DEFAULT_DAEMON_URL;
	const bridgeUrl = opts.bridgeUrl ?? env("LIFE_AGENT_BRIDGE_URL") ?? DEFAULT_BRIDGE_URL;
	const log = opts.log ?? ((m: string) => console.warn(`[answer-brain] ${m}`));
	const bridge = opts.bridge ?? createBridgeClient(bridgeUrl, { fetchImpl: opts.fetchImpl });

	const manifest = await fetchManifest(daemonUrl, { fetchImpl: opts.fetchImpl });
	verify(manifest, { effectors, features: extractors });
	log(`manifest verified — ${manifest.effectors.length} effectors, ${manifest.features.length} features`);

	installAnswerBrain(pi, {
		bridge,
		daemonUrl,
		fetchImpl: opts.fetchImpl,
		log,
		retrieveK: opts.retrieveK,
	});
}

/** Default ExtensionFactory shape: `(pi) => Promise<void>` (move-1 examples). The demo
 *  app passes the real pi adapted to PiLike. */
export default function answerBrainExtension(pi: PiLike): Promise<void> {
	return createAnswerBrainExtension(pi);
}

export { installAnswerBrain } from "./tools.js";
export { effectors } from "./effectors.js";
export { extractors } from "./features.js";
export { createBridgeClient } from "./bridge.js";
export type { BridgeClient } from "./bridge.js";
export * from "./pi.js";
export * from "./types.js";
