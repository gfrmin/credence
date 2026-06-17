// manifest.ts — fetch the body's vocabulary from the daemon and verify the body
// implements it. There is NO BDSL parser here (the de-dup cut, move-4-design §0,
// §5 Q1): the daemon owns the single library parser and serves the parsed
// manifest at GET /manifest; the body fetches it and checks every declared
// effector / feature has a registered impl / extractor. credence-pi's 159-line
// TS parser (extension/src/manifest.ts) is replaced by these two functions.

import type { Manifest } from "./wire.js";

export interface FetchManifestOptions {
	/** Abort the GET after this many ms. Default 10s. */
	timeoutMs?: number;
	/** Injectable fetch for tests (defaults to the global). */
	fetchImpl?: typeof fetch;
}

/**
 * GET {daemonUrl}/manifest → the effector/feature vocabulary. Throws on a
 * non-2xx, a transport error/timeout, or a structurally-malformed body — the
 * caller (the extension factory) decides whether that is fatal at startup.
 */
export async function fetchManifest(
	daemonUrl: string,
	opts: FetchManifestOptions = {},
): Promise<Manifest> {
	const fetchImpl = opts.fetchImpl ?? fetch;
	const base = daemonUrl.replace(/\/+$/, "");
	const ctrl = new AbortController();
	const timer = setTimeout(() => ctrl.abort(), opts.timeoutMs ?? 10_000);
	try {
		const resp = await fetchImpl(`${base}/manifest`, { signal: ctrl.signal });
		if (!resp.ok) {
			throw new Error(`fetchManifest: GET /manifest → ${resp.status}`);
		}
		const json = (await resp.json()) as Manifest;
		if (!Array.isArray(json?.effectors) || !Array.isArray(json?.features)) {
			throw new Error("fetchManifest: malformed manifest (missing effectors[] / features[])");
		}
		return json;
	} finally {
		clearTimeout(timer);
	}
}

/** The body's two registries, checked against the manifest. Values are opaque
 *  here (the harness in dispatch.ts knows how to call them) — `verify` only
 *  cares that a key exists for every declared name. */
export interface Registries {
	effectors: Record<string, unknown>;
	features: Record<string, unknown>;
}

/**
 * Fail closed if the body is missing an impl for any declared effector, or an
 * extractor for any declared feature. Extra registered impls are fine (the body
 * may map several daemon actions onto one effector); only *unimplemented
 * declarations* are an error. Names are reported verbatim (kebab-case) so a
 * developer can grep the bdsl directly.
 */
export function verify(manifest: Manifest, reg: Registries): void {
	const missingEffectors = manifest.effectors
		.filter((e) => !(e.name in reg.effectors))
		.map((e) => e.name);
	const missingFeatures = manifest.features
		.filter((f) => !(f.name in reg.features))
		.map((f) => f.name);

	const problems: string[] = [];
	if (missingEffectors.length > 0) {
		problems.push(`no impl for declared effector(s): ${missingEffectors.join(", ")}`);
	}
	if (missingFeatures.length > 0) {
		problems.push(`no extractor for declared feature(s): ${missingFeatures.join(", ")}`);
	}
	if (problems.length > 0) {
		throw new Error(`brain-body manifest verify: ${problems.join("; ")}`);
	}
}
