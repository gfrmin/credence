// dispatch.ts — the effector-registry + feature-extractor harness, parametric
// over the body's impl and context types. These are the two "plug points" a
// concrete body fills: a registry of effector implementations (keyed by the
// manifest's effector names) and a set of feature extractors (keyed by the
// manifest's feature names). The harness here is vocabulary-agnostic; the body
// supplies the impls and the names, and `verify` (manifest.ts) checks the two
// line up against what the daemon declares.

import type { EffectorSignal } from "./wire.js";

/** A body's effector implementations, keyed by manifest effector name. */
export type EffectorRegistry<TImpl> = Record<string, TImpl>;

/**
 * Look up the impl for a signal's effector and run it via the body-supplied
 * `run` adapter (which knows how to call `TImpl` and shape the host result).
 *
 * Throws on an unknown effector — the brain named a tentacle the body never
 * registered. `govern` turns that throw into a fail-open, so an unrecognised
 * effector degrades safely instead of silently no-op'ing.
 */
export async function dispatchEffector<TImpl, TResult>(
	registry: EffectorRegistry<TImpl>,
	signal: EffectorSignal,
	run: (impl: TImpl, parameters: Record<string, unknown>) => TResult | Promise<TResult>,
): Promise<TResult> {
	const impl = registry[signal.effector];
	if (impl === undefined) {
		throw new Error(`brain-body: no impl registered for effector "${signal.effector}"`);
	}
	return run(impl, signal.parameters);
}

/** A feature extractor reads the body's per-call context and returns one
 *  kebab-case space member (matching the feature's declared space). */
export type FeatureExtractor<TCtx> = (ctx: TCtx) => string;

/**
 * Run every extractor over `ctx`, collecting the feature dict the body sends to
 * the daemon (the *projections* the daemon's policy reads; the daemon decides).
 */
export function assembleFeatures<TCtx>(
	extractors: Record<string, FeatureExtractor<TCtx>>,
	ctx: TCtx,
): Record<string, string> {
	const out: Record<string, string> = {};
	for (const [name, fn] of Object.entries(extractors)) {
		out[name] = fn(ctx);
	}
	return out;
}
