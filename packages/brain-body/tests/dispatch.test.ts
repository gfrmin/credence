import { test } from "node:test";
import assert from "node:assert/strict";

import { assembleFeatures, dispatchEffector, type EffectorRegistry } from "../src/dispatch.js";
import type { EffectorSignal } from "../src/wire.js";

test("assembleFeatures runs every extractor over the ctx", () => {
	const extractors = {
		"posterior-dispersion": (c: { disp: string }) => c.disp,
		"owner-scoped": (_c: { disp: string }) => "yes",
	};
	const out = assembleFeatures(extractors, { disp: "sharp" });
	assert.deepEqual(out, { "posterior-dispersion": "sharp", "owner-scoped": "yes" });
});

test("dispatchEffector looks up the impl and runs it via the adapter", async () => {
	const registry: EffectorRegistry<(p: Record<string, unknown>) => string> = {
		answer: (p) => `answered:${String(p.value)}`,
	};
	const sig: EffectorSignal = { effector: "answer", parameters: { value: "V" } };
	const res = await dispatchEffector(registry, sig, (impl, params) => impl(params));
	assert.equal(res, "answered:V");
});

test("dispatchEffector throws on an unknown effector", async () => {
	const registry: EffectorRegistry<() => void> = {};
	const sig: EffectorSignal = { effector: "ghost", parameters: {} };
	await assert.rejects(
		() => dispatchEffector(registry, sig, (impl) => impl()),
		/no impl registered for effector "ghost"/,
	);
});
