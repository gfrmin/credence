import { test } from "node:test";
import assert from "node:assert/strict";

import { govern, withTimeout } from "../src/governor.js";
import type { EffectorSignal } from "../src/wire.js";

const SIGNAL: EffectorSignal = { effector: "answer", parameters: { value: "V" } };

test("govern composes → asks → dispatches on the happy path", async () => {
	const seen: string[] = [];
	const res = await govern<{ q: string }, string>({
		buildRequest: () => {
			seen.push("build");
			return { q: "x" };
		},
		askBrain: async (req) => {
			seen.push(`ask:${req.q}`);
			return SIGNAL;
		},
		dispatch: (sig) => {
			seen.push(`dispatch:${sig.effector}`);
			return `done:${String(sig.parameters.value)}`;
		},
		failOpen: "FAILOPEN",
		timeoutMs: 1000,
	});
	assert.equal(res, "done:V");
	assert.deepEqual(seen, ["build", "ask:x", "dispatch:answer"]);
});

test("govern fails open when askBrain throws (brain unreachable)", async () => {
	let logged = false;
	const res = await govern<null, string>({
		buildRequest: () => null,
		askBrain: async () => {
			throw new Error("ECONNREFUSED");
		},
		dispatch: () => "NEVER",
		failOpen: "FAILOPEN",
		timeoutMs: 1000,
		onError: () => {
			logged = true;
		},
	});
	assert.equal(res, "FAILOPEN");
	assert.ok(logged, "onError logged once");
});

test("govern fails open when askBrain exceeds the timeout", async () => {
	const res = await govern<null, string>({
		buildRequest: () => null,
		askBrain: () => new Promise<EffectorSignal>(() => {}), // never resolves
		dispatch: () => "NEVER",
		failOpen: "FAILOPEN",
		timeoutMs: 20,
	});
	assert.equal(res, "FAILOPEN");
});

test("govern fails open when the brain names an unknown effector (dispatch throws)", async () => {
	const res = await govern<null, string>({
		buildRequest: () => null,
		askBrain: async () => ({ effector: "ghost", parameters: {} }),
		dispatch: () => {
			throw new Error('no impl registered for effector "ghost"');
		},
		failOpen: "FAILOPEN",
		timeoutMs: 1000,
	});
	assert.equal(res, "FAILOPEN");
});

test("withTimeout resolves a fast promise and rejects a slow one", async () => {
	assert.equal(await withTimeout(Promise.resolve(42), 1000), 42);
	await assert.rejects(() => withTimeout(new Promise(() => {}), 20), /timed out after 20ms/);
});
