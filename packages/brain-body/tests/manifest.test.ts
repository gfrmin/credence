import { test } from "node:test";
import assert from "node:assert/strict";

import { fetchManifest, verify } from "../src/manifest.js";
import type { Manifest } from "../src/wire.js";

const MANIFEST: Manifest = {
	effectors: [
		{ name: "answer", parameters: [{ name: "value", type: "string" }] },
		{ name: "abstain", parameters: [{ name: "reason", type: "string" }] },
	],
	features: [{ name: "owner-scoped", spaceName: "owner-scoped-space" }],
};

function fakeFetch(status: number, body: unknown): typeof fetch {
	return (async () => ({
		ok: status >= 200 && status < 300,
		status,
		json: async () => body,
	})) as unknown as typeof fetch;
}

test("fetchManifest GETs {daemonUrl}/manifest (trailing slashes trimmed)", async () => {
	let calledUrl = "";
	const fetchImpl = (async (url: string) => {
		calledUrl = url;
		return { ok: true, status: 200, json: async () => MANIFEST };
	}) as unknown as typeof fetch;
	const m = await fetchManifest("http://d:1//", { fetchImpl });
	assert.equal(calledUrl, "http://d:1/manifest");
	assert.deepEqual(m, MANIFEST);
});

test("fetchManifest throws on a non-2xx", async () => {
	await assert.rejects(
		() => fetchManifest("http://d:1", { fetchImpl: fakeFetch(500, {}) }),
		/GET \/manifest → 500/,
	);
});

test("fetchManifest throws on a structurally-malformed body", async () => {
	await assert.rejects(
		() => fetchManifest("http://d:1", { fetchImpl: fakeFetch(200, { effectors: [] }) }),
		/malformed manifest/,
	);
});

test("verify passes when every declared item has an impl", () => {
	verify(MANIFEST, {
		effectors: { answer: 1, abstain: 1 },
		features: { "owner-scoped": 1 },
	});
});

test("verify names a missing effector impl (fails closed)", () => {
	assert.throws(
		() => verify(MANIFEST, { effectors: { answer: 1 }, features: { "owner-scoped": 1 } }),
		/no impl for declared effector\(s\): abstain/,
	);
});

test("verify names a missing feature extractor (fails closed)", () => {
	assert.throws(
		() => verify(MANIFEST, { effectors: { answer: 1, abstain: 1 }, features: {} }),
		/no extractor for declared feature\(s\): owner-scoped/,
	);
});

test("verify allows extra impls beyond the manifest (action→effector fan-in)", () => {
	verify(MANIFEST, {
		effectors: { answer: 1, abstain: 1, gather: 1 },
		features: { "owner-scoped": 1, extra: 1 },
	});
});
