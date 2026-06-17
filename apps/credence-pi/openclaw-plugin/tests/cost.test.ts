import { test } from "node:test";
import assert from "node:assert/strict";

import { computeTurnCost, buildPriceTable } from "../src/cost.js";
import type { LlmOutputEvent } from "../src/openclaw-types.js";

test("known model reconstructs USD from token counts (opus: 5 in / 25 out per Mtok)", () => {
  const table = buildPriceTable(undefined);
  const ev: LlmOutputEvent = {
    model: "claude-opus-4-8",
    usage: { input: 1_000_000, output: 1_000_000 },
  };
  const tc = computeTurnCost(ev, table);
  assert.equal(tc.usd, 30); // 5 + 25 (verified 2026-06 prices)
  assert.equal(tc.total_tokens, 2_000_000);
  assert.equal(tc.model, "claude-opus-4-8");
});

test("family substring match (sonnet) prices a versioned id", () => {
  const table = buildPriceTable(undefined);
  const tc = computeTurnCost(
    { model: "claude-sonnet-4-6-20250101", usage: { input: 1_000_000, output: 0 } },
    table,
  );
  assert.equal(tc.usd, 3); // sonnet input 3 / Mtok
});

test("unknown model → usd null, token counts preserved", () => {
  const table = buildPriceTable(undefined);
  const tc = computeTurnCost(
    { model: "mystery-model-x", usage: { input: 100, output: 50, total: 150 } },
    table,
  );
  assert.equal(tc.usd, null);
  assert.equal(tc.total_tokens, 150);
  assert.equal(tc.input_tokens, 100);
});

test("config override prices an otherwise-unknown model", () => {
  const table = buildPriceTable({ "mystery-model-x": { input: 1, output: 2 } });
  const tc = computeTurnCost(
    { model: "mystery-model-x", usage: { input: 1_000_000, output: 1_000_000 } },
    table,
  );
  assert.equal(tc.usd, 3); // 1 + 2
});

test("missing usage → zero tokens, usd 0 for a priced model", () => {
  const table = buildPriceTable(undefined);
  const tc = computeTurnCost({ model: "claude-haiku-4-5" }, table);
  assert.equal(tc.total_tokens, 0);
  assert.equal(tc.usd, 0);
});
