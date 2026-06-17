import { test } from "node:test";
import assert from "node:assert/strict";

import { buildRoster, NOMINAL_OUTPUT_TOKENS } from "../src/roster.js";
import { buildPriceTable } from "../src/cost.js";
import type { OpenClawConfigSlice } from "../src/openclaw-types.js";

const TABLE = buildPriceTable(undefined);
// per-call cost = output-$/Mtok × NOMINAL_OUTPUT_TOKENS (the shipped default-roster scale)
const perCall = (outPerM: number) => (outPerM / 1_000_000) * NOMINAL_OUTPUT_TOKENS;

test("buildRoster: reads providers' models + derives per-call cost from config pricing", () => {
  const cfg: OpenClawConfigSlice = {
    models: {
      providers: {
        anthropic: {
          models: [
            { id: "claude-haiku-4-5", name: "haiku", cost: { input: 1, output: 5 } },
            { id: "claude-opus-4-8", name: "opus", cost: { input: 5, output: 25 } },
          ],
        },
      },
    },
  };
  const roster = buildRoster(cfg, TABLE);
  assert.equal(roster.length, 2);
  const haiku = roster.find((m) => m.model === "claude-haiku-4-5")!;
  assert.equal(haiku.provider, "anthropic");
  assert.equal(haiku.name, "haiku");
  assert.equal(haiku.cost, perCall(5)); // own config output price
  const opus = roster.find((m) => m.model === "claude-opus-4-8")!;
  assert.equal(opus.cost, perCall(25));
  // matches the shipped default roster scale exactly (haiku 0.0035, opus 0.0175)
  assert.ok(Math.abs(haiku.cost - 0.0035) < 1e-9 && Math.abs(opus.cost - 0.0175) < 1e-9);
});

test("buildRoster: falls back to the price table when config omits per-model cost", () => {
  const cfg: OpenClawConfigSlice = {
    models: { providers: { anthropic: { models: [{ id: "claude-sonnet-4-6" }] } } },
  };
  const roster = buildRoster(cfg, TABLE);
  assert.equal(roster.length, 1);
  assert.equal(roster[0].cost, perCall(15)); // sonnet output 15 from the price table
});

test("buildRoster: the agent allow-list narrows the roster", () => {
  const cfg: OpenClawConfigSlice = {
    models: {
      providers: {
        anthropic: {
          models: [
            { id: "claude-haiku-4-5", cost: { input: 1, output: 5 } },
            { id: "claude-opus-4-8", cost: { input: 5, output: 25 } },
          ],
        },
      },
    },
    agents: { defaults: { models: { "anthropic/claude-haiku-4-5": {} } } },
  };
  const roster = buildRoster(cfg, TABLE);
  assert.equal(roster.length, 1);
  assert.equal(roster[0].model, "claude-haiku-4-5");
});

test("buildRoster: unpriceable models are skipped (no cost-aware EU without a cost)", () => {
  const cfg: OpenClawConfigSlice = {
    models: { providers: { custom: { models: [{ id: "totally-unknown-model" }] } } },
  };
  assert.equal(buildRoster(cfg, TABLE).length, 0);
});

test("buildRoster: missing/empty config → empty roster (routing stays inactive)", () => {
  assert.deepEqual(buildRoster(undefined, TABLE), []);
  assert.deepEqual(buildRoster({}, TABLE), []);
  assert.deepEqual(buildRoster({ models: { providers: {} } }, TABLE), []);
});
