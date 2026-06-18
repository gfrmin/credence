import { test } from "node:test";
import assert from "node:assert/strict";

import {
  PRESETS,
  PRESET_NAMES,
  DEFAULT_PROFILE,
  resolveProfile,
  PROFILE_KEYS,
} from "../src/profile.js";

test("the four presets exist with every dial", () => {
  assert.deepEqual(PRESET_NAMES.sort(), ["balanced", "cost-saver", "quality-first", "speed-first"]);
  for (const name of PRESET_NAMES) {
    for (const k of PROFILE_KEYS) {
      assert.equal(typeof PRESETS[name][k], "number", `${name}.${k}`);
    }
  }
});

test("resolveProfile: a named preset returns that vector", () => {
  assert.deepEqual(resolveProfile("speed-first", undefined), PRESETS["speed-first"]);
  assert.deepEqual(resolveProfile("cost-saver", undefined), PRESETS["cost-saver"]);
});

test("resolveProfile: default + unknown preset fall back to balanced", () => {
  assert.equal(DEFAULT_PROFILE, "balanced");
  assert.deepEqual(resolveProfile(undefined, undefined), PRESETS["balanced"]);
  assert.deepEqual(resolveProfile("nonsense", undefined), PRESETS["balanced"]);
});

test("resolveProfile: a per-dial override merges on top of the preset", () => {
  const p = resolveProfile("cost-saver", { w_time: 0.5, reward: 3.0 });
  assert.equal(p.w_time, 0.5); // overridden
  assert.equal(p.reward, 3.0); // overridden
  assert.equal(p.lambda, PRESETS["cost-saver"].lambda); // untouched
  assert.equal(p.q, PRESETS["cost-saver"].q); // untouched
});

test("resolveProfile: non-number / non-finite override values are ignored", () => {
  const p = resolveProfile("balanced", { w_time: "fast" as unknown as number, q: NaN, harm: 9 });
  assert.equal(p.w_time, PRESETS["balanced"].w_time); // string ignored
  assert.equal(p.q, PRESETS["balanced"].q); // NaN ignored
  assert.equal(p.harm, 9); // valid number applied
});

test("the presets encode the intended trade-off ordering", () => {
  // time dial: speed-first values a second most, cost-saver least.
  assert.ok(PRESETS["speed-first"].w_time > PRESETS["balanced"].w_time);
  assert.ok(PRESETS["balanced"].w_time > PRESETS["cost-saver"].w_time);
  // quality dial: quality-first pays the most for a correct answer, cost-saver the least.
  assert.ok(PRESETS["quality-first"].reward > PRESETS["balanced"].reward);
  assert.ok(PRESETS["balanced"].reward > PRESETS["cost-saver"].reward);
  // speed-first rarely interrupts (asking costs the user's time): highest q among non-quality.
  assert.ok(PRESETS["speed-first"].q > PRESETS["balanced"].q);
});
