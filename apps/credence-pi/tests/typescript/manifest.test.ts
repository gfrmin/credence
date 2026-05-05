// manifest.test.ts — step 5 of credence-pi: extension scaffolding and
// manifest parsing. Seven cases via Node's built-in test runner:
// parse capabilities, parse features, happy-path verifier, missing
// implementation, missing extractor, tokeniser smoke, bad form shape.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  parseCapabilities,
  parseFeatures,
  readCapabilities,
  readFeatures,
  verifyEffectors,
  verifyFeatures,
  type EffectorDecl,
  type FeatureDecl,
} from "../../extension/src/manifest.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BDSL_DIR = path.resolve(HERE, "..", "..", "bdsl");
const CAPABILITIES_PATH = path.join(BDSL_DIR, "capabilities.bdsl");
const FEATURES_PATH = path.join(BDSL_DIR, "features.bdsl");

test("parseCapabilities: capabilities.bdsl yields three effectors in order", () => {
  const decls = readCapabilities(CAPABILITIES_PATH);
  const expected: EffectorDecl[] = [
    { name: "ask",     parameters: [{ name: "text",   type: "string" }] },
    { name: "proceed", parameters: [] },
    { name: "block",   parameters: [{ name: "reason", type: "string" }] },
  ];
  assert.deepEqual(decls, expected);
});

test("parseFeatures: features.bdsl yields five features in order", () => {
  const decls = readFeatures(FEATURES_PATH);
  const expected: FeatureDecl[] = [
    { name: "tool-name",                    spaceName: "tool-name-space" },
    { name: "working-directory-relative",   spaceName: "wd-relative-space" },
    { name: "parent-tool-call-name",        spaceName: "parent-tool-name-space" },
    { name: "recent-repetition-count",      spaceName: "rep-count-space" },
    { name: "time-since-last-user-message", spaceName: "time-since-user-space" },
  ];
  assert.deepEqual(decls, expected);
});

test("verifyEffectors / verifyFeatures: happy path with all impls registered", () => {
  const effectorRegistry = {
    "ask":     () => {},
    "proceed": () => {},
    "block":   () => {},
  };
  const featureRegistry = {
    "tool-name":                    () => {},
    "working-directory-relative":   () => {},
    "parent-tool-call-name":        () => {},
    "recent-repetition-count":      () => {},
    "time-since-last-user-message": () => {},
  };
  verifyEffectors(readCapabilities(CAPABILITIES_PATH), effectorRegistry);
  verifyFeatures(readFeatures(FEATURES_PATH), featureRegistry);
  // No throw = pass.
});

test("verifyEffectors: missing implementation names the missing effector", () => {
  const decls = readCapabilities(CAPABILITIES_PATH);
  const registry = {
    "ask":   () => {},
    "block": () => {},
    // "proceed" deliberately omitted
  };
  assert.throws(
    () => verifyEffectors(decls, registry),
    { message: /proceed/ },
  );
});

test("verifyFeatures: missing extractor names the missing feature", () => {
  const decls = readFeatures(FEATURES_PATH);
  const registry = {
    "tool-name":                    () => {},
    "working-directory-relative":   () => {},
    "parent-tool-call-name":        () => {},
    // "recent-repetition-count" deliberately omitted
    "time-since-last-user-message": () => {},
  };
  assert.throws(
    () => verifyFeatures(decls, registry),
    { message: /recent-repetition-count/ },
  );
});

test("tokeniser: handles ; comments and the (define …) outer wrap", () => {
  const src = `
    ; leading comment line
    (define manifest
      (list
        (effector wave (parameters (hand string)))   ; trailing comment
        ; comment between
        (effector hush (parameters))))
  `;
  const decls = parseCapabilities(src);
  assert.deepEqual(decls, [
    { name: "wave", parameters: [{ name: "hand", type: "string" }] },
    { name: "hush", parameters: [] },
  ]);
});

test("bad form shape: (parameters bad-shape) raises with the offending form", () => {
  const src = "(effector ask (parameters bad-shape))";
  assert.throws(
    () => parseCapabilities(src),
    { message: /effector ask.*\(name type\).*bad-shape/ },
  );
});
