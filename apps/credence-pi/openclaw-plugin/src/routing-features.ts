// routing-features.ts — the body's sensory periphery for MODEL ROUTING.
//
// before_model_resolve hands the plugin the prompt (and, on current OpenClaw, attachment
// metadata) — so the honestly-extractable routing feature is the prompt's LENGTH, now in
// finer short/medium/long buckets. Semantic category — the signal the offline oracle
// measured — is NOT available live without a classifier, an arbitrary unmeasured component
// the project bars. The daemon's structure-BMA discovers which length splits predict
// accuracy and collapses the rest to the marginal, so finer buckets are self-correcting.
// (attachments is the natural next covariate: BeforeModelResolveEvent.attachments carries
// kind/mimeType for file-aware routing — added once it has a warm/online path.)
//
// The bucket boundaries are the train/live SYNCHRONISATION POINT. The warm belief
// (apps/credence-pi/eval/train_routing_from_matrix.jl) seeds each prompt-length cell from
// the agentic Terminal-Bench matrix's instruction lengths bucketed the SAME way — so that
// is the regime the daemon's warm belief was measured on. Change a boundary here and you
// must re-seed (re-run that trainer), or the body's "short" stops matching the belief's
// "short". Keep the values here and documented.

import type { BeforeModelResolveEvent } from "./openclaw-types.js";

export type RoutingFeatures = Record<string, string>;

// Finer than the old binary split: short ≤ 400 chars, medium 401–1000, long > 1000. (Prefer
// fine parameterisation — the structure-BMA collapses unhelpful splits.) The warm trainer
// buckets the matrix's instruction lengths with these SAME boundaries.
export const PROMPT_LENGTH_SHORT_MAX_CHARS = 400;
export const PROMPT_LENGTH_MEDIUM_MAX_CHARS = 1000;

export function promptLengthBucket(len: number): string {
  if (len <= PROMPT_LENGTH_SHORT_MAX_CHARS) return "short";
  if (len <= PROMPT_LENGTH_MEDIUM_MAX_CHARS) return "medium";
  return "long";
}

export function extractRoutingFeatures(event: BeforeModelResolveEvent): RoutingFeatures {
  const prompt = typeof event.prompt === "string" ? event.prompt : "";
  return {
    "prompt-length": promptLengthBucket(prompt.length),
  };
}
