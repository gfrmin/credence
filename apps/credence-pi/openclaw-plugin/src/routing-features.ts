// routing-features.ts — the body's sensory periphery for MODEL ROUTING.
//
// before_model_resolve hands the plugin the prompt and nothing else, so the only
// HONESTLY-extractable routing feature is the prompt's length. Semantic category — the
// signal the offline oracle actually measured — is NOT available live without a
// classifier, which would be an arbitrary, unmeasured component (and the project bars
// arbitrary constants/heuristics in the reasoning path). The daemon's structure-BMA
// discovers whether length predicts accuracy and collapses to the marginal if it does
// not, so declaring only length is honest and self-correcting.
//
// The short/long boundary is the ONE train/live synchronisation point. Training
// (apps/credence-pi/eval/train_routing_brain.jl) seeds the "short" prompt-length cell
// from the short-MCQ oracle bank — so this boundary defines the regime the daemon's warm
// belief was measured on. If you change it, the warm belief's "short" cell no longer
// matches what the body now calls "short". Keep the value here and documented.

import type { BeforeModelResolveEvent } from "./openclaw-types.js";

export type RoutingFeatures = Record<string, string>;

// Prompts at or below this many characters are "short" (MCQ-scale and brief asks — the
// regime the warm belief was measured on); longer prompts are "long" (unmeasured warm
// belief leans on the marginal structure until online learning lands).
export const PROMPT_LENGTH_SHORT_MAX_CHARS = 1000;

export function extractRoutingFeatures(event: BeforeModelResolveEvent): RoutingFeatures {
  const prompt = typeof event.prompt === "string" ? event.prompt : "";
  return {
    "prompt-length": prompt.length <= PROMPT_LENGTH_SHORT_MAX_CHARS ? "short" : "long",
  };
}
