// types.ts — the canonical normalized event the replay harness emits, one per
// tool call. Both adapters (native OpenClaw, ATIF) produce this shape; the Julia
// replay stage consumes it. "Record everything": every field the brain or the
// counterfactual analyzer could need is carried, plus provenance.

import type { Features } from "../openclaw-plugin/src/features.js";

export interface NormalizedEvent {
  /** Stable id of the source session/trajectory. */
  session_id: string;
  /** 0-based index of this tool call within the session (call order). */
  idx: number;
  /** Epoch ms of the call, if the transcript carried a timestamp (else null). */
  ts: number | null;
  /** The raw tool name as it appeared in the transcript (pre-bucketing). */
  tool_name: string;
  /** Short JSON summary of the tool input (for audit; never fed to the brain). */
  input_summary: string;
  /** The 5 declared features, bucketed by the body's FeatureTracker. */
  features: Features;
  /** Which features were derivable from this corpus (the rest are best-effort
   *  defaults; structure-BMA still conditions on the available subset). */
  features_available: string[];
  /** Whether the tool call failed in the transcript (for the waste criterion). */
  failed: boolean;
  /** Tokens attributed to the turn this call belongs to (per-turn, not per-call). */
  turn_tokens: number | null;
  /** Real USD cost of the turn from the transcript, if present (else null). */
  turn_cost_usd: number | null;
  /** Independent ground-truth label if the corpus ships one: "anomaly" |
   *  "normal" | null. Used for calibration, never fed to the decision. */
  label: string | null;
  /** Free-form provenance (corpus name, model, anomaly subtype, …). */
  meta: Record<string, unknown>;
}
