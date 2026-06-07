// clawsbench.ts — adapt a benchflow/ClawsBench record into the harness's
// NormalizedEvent stream.
//
// ClawsBench (HF benchflow/ClawsBench, CC-BY-NC-SA-4.0): one JSONL record per
// agent run. Relevant fields:
//   harness    "openclaw" | "gemini" | "claude-agent-acp" | "codex-acp"
//   model      e.g. "anthropic-vertex/claude-opus-4-6"
//   task_name, task_category, condition, skills, run  (join keys → results CSV)
//   traces[]   { step_id, source:"agent", message, tool_calls[], observation }
//     tool_calls[] { tool_call_id, function_name, arguments:{command} }
//
// The openclaw subset carries real tool names (exec/read/write/edit/process/…)
// matching the brain's vocabulary, and real shell commands in arguments.command
// (loops show up as repeated identical commands). No per-step tokens/timestamps
// → cost/time features are unavailable here (recorded as such); detection +
// calibration come from the objective waste criterion and the results-CSV
// outcomes joined later.
//
// Feature bucketing is delegated to the body's FeatureTracker — the ONE source
// of truth — exactly as the native adapter does.

import { FeatureTracker } from "../../openclaw-plugin/src/features.js";
import type {
  BeforeToolCallEvent,
  ToolContext,
} from "../../openclaw-plugin/src/openclaw-types.js";
import type { NormalizedEvent } from "../types.js";

// Reconstructable from the call sequence; tool-name comes straight from
// function_name. working-directory-relative needs paths (only exec commands
// carry them); time-since-last-user-message needs timestamps (absent here).
const ALWAYS_AVAILABLE = [
  "tool-name",
  "parent-tool-call-name",
  "recent-repetition-count",
];

interface ToolCall {
  tool_call_id?: string;
  function_name?: string;
  arguments?: { command?: string } & Record<string, unknown>;
}

interface Step {
  step_id?: number;
  source?: string;
  message?: string;
  tool_calls?: ToolCall[];
  observation?: { results?: Array<{ content?: string; is_error?: boolean }> };
}

export interface ClawsbenchRecord {
  harness?: string;
  session_id?: string;
  model?: string;
  task_name?: string;
  task_category?: string;
  condition?: string;
  skills?: string;
  run?: string;
  traces?: Step[];
}

function derivePathsFromCommand(cmd: string | undefined): string[] {
  if (!cmd) return [];
  // Pull whitespace-separated tokens that look like filesystem paths.
  return cmd
    .split(/\s+/)
    .filter((t) => /[/]/.test(t) && !/^-/.test(t))
    .map((t) => t.replace(/^["']|["',;]+$/g, ""))
    .filter((t) => t.length > 1);
}

function stepObservationErrored(step: Step): boolean {
  const results = step.observation?.results;
  if (!Array.isArray(results)) return false;
  return results.some((r) => r?.is_error === true);
}

/**
 * Adapt one ClawsBench record into its ordered NormalizedEvent stream. Returns
 * [] for records with no tool calls. `meta` provenance carries the join keys
 * (model/task/condition/run) so the results CSV can be joined downstream and so
 * the raw command survives for the objective waste criterion.
 */
export function adaptClawsbench(rec: ClawsbenchRecord): NormalizedEvent[] {
  const sessionId = String(rec.session_id ?? "");
  const tracker = new FeatureTracker();
  const ctx: ToolContext = { runId: sessionId, sessionId };

  const steps = (rec.traces ?? [])
    .slice()
    .sort((a, b) => (a.step_id ?? 0) - (b.step_id ?? 0));

  const events: NormalizedEvent[] = [];
  let idx = 0;

  for (const step of steps) {
    const errored = stepObservationErrored(step);
    for (const tc of step.tool_calls ?? []) {
      const toolName = String(tc.function_name ?? "");
      if (toolName === "") continue;
      const command = tc.arguments?.command;
      const input = tc.arguments ?? {};
      const event: BeforeToolCallEvent = {
        toolName,
        params: input,
        runId: sessionId,
        derivedPaths: derivePathsFromCommand(command),
      };
      // No timestamps in ClawsBench → let the tracker use its default time
      // bucket; time-since-last-user-message is recorded as unavailable.
      const features = tracker.extractAndRecord(event, ctx);

      const featuresAvailable = [...ALWAYS_AVAILABLE];
      if ((event.derivedPaths?.length ?? 0) > 0) {
        // Still no workspaceDir, so working-directory-relative will be
        // "outside-project"/"no-path"; mark it derivable-but-rootless.
        featuresAvailable.push("working-directory-relative(rootless)");
      }

      events.push({
        session_id: sessionId,
        idx,
        ts: null,
        tool_name: toolName,
        input_summary:
          typeof command === "string"
            ? command.slice(0, 120)
            : JSON.stringify(input).slice(0, 120),
        features,
        features_available: featuresAvailable,
        failed: errored,
        turn_tokens: null,
        turn_cost_usd: null,
        label: null,
        meta: {
          corpus: "clawsbench",
          harness: rec.harness,
          model: rec.model,
          task_name: rec.task_name,
          task_category: rec.task_category,
          condition: rec.condition,
          skills: rec.skills,
          run: rec.run,
          command: typeof command === "string" ? command : null,
        },
      });
      idx += 1;
    }
  }

  return events;
}
