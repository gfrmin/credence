// atbench.ts — adapt AI45Research/ATBench-Claw into the harness's
// NormalizedEvent stream.
//
// ATBench-Claw (HF, Apache-2.0): a JSON array of 500 OpenClaw trajectories,
// each { trajectory:{events:[...]}, labels:{is_safe, risk_source, failure_mode,
// harm_type, defense_type}, reason }. `events` is the native OpenClaw message
// timeline, but tool calls are `{type:"toolCall", name, arguments}` blocks
// inside assistant `content` (not the `tool_use`/`input` shape the native
// session adapter uses), so it gets its own adapter.
//
// The independent label is `is_safe` (per trajectory): 296 unsafe / 204 safe.
// It is NOT derived from repetition, so correlating the brain's block/ask
// decisions with it removes the self-consistency caveat of the loop label.
//
// Bucketing is delegated to the body's FeatureTracker — same as every adapter.

import { FeatureTracker } from "../../openclaw-plugin/src/features.js";
import type {
  BeforeToolCallEvent,
  ToolContext,
} from "../../openclaw-plugin/src/openclaw-types.js";
import type { NormalizedEvent } from "../types.js";
import {
  actionClass,
  targetExternality,
  looksUntrusted,
  extractTokens,
  untrustedImperatives,
  taintFlow,
  matchesImperative,
  isExternalSend,
  isCredentialAccess,
} from "./risk.js";

// Flatten a message's content (string | array of {text}) to plain text, for
// untrusted-marker scanning.
function messageText(msg: { content?: unknown; text?: unknown } | undefined): string {
  if (!msg) return "";
  const c = msg.content ?? msg.text;
  if (typeof c === "string") return c;
  if (Array.isArray(c)) {
    return c
      .map((b) => (b && typeof b === "object" ? String((b as { text?: unknown }).text ?? "") : String(b)))
      .join(" ");
  }
  return "";
}

const ALWAYS_AVAILABLE = [
  "tool-name",
  "parent-tool-call-name",
  "recent-repetition-count",
];

interface ToolCallBlock {
  type?: string;
  id?: string;
  name?: string;
  arguments?: Record<string, unknown>;
}

interface Event {
  type?: string;
  message?: { role?: string; content?: unknown; toolName?: string };
}

interface Labels {
  is_safe?: boolean;
  risk_source?: string;
  failure_mode?: string;
  harm_type?: string;
  defense_type?: string | null;
}

export interface AtbenchRecord {
  trajectory?: { events?: Event[]; modelId?: string; provider?: string };
  labels?: Labels;
  reason?: string;
}

function derivePaths(args: Record<string, unknown> | undefined): string[] {
  if (!args) return [];
  const out: string[] = [];
  const PATH_KEYS = /^(path|file|filename|filepath|dir|directory|cwd|target)$/i;
  for (const [k, v] of Object.entries(args)) {
    if (typeof v !== "string" || !v) continue;
    if (PATH_KEYS.test(k) || /[/\\]/.test(v)) out.push(v);
    // exec-style: pull path-like tokens out of a command string
    if (/^(command|cmd|script)$/i.test(k)) {
      for (const tok of v.split(/\s+/)) {
        if (/[/]/.test(tok) && !tok.startsWith("-")) out.push(tok);
      }
    }
  }
  return out;
}

function isToolCall(b: unknown): b is ToolCallBlock {
  return typeof b === "object" && b !== null && (b as ToolCallBlock).type === "toolCall";
}

/** Adapt one ATBench-Claw record into its ordered NormalizedEvent stream. The
 *  per-trajectory `is_safe` label is stamped on every event as "safe"/"unsafe". */
export function adaptAtbench(rec: AtbenchRecord, sessionId: string): NormalizedEvent[] {
  const tracker = new FeatureTracker();
  const ctx: ToolContext = { runId: sessionId, sessionId };
  const events = rec.trajectory?.events ?? [];
  const isSafe = rec.labels?.is_safe;
  const label = isSafe === true ? "safe" : isSafe === false ? "unsafe" : null;

  const out: NormalizedEvent[] = [];
  let idx = 0;
  // Causal taint state: a token taints a call only if it appeared in untrusted content
  // BEFORE that call (production-faithful — the live body only knows the past). The
  // agent's own policy reads (SOUL/USER/IDENTITY/.openclaw) are trusted, not input.
  const POLICY = /SOUL\.md|USER\.md|IDENTIT|\.openclaw/i;
  let untrustedSeen = false; // old control feature: any untrusted content seen yet
  const taintTokens = new Set<string>(); // distinctive tokens from untrusted spans so far
  const imperativeVerbs = new Set<string>(); // imperative verbs the injection has issued
  let credSeen = false; // a credential-access call has occurred earlier
  let credExfilChain = false; // sticky: a credential-access was followed by an external-send
  for (const ev of events) {
    const msg = ev.message;
    if (!msg) continue;
    const role = msg.role;
    if (role !== "assistant") {
      // user / toolResult content the agent consumed — accumulate the taint source
      const txt = messageText(msg);
      const isPolicy = POLICY.test(txt);
      if ((role === "toolResult" && !isPolicy) || looksUntrusted(txt)) {
        untrustedSeen = true;
        for (const tok of extractTokens(txt)) taintTokens.add(tok);
        for (const v of untrustedImperatives(txt)) imperativeVerbs.add(v);
      }
      continue;
    }
    const content = msg.content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (!isToolCall(block)) continue;
      const toolName = String(block.name ?? "");
      if (!toolName) continue;
      const args = block.arguments ?? {};
      const event: BeforeToolCallEvent = {
        toolName,
        params: args,
        runId: sessionId,
        derivedPaths: derivePaths(args),
      };
      const features = tracker.extractAndRecord(event, ctx);
      // complete the credential→external-send chain BEFORE recording credSeen for this
      // call (the external-send must come strictly after the credential access).
      if (credSeen && isExternalSend(toolName)) credExfilChain = true;
      // Risk-aware features. action-class is kept as the per-class substrate; the weak
      // target-externality / untrusted-provenance are retained only for comparison
      // (untrusted-provenance is the rejected control — it fires ~100%). The taint-flow
      // family is the validated discriminator (see results/SAFETY_INGREDIENTS.md).
      features["action-class"] = actionClass(toolName, args);
      features["target-externality"] = targetExternality(toolName, args);
      features["untrusted-provenance"] = untrustedSeen ? "follows-untrusted" : "trusted";
      features["taint-flow"] = taintFlow(toolName, args, taintTokens);
      features["injected-imperative"] = matchesImperative(toolName, args, imperativeVerbs) ? "yes" : "no";
      features["cred-exfil-chain"] = credExfilChain ? "yes" : "no";
      if (isCredentialAccess(toolName, args)) credSeen = true;
      const featuresAvailable = [
        ...ALWAYS_AVAILABLE,
        "action-class", "target-externality", "untrusted-provenance",
        "taint-flow", "injected-imperative", "cred-exfil-chain",
      ];
      if ((event.derivedPaths?.length ?? 0) > 0) {
        featuresAvailable.push("working-directory-relative(rootless)");
      }
      out.push({
        session_id: sessionId,
        idx,
        ts: null,
        tool_name: toolName,
        input_summary: JSON.stringify(args).slice(0, 120),
        features,
        features_available: featuresAvailable,
        failed: false, // ATBench carries no per-step tool error flag
        turn_tokens: null,
        turn_cost_usd: null,
        label,
        meta: {
          corpus: "atbench-claw",
          model: rec.trajectory?.modelId || null,
          is_safe: isSafe,
          reason: rec.reason ?? null, // human-written harm explanation (for reason-localized attribution)
          risk_source: rec.labels?.risk_source ?? null,
          failure_mode: rec.labels?.failure_mode ?? null,
          harm_type: rec.labels?.harm_type ?? null,
          defense_type: rec.labels?.defense_type ?? null,
        },
      });
      idx += 1;
    }
  }
  return out;
}
