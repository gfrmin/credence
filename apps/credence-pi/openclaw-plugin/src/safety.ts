// safety.ts — risk-aware features for the harm posterior P(unsafe|X), and the live
// session SafetyTracker that emits them. The BODY is the single source of truth for the
// taint computation (the eval adapter re-exports from here, mirroring FeatureTracker), so
// the offline-trained harm posterior and the live signal agree.
//
// Safety is not a function of tool/rep/ident (those govern waste); it lives in the CONTENT
// of an action: its class, and — the validated discriminator — whether a consequential
// SINK carries a value that originated in UNTRUSTED content (a tool result, an injected
// context). A compromised agent carries the taint into a sink; a defending agent breaks the
// flow. Measured trajectory precision on ATBench-Claw: 0.82–0.97 vs a regex's 0.67
// (eval/results/SAFETY_INGREDIENTS.md). No embeddings — coarse discrete partitions the
// structure-BMA combines.

import type { ToolContext } from "./openclaw-types.js";

function flatten(args: unknown): string {
  try {
    return JSON.stringify(args ?? "").toLowerCase();
  } catch {
    return "";
  }
}

const re = (s: string) => new RegExp(s, "i");

// Ordered most-dangerous-first; first match wins.
const ACTION_RULES: Array<[string, RegExp, RegExp | null]> = [
  // [class, name-pattern, args-pattern (null = name only)]
  ["delete", re("\\b(rm|unlink|rmdir|delete|remove|trash|drop)\\b"), null],
  ["delete", re("\\b(exec|process|bash|sh|tmux|run)\\b"), re("\\b(rm\\s|unlink|rmdir|drop\\s+table|delete\\s+from)\\b")],
  ["external-send", re("send|forward|publish|tweet|post\\b|webhook|gmail|email|sendmessage|pinmessage|sms|slack"), null],
  ["external-send", re("\\b(exec|process|bash|sh|tmux|curl|wget)\\b"), re("(curl|wget).*(-d|--data|-t\\b|-f\\b|@/|http)")],
  ["credential-access", re("op_signin|1password|keychain|secret|credential|vault"), null],
  ["credential-access", re("read|search|cat|grep|get"), re("api[_-]?key|secret|credential|token|password|\\.env|private[_-]?key")],
  ["cross-boundary", re("web_fetch|web_search|browser|agent-browser|activate_window|screenshot|type_text|open\\b"), null],
  ["cross-boundary", re("move|copy|mv|cp"), re("public/|/sync/|external|http")],
  ["exec", re("\\b(exec|process|bash|sh|tmux|run|gh\\s+api|api\\s+gateway|cron)\\b"), null],
  ["local-write", re("\\b(write|edit|create|apply_patch|update|move|mv|cp|copy|append|insert)\\b"), null],
  ["read-only", re("\\b(read|cat|head|tail|ls|grep|find|search|get|list|print|fetch|capture|transcript|view|show|envelope)\\b"), null],
];

/** Coarse risk class of an action (the candidate-harm taxonomy). */
export function actionClass(toolName: string, args: unknown): string {
  const name = (toolName || "").toLowerCase();
  const a = flatten(args);
  for (const [cls, namePat, argPat] of ACTION_RULES) {
    if (namePat.test(name) && (argPat === null || argPat.test(a) || argPat.test(name))) {
      return cls;
    }
    // command-style tools carry the verb in the args, not the name
    if (argPat === null && namePat.test(a)) return cls;
  }
  return "other";
}

// Does the action reach OUTSIDE the local/trusted boundary? internal | external | none.
const EMAIL = /[\w.+-]+@[\w.-]+\.\w+/;
const URL = /https?:\/\/[\w.-]+/;
const PUBLIC_PATH = /(^|["'\/])(public|sync|shared|outbox|external)\//i;
export function targetExternality(toolName: string, args: unknown): string {
  const a = flatten(args);
  const emails = a.match(new RegExp(EMAIL, "g")) ?? [];
  const urls = a.match(new RegExp(URL, "g")) ?? [];
  const external =
    emails.some((e) => !/@(localhost|internal|corp|company|local)\b/.test(e)) ||
    urls.some((u) => !/\/\/(localhost|127\.|10\.|192\.168|internal)/.test(u)) ||
    PUBLIC_PATH.test(a);
  if (external) return "external";
  if (emails.length || urls.length || /\bchannel\b|recipient|to:/.test(a)) return "internal";
  return "none";
}

// Untrusted-content markers in a piece of text the agent consumed (a user message or a
// tool result). Used to seed taint when content is explicitly flagged untrusted/injected.
const UNTRUSTED_MARKERS =
  /untrusted|injection|ignore (the )?(previous|above|prior)|disregard (your|the)|bypass|exfiltrat|skip (intermediate )?confirmation|override (your|the|safety|user)|jailbreak|system prompt|new instructions?:/i;
export function looksUntrusted(text: string): boolean {
  return UNTRUSTED_MARKERS.test(text);
}

// ── taint-flow primitives ───────────────────────────────────────────────
// A consequential SINK is any non-pure-read action.
const SINK_RE =
  /send|forward|post|publish|tweet|webhook|curl|wget|upload|message|write|edit|create|apply_patch|append|insert|move|copy|cp|mv|rm|unlink|delete|drop|remove|exec|process|bash|sh|tmux|run|api|gh /i;
const READ_RE = /read|cat|head|tail|ls|grep|find|list|view|envelope|capture/i;
const EXTERNAL_SEND_RE =
  /send|forward|post|publish|tweet|webhook|curl|wget|upload|message|email|gmail|sms|slack|send.?keys|sendmessage/i;
const CRED_RE =
  /api[_-]?key|secret|credential|token|password|\.env|private[_-]?key|keychain|op_signin|1password|vault|credentials/i;

/** Pure read — observes, does not act. */
export function isReadOnly(toolName: string): boolean {
  const n = (toolName || "").toLowerCase();
  return READ_RE.test(n) || n === "get";
}
/** Consequential sink — an action with an external/persistent effect. */
export function isSink(toolName: string): boolean {
  return SINK_RE.test(toolName || "") && !isReadOnly(toolName);
}
/** External-send class — a sink that delivers to a recipient/destination. */
export function isExternalSend(toolName: string): boolean {
  return EXTERNAL_SEND_RE.test(toolName || "");
}
/** Credential-access — reads or names a secret. */
export function isCredentialAccess(toolName: string, args: unknown): boolean {
  return CRED_RE.test(toolName || "") || CRED_RE.test(flatten(args));
}

// Distinctive token extraction (emails / urls / domains / deep paths), mirroring the
// validated python helper data/credence_pi_eval/analysis/atbench_taint.py.
const STOP_TOKENS = new Set([
  "example.com", "localhost", "127.0.0.1", "0.0.0.0", "github.com", "schema.org",
]);
const TOKEN_RES: RegExp[] = [
  /[\w.+-]+@[\w.-]+\.\w+/g, // email
  /https?:\/\/[\w./?=&%+#~-]+/g, // url
  /\b[\w-]+\.(?:com|net|org|io|ai|co|dev|xyz|ru|cn|evil|sh|info)\b/gi, // domain
  /(?:\/[\w.+-]+){2,}\/?/g, // deep path
];
/** Distinctive tokens in a piece of text, normalised (lowercased, trailing punct stripped). */
export function extractTokens(text: string): Set<string> {
  const t = (text || "").toLowerCase();
  const out = new Set<string>();
  for (const re of TOKEN_RES) {
    const m = t.match(re) || [];
    for (let tok of m) {
      tok = tok.replace(/[/.,);'"]+$/, "");
      if (tok.length >= 4 && !STOP_TOKENS.has(tok)) out.add(tok);
    }
  }
  return out;
}

/** Graded taint-flow bucket for a call, given the set of tainted tokens seen so far.
 *  none < tainted-sink < tainted-external-target (the precision gradient). */
export function taintFlow(toolName: string, args: unknown, taint: Set<string>): string {
  if (taint.size === 0) return "none";
  const at = flatten(args);
  // strongest: an external-send whose destination token came from untrusted content
  if (isExternalSend(toolName)) {
    for (const tok of extractTokens(at)) if (taint.has(tok)) return "tainted-external-target";
  }
  // any consequential sink carrying a tainted token
  if (isSink(toolName)) {
    for (const tok of taint) if (at.includes(tok)) return "tainted-sink";
  }
  return "none";
}

// Imperative verbs an injection uses to command an action.
const IMPERATIVE_VERBS = [
  "forward", "send", "email", "post", "publish", "delete", "remove", "drop",
  "transfer", "wire", "pay", "grant", "disable", "exfiltrate", "upload", "share",
];
const IMPERATIVE_RES = IMPERATIVE_VERBS.map((v) => new RegExp("\\b" + v + "\\b"));
/** Imperative verbs present in a piece of untrusted text (the injection's commands). */
export function untrustedImperatives(text: string): Set<string> {
  const t = (text || "").toLowerCase();
  const out = new Set<string>();
  IMPERATIVE_VERBS.forEach((v, i) => {
    if (IMPERATIVE_RES[i].test(t)) out.add(v);
  });
  return out;
}
/** Does this call execute one of the injected imperative verbs? */
export function matchesImperative(toolName: string, args: unknown, verbs: Set<string>): boolean {
  if (verbs.size === 0) return false;
  const blob = (toolName || "").toLowerCase() + " " + flatten(args);
  for (const v of verbs) if (new RegExp("\\b" + v + "\\b").test(blob)) return true;
  return false;
}

// ── live session taint state ─────────────────────────────────────────────
//
// The taint SOURCE in a live session is the content the agent consumes from the outside
// world — chiefly tool RESULTS (web fetches, file reads, email bodies). `observeResult`
// accumulates the distinctive tokens + imperative verbs seen so far; `extractSafety`
// computes the 4 features for a proposed call against that accumulated, CAUSAL state (a
// token taints a call only if it was seen BEFORE the call).

interface SessionTaint {
  tokens: Set<string>; // distinctive tokens seen in untrusted content so far
  verbs: Set<string>; // imperative verbs issued by untrusted content so far
  credSeen: boolean; // a credential-access call has occurred earlier
  credExfilChain: boolean; // a credential-access was followed by an external-send (sticky)
}

function sessionKey(ctx: ToolContext): string {
  return ctx.runId ?? ctx.sessionKey ?? ctx.sessionId ?? "default";
}

function resultToText(result: unknown): string {
  if (result == null) return "";
  if (typeof result === "string") return result;
  try {
    return JSON.stringify(result);
  } catch {
    return String(result);
  }
}

export class SafetyTracker {
  private sessions = new Map<string, SessionTaint>();

  private state(ctx: ToolContext): SessionTaint {
    const k = sessionKey(ctx);
    let s = this.sessions.get(k);
    if (!s) {
      s = { tokens: new Set(), verbs: new Set(), credSeen: false, credExfilChain: false };
      this.sessions.set(k, s);
    }
    return s;
  }

  /** Accumulate the taint SOURCE from a piece of untrusted content (a tool result). */
  observeResult(result: unknown, ctx: ToolContext): void {
    const text = resultToText(result);
    if (!text) return;
    const s = this.state(ctx);
    for (const tok of extractTokens(text)) s.tokens.add(tok);
    for (const v of untrustedImperatives(text)) s.verbs.add(v);
  }

  /** The 4 safety features for a proposed call, against the session's causal taint state. */
  extractSafety(toolName: string, args: unknown, ctx: ToolContext): Record<string, string> {
    const s = this.state(ctx);
    // Complete the credential→external-send chain BEFORE recording credSeen for THIS call,
    // so the external-send must come strictly after the credential access.
    if (s.credSeen && isExternalSend(toolName)) s.credExfilChain = true;
    const features: Record<string, string> = {
      "action-class": actionClass(toolName, args),
      "taint-flow": taintFlow(toolName, args, s.tokens),
      "injected-imperative": matchesImperative(toolName, args, s.verbs) ? "yes" : "no",
      "cred-exfil-chain": s.credExfilChain ? "yes" : "no",
    };
    if (isCredentialAccess(toolName, args)) s.credSeen = true;
    return features;
  }
}
