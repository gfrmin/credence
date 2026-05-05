// types.ts — shared types for the credence-pi extension body.
//
// These shapes are minimal and deliberately decoupled from pi's
// runtime types. Step 7 binds them to pi's actual extension API; for
// step 6, the extractors and client work against them directly so
// tests can construct mock sessions without depending on pi.

export interface ToolCallEvent {
  toolName: string;
  input: unknown;
}

export type MessageRole = "user" | "assistant" | "tool_call" | "tool_result";

export interface Message {
  role: MessageRole;
  toolName?: string;
  // ISO-8601 timestamp; absent on synthetic / pre-existing messages.
  timestamp?: string;
}

export interface Session {
  // Absolute path to the current working directory; empty string is
  // treated as "unknown / no path" and bucketed into "no-path".
  cwd: string;
  // Absolute path to the project root, used to classify cwd against
  // the project boundary. Empty string means "unknown".
  projectRoot: string;
  messages: Message[];
}

// The features dict the body POSTs to the daemon as part of a
// tool-proposed sensor event. Keys are kebab-case BDSL feature
// names; values are kebab-case BDSL space members. Both match
// features.bdsl verbatim — see manifest.ts's Registry comment.
export type FeatureDict = Record<string, string>;
