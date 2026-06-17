// pi.ts — the minimal slice of the pi-mono extension API the body uses, so the
// extension is unit-testable WITHOUT pi-mono (a fake `PiLike` in the tests). The
// real `ExtensionAPI` (@earendil-works/pi-coding-agent) satisfies this shape; the
// demo app (apps/answer-brain/app) adapts the real `pi` to it at its boundary.
// Tool parameter schemas are real TypeBox (what pi validates LLM args against).

import type { TSchema } from "typebox";

/** A pi tool result: text content + structured details (move-1 examples). */
export interface PiToolResult {
	content: Array<{ type: "text"; text: string }>;
	details: unknown;
}

export interface PiToolDefinition {
	name: string;
	label: string;
	description: string;
	parameters: TSchema;
	promptGuidelines?: string[];
	execute(
		toolCallId: string,
		params: Record<string, unknown>,
		signal?: AbortSignal,
		onUpdate?: unknown,
		ctx?: unknown,
	): Promise<PiToolResult>;
}

/** The tool_call event the governor intercepts. `input` is the live args object the
 *  tool will execute with (agent-loop passes the same reference), so a mutation here
 *  reaches execute. */
export interface PiToolCallEvent {
	toolName: string;
	input: Record<string, unknown>;
}

/** undefined = allow; {block} = stop + steer (createErrorToolResult(reason) feeds the LLM). */
export type ToolCallOutcome = { block: true; reason: string } | undefined;

export type ToolCallHandler = (
	event: PiToolCallEvent,
	ctx?: unknown,
) => Promise<ToolCallOutcome> | ToolCallOutcome;

export interface PiLike {
	registerTool(tool: PiToolDefinition): void;
	on(event: "tool_call", handler: ToolCallHandler): void;
}
