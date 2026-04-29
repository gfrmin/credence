import { SidecarClient } from "./sidecar-client.js";
import type { EvaluateRequest } from "./sidecar-client.js";

type ToolHistoryEntry = {
  toolName: string;
  params: Record<string, unknown>;
  timestamp: number;
};

const HISTORY_WINDOW = 50;

export default {
  id: "credence-governance",
  name: "Credence Governance",
  description:
    "Bayesian governance sidecar — intercepts tool calls via expected utility",

  register(api: any) {
    const config = api.config ?? {};
    const sidecarUrl =
      (config.sidecarUrl as string) ?? "http://localhost:3100";
    const timeoutMs = (config.timeoutMs as number) ?? 200;

    const client = new SidecarClient(sidecarUrl, timeoutMs);
    const recentHistory: ToolHistoryEntry[] = [];

    api.on(
      "before_tool_call",
      async (event: {
        toolName: string;
        params: Record<string, unknown>;
      }) => {
        const req: EvaluateRequest = {
          toolName: event.toolName,
          params: event.params,
          recentHistory: recentHistory.slice(-HISTORY_WINDOW),
        };

        const decision = await client.evaluate(req);

        if (decision.action === "block") {
          return {
            block: true,
            blockReason: `Credence governance: ${decision.reason ?? "tool call vetoed by expected utility calculation"}`,
          };
        }

        return undefined;
      },
      { priority: 100 },
    );

    api.on("after_tool_call", async (event: {
      toolName: string;
      params: Record<string, unknown>;
      result?: unknown;
      error?: string;
      durationMs?: number;
    }) => {
      const entry: ToolHistoryEntry = {
        toolName: event.toolName,
        params: event.params,
        timestamp: Date.now(),
      };
      recentHistory.push(entry);
      if (recentHistory.length > HISTORY_WINDOW) {
        recentHistory.shift();
      }

      client.observe({
        toolName: event.toolName,
        params: event.params,
        result: event.result,
        error: event.error,
        durationMs: event.durationMs,
        timestamp: entry.timestamp,
      });
    });

    api.on("agent_end", async () => {
      recentHistory.length = 0;
    });
  },
};
