import { SidecarClient } from "./sidecar-client.js";
import type {
  EvaluateRequest,
  EvaluateResponse,
  RequireApprovalPayload,
} from "./sidecar-client.js";

type ToolHistoryEntry = {
  toolName: string;
  params: Record<string, unknown>;
  timestamp: number;
};

const HISTORY_WINDOW = 50;

function renderDecision(response: EvaluateResponse): Record<string, unknown> | undefined {
  const decision = response.decision ?? (response.action === "block" ? "halt" : "proceed");

  switch (decision) {
    case "proceed":
    case "route":
      return undefined;

    case "halt":
      return {
        block: true,
        blockReason: `Credence governance: ${response.reason ?? "tool call vetoed by expected utility calculation"}`,
      };

    case "downgrade":
      return {
        block: true,
        blockReason: `Credence governance: ${response.reason ?? "alternative tool has higher expected utility"}`,
      };

    case "escalate": {
      const payload = response.requireApproval;
      if (payload != null) {
        return { requireApproval: payload };
      }
      return {
        requireApproval: {
          title: "Credence governance check",
          description: response.reason ?? "The proposed action has uncertain expected utility. Confirm to proceed.",
          severity: "warning",
          timeoutMs: 120000,
          timeoutBehavior: "deny",
        } satisfies RequireApprovalPayload,
      };
    }

    default:
      return undefined;
  }
}

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
    let pendingEscalation: { toolName: string; params: Record<string, unknown> } | null = null;

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

        const response = await client.evaluate(req);
        const decision = response.decision ?? (response.action === "block" ? "halt" : "proceed");

        if (decision === "escalate") {
          pendingEscalation = { toolName: event.toolName, params: event.params };
        }

        return renderDecision(response);
      },
      { priority: 100 },
    );

    api.on("after_tool_call", async (event: {
      toolName: string;
      params: Record<string, unknown>;
      result?: unknown;
      error?: string;
      durationMs?: number;
      userApproval?: boolean;
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

      let approval: boolean | null = null;
      if (pendingEscalation != null &&
          pendingEscalation.toolName === event.toolName) {
        approval = event.userApproval ?? true;
        pendingEscalation = null;
      }

      client.observe({
        toolName: event.toolName,
        params: event.params,
        result: event.result,
        error: event.error,
        durationMs: event.durationMs,
        timestamp: entry.timestamp,
        userApproval: approval,
      });
    });

    api.on("before_compaction", (event: { messages: unknown[] }) => {
      client.compactionPreview(event.messages);
    });

    api.on("agent_end", async () => {
      recentHistory.length = 0;
      pendingEscalation = null;
    });
  },
};
