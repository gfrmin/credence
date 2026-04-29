export type EvaluateRequest = {
  toolName: string;
  params: Record<string, unknown>;
  recentHistory: Array<{
    toolName: string;
    params: Record<string, unknown>;
    timestamp: number;
  }>;
};

export type EvaluateSignals = {
  alpha: number;
  beta: number;
  comparison_p: number;
  cv: number;
  eu_proceed: number;
  eu_halt: number;
  eu_downgrade: number;
  eu_escalate: number;
};

export type RequireApprovalPayload = {
  title: string;
  description: string;
  severity: "warning" | "error" | "info";
  timeoutMs: number;
  timeoutBehavior: "deny" | "allow";
};

export type EvaluateResponse = {
  action: "proceed" | "block" | "escalate";
  decision?: "proceed" | "halt" | "downgrade" | "route" | "escalate";
  reason?: string;
  signals?: EvaluateSignals;
  requireApproval?: RequireApprovalPayload | null;
};

export type ObserveRequest = {
  toolName: string;
  params: Record<string, unknown>;
  result?: unknown;
  error?: string;
  durationMs?: number;
  timestamp: number;
  userApproval?: boolean | null;
};

export class SidecarClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;

  constructor(baseUrl: string, timeoutMs: number) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.timeoutMs = timeoutMs;
  }

  async evaluate(req: EvaluateRequest): Promise<EvaluateResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      if (!response.ok) {
        return { action: "proceed" };
      }
      return (await response.json()) as EvaluateResponse;
    } catch {
      return { action: "proceed" };
    }
  }

  async observe(req: ObserveRequest): Promise<void> {
    try {
      await fetch(`${this.baseUrl}/observe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch {
      // fire-and-forget: sidecar unavailability doesn't block the agent
    }
  }

  async compactionPreview(messages: unknown[]): Promise<void> {
    try {
      await fetch(`${this.baseUrl}/compaction-preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages }),
        signal: AbortSignal.timeout(5000),
      });
    } catch {
      // fire-and-forget: sidecar unavailability doesn't block compaction
    }
  }

  async health(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}
