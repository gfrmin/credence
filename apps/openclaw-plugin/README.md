# Credence Governance Plugin for OpenClaw

Prototype OpenClaw plugin that intercepts tool calls via a Credence governance
sidecar. The sidecar evaluates each candidate tool call and can veto it when
the expected utility calculation indicates intervention (e.g., loop detection).

## Prerequisites

- OpenClaw installed and configured
- Credence governance sidecar running (see `../credence-governance-sidecar/`)
- Node.js >= 22

## Install

```bash
# From this directory
openclaw plugins install . --link
```

## Configure

In `~/.openclaw/config.yaml` (or equivalent):

```yaml
plugins:
  entries:
    credence-governance:
      enabled: true
      config:
        sidecarUrl: "http://localhost:3100"
        timeoutMs: 200
```

## How it works

1. Plugin registers on `before_tool_call` and `after_tool_call` hooks.
2. Before each tool call, the plugin sends the candidate call and recent
   history to the sidecar at `POST /evaluate`.
3. The sidecar returns an enriched response with a `decision` field
   (`proceed`, `halt`, `downgrade`, `route`, or `escalate`) and optional
   `signals` (alpha, beta, EU values) plus `requireApproval` payload.
4. The plugin renders the decision as an OpenClaw hook return value:
   - **proceed / route** — no intervention (`undefined`).
   - **halt** — block with reason (`{ block: true, blockReason: "..." }`).
   - **downgrade** — block with reason suggesting an alternative tool.
   - **escalate** — request user approval (`{ requireApproval: { ... } }`).
5. After each tool call, the plugin sends the outcome to the sidecar at
   `POST /observe` (fire-and-forget). If the call was an escalation, the
   user's approval decision is forwarded as `userApproval: boolean`.
6. If the sidecar is unavailable, the plugin fails open (no intervention).

The sidecar response is backwards-compatible with the Move 1 `action`
field: if `decision` is absent, the plugin falls back to mapping
`action: "block"` to `halt` and `action: "proceed"` to `proceed`.
