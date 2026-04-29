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
3. If the sidecar returns `{ action: "block" }`, the plugin vetoes the
   tool call with a descriptive reason.
4. After each tool call, the plugin sends the outcome to the sidecar at
   `POST /observe` (fire-and-forget, no latency impact).
5. If the sidecar is unavailable, the plugin fails open (no intervention).

## Prototype scope

This prototype demonstrates only **loop detection** (veto when the same tool
with the same arguments is called more than N times). Move 2 wires up the
full Credence brain with four intervention types.
