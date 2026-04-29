# Credence Governance Sidecar

Prototype Julia HTTP server that evaluates tool-call decisions for the
Credence governance plugin. The sidecar receives candidate tool calls,
checks for repetition patterns, and returns proceed/block decisions.

## Prerequisites

- Julia >= 1.9
- HTTP.jl and JSON3.jl (`julia -e 'using Pkg; Pkg.add(["HTTP", "JSON3"])'`)

## Run

```bash
julia apps/credence-governance-sidecar/server.jl
```

Environment variables:

- `CREDENCE_SIDECAR_PORT` — listen port (default: 3100)
- `CREDENCE_MAX_REPETITIONS` — loop detection threshold (default: 3)

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/evaluate` | Evaluate a candidate tool call; returns `{ action: "proceed" }` or `{ action: "block", reason: "..." }` |
| `POST` | `/observe` | Record a completed tool call outcome (fire-and-forget from plugin) |
| `GET` | `/health` | Health check with uptime and history length |
| `POST` | `/reset` | Clear tool call history (for testing) |

## Prototype scope

This prototype uses a simple repetition counter for loop detection. If
the same tool with the same arguments has been called more than N times,
the sidecar returns a block decision. `Read` tool calls are excluded
(reading files repeatedly is normal agent behaviour).

Move 2 replaces this with the actual Credence brain: posterior over
(model × tool × task-category), updated via Bayesian conditioning,
decisions via expected utility maximisation.
