# Credence Governance Plugin for OpenClaw

Intercepts tool calls via a Bayesian governance sidecar. The sidecar evaluates
each candidate tool call by expected utility and can veto runaway loops,
escalate uncertain actions to the user, or suggest alternatives.

## Prerequisites

- [OpenClaw](https://openclaw.ai) installed (`npm install -g openclaw@latest`)
- Node.js >= 22
- Julia >= 1.9 (for the governance sidecar)

## Setup

### 1. Start the governance sidecar

```bash
cd apps/credence-governance-sidecar
julia --project=. server.jl
```

The sidecar listens on `http://localhost:3100` by default. If the sidecar is not
running when the plugin loads, the plugin warns once and fails open (no
governance protection until the sidecar comes back).

### 2. Build and install the plugin

```bash
cd apps/openclaw-plugin
npm install
npm run build

# Developer (link mode — picks up source changes after gateway restart):
openclaw plugins install -l "$(pwd)"

# User (copy mode):
openclaw plugins install "$(pwd)"
```

### 3. Configure

Add to `~/.openclaw/openclaw.json`:

```json5
{
  plugins: {
    entries: {
      "credence-governance": {
        enabled: true,
        config: {
          sidecarUrl: "http://localhost:3100",
          timeoutMs: 200
        }
      }
    }
  }
}
```

Both config values are optional — the defaults shown above apply if omitted.

### 4. Restart the gateway

```bash
openclaw gateway restart
```

### 5. Verify

```bash
openclaw plugins list                          # should show credence-governance enabled
openclaw plugins inspect credence-governance   # should show all 4 hooks registered
openclaw plugins doctor                        # should report no errors for this plugin
```

## How it works

1. `before_tool_call`: sends the candidate tool call + recent history to the
   sidecar at `POST /evaluate`.
2. Sidecar returns a decision: **proceed**, **halt** (block with reason),
   **downgrade** (block suggesting alternative), or **escalate** (request user
   approval).
3. `after_tool_call`: sends outcome to `POST /observe` (fire-and-forget) so the
   sidecar can update its posterior.
4. `before_compaction`: sends messages about to be compacted to
   `POST /compaction-preview` so the sidecar can detect instruction patterns.
5. `agent_end`: clears the local history buffer.

## Troubleshooting

**Sidecar not running.** The plugin warns once in the gateway logs and proceeds
without governance. Start the sidecar and the plugin resumes automatically.

**Plugin not loading.** Run `openclaw plugins doctor` to surface manifest issues.
Ensure `npm run build` completed successfully (check that `dist/index.js` exists).

**Config not taking effect.** Verify config is under `plugins.entries.credence-governance.config`
in `~/.openclaw/openclaw.json` (JSON5 format), not at the top level. Restart the
gateway after config changes.

**Gateway hasn't picked up changes.** After installing or updating the plugin,
run `openclaw gateway restart`.
