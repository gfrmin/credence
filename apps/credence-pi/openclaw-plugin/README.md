# credence-pi — OpenClaw plugin body

The **body** that lets the credence-pi Bayesian brain govern a real
pi/OpenClaw agent. It registers an OpenClaw `before_tool_call` hook,
forwards each proposed tool call to the credence-pi daemon (the opaque
brain), and maps the brain's decision back to OpenClaw:

| brain effector | OpenClaw result |
|---|---|
| `proceed` | allow |
| `block`   | `{ block: true, blockReason }` |
| `ask`     | `{ requireApproval }` — OpenClaw's native approval dialog; the user's choice is posted back so the brain learns |

It also logs tool **outcomes** (`after_tool_call`) and reconstructs
**per-turn cost** (`llm_output` token counts × a price table) so the
observation log accumulates the data the dollars-saved surface needs.

**Fail-open:** if the daemon is unreachable or slow, the tool proceeds
(one warning per outage). Governance never blocks the agent on
infrastructure failure.

This is one of two bodies over the same brain; the other,
`apps/credence-pi/extension/`, targets the pi coding agent directly.
Brain + wire (POST `/sensor`, SSE `/signals`) are shared and unchanged.

## Why a plugin (not a pi extension)

Current OpenClaw vendors pi's coding-agent and runs its gateway agent with
`noExtensions: true`, so a pi `ExtensionFactory` never loads. The supported
interception point is an OpenClaw **plugin** `before_tool_call` hook. See
`docs/credence-pi-pass-2/move-1-design.md`.

## Install (operator)

1. Start the brain daemon — it listens on `http://127.0.0.1:8787`. Either run
   the published image
   (`docker run -p 8787:8787 -v ~/.credence-pi:/root/.credence-pi ghcr.io/gfrmin/credence-pi-daemon`)
   or run it from source
   (`julia --project=<repo-root> apps/credence-pi/daemon/main.jl`).
   See `apps/credence-pi/daemon/README.md`.
2. Build the plugin: `cd apps/credence-pi/openclaw-plugin && npm install && npm run build`.
3. Install it into OpenClaw. From a published registry:
   `openclaw plugins install @gfrmin/credence-pi-openclaw`; or link a local
   checkout for development: `openclaw plugins install -l apps/credence-pi/openclaw-plugin`.
   Then `openclaw plugins enable credence-pi`.
4. **Per-turn cost signal.** On current OpenClaw (≥ 2026.6.2) the `llm_output`
   cost hook is active out of the box — no extra config. (Older builds gated it
   behind a since-removed `plugins.entries.credence-pi.hooks.allowConversationAccess`
   flag; that key is now rejected by the config schema.) Governance —
   allow/block/ask — never depended on it.
5. Restart the gateway so it picks up the plugin.
6. Verify it loaded: `openclaw plugins list` shows `credence-pi` as `loaded`.

## Config (`openclaw.plugin.json` → `configSchema`)

| key | default | meaning |
|---|---|---|
| `daemonUrl` | `http://127.0.0.1:8787` | credence-pi daemon base URL |
| `hookTimeoutMs` | `3000` | max wait for the daemon decision before failing open |
| `approvalTimeoutMs` | `120000` | how long OpenClaw waits for the user on an `ask` before denying |
| `redactToolInputs` | `false` | omit tool-call inputs from sensor events (they can carry secrets); ask-preview becomes generic |
| `shadowMode` | `false` | observe-only: brain decides + daemon logs, body never enforces or overrides the model |
| `routing` | `false` | enable EU-max **model routing** via `before_model_resolve` (see below) |
| `silent` | `false` | suppress info/warn logs |
| `pricing` | — | per-model USD/Mtok overrides: `{ "<model>": { "input": n, "output": n, "cacheRead": n, "cacheWrite": n } }` |

**Privacy:** by default the daemon logs `proposed_call.input` (commands, paths) to
`~/.credence-pi/observations.jsonl`. Set `redactToolInputs: true` to keep inputs out of the log.
Fail-open warnings are emitted once per outage (re-armed when the daemon recovers).

The built-in price table is approximate; set `pricing` for an exact
dollars-saved figure for your providers. Unknown/unpriced models log
`usd: null` (token counts still recorded; the Move-2 surface applies a
token×price fallback).

## Model routing (opt-in)

With `routing: true` the plugin also registers a `before_model_resolve`
hook: for each turn it posts a `route-request` (carrying the prompt's
length feature) to the daemon and applies the brain's chosen model as
OpenClaw's `modelOverride` / `providerOverride`. The decision is the
**same EU-max** the governor runs — `argmax_a [ value·P(correct|X) − cost_a ]`
over the declared roster, through the one canonical `optimise` — so
credence-pi picks the cheapest model whose expected accuracy justifies it
under the active profile.

| brain effector | OpenClaw result |
|---|---|
| `route` | `{ modelOverride, providerOverride }` from the chosen model |
| (timeout / daemon down / no roster) | no override — OpenClaw keeps its configured model (fail open) |

**To enable:**

1. The daemon must have a routing roster. The shipped `bdsl/routing.bdsl`
   declares one (haiku / sonnet / opus) plus `correct-answer-value` in
   `utility.bdsl` (the per-profile dial: low ⇒ cost-sensitive, high ⇒
   quality-sensitive). Absent ⇒ routing stays inert and the daemon is
   governance-only.
2. Set `routing: true` in the plugin config. The `before_model_resolve`
   hook is active out of the box on current OpenClaw (like `llm_output`);
   the plugin registers it defensively, so an older host that lacks it just
   runs without routing.
3. `shadowMode: true` logs the would-route per turn without changing the
   model — use it to see routing decisions on real traffic first.

**What the belief is.** The per-model accuracy belief is **warm-seeded from
measured data** (`brain/routing_brain.counts.json`, distilled from the
3-frontier-model oracle grid by `eval/train_routing_brain.jl`) so routing is
calibrated from install, and it **learns online**: each routed turn's
per-turn outcome — did the proposed call execute cleanly
(`tool-completed.success`)? — updates the routed model's belief. That signal
is noisy (a correct call can fail on a flaky tool), so the brain models the
confound explicitly and learns it (per-context tool-reliability), and a flaky
tool is absorbed by the reliability latent rather than blamed on the model;
the belief conditions on the decoded soft correctness through the canonical
`condition` (mean-exact, so routing's EU is exact). A human approve/reject is
the gold anchor. Honest limits — per-turn proxy ≠ ground-truth correctness;
the false-success rate is only weakly identified; run-level / multi-call
credit assignment stays deferred — are in `eval/results/ROUTING_DOMINANCE.md`.
The belief is durable (route-outcomes are logged and replayed on restart).
Only `prompt-length` is conditioned on, because it is the one feature honestly
extractable from a raw prompt; the structure-BMA collapses it to the marginal
if it doesn't predict accuracy.

## Develop

```
npm install
npm run build      # tsc → dist/
npm run typecheck  # tsc --noEmit
npm test           # node --test (tsx) over tests/*.test.ts
```

## Known limitations (Move 1 / MVP-0)

- `working-directory-relative` and `time-since-last-user-message` features
  are best-effort (OpenClaw doesn't put cwd or message timestamps on the
  tool ctx); the loop-relevant features (tool-name, parent, repetition)
  are exact. In Move 1 the brain does not condition on features at decision
  time, so this only affects future feature-conditioned learning.
- Cost USD is reconstructed from a local price table (no host
  `calculateCost` dependency); override via `pricing`.
- The per-run feature buffer is bounded per run but the run map is not
  evicted on a long-lived gateway — a PASS-2 cleanup item.
