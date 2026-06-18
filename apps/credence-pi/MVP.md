# credence-pi MVP — a proxy welfare-maximiser for OpenClaw

**What it is.** credence-pi is a sidecar that reduces the *negatives* of using OpenClaw —
money, time, risk, wasted effort/loops, bad outcomes, interruptions — and adapts to **each
user's trade-offs** (time vs money, money vs quality, …). It is a **proxy welfare-maximiser**:
ONE shared learned belief, the USER's utility, ONE expected-utility maximisation (the Savage
factorisation). Every cost-reducing decision flows through that one mechanism.

A Julia daemon (the brain) holds the belief and does all the math; a TypeScript OpenClaw plugin
(the body) ships observations + the user's preferences and applies the brain's decisions. The
body is math-free.

## The decisions it makes (all one EU-max over one belief)

| decision | hook | reduces |
|---|---|---|
| **govern** waste/loops (proceed/block) | `before_tool_call` | wasted money + time on repeated/looping calls (tail-aware look-ahead) |
| **ask vs act** | `before_tool_call` | bad outcomes, via VOI-gated confirmation — asks only when the info beats the interruption |
| **flag risk/harm** | `before_tool_call` | harmful actions (the waste cutoff slides with P(unsafe)) |
| **route** the model | `before_model_resolve` | money/time vs quality — picks the EU-max model |
| **escalate** (observe-then-escalate) | per turn | starts cheap, escalates only on observed struggle — the dominance-proof winner |

## The coordinates (each a real unit, no arbitrary constants)

A **user = a utility profile** = exchange rates across the coordinates:

- **money** — per-call $ (implicit weight 1).
- **time** — `w_time`, $ per **second** of the user's wall-clock ("what is your time worth?").
  `E[time]` is a *learned* Poisson-Gamma latency belief (turns × measured s/turn).
- **attention** — `q`, $ per interruption.
- **risk** — `H`, $ per unit expected harm.
- **quality** — `reward` (`correct-answer-value`), $ value of a correct answer.

Trade-offs are **uncertainty-gated**: profiles only diverge where the belief is unsure; when the
brain is confident, every profile agrees — so the user's preferences govern exactly the
ambiguous calls.

## How a user expresses their trade-off

One knob in the OpenClaw plugin config — a named preset, switchable per-request, **no restart**:

```json
{ "plugins": { "entries": { "credence-pi": { "config": { "profile": "speed-first" } } } } }
```

| preset | routes to (same task) | governs by |
|---|---|---|
| `cost-saver` | cheapest viable | blocks waste readily, asks freely |
| `balanced` | the accuracy winner | the middle |
| `quality-first` | the best model, escalates fully | fewer false blocks, more harm-averse |
| `speed-first` | the **faster** model | rarely interrupts (your time is valuable) |

Power users override any individual dial via `profileOverride` (e.g. `w_time` = your $/second).
Presets are *preference data the body ships per request* (like the model roster); `utility.bdsl`
is the daemon's default. The daemon applies the per-request profile to that one decision;
beliefs are untouched.

## See the value

`GET /report` on the daemon returns the governance + spend tallies as JSON (no more manual CLI),
and the plugin logs a one-line in-session summary on session end:

> credence-pi: this run — spend $X; blocked N waste call(s) (~$Y saved, est.); asked M, you
> denied K. Full report: GET …/report

## What's landed (each a tested, CI-green PR on master)

- **Escalation backbone** (#131) — observe-then-escalate wired into the live daemon
  (`escalate-request` ↔ in-process `escalation_next`, equivalence-tested). Best-welfare arm on
  every profile vs every fixed single-model policy, through real daemon code.
- **MVP-A: time as a decision coordinate** (#132) — latency belief (reuses the Poisson-Gamma
  turns belief) + `w_time` folded into the EU offset across routing/escalation/governance.
  **Bit-identical at `w_time=0`.** The time/quality flip is live: reward 1.0 + `w-time` 0 →
  sonnet (accurate); + 0.03 → haiku (faster).
- **MVP-B: user-selectable profiles** (#133) — 4 presets + advanced override, per-request, no
  restart. Proven: same daemon, `speed-first`→haiku, `quality-first`→sonnet.
- **MVP-C: value reporting** (#134) — `GET /report` + in-session summary.

**Evidence.** All tested (40 routing + full feature-brain + 52 plugin assertions, lint-clean),
plus the Terminal-Bench **dominance proof** (`eval/live_ab/EXPERIMENT.md`: escalation is the
best deployable policy, minimax regret 0.069 vs every fixed/baseline router) and the
**live-daemon escalation A/B** (`eval/live_ab/escalation_live.txt`: escalation is the
best-welfare arm on all three profiles).

## Remaining: MVP-D — the live real-usage proof

The last gate is a **live real-OpenClaw welfare A/B**: run real OpenClaw over a range of regular
tasks under two contrasting profiles (cost-saver vs speed-first) and report **net ΔWelfare per
coordinate**, demonstrating the trade-off on real usage. Planned harness:
`eval/live_ab/oc_welfare_ab.sh` (extends the proven `oc_container_smoke.sh` — daemon networked
into the container) + `eval/live_ab/oc_welfare_score.jl` (reuses `welfare.jl` +
`escalation_live.jl`). It spends real API budget and re-enters the container-orchestration path,
so it is best run as its own focused session.

## Honest limits

- The value report surfaces **governance + spend** today; **routing-savings** and the **time**
  coordinate are a noted enrichment (the observation log already carries the `route-decision` +
  `turn-cost` events to compute them).
- Latency is learned offline (warm counts-JSON) and consulted at decision time; **online**
  latency learning (conditioning the Gamma on each turn's observed duration) is the natural
  next step.
- The free local (e.g. ollama) tier is out of MVP scope: measured weak on agentic tasks; the
  broadly-valuable result is priced-roster escalation + the profile trade-offs.
- The per-request profile is trusted from the body (the user's own config); no clamping.
