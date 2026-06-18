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

## MVP-D — the live proof on Terminal-Bench (DONE, 2026-06-18)

MVP-D proves the product on **Terminal-Bench** with the **Anthropic models the vast majority of
OpenClaw users run** (haiku/sonnet/opus) — where real tasks genuinely differentiate the models
(measured solve: haiku 41%, sonnet 71%, opus 63%, at a 3–4× cost spread), so the **quality⟷cost**
routing trade is real (unlike trivial tasks, where the cheapest reliable model dominates and
there is nothing to route).

**The product runs live on real TB tasks, graded by TB's own tests** (`oc_tb_spotcheck.sh`,
`results/tb_spotcheck.txt`): OpenClaw, in each task's own container, solved hello-world and
fix-permissions; and — the routing-relevant case — on the medium task `heterogeneous-dates`,
**OpenClaw+haiku FAILS (computes the wrong average) while OpenClaw+sonnet SOLVES.** The model
choice decides success: a cost-saver routed to haiku saves money but fails it; a quality-first
routed to sonnet pays more and gets the answer. OpenClaw reproduces the capability×cost matrix's
per-(model,task) solve pattern exactly (fix-permissions haiku ✓; het-dates haiku ✗ / sonnet ✓;
sqlite-db-truncate haiku ✗), validating that matrix as a faithful OpenClaw proxy.

**Routing dominates every fixed router, per profile, across 17 real TB tasks**
(`results/escalation_live.txt`, scored through the **live daemon**, leakage-free cold belief).
credence-pi's observe-then-escalate is the best-welfare arm on every profile:

| profile | credence-pi (escalate) | always-haiku | always-sonnet | always-opus |
|---|---|---|---|---|
| cost-saver | **0.0289** | 0.0237 | 0.0019 | −0.193 |
| balanced | **0.5437** | 0.333 | 0.5313 | 0.277 |
| quality-first | **3.6151** | 1.98 | 3.355 | 2.787 |

No single fixed model is best for every profile; credence-pi's per-profile EU-max is — "smarter
than every fixed router," measured on real TB tasks.

**The plugin + daemon route + govern LIVE** (`oc_welfare_run.sh`): OpenClaw + the credence-pi
plugin + the daemon, in-container, with `before_tool_call` governance and the daemon's routing
decision over the wire — the de-risk that closes `oc_container_smoke.sh`'s stated remainder.

**Honest framing.** The per-(model,task) capability matrix was gathered with the claude-CLI
agent — an OpenClaw-class native-tool-calling loop; the spot-check confirms OpenClaw reproduces
its solve pattern, and routing value is **model-capability-driven** (which model solves which
task at what cost), which is agent-invariant — so the dominance transfers to the product. Full
OpenClaw-on-every-TB-task graded (rebuilding tb's per-task container + grading around OpenClaw's
2.3GB bind-mount) is the heavier task #21, deferred.

## Honest limits

- The value report surfaces **governance + spend** today; **routing-savings** and the **time**
  coordinate are a noted enrichment (the observation log already carries the `route-decision` +
  `turn-cost` events to compute them).
- Latency is learned offline (warm counts-JSON) and consulted at decision time; **online**
  latency learning (conditioning the Gamma on each turn's observed duration) is the natural
  next step.
- **Free local models are a measured niche, not the default** (the common case is Anthropic
  models). We measured when a free local model (`qwen2.5:7b`) beats cheap cloud (haiku) on a
  live 6-task suite (`oc_welfare_matrix.py`, `welfare_matrix.jsonl`): free wins only when
  `reward ≤ $0.044 − 141·w_time` — i.e. a user who values a correct answer at under ~4¢ **and**
  their time at under ~$1/hr (true batch/overnight). For cheap agentic tasks haiku
  (~$0.007/task, faster, more reliable: 6/6 vs qwen 5/6) dominates; the free tier's slowness +
  lower reliability outweigh its ~$0.007 saving. credence-pi's EU-max draws this crossover
  correctly — it does not naively chase the free model (always-qwen is 8× worse than
  credence-pi's pick for a speed-first user).
- The per-request profile is trusted from the body (the user's own config); no clamping.
