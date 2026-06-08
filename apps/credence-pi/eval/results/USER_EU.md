# Maximising the OpenClaw user's expected utility (the unifying frame)

credence-pi's job is one EU calculation: for each proposed action, choose
`e ∈ {proceed, ask, block}` maximising the **OpenClaw user's** expected utility, in one
currency:

```
EU(user, e | X) = V·P(success|X) − H·P(unsafe|X) − c·cost(X) − q·interrupt(e)
```

Safety, task-value, and waste are *terms in the user's utility*, not three bolted-on
classifiers. This note records what each term actually is, measured on real data
(`benchflow/ClawsBench` outcomes + `AI45Research/ATBench-Claw` safety), and which are real
learnable levers versus protective constraints versus metareasoning.

Reproduce: `analysis/clawsbench_user_eu.py` (numbers below), `safety_eval.jl` (harm term),
`results/FINDINGS.md` (waste term), `results/SAFETY_INGREDIENTS.md` (taint).

## The user's EU problem is real and large

On 1,132 ClawsBench runs with real cost+outcome data (Gemini-flash, 40 task types):

- **23% pass / 77% fail.** (These are hard benchmark tasks; a typical user's mix fails less.)
- **Failed runs burn 45% more tool calls** than passed (13.8 vs 9.5; median 10 vs 7).
- **83% of all tool-call spend and 82% of agent time is on runs that FAIL.**
- **Cost rises monotonically as the outcome worsens:** success 9.5 → partial 12.9 →
  zero 13.1 → **negative (actively harmful) 15.8**. Thrashing *is* the failure signature,
  and the harmful runs cost the most.

So the user's EU is dominated by money+time poured into doomed runs. That is the prize.

## Term by term

### c·cost — the waste lever (PROVEN, the dominant user-EU lever)
The governor blocks repeated-identical calls at precision 1.0 / recall 1.0 on held-out
real sessions (`FINDINGS.md`). Direct, measured, EU-positive (identical re-execution
produces nothing). This is the one term the governor can move *up* with certainty.

### H·P(unsafe) — the harm lever (PROVEN, high value-per-event)
The taint-flow features (`SAFETY_INGREDIENTS.md`) give a calibrated `P(unsafe|X)` reaching
0.94 precision at 1.2% false-interrupt — blocking injected-data exfiltration with near-zero
collateral. And the data above shows harmful outcomes cost the *most* (negative = 15.8
calls), so cutting them saves money too. Recall is structurally bounded (~30%): a
tool-boundary governor cannot see harmful *output*.

### V·P(success) — task-value: NO learnable per-action lever (the attribution wall)
Per-*action* success credit is **unrecoverable** from per-session reward:
- Outcomes join to the event stream only at `(model, task_name)` granularity — **13% of
  calls** — i.e. task-type difficulty, not per-session, not per-action.
- Per-call budget-burn against that label is **flat** (0.27 / 0.25 / 0.24 / 0.26): the
  coarse label washes out the within-task "this run is thrashing" signal.

This is the exploration⊥attribution law (Paper 1) in the flesh: a governor cannot learn
"this action helps" from sparse session-level reward. A learned `P(success|action)` term
would be a fiction; we do not add one.

### The actionable task-value contribution is METAREASONING, not a classifier
What *is* learnable from per-run reward is per-session-**state** doom:

```
base P(fail) = 0.767
P(fail | budget-burn ≥ 10) = 0.875   (~9 calls still at risk)
P(fail | budget-burn ≥ 25) = 0.913   (~16 calls still at risk)
```

"How much more to compute" is itself an EU decision (the metareasoning principle, CLAUDE.md
§On metareasoning): as a session burns budget without resolution, the VOI of asking *"is
this going anywhere?"* rises, because the marginal budget is better spent on a cheap
confirmation than on continued likely-doomed execution. In one currency
(`EU(ask@K) − EU(continue) ≈ P(fail|≥K)·remaining·c − q`):

```
ask@burn≥10: fires on 50% of runs, P(doomed)=0.88, EU gain ≈ $3.93/run   (c=$0.50, q=$0.02)
ask@burn≥15: fires on 24% of runs, P(doomed)=0.88, EU gain ≈ $5.28/run
```

This is a session-level VOI rule (the existing `net_voi` machinery with cost-so-far in the
utility), using a belief learnable from per-run reward — **not** per-action credit. It is
EU-positive whenever call cost ≫ interrupt cost.

## What this resolves

The three-term EU, on real data, reduces to **two learned levers** (waste, harm), **one
protective constraint** (don't over-block ⇒ preserve P(success); confirmed by the 1.2%
false-interrupt and 0% over-block on distinct reads), and **one metareasoning escalation**
(ask as budget burns). The honest "task-value" answer is not a `P(success)` classifier —
it is metareasoning over the user's compute budget, which is the credence-native form and
needs no per-action credit.

The cross-outcome tradeoff is the single dial `H/V` (harm-aversion) plus `c`, `q`; all in
one currency so the user sets their own risk/cost/value point.

## Honest scope / what needs live telemetry

Benchmark task mix (77% fail) ≠ typical user mix — burn is a *stronger* relative signal at
lower base failure. The EU figures assume the user aborts a doomed run when asked and use
labelled default $; replay-stationarity holds for waste, weaker for the metareasoning
escalation. The causal "enforcing it raises net user EU" is settled by live enforcement
telemetry (shadow mode + opt-in users) — the same gate as the waste claim.
