# credence-pi routing proof — Terminal-Bench capability×cost matrix

**Goal.** Prove that a Bayesian router (credence-pi) which learns, per task
context X, beliefs over `P(resolved | model, X)` and `E[cost | model, X]`, then
selects the model maximising expected utility

    EU(model | X) = value · P(resolved | model, X) − E[cost | model, X]

**dominates every fixed single-model policy** (always-haiku / always-sonnet /
always-opus) and naive heuristics — on *real* agentic terminal tasks, not toy
coding. The dominance comes from exploiting the capability×cost spread: a cheap
model that thrashes on a hard task burns more turns/tokens and can cost **more**
than a dearer model that one-shots it, while still failing.

This is grounded in real-world OpenClaw-class usage: the instrument is
`claude-code` (native tool-calling, same agentic class as OpenClaw — Bash/Edit/
Write/Read/Grep loop), run on **Terminal-Bench 2.0** tasks (89 human-verified
end-to-end terminal tasks: compile/debug/configure/sysadmin/security/data-science,
each with a Docker env + deterministic test suite). OpenClaw-live is the
deployment grounding (separate, follow-on).

## Why Terminal-Bench (ROI argument)

A router's return = (spread in outcomes across model choices) × (frequency the
spread is decision-relevant). Saturated easy coding (Exercism) → all frontier
models solve everything → spread collapses → router ROI ≈ 0. Terminal-Bench
maximises spread: local fails, haiku thrashes/fails on hard tasks, sonnet is the
workhorse, opus is overkill on easy. High spread ⇒ high router ROI. Honest
boundary: ROI is realised on *divergent* tasks, not on uniformly-trivial or
uniformly-impossible tails.

## Instrument (validated 2026-06-17)

- Dataset: `terminal-bench-core==0.1.1` (80 tasks: 12 easy / 44 medium / 24 hard;
  9 categories; incl. 4 real SWE-bench issues and several `.easy`/`.hard` variants).
- Agent: `tb_agent.py::ClaudeCodeStreamJSONL` — subclasses tb's `claude-code`
  agent, redirecting `claude --output-format stream-json` to a mounted JSONL file
  for clean, cache-aware cost capture (the stock agent's asciinema capture mangles
  the JSON). Native tool-calling — NOT terminus (terminus's brittle text-JSON
  schema confounds capability with schema-compliance and records no usage).
- Runtime: podman (docker-CLI shim) + podman-compose; `--n-concurrent` for fan-out.
- Per run we record: `resolved` (tb's deterministic grade), `cost_usd`
  (claude's own `total_cost_usd` — the real bill, priced live; cross-checked to
  reproduce exactly from the verified token rates), `num_turns`, full token
  breakdown (input/output/cache_read/cache_create), `duration_s`, `failure_mode`.

### Prices (verified 2026-06-17, USD/MTok: input / output / cache_read / 5m_write)

| haiku | sonnet | opus |
|---|---|---|
| 1 / 5 / 0.10 / 1.25 | 3 / 15 / 0.30 / 3.75 | 5 / 25 / 0.50 / 6.25 |

NB Opus 4.8 is **$5/$25**, ~3× cheaper than the retired Opus-4.1 $15/$75 — only
~1.67× Sonnet. The cost claim must use current prices (a stale table is a
first-read killer); we report claude's realized bill, which encodes them.

## Method

1. **Matrix** (`tb_matrix.py`): for each tier × task (R reps), one `tb run` per
   tier fanning over tasks. → `results/tb_matrix.jsonl`. This is the empirical
   ground truth the router is a decision OVER.
2. **Beliefs** (credence-pi brain, all via `condition`/`expect` — no arbitrary
   constants): per (tier, context-cell) a Beta over `P(resolved)` [BetaBernoulli]
   and a Gamma over turns [Poisson-Gamma, already in `src/`]; `E[cost|tier,X]`
   from `E[tokens|tier,X]` × verified per-token price. Context X from declared
   task features (difficulty / category / instruction length — plus a
   prompt-only feature set to refute "you used the label as the answer").
3. **Split**: warm beliefs on a train split; route on held-out test tasks;
   tally realized (cost, correctness) of the router vs each fixed policy and vs
   naive heuristics (always-cheapest, always-best, random).
4. **Dominance test**: router's cost-per-solve and total cost ≤ every fixed
   policy at equal-or-better resolve rate, on held-out tasks.
5. **Adversarial verification** (Workflow tool): independent checks for
   confounds — train/test leakage, cost-accounting errors, cherry-picked tasks,
   feature = relabelled answer, variance/seed sensitivity.

## Honest limits (to state up front in the report)

- ROI is realised on divergent tasks; uniformly-easy/impossible tails give the
  router nothing to exploit (and we will show that, not hide it).
- The matrix instrument is `claude-code`, not OpenClaw; the routing math is
  agent-agnostic, but the OpenClaw-live A/B is the deployment proof (follow-on).
- Per-run cost has cold/warm-cache variance; reps + the Gamma belief absorb it.
- `local` (ollama) is excluded from this matrix (the claude CLI can't drive it);
  it is a separate, OpenClaw-only arm — and prior work shows local can't reliably
  do agentic tool-use, so it is the free-but-incapable floor, not a contender.

## Findings (2026-06-17, 17-task matrix × 3 tiers, 40 seeds, 60/40 split)

The matrix shows a real, multi-directional capability×cost spread: the cheapest-cost
tier that solves is haiku on 9 tasks, sonnet on 2, opus on 3 (and 3 tasks — fibonacci-
server, nginx-request-logging, polyglot-rust-c — no tier solves at budget).
`configure-git-webserver` is solved by sonnet but FAILS on opus (which also costs 2×);
`path-tracing` is solved by haiku but fails on sonnet (a single-rep inversion — likely a
sonnet-timeout fluke; ≥3 reps would de-risk). The per-task oracle (cheapest solver)
solves 14/17 at $2.72 vs always-opus's 13/17 at $7.30.

**Two routing strategies, opposite verdicts — the central result:**

1. **Predict-then-route (up-front, `eu-max`) does NOT win.** difficulty/category/length
   do not predict the idiosyncratic capability boundary (an "easy" task haiku fails; a
   "hard" task haiku solves), so the feature belief can't beat a strong all-rounder
   (always-sonnet). eu-max ties the other feature-based foils and beats only the
   over-spenders. Honest negative result; the features don't carry the signal.

2. **Observe-then-escalate (`escalation-eu`) is the best DEPLOYABLE policy.** Try cheapest;
   escalate only when the myopic single-step EU is positive, reward·E[θ_next|X] ≥
   E[cost_next] (the gate, same beliefs); stop at the first OBSERVED solve (the test suite
   is the verifier). It is `frugalgpt_cascade` PLUS that one gate — so the gate is the
   entire difference. Result, **minimax regret vs the best deployable arm** across
   {cost-hawk, balanced, quality-hawk} (100 seeds, lower=better): **escalation-eu 0.043**
   vs plain frugalgpt-cascade 1.35, always-opus 2.59, always-sonnet 4.75, every
   fixed/RouteLLM ~8.1. It **beats always-opus on every profile** and **dominates the plain
   (gateless) FrugalGPT cascade** (≥ all profiles, strictly > cost-hawk & balanced — the EU
   gate's value). It **recovers ~95% of the non-deployable clairvoyant-oracle ceiling** on
   quality-hawk and **trails that oracle by ~0.4–1.4 welfare** on every profile — as any
   deployable policy must (a per-task hindsight oracle weakly dominates by construction).
   This is the credence-pi metareasoning story: not "beats the clairvoyant bound" (that is
   impossible and was a mislabel — corrected after adversarial review), but "best
   deployable, recovering most of the ceiling."

**The win is cross-profile robustness, NOT per-profile dominance.** Per profile,
escalation-eu is *second-best*: 3rd on cost-hawk (behind always-haiku and argmax-accuracy
— it ~ties haiku, margin −0.04 but wins 64% of seeds), 2nd on balanced (behind
always-sonnet, ~59% seeds), ties the cascade on quality-hawk. No fixed policy is good
across the mix (each wins its own profile and is badly off elsewhere); escalation-eu is
near-best *everywhere*, which is the minimax-regret win (the Wald complete-class point).

**Honest limits:** (a) needs an observable success signal (here the test suite; elsewhere a
fallible verifier — the known FrugalGPT caveat). (b) The gate is MYOPIC, not exact
sequential EU-max; the exact backward-induction variant has worst-regret ~0.073 (the
myopic gate edges it on *this* sample via conservatism — not a general claim). (c) The
winning arm is not yet wired into the live daemon (offline eval result). (d) Headline
regret is a RANGE: 0.043 at 100 seeds / train-frac 0.6, rising to ~0.14 at frac 0.5 and
~0.06 at the binding reward r≈0.35 the 3-point profile set skips; report it as
~0.04–0.14, robust as the *lowest* worst-case across every reconfiguration tried, not as a
single point. (e) 1 rep/cell ⇒ outcome noise (path-tracing inversion); the 4 sonnet
timeouts are recovered lower-bound costs (under-counting flatters always-sonnet more than
escalation-eu, so correcting it doesn't break the result). (f) `cron-broken-network`
excluded (grader can't score it). (g) Cascade-dominance leans partly on the 3 all-fail
tasks (where the gate halts but the gateless cascade pays the full ladder); on the
decision-relevant subset the margin narrows but escalation-eu still leads.

*Verified by a 5-skeptic adversarial workflow (leakage clean; foil-fairness caught the
clairvoyant mislabel — now fixed; metric/data/mechanism minor caveats folded in above).*

## Status

- [x] Stack validated (oracle + 3 tiers on hello-world; clean cost capture)
- [x] Prices verified
- [x] Pilot matrix (6 tasks × 3 tiers) — harness validated; spread too small to route
- [x] Full matrix (17 tasks × 3 tiers) — `results/tb_matrix_full.jsonl`
- [x] Beliefs + train/test split + dominance test — `tb_dominance.jl`; escalation-eu best deployable
- [x] Adversarial verification (5-skeptic workflow) — SURVIVES-WITH-CAVEATS; clairvoyant
      mislabel caught & fixed; minor caveats folded into Findings above
- [~] Report: measured savings vs every fixed policy + honest limits (this doc; ≥3 reps next)
- [ ] (next capability) live myopic-EU escalation in the daemon (via `optimise`); ≥3 reps;
      OpenClaw-in-container grounding; exact sequential-EU gate variant
