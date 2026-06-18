# credence-pi routing proof — Terminal-Bench capability×cost matrix

**Goal.** Prove that a Bayesian router (credence-pi) which learns, per task
context X, beliefs over `P(resolved | model, X)` and `E[cost | model, X]`, then
selects the model maximising expected utility

    EU(model | X) = value · P(resolved | model, X) − E[cost | model, X]

**is the best deployable policy by cross-profile minimax regret** over every fixed
single-model policy (always-haiku / always-sonnet / always-opus) and naive heuristics —
on *real* agentic terminal tasks, not toy coding. The edge comes from exploiting the
capability×cost spread: a cheap model that thrashes on a hard task burns more
turns/tokens and can cost **more** than a dearer model that one-shots it, while still
failing.

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

## Findings (2026-06-17, 17-task matrix × 3 tiers × 3 reps, 100 seeds, 60/40 split)

The matrix shows a real, multi-directional capability×cost spread, and ≥3 reps confirm the
decisive inversions are NOT noise — they replicate 3/3. Two robust non-monotonicities
across the price order: `path-tracing` solves on haiku **3/3** but sonnet **0/3** (the
single-rep run flagged this as a likely sonnet-timeout fluke — the reps prove the opposite:
the cheapest model reliably beats a dearer one here), and `configure-git-webserver` /
`password-recovery` solve on sonnet **3/3** but opus only **0/3** and **1/3** (a dearer,
costlier model reliably *worse*). Three tasks (fibonacci-server, nginx-request-logging,
polyglot-rust-c) stay unsolved by every tier at budget. The de-risked cheapest-cost oracle
(majority-solve) clears 13/17 at $1.86. Non-monotonic, *replicated* capability is exactly
what no fixed tier order can exploit and observe-then-escalate can.

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
   {cost-hawk, balanced, quality-hawk} (3 reps/cell, 100 seeds, lower=better):
   **escalation-eu 0.069** vs plain frugalgpt-cascade 0.92, always-sonnet 2.06, always-opus
   5.68, up-front eu-max 6.11, every fixed/RouteLLM ~10.9 — a 13× gap to the next deployable
   arm. It **beats always-opus on every profile** and **dominates the plain (gateless)
   FrugalGPT cascade** (≥ all profiles, strictly > on cost-hawk +0.85 and balanced +0.29 —
   the EU gate's value; they tie on quality-hawk, where a $5 solve is worth full escalation
   and the gate never halts). It **recovers ~95% of the non-deployable clairvoyant-oracle
   ceiling** on quality-hawk (25.46 of 26.76) and **trails that oracle by ~0.6–1.3 welfare**
   on every profile — as any deployable policy must (a per-task hindsight oracle weakly
   dominates by construction). The credence-pi metareasoning story: not "beats the
   clairvoyant bound" (impossible — a mislabel corrected after adversarial review), but
   "best deployable, recovering most of the ceiling." The single-rep headline was 0.043;
   3 reps move it to 0.069 (the binding profile shifts to cost-hawk) — same ranking, same
   13×-plus margin, now de-risked against per-cell outcome noise.

**The win is cross-profile robustness, NOT per-profile dominance.** Per profile,
escalation-eu is near-best: on cost-hawk it trails the leader always-haiku by 0.069 (the
binding regret — at a $0.25 solve the gate occasionally escalates to a tier that doesn't
pay off), on balanced it trails always-sonnet by just 0.020 (a near-tie, 36% of seeds), and
on quality-hawk it IS the best deployable arm (tying the gateless cascade). No fixed policy
is good across the mix — always-haiku wins cost-hawk but is worst-but-one on quality-hawk
(regret 10.9); always-sonnet wins balanced but trails 2.06 worst-case. escalation-eu is
near-best *everywhere*, the minimax-regret win (the Wald complete-class point).

**Honest limits:** (a) needs an observable success signal (here the test suite; elsewhere a
fallible verifier — the known FrugalGPT caveat). (b) The gate is MYOPIC (single-step), not
the exact sequential EU-max: it ignores the option value of still-dearer rungs, so it
under-escalates (conservative). On this sample the conservatism doesn't hurt; the exact
backward-induction variant is future work, not a claimed result. (c) The gate now lives in
the brain (`RoutingBrain.escalation_next`, via `optimise`) AND is wired into the live daemon
loop (`escalate-request`, §"Live-daemon escalation" below) — a unit test asserts the daemon
decision equals in-process `escalation_next` on identical inputs, so this offline result
transfers to the live system. (d) Headline regret 0.069 is
the canonical config (3 reps, 100 seeds, train-frac 0.6, 3-point profile set); the
single-rep sensitivity sweep put the range at ~0.04–0.14 across reconfigurations and the
rep3 point sits inside it. Robust as the *lowest* worst-case among deployable arms by 13×,
not a knife-edge single value. (e) 3 reps/cell — the decisive inversions (path-tracing,
configure-git-webserver, password-recovery) replicate 3/3, so they are real capability
boundaries, not per-cell outcome noise; this directly de-risks the single-rep concern.
Sonnet timeouts remain recovered lower-bound costs (still under-counting always-sonnet's
cost, which flatters it over escalation-eu — correcting it only widens escalation-eu's
lead). (f) `cron-broken-network` excluded (grader can't score it). (g) Cascade-dominance
leans partly on the 3 all-fail tasks (where the gate halts but the gateless cascade pays the
full ladder); on the decision-relevant subset the margin narrows but escalation-eu still
leads.

*Verified by a 5-skeptic adversarial workflow (leakage clean; foil-fairness caught the
clairvoyant mislabel — now fixed; metric/data/mechanism minor caveats folded in above).*

## Live-daemon escalation (stage a) — the gate wired into the daemon loop

The dominance result above was an *in-process* eval. This stage runs the SAME observe-then-
escalate gate **as live daemon code**: `eval/live_ab/escalation_live.jl` drives the
try→observe→escalate loop entirely over the running daemon's `escalate-request` HTTP handler
(280 round-trips per run, logged), scoring realized welfare on the 17-task × 3-rep matrix (51
worlds). Two guards make it honest:

- **Cold belief** (`CREDENCE_PI_ROUTING_BRAIN=""`): the shipped warm belief was trained on
  this matrix, so we run the gate with the *cold* prior (θ=0.5). This is leakage-free AND a
  **conservative lower bound** — the deployed warm belief can only sharpen the gate. With cold
  θ the gate is purely "is the next rung worth its cost vs reward·0.5"; the observed `resolved`
  (ground truth) drives escalation.
- **Leave-one-task-out cost**: the gate's only matrix-derived input is per-(tier,difficulty)
  expected cost, computed excluding the scored task. (Cost is not the `resolved` label.)

Realized per-world means (reward = profile value of a correct answer):

| profile | arm | welfare | solves | cost ($) |
|---|---|---:|---:|---:|
| **cost-hawk** (0.25) | **escalation** | **0.0289** | 0.451 | 0.0839 |
| | always-haiku | 0.0237 | 0.412 | 0.0793 |
| | always-sonnet | 0.0019 | 0.706 | 0.1745 |
| | always-opus | −0.1932 | 0.627 | 0.3501 |
| **balanced** (1.0) | **escalation** | **0.5437** | 0.784 | 0.2406 |
| | always-haiku | 0.3325 | 0.412 | 0.0793 |
| | always-sonnet | 0.5313 | 0.706 | 0.1745 |
| | always-opus | 0.2774 | 0.627 | 0.3501 |
| **quality-hawk** (5.0) | **escalation** | **3.6151** | 0.784 | 0.3064 |
| | always-sonnet | 3.3549 | 0.706 | 0.1745 |
| | always-opus | 2.7872 | 0.627 | 0.3501 |
| | always-haiku | 1.9795 | 0.412 | 0.0793 |

**Escalation's welfare point estimate is best on every profile**, through the live daemon, with a
cold (leakage-free) belief. Versus the A/B baseline always-sonnet: **+0.027 / +0.012 / +0.260
welfare** (cost-hawk / balanced / quality-hawk). A cluster bootstrap over the 17 tasks (reps within
a task are correlated, so we resample TASKS, not the 51 rows; percentile, under-covering at K=17)
quantifies the *precision* — and the honest reading is **under-powered, not refuted**: the per-call
welfare margins are small relative to task-to-task variance, so 17 tasks give wide intervals
(cost-hawk +0.005, CI [+0.001, +0.010], p=0.009 — significant but economically negligible; balanced
+0.012, CI [−0.10, +0.18], p=0.47; quality-hawk +0.260, CI [−0.20, +0.95], p=0.23). A wide interval
here is a sample-size statement, **not** evidence the margin is zero — the point estimates favour
escalation on all three. The **lower-variance** capability-union signal is directionally firm:
escalation **solves more than any single tier** (0.784 vs sonnet 0.706 on balanced/quality-hawk;
solve-rate Δ +0.078, CI [0.00, +0.22] — never crosses zero), capturing the *union* of tier
capabilities — the non-monotonic tasks (path-tracing only haiku; configure-git-webserver only
sonnet) no fixed order exploits. On cost-hawk it is frugal instead (−52% cost: $0.084 vs $0.175,
stopping early when a solve isn't worth its price). So the live A/B confirms the mechanism and the
direction end-to-end; tightening the per-profile *magnitude* needs a larger sample (task #21) or
the better-powered offline minimax sim above — consistent with the "cross-profile robustness"
framing, which is the structural fact that no single model wins all profiles, independent of the
per-profile error bars. (Output: `results/escalation_live.txt`, with the bootstrap CIs. Stage b
adds a free local row + a live OpenClaw confirmation.)

## Calibration of the up-front belief (`eval/calibration.jl`)

Is the belief's θ a *true* probability? The EU trade reward·θ − cost is only as good as θ's
calibration, not just its ranking. Held-out predicted-θ vs realized-solve over the 3-rep
matrix (100 seeds, 6300 pairs, difficulty/category/length features):

| metric | value | read |
|---|---|---|
| ECE (10-bin) | **0.074** | moderate; under-confident at the low end (predicts ~0.17 where reality is ~0.77), well-calibrated mid-range |
| AUROC | **0.59** | near-chance — the up-front belief barely *ranks* solve-vs-fail on held-out tasks |
| Brier / log-score | 0.243 / 0.679 | — |

The weak AUROC has a real, honest root cause: the warm belief orders haiku<sonnet<opus (from
the MCQ oracle grid), but tb reality is sonnet>opus (sonnet solves 36/51, opus 32/51), so it
**mis-ranks tiers** on agentic tasks. This is the dominance eval's verdict — **difficulty/
category/length + the MCQ warm-seed don't predict the agentic capability boundary** — now
quantified. It is exactly why **observe-then-escalate wins**: it needs no calibrated up-front
θ, just an observed failure. The implication for live up-front routing: re-warm the belief on
agentic outcomes and/or enrich the features (the structure-BMA auto-sophisticates) — bounded
by what is honestly extractable at routing time. (Non-causal diagnostic; never feeds a
decision — `posterior_accuracy` read-out only.)

## Status

- [x] Stack validated (oracle + 3 tiers on hello-world; clean cost capture)
- [x] Prices verified
- [x] Pilot matrix (6 tasks × 3 tiers) — harness validated; spread too small to route
- [x] Full matrix (17 tasks × 3 tiers) — `results/tb_matrix_full.jsonl`
- [x] Beliefs + train/test split + dominance test — `tb_dominance.jl`; escalation-eu best deployable
- [x] Adversarial verification (5-skeptic workflow) — SURVIVES-WITH-CAVEATS; clairvoyant
      mislabel caught & fixed; minor caveats folded into Findings above
- [x] ≥3 reps/cell — de-risked (`results/tb_matrix_rep3.jsonl`; rep-aligned aggregation in
      `tb_dominance.jl`, verified behavior-identical at R=1). Regret 0.043→0.069; ranking and
      13×-margin hold; the flagged inversions confirmed real (3/3), not noise.
- [x] Myopic-EU escalation gate in the brain via `optimise` (`RoutingBrain.escalation_next`);
      the eval calls it (no host reimplementation); tests assert the exact gate boundary.
- [x] OpenClaw-in-container smoke (`live_ab/oc_container_smoke.sh`) — OpenClaw runs in a
      Terminal-Bench container and completes a real task (haiku, transport=embedded).
- [x] Report: measured savings vs every fixed policy + honest limits (this doc)
- [x] **Escalation gate wired into the LIVE daemon loop** (`escalate-request`; brain
      `escalate-decide`); equivalence-tested (`daemon decision == in-process escalation_next`,
      `tests/julia/test_routing.jl` §B'); stage-(a) live-daemon A/B below — closes limit (c).
- [x] **MVP-D (live TB MVP)** — OpenClaw run LIVE on real TB tasks in their own containers,
      graded by the tasks' own tests (`oc_tb_spotcheck.sh`, `results/tb_spotcheck.txt`):
      hello-world ✓, fix-permissions ✓ (haiku), and the routing-relevant differentiator
      `heterogeneous-dates` — **haiku FAILS / sonnet SOLVES**, reproducing the matrix split
      live → validates the claude-CLI matrix as a faithful OpenClaw proxy. Cross-profile breadth =
      `escalation_live.txt` (escalation's point estimate best on all profiles; per-profile welfare
      bars wide at 17 tasks — precision, not refutation; bootstrap CIs in that file).
      Free-local (qwen) row done as a measured **crossover** (`oc_welfare_matrix.py` /
      `welfare_matrix.jsonl` / `oc_welfare_score.jl`): free wins only at `reward ≤ 0.044 −
      141·w_time` — a near-zero answer-value AND time-value (batch) user; else cheap-fast-paid
      dominates. Plugin+daemon route+govern live = `oc_welfare_run.sh`.
- [ ] (deferred = task #21) full OpenClaw-on-every-TB-task graded matrix (rebuild tb's per-task
      container+grading around OpenClaw's 2.3GB bind-mount); the spot-check + proxy cover the MVP.
- [ ] (future) exact sequential-EU (backward-induction) gate variant vs the myopic gate
