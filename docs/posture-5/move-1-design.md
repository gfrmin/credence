# Move 1 — Benchmark methodology for credence-proxy v0.1

## 0. Strategic context

Posture 5 path 2: ship credence-proxy v0.1 against the current OAI-compatible
architecture. Cache savings are out of scope (Move 0 found prompt caching is
unsupported on the OAI-compatible endpoint). The proxy's cost-savings story is
**model-tier routing only**: Bayesian EU maximisation selects cheaper models
(Haiku) when expected quality is sufficient, falling back to costlier models
(Sonnet, Opus) when the task demands it.

Move 1 designs a benchmark methodology that validates this claim on
representative workloads. The methodology must be defensible *for what it
claims* — model-tier routing savings on coding-shaped workloads — and honest
about what it doesn't measure (cache savings, multi-provider routing beyond
Anthropic+OpenAI, production-scale workloads).

A named methodological feature: the design isolates routing-induced variance
from LLM-output variance. Baseline repetitions (always-Sonnet, no routing)
measure output variance alone; routing repetitions combine output variance with
routing variance (cold-start learning, model-selection stochasticity). This
decomposition lets the benchmark report routing's own statistical reliability
separately from the underlying model stochasticity — a precise answer to "are
the savings real or noise?" rather than an aggregate one.

Move 0's audit report is a named input: any benchmark that implies cache
savings contradicts the audit's findings and is dishonest.

## 1. Scope

Two deliverables, sequenced:

1. **Upfront feasibility check** (pre-benchmark gate). Validates path 2's
   premise: that Bayesian routing produces meaningful cost savings on
   representative coding workloads. If the check fails, the methodology design
   pauses and the value proposition is re-examined. Specified in §Pre-design.

2. **Benchmark methodology** (the main artefact). Specifies workload selection
   (§5.1), quality measurement (§5.2), baseline selection (§5.3), cost
   measurement (§5.4), statistical robustness (§5.5), and deliverable shape
   (§5.6). Move 2 executes this methodology and produces results.

## 2. Out of scope

- Cache savings measurement (out of v0.1 per path 2).
- Native Messages API migration design or execution.
- Multi-provider routing evaluation beyond what the current proxy supports
  (Anthropic + OpenAI are the two configured providers; Google is named in
  the provider type but not configured).
- OpenClaw-version-specific behaviour or OpenClaw integration testing.
- The benchmark execution itself (Move 2).
- Publication artefact (post-MVP, post-Move 2).

## Pre-design: upfront feasibility check

### Purpose

The existing `scripts/openclaw_eval.py` 50-categorical-query smoke test reports
significant savings, but the workload is synthetic (single-turn chat/factual/
creative queries) and unrepresentative of real coding use. Real agentic-coding
workloads are multi-turn, dominated by code-category turns where Sonnet's
reasoning may be genuinely needed, and the Haiku-eligible fraction may be
smaller than the synthetic eval suggests.

The feasibility check validates that routing produces meaningful savings on
coding-shaped workloads before committing to the full benchmark methodology.
Note: the feasibility check's savings numbers are upper-bound estimates because
beliefs accumulate across workloads (warm-start after the first few). The full
benchmark's fresh-priors-per-repetition design (§5.5) produces more
conservative figures that include the cold-start cost every real user pays.

### Specification

**Workloads.** 5 hand-curated multi-turn coding workloads, each 5–8 turns:

| # | Task type | Description shape |
|---|-----------|-------------------|
| 1 | Debug | "Here's a traceback, find the bug, propose a fix" |
| 2 | Implement feature | "Add pagination to this API endpoint" |
| 3 | Refactor | "Extract this repeated pattern into a helper" |
| 4 | Write tests | "Write unit tests for this function" |
| 5 | Explain + modify | "Explain what this code does, then change X" |

Each workload is a conversation driven by a simple client script. The client
sends OpenAI-format chat-completion requests to the proxy, includes the
response in the next turn's message history, and records the routing decision,
token counts, and cost per turn.

**Execution.** Two phases, same as the existing eval's structure:

1. **Baseline phase**: `CREDENCE_FORCE_MODEL=claude-sonnet-4-6`. All turns use
   Sonnet. Measures the "naive" cost of running everything on Sonnet.
2. **Routing phase**: no forced model. Bayesian routing enabled with fresh
   priors. The 5 workloads run in sequence under a single routing state, so
   beliefs accumulate across workloads. The first 1–2 workloads serve partially
   as warm-up; the last 2–3 run against informed beliefs. Per-workload costs
   are reported to show the learning curve.

**Metrics.** Per workload:
- Total cost under baseline (always-Sonnet).
- Total cost under routing.
- Savings percentage.
- Per-turn routing decisions (which model was selected).
- Quality signal: task completion (did the conversation produce working code /
  a correct explanation / passing tests?).

Aggregate: mean savings across the 5 workloads, weighted by token volume.

**Pass threshold: ≥15% aggregate cost savings with no quality degradation.**

Justification: at 15% savings, a team spending $10K/month on LLM API calls
saves $1,500/month — meaningful enough to justify the proxy's operational
overhead. Below 10%, the savings are marginal compared to the complexity of
running a routing proxy. The threshold is conservative: the synthetic eval
reports 40–60% savings, but representative workloads will have fewer
Haiku-eligible turns. 15% is the floor for a credible v0.1 cost-savings claim.

**Quality degradation criterion:** the routing phase must achieve the same
task-completion rate as the baseline phase. A workload that completes
successfully under Sonnet but fails under routing counts as quality
degradation, regardless of cost savings.

**Fail outcome.** If aggregate savings < 15%, or if any workload shows quality
degradation:
- Report the findings honestly.
- Pause the methodology design.
- Re-examine: is the routing's category inference too coarse to distinguish
  Haiku-eligible turns? Is the EU cost-quality tradeoff miscalibrated? Is the
  value proposition weaker than assumed?
- Path 2 may need reshaping (e.g., narrower workload scope, different model
  tier split, or acknowledging that routing savings are modest).

### Warm-start vs cold-start

With fresh Beta(1,1) priors, the routing's EU calculation reduces to:

```
EU(model) = reward × 0.5 − cost(model)
```

Since all models have identical expected reliability (0.5), the cheapest model
always wins. This is the cold-start problem: uninformed routing always picks
the cheapest model, which maximises savings but may sacrifice quality.

The feasibility check handles this by running workloads in sequence: beliefs
update after each turn's quality-judge signal. By workload 3, the routing has
~15 quality observations and beliefs are meaningfully informed. The per-workload
cost breakdown shows the learning curve — early workloads may show excessive
Haiku selection (cold-start), later workloads show calibrated routing.

The full benchmark (Move 2) should report both cold-start and warm-state
results separately, since users in production will also experience a cold-start
phase.

## 5. Design decisions

### 5.1 — Workload selection

**The question.** What constitutes a "representative coding workload" for the
full benchmark?

**Sources, in order of preference:**

1. **SWE-bench trajectory datasets.** Multiple public datasets on HuggingFace
   contain full multi-turn agentic coding traces:
   - `nebius/SWE-agent-trajectories` — 80K trajectories from SWE-agent.
   - `SWE-bench/SWE-smith-trajectories` — 5K trajectories.
   - `nebius/SWE-rebench-openhands-trajectories` — OpenHands agent traces.

   These contain realistic agent-tool interaction sequences. **Adaptation
   required**: the trajectories use SWE-agent/OpenHands message formats, not
   standard OpenAI chat-completion format. Adaptation cost is the limiting
   factor. The methodological payoff of adaptation is quality-signal
   robustness: SWE-bench instances have ground-truth test suites, so quality
   measurement uses pass/fail execution rather than LLM judging.

2. **Hand-curated workloads.** Constructed from common agentic-coding patterns
   (debug, implement, refactor, test, explain). Cheaper to produce, fully
   controlled, but less representative than real agentic traces.

3. **Personal-use transcripts.** Captured from brief personal use of the proxy
   against real coding tasks. Most representative but smallest sample.

**Proposal.** The benchmark uses 20 workloads:

- **10 hand-curated** (the feasibility check's 5 plus 5 additional, covering
  diverse task types and complexity levels).
- **10 adapted from SWE-bench trajectories** (selected for task-type diversity
  and moderate turn count — 5–15 turns, filtering out trivially short or
  excessively long traces).

If SWE-bench adaptation proves heavier than expected (halting trigger: more
than 4 hours of adaptation work for 10 trajectories), fall back to 20
hand-curated workloads. The report names which workloads came from which
source.

**Task-type distribution target:** debug 25%, implement-feature 25%, refactor
20%, test-writing 15%, explain/chat 15%. The distribution is a target, not a
rigid requirement; available SWE-bench trajectories may not match this
distribution exactly.

### 5.2 — Quality measurement

**The question.** How does the benchmark measure whether routing degrades
quality?

**Ground-truth tasks** (preferred): workloads where the outcome is verifiable
by execution. "Does the generated code compile?", "Do the generated tests
pass?", "Does the patch resolve the SWE-bench issue?". Ground truth is
binary (pass/fail) and sidesteps judging entirely.

For hand-curated workloads: the workload specification includes an expected
outcome (a function signature, a test that should pass, an error message that
should disappear). The quality signal is: does the routing phase's output
meet the same outcome specification the baseline's output meets?

For SWE-bench-adapted workloads: ground truth is the SWE-bench test suite —
does the generated patch pass the instance's test harness?

**Judged tasks** (fallback): workloads without verifiable outcomes (explain,
chat-type turns within otherwise-coding workloads). Use Claude Opus 4.6 as
judge with a structured scoring rubric (correctness 0–10, completeness 0–10,
relevance 0–10). Composite score: unweighted mean.

**Same-tier-judge concern.** The existing eval uses Haiku judging Haiku outputs
— a circular quality signal. This benchmark uses:
- Ground truth (binary, no judge) for all coding outcomes.
- Opus judging (higher-tier) for non-coding outputs.
- No model judges its own outputs.

**Quality-degradation threshold.** A workload is quality-degraded if:
- Ground-truth task: baseline passes but routing fails.
- Judged task: routing's composite score < baseline's composite score by >1.0
  point (on 0–10 scale).

The benchmark reports both per-workload quality comparisons and the aggregate
quality-degradation rate.

### 5.3 — Baseline selection

**The question.** What is cost-savings measured against?

**Primary baseline: always-Sonnet.** Every turn uses `claude-sonnet-4-6` via
`CREDENCE_FORCE_MODEL`. This is the most defensible baseline because:
- Most coding harnesses (Cursor, Claude Code, Windsurf) default to a
  Sonnet-tier model for coding tasks.
- Sonnet is the proxy's natural comparison point — it's the model users would
  use if they weren't routing.
- Savings against always-Sonnet are meaningful and not cherry-picked.

**Secondary baseline: always-Opus.** Reported alongside but not the primary
claim. Some users run Opus for high-stakes tasks; the always-Opus comparison
answers "what if you'd been paying Opus prices?" without making it the
headline number.

**Not included:** always-Haiku baseline (the floor — it's what the routing
would converge to if quality didn't matter, which is uninteresting) or
user's-actual-behaviour (requires users in hand, post-MVP).

### 5.4 — Cost measurement

**The question.** How is cost computed, and at what prices?

**Pricing source.** Official Anthropic and OpenAI pricing pages, snapshot at
benchmark execution date. The report names the snapshot date and the exact
per-million-token rates used.

**Note: proxy cost table accuracy.** The proxy's hardcoded `ModelSpec` prices
in `provider.py:62-92` are used for EU routing decisions, not for benchmark
cost accounting. However, if the proxy's internal prices diverge from actual
pricing, routing decisions are suboptimal (the EU tradeoff is miscalibrated).
Any historical cost-savings claims produced using the proxy's current pricing
should be treated as preliminary until the corrected-pricing benchmark runs.
Before benchmark execution, the proxy's cost table must be verified against
current pricing and corrected if needed. Current discrepancies observed:

| Model | Proxy (per 1M input/output) | Actual (per 1M input/output) | Status |
|-------|-----------------------------|------------------------------|--------|
| Haiku 4.5 | $0.80 / $4.00 | $1.00 / $5.00 | Uses Haiku 3.5 prices; 20% under |
| Sonnet 4.6 | $3.00 / $15.00 | $3.00 / $15.00 | Correct |
| Opus 4.6 | $15.00 / $75.00 | $5.00 / $25.00 | Uses Opus 4.0 prices; **3× over** |

Source: [Anthropic pricing page](https://platform.claude.com/docs/en/docs/about-claude/pricing)
(snapshot 2026-04-27). Opus 4.5/4.6/4.7 are priced at $5/$25; the proxy's
$15/$75 matches the deprecated Opus 4.0/4.1 pricing. Haiku 4.5 is $1/$5;
the proxy's $0.80/$4.00 matches the deprecated Haiku 3.5 pricing.

**Routing impact.** The Opus 3× overpricing means the EU calculation massively
overestimates Opus's cost, so the routing almost never selects Opus even when
its quality would justify the actual (much lower) price. The Haiku
underpricing mildly favours Haiku over what it should. These discrepancies
affect routing decisions and therefore benchmark results. Correcting the cost
table is a **pre-benchmark prerequisite**, not a nice-to-have. Post-Move-2
hardening work should decouple pricing from source code (e.g., a config file
or API-fetched rates) so the cost table cannot silently drift again.

**Per-turn cost formula:**

```
cost = (input_tokens × input_rate + output_tokens × output_rate) / 1_000_000
```

Where `input_rate` and `output_rate` are the selected model's per-million-token
prices from the pricing snapshot.

**Aggregate cost metrics:**
- Per-workload total cost (sum of per-turn costs).
- Per-workload savings percentage vs baseline: `(baseline_cost - routing_cost) / baseline_cost × 100`.
- Aggregate savings: mean of per-workload savings percentages.
- Token-volume-weighted aggregate: per-workload savings weighted by that
  workload's total token volume (so large workloads contribute proportionally
  more to the aggregate).
- Both aggregates reported; the token-volume-weighted figure is the headline
  number (it represents actual dollar savings better than unweighted mean).

**Cache savings: explicitly not measured.** Per Move 0's verdict, no prompt
caching occurs on the OAI-compatible endpoint. Any reader asking about cache
savings gets the v0.1 answer: out of scope; native API migration deferred to
v0.2; see `docs/posture-5/move-0-audit-report.md`.

### 5.5 — Statistical robustness

**The question.** How does the methodology handle variance from stochastic LLM
outputs?

**Repetitions.** N=3 per workload per mode (baseline and routing). Each
repetition uses a fresh routing state (for the routing phase) to avoid
belief-state contamination across repetitions.

N=3 is the minimum for computing a standard deviation; N=5 would be
preferable if API budget permits. The design doc commits to N=3 as the floor;
Move 2 may run N=5 if the feasibility check's cost data shows the budget is
comfortable.

**Reported statistics per workload:**
- Cost: mean, std, min, max across N repetitions.
- Savings %: mean, std.
- Quality: pass rate across N repetitions (for ground-truth tasks); mean
  judge score ± std (for judged tasks).

**Aggregate statistics:**
- Mean savings % ± std across workloads.
- Token-volume-weighted savings % ± propagated std.
- Quality-degradation rate: fraction of workload-repetition pairs where
  routing degraded quality vs baseline.

**Variance sources.** Two distinct sources:
1. *Output variance*: the same model produces slightly different outputs per
   run, leading to different token counts and different task outcomes.
2. *Routing variance*: with fresh priors per repetition, the routing's
   cold-start learning curve varies. This is intentional — it captures the
   real variance a new user would experience.

The benchmark reports both sources' contributions where distinguishable
(baseline repetitions isolate output variance; routing repetitions combine
both).

### 5.6 — Deliverable shape

**The question.** What does the benchmark output look like?

**Report location:** `docs/posture-5/move-2-benchmark-results.md`.

**Structure:**

1. **Executive summary** — one paragraph: savings claim, quality finding,
   honest scope statement.

2. **Methodology** — compact recapitulation of this design doc's decisions:
   workload sources, quality measurement, baselines, pricing snapshot, N.

3. **Per-workload results table:**

   | Workload | Type | Turns | Sonnet cost | Routing cost | Savings % | Quality |
   |----------|------|-------|-------------|--------------|-----------|---------|
   | debug-1 | debug | 6 | $0.042 ± 0.003 | $0.028 ± 0.004 | 33% ± 5% | 3/3 pass |

4. **Aggregate results:**
   - Mean savings, token-volume-weighted savings.
   - Quality-degradation rate.
   - Always-Opus secondary comparison.

5. **Routing distribution** — per workload: how often each model was selected.
   Across all workloads: aggregate model-selection distribution. Learning-curve
   visualization: routing decisions by turn position (early turns vs late
   turns).

6. **Honest limitations:**
   - Cache savings not measured (Move 0 finding).
   - Workloads are hand-curated / SWE-bench-adapted, not production transcripts.
   - N=3 (or 5) repetitions; statistical power is limited.
   - Cold-start learning curve affects early workloads.
   - Routing is based on keyword-category inference, not semantic understanding.

7. **Move 1 input absorbed** — names the Move 0 audit findings that shaped
   this methodology (no cache measurement, OAI-compatible endpoint only).

The deliverable doubles as draft material for the eventual blog post or
technical write-up. The executive summary and per-workload table are designed
to be extractable as publication-ready content.

## 6. Risks

**Risk 1: Workload availability.** If SWE-bench trajectory adaptation is
heavier than expected, the benchmark falls back to 20 hand-curated workloads.
Hand-curated workloads are less representative than real agentic traces.
Mitigation: the report names the workload source per workload; readers can
assess representativeness.

**Risk 2: Feasibility check fails.** If aggregate savings < 15% on
representative coding workloads, path 2's value proposition is weaker than
assumed. Mitigation: the feasibility check runs before the full benchmark;
failure pauses the methodology and triggers a value-proposition conversation
rather than producing misleading benchmark results.

**Risk 3: Judging cost for non-ground-truth tasks.** Opus judging at N=3
repetitions per workload is expensive if many workloads lack ground truth.
Mitigation: prioritise ground-truth-bearing workloads (coding outcomes);
limit judged workloads to ≤5 of the 20; report the asymmetry honestly.

**Risk 4: Pricing snapshot drift.** Anthropic's pricing changes may invalidate
cost numbers between benchmark execution and publication. Mitigation: the
report names the snapshot date; cost calculations are parameterised (re-running
with updated prices is mechanical, not methodological work).

**Risk 5: Routing logic pathology.** Representative workloads may surface
routing bugs not visible in the synthetic eval (always picks one model
regardless of category, oscillates between turns, never learns from quality
signals). Mitigation: the per-workload routing-distribution table makes
pathologies visible. If routing behaviour is pathological, the benchmark halts
— the methodology is sound but the substrate isn't ready for benchmarking.

**Risk 6: Cold-start dominance.** If most of the 20 workloads run during the
cold-start phase (before beliefs stabilise), the benchmark measures cold-start
behaviour rather than steady-state routing. Mitigation: run workloads in
sequence with accumulating beliefs; report per-workload savings to show the
learning curve; distinguish cold-start from warm-state results.

**Risk 7: Proxy cost-table miscalibration (confirmed).** The proxy's hardcoded
model prices (`provider.py:62-92`) are already known to be wrong: Opus 4.6 is
priced at 3× actual ($15/$75 vs $5/$25, using deprecated Opus 4.0 rates) and
Haiku 4.5 is priced 20% under actual ($0.80/$4.00 vs $1.00/$5.00, using Haiku
3.5 rates). The Opus overpricing means the routing almost never selects Opus
even when it would be EU-optimal at the real price; this suppresses a routing
option the benchmark should evaluate. Mitigation: correct the cost table in a
pre-benchmark PR. The benchmark must run against corrected prices; running
against miscalibrated prices would measure the performance of a miscalibrated
router, not the routing methodology.

## 7. Test plan (execution shape for Move 2)

1. **Verify proxy cost table** against official pricing page. Verification:
   compare each model's `input_price_per_1k` and `output_price_per_1k` in
   `provider.py` against the snapshotted pricing page; any discrepancy beyond
   ±5% (to allow for minor rounding in per-1K vs per-1M conversion) is
   corrected in a pre-benchmark PR before proceeding.

2. **Run feasibility check** per §Pre-design:
   - 5 hand-curated multi-turn workloads.
   - Baseline phase (forced Sonnet) + routing phase (fresh priors, sequential).
   - Report cost-savings table and quality.
   - Gate: ≥15% savings, no quality degradation → proceed. Otherwise halt.

3. **Prepare full workload set** per §5.1:
   - 10 hand-curated + 10 SWE-bench-adapted (or 20 hand-curated if adaptation
     is too heavy).
   - Verify task-type distribution approximates target.

4. **Run full benchmark** per §5.2–§5.5:
   - N=3 (or 5) repetitions.
   - Baseline (forced Sonnet) + routing (fresh priors per rep, sequential
     across workloads within each rep).
   - Collect per-turn metrics: model selected, token counts, cost, quality.

5. **Compute secondary baseline** (always-Opus): re-price the baseline
   workloads at Opus rates. No additional API calls needed.

6. **Write report** per §5.6 at `docs/posture-5/move-2-benchmark-results.md`.

## Closure

After the benchmark report lands:
- Move 3 (release engineering) opens, using the benchmark results as the
  v0.1 claim's evidential basis.
- The report's honest-limitations section becomes the "future work" input for
  post-v0.1 (cache savings via native API migration, production-scale
  evaluation, broader workload coverage).
- Posture 5's master plan is provisional; refinement after Move 2's findings
  is appropriate.
