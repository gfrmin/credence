# Move 2 — Feasibility check

Gate for the full benchmark (Move 2 sessions 2–3). Validates that Bayesian
model-tier routing produces meaningful cost savings on representative coding
workloads. N=1 per workload per phase; statistical robustness deferred to the
full benchmark. Pass threshold: ≥15% token-volume-weighted savings with zero
quality degradation.

## Per-workload results

| Workload | Type | Turns | Baseline cost | Routing cost | Savings % | Baseline outcome | Routing outcome |
|----------|------|-------|---------------|--------------|-----------|------------------|-----------------|
| debug-binary-search | debug | 5 | $0.0446 | $0.0085 | 81.0% | pass | pass |
| implement-pagination | implement | 5 | $0.0931 | $0.0190 | 79.6% | pass | pass |
| refactor-error-handling | refactor | 5 | $0.0928 | $0.0177 | 80.9% | pass | pass |
| write-tests-lru-cache | test | 5 | $0.0666 | $0.0372 | 44.1% | pass | pass |
| explain-modify-fibonacci | explain+modify | 5 | $0.0630 | $0.0186 | 70.4% | pass | pass |

## Routing distribution

Model selection per workload (workload 1 = cold-start, workload 5 = warmest):

**debug-binary-search** (debug): claude-haiku-4-5: 5
  Turn sequence: haiku → haiku → haiku → haiku → haiku

**implement-pagination** (implement): claude-haiku-4-5: 5
  Turn sequence: haiku → haiku → haiku → haiku → haiku

**refactor-error-handling** (refactor): claude-haiku-4-5: 5
  Turn sequence: haiku → haiku → haiku → haiku → haiku

**write-tests-lru-cache** (test): claude-haiku-4-5: 5
  Turn sequence: haiku → haiku → haiku → haiku → haiku

**explain-modify-fibonacci** (explain+modify): claude-haiku-4-5: 5
  Turn sequence: haiku → haiku → haiku → haiku → haiku

### Learned reliability (end of routing phase)

| Model | code | reasoning | creative | factual | chat |
|-------|------|-----------|----------|---------|------|
| claude-haiku-4-5 | 0.848 | 0.500 | 0.500 | 0.500 | 0.837 |
| claude-sonnet-4-6 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| claude-opus-4-6 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |

## Aggregate

- **Token-volume-weighted savings: 68.9%**
- Unweighted mean savings: 71.2%
- Quality-degradation count: 0/5

## Gate decision

Threshold: ≥15% weighted savings, 0 quality degradations.

**PASS.** Weighted savings 68.9% ≥ 15%, quality degradation count 0 = 0.

The full benchmark (Move 2 session 3) proceeds against the methodology
specified in `docs/posture-5/move-1-design.md`.

## Findings

1. **Cold-start routing sends all traffic to cheapest model.** With fresh
   Beta(1,1) priors, E[theta]=0.5 for all models. EU-maximisation then
   selects on cost alone, which means Haiku wins every turn. This is
   mathematically correct behaviour — not a bug — but it means this
   feasibility check measures "Haiku vs Sonnet cost ratio", not "smart
   routing intelligence".

2. **Quality judge fires and updates beliefs.** The learned reliability
   table above shows Haiku's code and chat posteriors moved from 0.500
   (prior) to ~0.84, confirming the async judge pipeline works end-to-end.
   However, at current pricing (Haiku $0.001/$0.005 vs Sonnet $0.003/$0.015),
   Haiku's cost advantage is large enough that even moderate reliability
   differences don't shift the EU-optimal choice. A warm-start scenario
   (where some categories have low Haiku reliability) is needed to test
   actual model-switching behaviour — deferred to the full benchmark.

3. **All quality outcomes pass.** Haiku produces code that matches the
   expected regex patterns for all five workloads. The patterns test for
   structural correctness (e.g. `lo = mid + 1` for the binary search fix,
   `functools.wraps` for the decorator refactor), not deep quality. The
   full benchmark should use the judge's 0-10 scale, not binary regex.

4. **Proxy metrics collection bug.** The server's `_request_log.append()`
   runs in a `generate()` async generator's post-yield code. When clients
   disconnect after `data: [DONE]` (standard SSE behaviour), the generator
   is abandoned before the metrics code executes. Worked around in this
   check by consuming the full stream; the server-side bug remains.

## Limitations

- N=1 per workload per phase — no statistical significance.
- Only Anthropic models tested (OpenAI excluded due to `max_tokens` →
  `max_completion_tokens` incompatibility).
- Quality checked by regex pattern matching, not human evaluation.
- Cold-start only — no warm-start or adversarial routing scenarios.
