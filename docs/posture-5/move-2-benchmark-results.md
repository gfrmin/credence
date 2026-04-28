# Move 2 — Benchmark results

## Executive summary

Bayesian model-tier routing achieved **67.4%** token-volume-weighted cost savings vs always-Sonnet across 20 coding workloads (N=3 repetitions). Quality degradation rate: 0/60 workload-repetition pairs (regex), 8/20 workloads (Opus judge, >1pt drop). Scope: model-tier routing on Anthropic models only; cache savings explicitly not measured (Move 0 finding: prompt caching unsupported on OAI-compatible endpoint).

## Methodology

- **Workloads:** 20 hand-curated multi-turn coding workloads
- **Repetitions:** N=3 per phase, fresh routing state per repetition
- **Baseline:** always-Sonnet (`CREDENCE_FORCE_MODEL=claude-sonnet-4-6`)
- **Secondary baseline:** always-Opus (repriced from Sonnet baseline token counts)
- **Quality measurement:** regex pattern matching (binary) + Claude Opus 4.6 judge (0-10 scale)
- **Pricing snapshot:** 2026-04-28 (Haiku $1/$5, Sonnet $3/$15, Opus $5/$25 per 1M tokens)
- **Regime tags:** cold-start (workloads 1-5), transition (6-10), warm-state (11-20)

## Per-workload results

| # | Workload | Type | Turns | Regime | Sonnet cost | Routing cost | Savings % | Quality (regex) | Opus score |
|---|----------|------|-------|--------|-------------|--------------|-----------|-----------------|------------|
| 1 | debug-binary-search | debug | 5 | cold-start | $0.0403 +/- 0.0031 | $0.0092 +/- 0.0015 | 76.9% +/- 5.3% | 100%/100% | 6.3/7.1 |
| 2 | implement-pagination | implement | 5 | cold-start | $0.0743 +/- 0.0062 | $0.0199 +/- 0.0006 | 73.1% +/- 1.7% | 100%/100% | 3.8/3.9 |
| 3 | refactor-error-handling | refactor | 5 | cold-start | $0.0765 +/- 0.0142 | $0.0183 +/- 0.0006 | 75.5% +/- 5.2% | 100%/100% | 4.0/3.2 |
| 4 | write-tests-lru-cache | test | 5 | cold-start | $0.1025 +/- 0.0044 | $0.0372 +/- 0.0000 | 63.7% +/- 1.6% | 100%/100% **DEGRADED** | 4.4/2.3 |
| 5 | explain-modify-fibonacci | explain+modify | 5 | cold-start | $0.0687 +/- 0.0024 | $0.0172 +/- 0.0015 | 74.9% +/- 2.9% | 100%/100% | 6.8/7.0 |
| 6 | code-review-race-condition | code-review | 6 | transition | $0.1382 +/- 0.0015 | $0.0428 +/- 0.0020 | 69.0% +/- 1.8% | 100%/100% **DEGRADED** | 5.1/4.0 |
| 7 | generate-api-docs | documentation | 5 | transition | $0.1030 +/- 0.0009 | $0.0323 +/- 0.0004 | 68.6% +/- 0.4% | 100%/100% | 5.3/4.9 |
| 8 | migrate-sync-to-async | migration | 6 | transition | $0.1182 +/- 0.0077 | $0.0488 +/- 0.0000 | 58.6% +/- 2.7% | 100%/100% | 5.4/5.0 |
| 9 | diagnose-from-logs | bug-triage | 5 | transition | $0.1119 +/- 0.0005 | $0.0347 +/- 0.0011 | 69.0% +/- 1.1% | 100%/100% | 6.1/5.7 |
| 10 | design-rest-api | api-design | 6 | transition | $0.1417 +/- 0.0000 | $0.0472 +/- 0.0000 | 66.7% +/- 0.0% | 100%/100% **DEGRADED** | 6.4/5.1 |
| 11 | optimize-database-queries | performance | 5 | warm-state | $0.1126 +/- 0.0000 | $0.0375 +/- 0.0000 | 66.7% +/- 0.0% | 100%/100% | 5.4/5.4 |
| 12 | security-audit-jwt | security-audit | 5 | warm-state | $0.1148 +/- 0.0000 | $0.0382 +/- 0.0000 | 66.7% +/- 0.0% | 100%/100% **DEGRADED** | 6.3/5.0 |
| 13 | implement-data-pipeline | implement | 5 | warm-state | $0.1109 +/- 0.0000 | $0.0370 +/- 0.0000 | 66.7% +/- 0.0% | 100%/100% | 4.3/4.0 |
| 14 | build-cli-tool | implement | 5 | warm-state | $0.1114 +/- 0.0000 | $0.0371 +/- 0.0000 | 66.7% +/- 0.0% | 100%/100% **DEGRADED** | 4.3/2.8 |
| 15 | fix-memory-leak | debug | 5 | warm-state | $0.1148 +/- 0.0000 | $0.0382 +/- 0.0000 | 66.7% +/- 0.0% | 100%/100% | 5.3/4.3 |
| 16 | implement-state-machine | implement | 5 | warm-state | $0.1118 +/- 0.0000 | $0.0373 +/- 0.0000 | 66.7% +/- 0.0% | 100%/100% | 3.2/2.3 |
| 17 | design-websocket-protocol | system-design | 5 | warm-state | $0.1099 +/- 0.0000 | $0.0366 +/- 0.0000 | 66.7% +/- 0.0% | 100%/100% **DEGRADED** | 5.0/3.7 |
| 18 | parse-and-transform-ast | implement | 5 | warm-state | $0.1112 +/- 0.0000 | $0.0371 +/- 0.0000 | 66.7% +/- 0.0% | 100%/100% **DEGRADED** | 3.9/2.7 |
| 19 | debug-distributed-system | debug | 5 | warm-state | $0.1146 +/- 0.0000 | $0.0382 +/- 0.0000 | 66.7% +/- 0.0% | 100%/100% **DEGRADED** | 6.2/4.3 |
| 20 | implement-caching-layer | implement | 5 | warm-state | $0.1109 +/- 0.0000 | $0.0370 +/- 0.0000 | 66.7% +/- 0.0% | 100%/100% | 4.0/4.0 |

## Regime-tagged results

| Regime | Workloads | Mean savings % | Weighted savings % | Quality degradations |
|--------|-----------|----------------|--------------------|----------------------|
| cold-start | 5 | 72.8% +/- 5.3% | 71.7% | 0 (regex) / 1 (opus) |
| transition | 5 | 66.4% +/- 4.5% | 66.4% | 0 (regex) / 2 (opus) |
| warm-state | 10 | 66.7% +/- 0.0% | 66.7% | 0 (regex) / 5 (opus) |

## Task-type breakdown

| Task type | N | Mean savings % |
|-----------|---|----------------|
| api-design | 1 | 66.7% +/- 0.0% |
| bug-triage | 1 | 69.0% +/- 0.0% |
| code-review | 1 | 69.0% +/- 0.0% |
| debug | 3 | 70.1% +/- 5.9% |
| documentation | 1 | 68.6% +/- 0.0% |
| explain+modify | 1 | 74.9% +/- 0.0% |
| implement | 6 | 67.7% +/- 2.6% |
| migration | 1 | 58.6% +/- 0.0% |
| performance | 1 | 66.7% +/- 0.0% |
| refactor | 1 | 75.5% +/- 0.0% |
| security-audit | 1 | 66.7% +/- 0.0% |
| system-design | 1 | 66.7% +/- 0.0% |
| test | 1 | 63.7% +/- 0.0% |

## Variance decomposition

Baseline repetitions isolate output variance (model stochasticity). Routing repetitions combine output variance with routing variance (cold-start learning, model-selection stochasticity).

- Mean output variance (from baseline): 0.00001690
- Mean total variance (from routing): 0.00000054
- Routing variance contribution: 0.00000000
- Routing fraction of total variance: 0.0%

## Routing distribution

**debug-binary-search** (cold-start): claude-haiku-4-5: 15/15 (100%)
**implement-pagination** (cold-start): claude-haiku-4-5: 15/15 (100%)
**refactor-error-handling** (cold-start): claude-haiku-4-5: 15/15 (100%)
**write-tests-lru-cache** (cold-start): claude-haiku-4-5: 15/15 (100%)
**explain-modify-fibonacci** (cold-start): claude-haiku-4-5: 15/15 (100%)
**code-review-race-condition** (transition): claude-haiku-4-5: 18/18 (100%)
**generate-api-docs** (transition): claude-haiku-4-5: 15/15 (100%)
**migrate-sync-to-async** (transition): claude-haiku-4-5: 18/18 (100%)
**diagnose-from-logs** (transition): claude-haiku-4-5: 15/15 (100%)
**design-rest-api** (transition): claude-haiku-4-5: 18/18 (100%)
**optimize-database-queries** (warm-state): claude-haiku-4-5: 15/15 (100%)
**security-audit-jwt** (warm-state): claude-haiku-4-5: 15/15 (100%)
**implement-data-pipeline** (warm-state): claude-haiku-4-5: 15/15 (100%)
**build-cli-tool** (warm-state): claude-haiku-4-5: 15/15 (100%)
**fix-memory-leak** (warm-state): claude-haiku-4-5: 15/15 (100%)
**implement-state-machine** (warm-state): claude-haiku-4-5: 15/15 (100%)
**design-websocket-protocol** (warm-state): claude-haiku-4-5: 15/15 (100%)
**parse-and-transform-ast** (warm-state): claude-haiku-4-5: 15/15 (100%)
**debug-distributed-system** (warm-state): claude-haiku-4-5: 15/15 (100%)
**implement-caching-layer** (warm-state): claude-haiku-4-5: 15/15 (100%)

## Belief state snapshots

### Repetition 1

After workload 5:
  claude-haiku-4-5: chat=0.794 code=0.848 creative=0.500 factual=0.500 reasoning=0.500
  claude-sonnet-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500
  claude-opus-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500

After workload 10:
  claude-haiku-4-5: chat=0.786 code=0.783 creative=0.764 factual=0.500 reasoning=0.780
  claude-sonnet-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500
  claude-opus-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500

After workload 15:
  claude-haiku-4-5: chat=0.757 code=0.725 creative=0.778 factual=0.500 reasoning=0.827
  claude-sonnet-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500
  claude-opus-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500

After workload 20:
  claude-haiku-4-5: chat=0.769 code=0.737 creative=0.809 factual=0.500 reasoning=0.827
  claude-sonnet-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500
  claude-opus-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500

### Repetition 2

After workload 5:
  claude-haiku-4-5: chat=0.749 code=0.924 creative=0.500 factual=0.500 reasoning=0.500
  claude-sonnet-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500
  claude-opus-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500

After workload 10:
  claude-haiku-4-5: chat=0.738 code=0.954 creative=0.590 factual=0.500 reasoning=0.740
  claude-sonnet-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500
  claude-opus-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500

After workload 15:
  claude-haiku-4-5: chat=0.721 code=0.961 creative=0.611 factual=0.500 reasoning=0.765
  claude-sonnet-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500
  claude-opus-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500

After workload 20:
  claude-haiku-4-5: chat=0.715 code=0.967 creative=0.644 factual=0.500 reasoning=0.765
  claude-sonnet-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500
  claude-opus-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500

### Repetition 3

After workload 5:
  claude-haiku-4-5: chat=0.761 code=0.774 creative=0.500 factual=0.500 reasoning=0.500
  claude-sonnet-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500
  claude-opus-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500

After workload 10:
  claude-haiku-4-5: chat=0.763 code=0.786 creative=0.736 factual=0.500 reasoning=0.726
  claude-sonnet-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500
  claude-opus-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500

After workload 15:
  claude-haiku-4-5: chat=0.754 code=0.817 creative=0.736 factual=0.500 reasoning=0.763
  claude-sonnet-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500
  claude-opus-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500

After workload 20:
  claude-haiku-4-5: chat=0.767 code=0.845 creative=0.732 factual=0.500 reasoning=0.763
  claude-sonnet-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500
  claude-opus-4-6: chat=0.500 code=0.500 creative=0.500 factual=0.500 reasoning=0.500

## Secondary baseline: always-Opus

| Workload | Opus cost | Routing cost | Savings vs Opus |
|----------|-----------|--------------|-----------------|
| debug-binary-search | $0.0672 | $0.0092 | 86.3% |
| implement-pagination | $0.1238 | $0.0199 | 83.9% |
| refactor-error-handling | $0.1275 | $0.0183 | 85.7% |
| write-tests-lru-cache | $0.1708 | $0.0372 | 78.2% |
| explain-modify-fibonacci | $0.1145 | $0.0172 | 85.0% |
| code-review-race-condition | $0.2304 | $0.0428 | 81.4% |
| generate-api-docs | $0.1717 | $0.0323 | 81.2% |
| migrate-sync-to-async | $0.1971 | $0.0488 | 75.2% |
| diagnose-from-logs | $0.1864 | $0.0347 | 81.4% |
| design-rest-api | $0.2362 | $0.0472 | 80.0% |
| optimize-database-queries | $0.1876 | $0.0375 | 80.0% |
| security-audit-jwt | $0.1913 | $0.0382 | 80.0% |
| implement-data-pipeline | $0.1849 | $0.0370 | 80.0% |
| build-cli-tool | $0.1856 | $0.0371 | 80.0% |
| fix-memory-leak | $0.1913 | $0.0382 | 80.0% |
| implement-state-machine | $0.1864 | $0.0373 | 80.0% |
| design-websocket-protocol | $0.1831 | $0.0366 | 80.0% |
| parse-and-transform-ast | $0.1854 | $0.0371 | 80.0% |
| debug-distributed-system | $0.1909 | $0.0382 | 80.0% |
| implement-caching-layer | $0.1848 | $0.0370 | 80.0% |

## Aggregate

- **Token-volume-weighted savings: 67.4%**
- Unweighted mean savings: 68.1% +/- 4.2%
- Quality degradations (regex): 0/60
- Quality degradations (Opus judge >1pt): 8/20

## Findings

1. **Routing selects Haiku 100% of the time.** Across all 3 routing repetitions
   (309 turns), the router never selected Sonnet or Opus. With fresh Beta(1,1)
   priors, all models start at E[theta]=0.5. The EU calculation then selects on
   cost alone: Haiku ($1/$5) vs Sonnet ($3/$15) vs Opus ($5/$25). The Haiku
   quality judge scores (0.7-0.96 across categories) never drop low enough to
   shift EU toward costlier models. This is the dominant finding: at current
   Anthropic pricing ratios, the Bayesian router converges to always-Haiku
   because Haiku's quality-per-dollar is too high for the EU tradeoff to favour
   Sonnet.

2. **Opus judge detects quality degradation that regex misses.** Regex pattern
   matching shows 0/60 degradations — Haiku's responses contain the right
   keywords. But the Opus judge (0-10 scoring on correctness, completeness,
   helpfulness) flags 8/20 workloads with >1.0pt drops. The affected workloads
   skew toward complex tasks: code-review, security-audit, system-design,
   AST-parsing. Haiku's responses are structurally correct but less thorough.

3. **Variance decomposition is degenerate.** Because routing is deterministic
   (always Haiku), routing variance is zero. Baseline (Sonnet) has higher
   variance than routing (Haiku) because Sonnet produces longer, more variable
   responses. The decomposition framework from Move 1 would become meaningful
   if the cost ratio were tighter or if quality thresholds were configurable.

4. **Belief learning works but doesn't trigger model switching.** The belief
   snapshots show Haiku's category posteriors moving from 0.5 to 0.7-0.96
   across 20 workloads, confirming the judge pipeline updates beliefs
   correctly. Sonnet/Opus stay at 0.5 (never tried). The gap between
   Haiku's learned reliability and its cost advantage over Sonnet is too
   large for the EU calculation to explore Sonnet — Haiku would need to drop
   below ~0.33 reliability for Sonnet to become EU-optimal at 3x the price.

5. **The max_tokens=1024 cap produces ceiling effects.** Many warm-state
   workloads show zero cost variance across repetitions because every turn
   hits the 1024-token output cap. This is an artefact of the benchmark
   configuration, not of routing. Production use would set higher limits.

## Honest limitations

- Cache savings not measured (Move 0 finding: prompt caching unsupported on OAI-compatible endpoint).
- All 20 workloads are hand-curated, not production transcripts or SWE-bench traces.
- N=3 repetitions; statistical power is limited.
- Cold-start learning curve affects workloads 1-5 in routing phase.
- Only Anthropic models tested (OpenAI excluded due to `max_tokens` / `max_completion_tokens` incompatibility).
- Routing is based on keyword-category inference, not semantic understanding.
- Opus judge truncates conversations to ~6000 chars; long conversations lose context.

## Move 1 input absorbed

- Move 0 audit: no cache measurement (OAI-compatible endpoint only).
- Move 1 methodology: N=3, regime-tagged, variance decomposition, Opus judging.
- Proxy cost table verified against official pricing (snapshot 2026-04-28).
