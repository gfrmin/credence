# Dominance benchmark — results

Task: `[:colour_typed, :motion_typed, :territorial]`, regime changes at `[70, 140]`, 210 steps, respawn on, 20 seeds, paired-seed percentile bootstrap (10 000 resamples).

Primary realised-value measure (§8, ratified 2026-07-03): the mean per-step energy rate `ce[end]/n` — the uniform-weight statistic the agent's declared utility maximises. AUC of the cumulative trajectory (front-loaded) is reported-only, for cross-round comparability.

| policy | mean rate | mean final-window rate | mean AUC (reported) | mean steps-to-half | mean meta-actions |
|---|---|---|---|---|---|
| clairvoyant | 0.429 | 0.875 | 47.59 | 51.8 | 33.0 |
| eu_max | 0.424 | 0.881 | 47.23 | 52.1 | 27.0 |
| fixed_k10 | 0.393 | 0.72 | 48.22 | 67.6 | 21.0 |
| fixed_k25 | 0.401 | 0.488 | 50.51 | 52.8 | 8.0 |
| fixed_k5 | 0.388 | 0.667 | 48.49 | 57.6 | 42.0 |
| fixed_k50 | 0.45 | 0.869 | 49.79 | 76.8 | 4.0 |
| never_explore | 0.274 | 0.125 | 47.11 | 28.9 | 2.0 |
| random_p005 | 0.489 | 1.006 | 51.08 | 86.6 | 12.3 |
| random_p015 | 0.445 | 0.875 | 49.71 | 85.9 | 37.7 |
| random_p04 | 0.373 | 0.75 | 46.65 | 51.6 | 131.0 |

## Paired gaps (eu_max − baseline; efficiency = steps to the shared per-seed
level, sign-flipped so + favours eu_max — belief-derived-valuation §2c)

| baseline | rate gap [95% CI] | final-window gap [95% CI] | efficiency gap [95% CI] | worst-seed rate gap | q10 rate gap |
|---|---|---|---|---|---|
| random_p005 | -0.065 [-0.193, 0.061] | -0.125 [-0.625, 0.363] | -3.6 [-13.2, 5.6] | -0.5 | -0.476 |
| fixed_k50 | -0.026 [-0.117, 0.064] | 0.012 [-0.339, 0.375] | -7.0 [-16.4, 3.4] | -0.381 | -0.333 |
| never_explore | 0.15 [0.062, 0.235] | 0.756 [0.363, 1.125] | 11.4 [2.4, 21.0] | -0.381 | 0.024 |

`eu_max − never_explore` is the headline: the learned-returns escape ops are identical on both sides, so this gap is exploration's isolated value.

## Alternative dominance measures (reported, not asserted)

Complementary dominance notions on the same per-seed paired gaps: per-seed win
rate with an exact two-sided sign test (distribution-free — robust to the
winner's-curse outlier seeds the mean bootstrap is sensitive to), the median
gap, the 10th-percentile gap (tail regret softer than minimax), and the
final-regime rate (converged behaviour: AUC charges exploration's tuition over
the whole run; this reads what it bought after the last change).

| baseline | metric | mean gap | median gap | win rate | sign-test p | q10 gap |
|---|---|---|---|---|---|---|
| random_p005 | mean rate (primary) | -0.065 | -0.083 | 8–11 of 20 | 0.6476 | -0.476 |
| random_p005 | AUC (front-loaded, reported-only) | -3.848 | -4.464 | 7–13 of 20 | 0.2632 | -16.595 |
| random_p005 | final-window rate | -0.125 | -0.357 | 9–11 of 20 | 0.8238 | -1.786 |
| random_p005 | final-regime rate | -0.074 | -0.423 | 9–11 of 20 | 0.8238 | -1.127 |
| fixed_k50 | mean rate (primary) | -0.026 | 0.012 | 10–9 of 20 | 1.0 | -0.333 |
| fixed_k50 | AUC (front-loaded, reported-only) | -2.555 | -3.262 | 9–11 of 20 | 0.8238 | -16.619 |
| fixed_k50 | final-window rate | 0.012 | -0.238 | 8–12 of 20 | 0.5034 | -0.952 |
| fixed_k50 | final-regime rate | -0.039 | -0.141 | 8–12 of 20 | 0.5034 | -0.634 |
| never_explore | mean rate (primary) | 0.15 | 0.119 | 19–1 of 20 | 0.0 | 0.024 |
| never_explore | AUC (front-loaded, reported-only) | 0.121 | 0.143 | 10–10 of 20 | 1.0 | -10.595 |
| never_explore | final-window rate | 0.756 | 0.774 | 15–2 of 20 | 0.0023 | -0.476 |
| never_explore | final-regime rate | 0.627 | 0.599 | 18–2 of 20 | 0.0004 | -0.07 |

## Behaviour-verified inversions

- seed 0: eu_max takes gw_add_feature at step 95 (never_explore: growth vetoed by construction; rate gap 0.024)
- seed 1: eu_max takes gw_add_feature at step 79 (never_explore: growth vetoed by construction; rate gap 0.429)
- seed 2: eu_max takes gw_add_feature at step 91 (never_explore: growth vetoed by construction; rate gap 0.095)
- seed 3: eu_max takes gw_add_feature at step 88 (never_explore: growth vetoed by construction; rate gap 0.119)
- seed 4: eu_max takes gw_add_feature at step 98 (never_explore: growth vetoed by construction; rate gap 0.286)
- seed 5: eu_max takes gw_add_feature at step 91 (never_explore: growth vetoed by construction; rate gap 0.5)
- seed 6: eu_max takes gw_add_feature at step 6 (never_explore: growth vetoed by construction; rate gap -0.381)
- seed 7: eu_max takes gw_add_feature at step 4 (never_explore: growth vetoed by construction; rate gap 0.024)
- seed 8: eu_max takes gw_add_feature at step 8 (never_explore: growth vetoed by construction; rate gap 0.024)
- seed 9: eu_max takes gw_add_feature at step 89 (never_explore: growth vetoed by construction; rate gap 0.048)

Bracket (mean rate): never_explore 0.274 ≤ eu_max 0.424 ≤ clairvoyant 0.429
