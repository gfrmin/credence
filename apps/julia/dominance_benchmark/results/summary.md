# Dominance benchmark — results

Task: `[:colour_typed, :motion_typed, :territorial]`, regime changes at `[70, 140]`, 210 steps, respawn on, 20 seeds, paired-seed percentile bootstrap (10 000 resamples).

Primary realised-value measure (§8, ratified 2026-07-03): the mean per-step energy rate `ce[end]/n` — the uniform-weight statistic the agent's declared utility maximises. AUC of the cumulative trajectory (front-loaded) is reported-only, for cross-round comparability.

| policy | mean rate | mean final-window rate | mean AUC (reported) | mean steps-to-half | mean meta-actions |
|---|---|---|---|---|---|
| clairvoyant | 0.38 | 0.69 | 46.29 | 58.0 | 19.1 |
| eu_max | 0.38 | 0.69 | 46.29 | 58.0 | 13.1 |
| fixed_k10 | 0.433 | 0.75 | 49.48 | 75.5 | 21.0 |
| fixed_k25 | 0.408 | 0.524 | 50.74 | 55.0 | 8.0 |
| fixed_k5 | 0.517 | 0.929 | 52.9 | 70.8 | 42.0 |
| fixed_k50 | 0.431 | 0.732 | 49.69 | 69.8 | 4.0 |
| never_explore | 0.274 | 0.125 | 47.11 | 28.9 | 2.0 |
| random_p005 | 0.449 | 0.893 | 50.32 | 70.2 | 12.3 |
| random_p015 | 0.402 | 0.75 | 48.58 | 69.2 | 37.7 |
| random_p04 | 0.407 | 0.833 | 46.31 | 75.0 | 131.0 |

## Paired gaps (eu_max − baseline; efficiency = steps to the shared per-seed
level, sign-flipped so + favours eu_max — belief-derived-valuation §2c)

| baseline | rate gap [95% CI] | final-window gap [95% CI] | efficiency gap [95% CI] | worst-seed rate gap | q10 rate gap |
|---|---|---|---|---|---|
| random_p005 | -0.069 [-0.189, 0.057] | -0.202 [-0.702, 0.339] | -1.8 [-12.9, 9.8] | -0.452 | -0.452 |
| fixed_k5 | -0.137 [-0.254, -0.017] | -0.238 [-0.845, 0.393] | -9.8 [-20.8, 0.6] | -0.714 | -0.571 |
| never_explore | 0.106 [0.004, 0.21] | 0.565 [0.149, 1.018] | 16.0 [5.6, 26.6] | -0.452 | -0.095 |

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
| random_p005 | mean rate (primary) | -0.069 | -0.06 | 9–11 of 20 | 0.8238 | -0.452 |
| random_p005 | AUC (front-loaded, reported-only) | -4.037 | -4.381 | 7–13 of 20 | 0.2632 | -22.238 |
| random_p005 | final-window rate | -0.202 | -0.179 | 7–12 of 20 | 0.3593 | -1.786 |
| random_p005 | final-regime rate | -0.032 | 0.035 | 10–9 of 20 | 1.0 | -1.197 |
| fixed_k5 | mean rate (primary) | -0.137 | -0.167 | 5–15 of 20 | 0.0414 | -0.571 |
| fixed_k5 | AUC (front-loaded, reported-only) | -6.614 | -5.774 | 4–16 of 20 | 0.0118 | -24.429 |
| fixed_k5 | final-window rate | -0.238 | -0.357 | 8–12 of 20 | 0.5034 | -2.143 |
| fixed_k5 | final-regime rate | -0.299 | -0.387 | 6–13 of 20 | 0.1671 | -1.972 |
| never_explore | mean rate (primary) | 0.106 | 0.06 | 12–8 of 20 | 0.5034 | -0.095 |
| never_explore | AUC (front-loaded, reported-only) | -0.825 | -1.25 | 9–11 of 20 | 0.8238 | -22.238 |
| never_explore | final-window rate | 0.565 | 0.417 | 13–5 of 20 | 0.0963 | -0.714 |
| never_explore | final-regime rate | 0.528 | 0.317 | 17–3 of 20 | 0.0026 | -0.141 |

## Behaviour-verified inversions

- seed 0: eu_max takes gw_add_feature at step 96 (never_explore: growth vetoed by construction; rate gap 0.31)
- seed 1: eu_max takes gw_add_feature at step 84 (never_explore: growth vetoed by construction; rate gap -0.024)
- seed 2: eu_max takes gw_add_feature at step 96 (never_explore: growth vetoed by construction; rate gap 0.643)
- seed 3: eu_max takes gw_add_feature at step 98 (never_explore: growth vetoed by construction; rate gap -0.095)
- seed 4: eu_max takes gw_add_feature at step 100 (never_explore: growth vetoed by construction; rate gap -0.048)
- seed 5: eu_max takes gw_add_feature at step 110 (never_explore: growth vetoed by construction; rate gap 0.357)
- seed 6: eu_max takes gw_add_feature at step 89 (never_explore: growth vetoed by construction; rate gap -0.452)
- seed 7: eu_max takes gw_add_feature at step 95 (never_explore: growth vetoed by construction; rate gap 0.071)
- seed 8: eu_max takes gw_add_feature at step 90 (never_explore: growth vetoed by construction; rate gap 0.238)
- seed 9: eu_max takes gw_add_feature at step 89 (never_explore: growth vetoed by construction; rate gap -0.048)

Bracket (mean rate): never_explore 0.274 ≤ eu_max 0.38 ≤ clairvoyant 0.38
