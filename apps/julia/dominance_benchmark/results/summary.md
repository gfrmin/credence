# Dominance benchmark — results

Task: `[:colour_typed, :motion_typed, :territorial]`, regime changes at `[70, 140]`, 210 steps, respawn on, 20 seeds, paired-seed percentile bootstrap (10 000 resamples).

| policy | mean AUC | mean final-window rate | mean steps-to-half | mean meta-actions |
|---|---|---|---|---|
| clairvoyant | 47.59 | 0.875 | 51.8 | 33.0 |
| eu_max | 47.23 | 0.881 | 52.1 | 27.0 |
| fixed_k10 | 48.22 | 0.72 | 67.6 | 21.0 |
| fixed_k25 | 50.51 | 0.488 | 52.8 | 8.0 |
| fixed_k5 | 48.49 | 0.667 | 57.6 | 42.0 |
| fixed_k50 | 49.79 | 0.869 | 76.8 | 4.0 |
| never_explore | 47.11 | 0.125 | 28.9 | 2.0 |
| random_p005 | 51.08 | 1.006 | 86.6 | 12.3 |
| random_p015 | 49.71 | 0.875 | 85.9 | 37.7 |
| random_p04 | 46.65 | 0.75 | 51.6 | 131.0 |

## Paired gaps (eu_max − baseline; efficiency = steps to the shared per-seed
level, sign-flipped so + favours eu_max — belief-derived-valuation §2c)

| baseline | AUC gap [95% CI] | final-window gap [95% CI] | efficiency gap [95% CI] | worst-seed AUC gap |
|---|---|---|---|---|
| random_p005 | -3.85 [-8.15, 0.54] | -0.125 [-0.625, 0.363] | -3.6 [-13.2, 5.6] | -20.81 |
| fixed_k25 | -3.28 [-6.3, -0.35] | 0.393 [-0.071, 0.857] | -2.1 [-10.2, 5.6] | -19.67 |
| never_explore | 0.12 [-2.86, 3.0] | 0.756 [0.363, 1.125] | 11.4 [2.4, 21.0] | -16.19 |

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
| random_p005 | AUC | -3.848 | -4.464 | 7–13 of 20 | 0.2632 | -16.595 |
| random_p005 | final-window rate | -0.125 | -0.357 | 9–11 of 20 | 0.8238 | -1.786 |
| random_p005 | final-regime rate | -0.074 | -0.423 | 9–11 of 20 | 0.8238 | -1.127 |
| fixed_k25 | AUC | -3.276 | -2.393 | 6–14 of 20 | 0.1153 | -14.857 |
| fixed_k25 | final-window rate | 0.393 | 0.476 | 12–8 of 20 | 0.5034 | -1.071 |
| fixed_k25 | final-regime rate | 0.225 | 0.317 | 13–7 of 20 | 0.2632 | -0.775 |
| never_explore | AUC | 0.121 | 0.143 | 10–10 of 20 | 1.0 | -10.595 |
| never_explore | final-window rate | 0.756 | 0.774 | 15–2 of 20 | 0.0023 | -0.476 |
| never_explore | final-regime rate | 0.627 | 0.599 | 18–2 of 20 | 0.0004 | -0.07 |

## Behaviour-verified inversions

- seed 0: eu_max takes gw_add_feature at step 95 (never_explore: growth vetoed by construction; auc gap -4.31)
- seed 1: eu_max takes gw_add_feature at step 79 (never_explore: growth vetoed by construction; auc gap 11.71)
- seed 2: eu_max takes gw_add_feature at step 91 (never_explore: growth vetoed by construction; auc gap -5.9)
- seed 3: eu_max takes gw_add_feature at step 88 (never_explore: growth vetoed by construction; auc gap -0.07)
- seed 4: eu_max takes gw_add_feature at step 98 (never_explore: growth vetoed by construction; auc gap 6.55)
- seed 5: eu_max takes gw_add_feature at step 91 (never_explore: growth vetoed by construction; auc gap 11.55)
- seed 6: eu_max takes gw_add_feature at step 6 (never_explore: growth vetoed by construction; auc gap -16.19)
- seed 7: eu_max takes gw_add_feature at step 4 (never_explore: growth vetoed by construction; auc gap -0.31)
- seed 8: eu_max takes gw_add_feature at step 8 (never_explore: growth vetoed by construction; auc gap 4.17)
- seed 9: eu_max takes gw_add_feature at step 89 (never_explore: growth vetoed by construction; auc gap -10.6)

Bracket: never_explore 47.11 ≤ eu_max 47.23 ≤ clairvoyant 47.59
