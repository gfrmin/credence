# Dominance benchmark — results

Task: `[:colour_typed, :motion_typed, :territorial]`, regime changes at `[70, 140]`, 210 steps, respawn on, 20 seeds, paired-seed percentile bootstrap (10 000 resamples).

| policy | mean AUC | mean final-window rate | mean steps-to-half | mean meta-actions |
|---|---|---|---|---|
| clairvoyant | 47.59 | 0.875 | 51.8 | 32.2 |
| eu_max | 47.23 | 0.881 | 52.1 | 26.1 |
| fixed_k10 | 49.21 | 0.875 | 73.8 | 21.0 |
| fixed_k25 | 51.23 | 0.518 | 62.0 | 8.0 |
| fixed_k5 | 51.06 | 0.899 | 64.1 | 42.0 |
| fixed_k50 | 48.61 | 0.792 | 62.3 | 4.0 |
| never_explore | 47.11 | 0.125 | 28.9 | 2.0 |
| random_p005 | 49.96 | 0.583 | 101.0 | 12.3 |
| random_p015 | 48.45 | 0.887 | 72.2 | 37.7 |
| random_p04 | 49.07 | 0.696 | 51.4 | 131.0 |

## Paired gaps (eu_max − baseline; efficiency = steps to the shared per-seed
level, sign-flipped so + favours eu_max — belief-derived-valuation §2c)

| baseline | AUC gap [95% CI] | final-window gap [95% CI] | efficiency gap [95% CI] | worst-seed AUC gap |
|---|---|---|---|---|
| random_p005 | -2.72 [-6.84, 1.44] | 0.298 [-0.274, 0.863] | -3.2 [-12.4, 5.6] | -16.6 |
| fixed_k25 | -4.0 [-7.15, -0.8] | 0.363 [-0.095, 0.839] | -5.6 [-14.9, 3.6] | -19.67 |
| never_explore | 0.12 [-2.86, 3.0] | 0.756 [0.363, 1.125] | 12.0 [3.0, 21.6] | -16.19 |

`eu_max − never_explore` is the headline: the learned-returns escape ops are identical on both sides, so this gap is exploration's isolated value.

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
