# Dominance benchmark — results

Task: `[:colour_typed, :motion_typed, :territorial]`, regime changes at `[70, 140]`, 210 steps, respawn on, 20 seeds, paired-seed percentile bootstrap (10 000 resamples).

| policy | mean AUC | mean final-window rate | mean steps-to-half | mean meta-actions |
|---|---|---|---|---|
| clairvoyant | 40.96 | 0.798 | 52.7 | 629.8 |
| eu_max | 40.96 | 0.798 | 52.7 | 629.8 |
| fixed_k10 | 46.9 | 0.655 | 56.8 | 21.0 |
| fixed_k25 | 46.74 | 0.911 | 61.2 | 8.0 |
| fixed_k5 | 43.45 | 0.917 | 74.3 | 42.0 |
| fixed_k50 | 51.54 | 0.673 | 93.6 | 4.0 |
| never_explore | 34.9 | 0.804 | 42.5 | 335.4 |
| random_p005 | 48.72 | 0.673 | 36.6 | 12.3 |
| random_p015 | 44.12 | 0.601 | 52.9 | 37.7 |
| random_p04 | 44.78 | 0.714 | 46.2 | 131.0 |

## Paired gaps (eu_max − baseline; efficiency sign-flipped so + favours eu_max)

| baseline | AUC gap [95% CI] | efficiency gap [95% CI] | worst-seed AUC gap |
|---|---|---|---|
| random_p005 | -7.75 [-15.48, -0.92] | -16.2 [-42.2, 6.6] | -56.31 |
| fixed_k50 | -10.58 [-17.17, -4.09] | 41.0 [16.4, 67.1] | -46.24 |
| never_explore | 6.06 [0.89, 10.97] | -10.2 [-37.2, 17.6] | -17.52 |

`eu_max − never_explore` is the headline: the escape-mass heuristic is identical on both sides, so this gap is exploration's isolated value.

## Behaviour-verified inversions

- seed 0: eu_max takes gw_add_feature at step 92 (never_explore: growth vetoed by construction; auc gap 9.1)
- seed 1: eu_max takes gw_add_feature at step 81 (never_explore: growth vetoed by construction; auc gap 7.14)
- seed 2: eu_max takes gw_add_feature at step 93 (never_explore: growth vetoed by construction; auc gap 20.52)
- seed 3: eu_max takes gw_add_feature at step 89 (never_explore: growth vetoed by construction; auc gap 10.5)
- seed 4: eu_max takes gw_add_feature at step 98 (never_explore: growth vetoed by construction; auc gap 17.45)
- seed 5: eu_max takes gw_add_feature at step 92 (never_explore: growth vetoed by construction; auc gap -17.52)
- seed 6: eu_max takes gw_add_feature at step 92 (never_explore: growth vetoed by construction; auc gap 23.5)
- seed 7: eu_max takes gw_add_feature at step 95 (never_explore: growth vetoed by construction; auc gap -13.93)
- seed 8: eu_max takes gw_add_feature at step 95 (never_explore: growth vetoed by construction; auc gap -2.14)
- seed 9: eu_max takes gw_add_feature at step 90 (never_explore: growth vetoed by construction; auc gap 6.4)

Bracket: never_explore 34.9 ≤ eu_max 40.96 ≤ clairvoyant 40.96
