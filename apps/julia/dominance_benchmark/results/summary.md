# Dominance benchmark — results

Task: `[:colour_typed, :motion_typed, :territorial]`, regime changes at `[70, 140]`, 210 steps, respawn on, 20 seeds, paired-seed percentile bootstrap (10 000 resamples).

| policy | mean AUC | mean final-window rate | mean steps-to-half | mean meta-actions |
|---|---|---|---|---|
| clairvoyant | 46.49 | 0.565 | 45.0 | 625.4 |
| eu_max | 46.49 | 0.565 | 45.0 | 625.4 |
| fixed_k10 | 49.21 | 0.875 | 73.8 | 21.0 |
| fixed_k25 | 51.23 | 0.518 | 62.0 | 8.0 |
| fixed_k5 | 51.06 | 0.899 | 64.1 | 42.0 |
| fixed_k50 | 48.61 | 0.792 | 62.3 | 4.0 |
| never_explore | 46.69 | 0.244 | 25.2 | 301.1 |
| random_p005 | 49.96 | 0.583 | 101.0 | 12.3 |
| random_p015 | 48.45 | 0.887 | 72.2 | 37.7 |
| random_p04 | 49.07 | 0.696 | 51.4 | 131.0 |

## Paired gaps (eu_max − baseline; efficiency sign-flipped so + favours eu_max)

| baseline | AUC gap [95% CI] | efficiency gap [95% CI] | worst-seed AUC gap |
|---|---|---|---|
| random_p005 | -3.47 [-9.24, 2.78] | 56.1 [24.0, 89.7] | -25.64 |
| fixed_k25 | -4.74 [-9.41, 0.17] | 17.0 [-14.9, 48.2] | -22.95 |
| never_explore | -0.2 [-3.53, 2.99] | -19.7 [-41.5, -1.2] | -16.19 |

`eu_max − never_explore` is the headline: the escape-mass heuristic is identical on both sides, so this gap is exploration's isolated value.

## Behaviour-verified inversions

- seed 0: eu_max takes gw_add_feature at step 95 (never_explore: growth vetoed by construction; auc gap -7.6)
- seed 1: eu_max takes gw_add_feature at step 80 (never_explore: growth vetoed by construction; auc gap 8.05)
- seed 2: eu_max takes gw_add_feature at step 93 (never_explore: growth vetoed by construction; auc gap -5.9)
- seed 3: eu_max takes gw_add_feature at step 89 (never_explore: growth vetoed by construction; auc gap -3.36)
- seed 4: eu_max takes gw_add_feature at step 98 (never_explore: growth vetoed by construction; auc gap 6.48)
- seed 5: eu_max takes gw_add_feature at step 92 (never_explore: growth vetoed by construction; auc gap 2.98)
- seed 6: eu_max takes gw_add_feature at step 89 (never_explore: growth vetoed by construction; auc gap 3.17)
- seed 7: eu_max takes gw_add_feature at step 95 (never_explore: growth vetoed by construction; auc gap 0.69)
- seed 8: eu_max takes gw_add_feature at step 91 (never_explore: growth vetoed by construction; auc gap 13.24)
- seed 9: eu_max takes gw_add_feature at step 93 (never_explore: growth vetoed by construction; auc gap 7.55)

Bracket: never_explore 46.69 ≤ eu_max 46.49 ≤ clairvoyant 46.49
