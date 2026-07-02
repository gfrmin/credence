# Priced-exploration sweep — dominance benchmark

Task: `[:colour_typed, :motion_typed, :territorial]`, regime changes at `[70, 140]`, 210 steps, respawn on, 20 seeds, paired-seed percentile bootstrap (10 000 resamples).

eu_max swept over declared `exploration_cost` (Δ log-evidence nats, priced into BOTH the exact VOI lookahead and its matching execution). Baselines run at their own tuned behaviour (cost 0). A positive gap (eu_max − baseline) favours eu_max; efficiency sign-flipped so + favours eu_max.

## Baseline reference (fixed)

| baseline | mean AUC | mean growth ops |
|---|---|---|
| never_explore | 34.9 | 0.0 |
| random_p005 | 48.72 | 9.4 |
| fixed_k50 | 51.54 | 3.0 |

## eu_max priced sweep

| cost | mean AUC | mean growth ops | AUC gap vs never_explore [CI] | AUC gap vs random_p005 [CI] | AUC gap vs fixed_k50 [CI] | worst-seed gap vs random | worst-seed gap vs fixed |
|---|---|---|---|---|---|---|---|
| 0.0 | 40.96 | 10.6 | 6.06 [0.89, 10.97] | -7.75 [-15.48, -0.92] | -10.58 [-17.17, -4.09] | -56.31 | -46.24 |
| 0.25 | 39.91 | 5.6 | 5.01 [0.24, 9.62] | -8.81 [-15.86, -2.43] | -11.63 [-17.77, -5.3] | -52.38 | -42.31 |
| 0.5 | 39.23 | 4.5 | 4.33 [0.16, 8.31] | -9.48 [-16.75, -2.96] | -12.3 [-18.49, -6.21] | -52.38 | -42.31 |
| 1.0 | 39.2 | 3.8 | 4.3 [0.5, 8.02] | -9.51 [-16.49, -3.12] | -12.33 [-18.58, -6.27] | -52.57 | -42.5 |
| 2.0 | 38.73 | 3.6 | 3.83 [0.12, 7.18] | -9.99 [-17.15, -3.27] | -12.81 [-19.14, -6.68] | -52.57 | -42.5 |

## Efficiency gap (steps-to-own-half; baseline − eu_max, + favours eu_max)

| cost | eff gap vs random_p005 [CI] | eff gap vs fixed_k50 [CI] |
|---|---|---|
| 0.0 | -16.2 [-42.2, 6.6] | 41.0 [16.4, 67.1] |
| 0.25 | -24.3 [-52.6, 1.0] | 32.8 [-0.6, 64.8] |
| 0.5 | -24.7 [-55.8, 2.5] | 32.4 [-2.6, 66.2] |
| 1.0 | -10.9 [-36.2, 10.1] | 46.2 [18.2, 75.0] |
| 2.0 | -3.1 [-22.6, 13.9] | 54.0 [24.0, 84.6] |

## Reading the sweep

- **Headline held** at a cost iff the `AUC gap vs never_explore [CI]` lower bound stays > 0 (exploration still pays for itself).
- **Dominance restored** at a cost iff the `AUC gap vs random_p005` AND `AUC gap vs fixed_k50` CI lower bounds are both > 0 (or at minimum the point gaps turn non-negative and worst-seed gaps stop being deeply negative).
- **Priced region exists** iff some cost row satisfies BOTH above.
- **Mechanism**: mean growth ops should fall monotonically with cost; the failure mode is a cost so high that growth never fires and eu_max collapses into never_explore (headline gap → 0).
