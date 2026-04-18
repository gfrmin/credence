# Benchmark Results

## Experiment 1: Stationary Tool Reliabilities (20 seeds)

All agents face 50 questions with fixed tool reliabilities across 4 simulated tools.
Fast agents (oracle, bayesian, baselines) run 20 seeds; LangChain agents run 5 seeds
due to Ollama inference cost.

| Agent | N | Score | Accuracy | Abstain% | Tools/Q | CostEff | ECE |
|---|---|---|---|---|---|---|---|
| oracle | 20 | 188.2 +/- 49.1 | 0.706 +/- 0.068 | 0.025 +/- 0.015 | 1.08 +/- 0.00 | 3.24 +/- 0.58 | 0.091 +/- 0.047 |
| bayesian | 20 | 112.6 +/- 44.6 | 0.596 +/- 0.037 | 0.110 +/- 0.216 | 0.99 +/- 0.14 | 2.83 +/- 0.80 | 0.101 +/- 0.056 |
| single_best | 20 | 34.5 +/- 48.9 | 0.446 +/- 0.065 | 0.000 +/- 0.000 | 1.00 +/- 0.00 | 1.69 +/- 0.98 | 0.000 +/- 0.000 |
| all_tools | 20 | -88.0 +/- 57.0 | 0.616 +/- 0.076 | 0.000 +/- 0.000 | 4.00 +/- 0.00 | 0.71 +/- 0.19 | 0.000 +/- 0.000 |
| random | 20 | 14.0 +/- 53.3 | 0.448 +/- 0.071 | 0.000 +/- 0.000 | 1.00 +/- 0.00 | 1.19 +/- 0.74 | 0.000 +/- 0.000 |
| langchain_react | 5 | -7.4 +/- 16.2 | 0.640 +/- 0.026 | 0.012 +/- 0.010 | 3.22 +/- 0.04 | 0.97 +/- 0.07 | 0.000 +/- 0.000 |
| langchain_enhanced | 5 | -68.2 +/- 13.1 | 0.660 +/- 0.022 | 0.072 +/- 0.010 | 3.94 +/- 0.03 | 0.77 +/- 0.04 | 0.000 +/- 0.000 |

**Key findings:**

- **Bayesian agent scores +112.6 vs LangChain react -7.4** despite lower raw accuracy
  (59.6% vs 64.0%). The Bayesian agent wins on cost efficiency (2.83 vs 0.97) by
  querying fewer tools per question (0.99 vs 3.22).
- **Enhanced LangChain prompting makes things worse.** The enhanced variant is more
  accurate (66.0% vs 64.0%) but scores -68.2 because it queries nearly every tool
  on every question (3.94 tools/Q), destroying its cost efficiency (0.77).
- **The all_tools baseline (-88.0) demonstrates the cost trap.** Querying all 4 tools
  every time yields 61.6% accuracy but the tool costs overwhelm the reward.

**Plots:** `stationary_full/cumulative_score.png`, `stationary_full/score_comparison.png`,
`stationary_full/tool_selection_heatmap.png`, `stationary_full/calibration_bayesian.png`,
`stationary_full/calibration_oracle.png`


## Experiment 2: Tool Reliability Drift (20 seeds)

Tool A's reliability degrades at question 25 (midpoint). Tests whether agents adapt
to non-stationarity.

| Agent | Before | After | Delta |
|---|---|---|---|
| oracle | 90.2 +/- 35.0 | 73.0 +/- 40.7 | -17.2 |
| bayesian_forget | 47.6 +/- 30.9 | 25.9 +/- 27.5 | -21.8 |
| bayesian_no_forget | 40.8 +/- 35.6 | 37.2 +/- 39.9 | -3.5 |
| single_best | 14.2 +/- 36.0 | -54.8 +/- 31.2 | -69.0 |
| random | 3.4 +/- 27.8 | -3.6 +/- 28.4 | -7.0 |

**Key findings:**

- **single_best collapses under drift (-69.0 delta).** It commits to tool A based on
  early experience and cannot adapt when reliability drops, going from +14.2 to -54.8.
- **Bayesian agents adapt gracefully.** The forgetting variant (lambda decay on old
  observations) shows a -21.8 delta vs -3.5 for the no-forget variant. The no-forget
  agent's prior inertia buffers it — learned reliability drifts slowly when mixed with
  many pre-drift observations.
- **Even the oracle suffers (-17.2)** because it still pays tool costs in the degraded
  regime where fewer tools are worth querying.

**Plots:** `drift_full/drift_cumulative.png`, `drift_full/reliability_curve_bayesian_forget.png`,
`drift_full/reliability_curve_bayesian_no_forget.png`, `drift_full/reliability_curve_oracle.png`


## Experiment 3: Ablation Study (20 seeds)

Each variant removes one component from the full Bayesian agent. Delta is relative
to the full agent's score of 112.6.

| Variant | Score | Accuracy | Tools/Q | Delta |
|---|---|---|---|---|
| full_agent | 112.6 +/- 44.6 | 0.596 +/- 0.037 | 0.99 +/- 0.14 | -- |
| no_voi | 34.5 +/- 48.9 | 0.446 +/- 0.065 | 1.00 +/- 0.00 | -78.1 |
| no_category | 10.6 +/- 55.8 | 0.316 +/- 0.178 | 0.49 +/- 0.39 | -102.0 |
| no_abstention | 91.1 +/- 82.2 | 0.539 +/- 0.121 | 0.99 +/- 0.14 | -21.5 |
| fixed_reliability | 34.5 +/- 48.9 | 0.446 +/- 0.065 | 1.00 +/- 0.00 | -78.1 |
| no_crossverify | 117.6 +/- 40.8 | 0.600 +/- 0.052 | 0.95 +/- 0.11 | +5.0 |

**Component importance ranking (by score delta):**

1. **Category inference (-102.0):** Most critical. Without category-aware priors the
   agent cannot route questions to appropriate tools. Accuracy collapses to 31.6%.
2. **VOI-based tool selection (-78.1):** Without value-of-information, the agent picks
   the cheapest tool rather than the most informative, matching single_best performance.
3. **Reliability learning (-78.1):** Same impact as removing VOI — without learned
   reliability, VOI calculations use uninformative priors and degenerate to cost-based
   selection.
4. **Abstention (-21.5):** Moderate impact. Forcing submission on low-confidence
   questions adds wrong answers that cost -10 each. Variance doubles (82.2 vs 44.6)
   indicating less consistent performance.
5. **Cross-verification (+5.0):** Removing second-tool verification *slightly improves*
   score. The cost of a second query sometimes outweighs the information gain in this
   environment. VOI correctly identifies this most of the time, but occasionally
   over-queries.

**Plots:** `ablation/ablation_comparison.png`, `ablation/ablation_tool_calls.png`


## Headline Findings

1. **EU maximisation beats prompt engineering.** The Bayesian agent (+112.6) outscores
   LangChain react (-7.4) by 120 points despite 4 percentage points lower accuracy.
   The mechanism: principled tool selection via VOI queries ~1 tool per question instead
   of ~3.2, and strategic abstention avoids costly wrong answers.

2. **More prompting makes LangChain worse.** The enhanced LangChain variant with
   reliability-aware prompts and self-reflection scores -68.2 — worse than the basic
   react agent. Extra reasoning triggers more tool calls (3.94/Q) without proportional
   accuracy gains, a concrete example of the "expensive flowchart" problem.

3. **Category inference is the most valuable component.** Ablation shows removing
   category-aware priors costs -102 points — more than removing VOI (-78.1) or
   reliability learning (-78.1). Knowing *which kind* of question you're facing is
   the single most important input to tool selection.
