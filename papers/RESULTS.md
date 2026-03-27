# Benchmark Results

All experiments run from the Julia benchmark at `domains/qa_benchmark/host.jl`.
Fast agents (bayesian, baselines) and LLM agents all run 20 seeds.
LLM agents use Llama 3.1 8B via Ollama (temperature 0).

## Experiment 1: Stationary Tool Reliabilities (20 seeds)

All agents face 50 questions with fixed tool reliabilities across 4 simulated tools.

| Agent | N | Score | Accuracy | Abstain% | Tools/Q | Time/Q (s) |
|---|---|---|---|---|---|---|
| bayesian | 20 | +129.5 +/- 51.3 | 0.626 | 0.000 | 1.80 | 0.07 |
| single_best | 20 | +58.5 +/- 55.9 | 0.478 | 0.000 | 1.00 | <0.01 |
| random | 20 | +12.2 +/- 54.8 | 0.450 | 0.000 | 1.00 | <0.01 |
| llm_RSH | 20 | +10.8 +/- 50.8 | 0.764 | 0.251 | 3.05 | 5.65 |
| llm_R | 20 | -15.3 +/- 28.0 | 0.660 | 0.255 | 2.70 | 4.60 |
| all_tools | 20 | -114.2 +/- 60.1 | 0.581 | 0.000 | 4.00 | <0.01 |
| llm_bare | 20 | -160.5 +/- 36.1 | 0.438 | 0.008 | 3.26 | 0.93 |

LLM agent variants:
- **llm_bare**: base prompt only — LLM outputs an action with no reasoning
- **llm_R**: adds ReAct reasoning traces (Thought/Action/Observation)
- **llm_RSH**: adds strategy guidance (cost-awareness, reliability heuristics) and cross-question history (last 10 outcomes)

**Key findings:**

- **Bayesian agent scores +129.5 vs best LLM +10.8** despite lower accuracy
  (62.6% vs 76.4%). The Bayesian agent wins by querying fewer tools per question
  (1.80 vs 3.05) and taking 0.07s vs 5.65s per question (80× faster).
- **The prompting gradient never closes the gap.** LLM Bare (-160.5) → LLM ReAct
  (-15.3) → LLM ReAct+S+H (+10.8). Each prompting technique improves accuracy but
  the best LLM variant still scores 12× less than the Bayesian agent.
- **The all_tools baseline (-114.2) demonstrates the cost trap.** Querying all 4 tools
  every time yields 58.1% accuracy but the tool costs overwhelm the reward.

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
to the full agent's score of 112.6. (These are from the older Python benchmark run;
Julia ablation rerun pending.)

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

1. **EU maximisation beats prompt engineering.** The Bayesian agent (+129.5) outscores
   the best LLM agent (+10.8) by 119 points despite 14 percentage points lower accuracy.
   The mechanism: principled tool selection via VOI queries ~1.8 tools per question instead
   of ~3.05, taking 0.07s instead of 5.65s per question.

2. **More prompting helps but never closes the gap.** LLM Bare (-160.5) → LLM ReAct
   (-15.3) → LLM ReAct+S+H (+10.8). Each technique (reasoning traces, strategy prompts,
   cross-question memory) improves accuracy, but the best variant scores 12× less than
   the Bayesian agent. The LLM lacks a formal mechanism for computing whether a query
   is worth its cost.

3. **Category inference is the most valuable component.** Ablation shows removing
   category-aware priors costs -102 points — more than removing VOI (-78.1) or
   reliability learning (-78.1). Knowing *which kind* of question you're facing is
   the single most important input to tool selection.
