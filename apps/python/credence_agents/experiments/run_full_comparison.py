# Role: body
"""Full comparison: 20-seed stationary + drift, 20-seed LangChain.

Runs non-LLM agents with full statistical power (20 seeds) and LangChain
agents with 20 seeds.

Usage:
    uv run python -m experiments.run_full_comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from credence_agents.agents.baselines import (
    AllToolsAgent,
    OracleAgent,
    RandomAgent,
    SingleBestToolAgent,
)
from credence_agents.agents.bayesian_agent import BayesianAgent
from credence_agents.agents.langchain_agent import LangChainAgent
from credence_agents.agents.langchain_enhanced import LangChainEnhancedAgent
from credence_agents.julia_bridge import CredenceBridge
from credence_agents.analysis.metrics import (
    abstention_rate,
    accuracy,
    cost_efficiency,
    expected_calibration_error,
    tool_calls_per_question,
    total_score,
    wall_time_per_question,
)
from credence_agents.analysis.visualisation import (
    calibration_plot,
    cumulative_score_plot,
    score_comparison_bar,
    tool_selection_heatmap,
)
from credence_agents.environment.benchmark import BenchmarkResult, run_benchmark
from credence_agents.environment.questions import get_questions
from credence_agents.environment.tools import make_spec_tools, tool_config_for

from experiments.run_ablation import (
    ablation_table,
    run_ablation,
    save_plots as save_ablation_plots,
    save_results as save_ablation_results,
)
from experiments.run_drift import (
    make_degraded_tool_a,
    run_drift_experiment,
    before_after_table,
    save_drift_plots,
    save_reliability_learning_curve,
)


RESULTS_DIR = Path("results")
N_SEEDS_FAST = 20
N_SEEDS_LLM = 20


def summary_table(results: dict[str, list[BenchmarkResult]]) -> str:
    header = (
        f"{'Agent':<22} {'N':>3} {'Score':>12} {'Accuracy':>12} {'Abstain%':>10} "
        f"{'Tools/Q':>10} {'CostEff':>10} {'ECE':>10} {'Time/Q':>12}"
    )
    lines = [header, "-" * len(header)]

    for agent_name, runs in results.items():
        n = len(runs)
        scores = [total_score(r) for r in runs]
        accs = [accuracy(r) for r in runs]
        abstains = [abstention_rate(r) for r in runs]
        tools_q = [tool_calls_per_question(r) for r in runs]
        cost_effs = [cost_efficiency(r) for r in runs]
        eces = [expected_calibration_error(r) for r in runs]
        times_q = [wall_time_per_question(r) for r in runs]

        lines.append(
            f"{agent_name:<22} {n:>3} "
            f"{np.mean(scores):>7.1f}±{np.std(scores):>3.1f} "
            f"{np.mean(accs):>7.3f}±{np.std(accs):.3f} "
            f"{np.mean(abstains):>7.3f}±{np.std(abstains):.3f} "
            f"{np.mean(tools_q):>7.2f}±{np.std(tools_q):.2f} "
            f"{np.mean(cost_effs):>7.2f}±{np.std(cost_effs):.2f} "
            f"{np.mean(eces):>7.3f}±{np.std(eces):.3f} "
            f"{np.mean(times_q):>7.2f}±{np.std(times_q):.2f}"
        )

    return "\n".join(lines)


def run_stationary_full() -> dict[str, list[BenchmarkResult]]:
    """Run stationary: 20 seeds for fast agents, 20 for LangChain."""
    spec_tools = make_spec_tools()
    tool_configs = [tool_config_for(t) for t in spec_tools]
    bridge = CredenceBridge()

    fast_agents = [
        ("oracle", lambda: OracleAgent(bridge=bridge, tools=list(spec_tools), tool_configs=tool_configs)),
        ("bayesian", lambda: BayesianAgent(bridge=bridge, tool_configs=tool_configs)),
        ("single_best", lambda: SingleBestToolAgent(tool_idx=0)),
        ("all_tools", lambda: AllToolsAgent(num_tools=4)),
        ("random", lambda: RandomAgent(num_tools=4, seed=0)),
    ]
    llm_agents = [
        ("langchain_react", lambda: LangChainAgent()),
        ("langchain_enhanced", lambda: LangChainEnhancedAgent()),
    ]

    results: dict[str, list[BenchmarkResult]] = {}

    # Fast agents: 20 seeds
    print(f"  Running fast agents ({N_SEEDS_FAST} seeds)...", file=sys.stderr)
    for seed in range(N_SEEDS_FAST):
        questions = get_questions(seed=seed)
        for name, factory in fast_agents:
            agent = factory()
            result = run_benchmark(agent, spec_tools, questions, seed=seed)
            results.setdefault(name, []).append(result)
        print(f"    Seed {seed} done", file=sys.stderr)

    # LangChain agents: 20 seeds
    print(f"  Running LangChain agents ({N_SEEDS_LLM} seeds)...", file=sys.stderr)
    for seed in range(N_SEEDS_LLM):
        questions = get_questions(seed=seed)
        for name, factory in llm_agents:
            agent = factory()
            result = run_benchmark(agent, spec_tools, questions, seed=seed)
            results.setdefault(name, []).append(result)
        print(f"    LLM seed {seed} done", file=sys.stderr)

    return results


def main():
    # --- Part 1: Stationary experiment ---
    print("=" * 60, file=sys.stderr)
    print("PART 1: Stationary experiment", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    stat_results = run_stationary_full()
    out_stat = RESULTS_DIR / "stationary_full"
    out_stat.mkdir(parents=True, exist_ok=True)

    print("\n" + summary_table(stat_results))

    # Plots
    fig = cumulative_score_plot(stat_results)
    fig.savefig(out_stat / "cumulative_score.png", dpi=150)
    plt.close(fig)

    fig = score_comparison_bar(stat_results)
    fig.savefig(out_stat / "score_comparison.png", dpi=150)
    plt.close(fig)

    fig = tool_selection_heatmap(stat_results)
    fig.savefig(out_stat / "tool_selection_heatmap.png", dpi=150)
    plt.close(fig)

    # Calibration for bayesian and oracle
    for agent_name in ("bayesian", "oracle"):
        if agent_name in stat_results:
            run = stat_results[agent_name][0]
            has_conf = any(
                r.belief_snapshot and "answer_posterior" in r.belief_snapshot
                for r in run.records if r.action_type == "submit"
            )
            if has_conf:
                fig = calibration_plot(run)
                fig.savefig(out_stat / f"calibration_{agent_name}.png", dpi=150)
                plt.close(fig)

    print(f"\nStationary plots saved to {out_stat}/", file=sys.stderr)

    # --- Part 2: Drift experiment ---
    print("\n" + "=" * 60, file=sys.stderr)
    print("PART 2: Drift experiment (20 seeds)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    drift_results, reliability_traces = run_drift_experiment(n_seeds=N_SEEDS_FAST)
    out_drift = RESULTS_DIR / "drift_full"
    out_drift.mkdir(parents=True, exist_ok=True)

    save_drift_plots(drift_results, out_drift)

    spec_tools = make_spec_tools()
    degraded_a = make_degraded_tool_a(spec_tools[0])
    save_reliability_learning_curve(
        reliability_traces,
        spec_tools[0].reliability_by_category,
        degraded_a.reliability_by_category,
        out_drift,
    )

    print("\n" + before_after_table(drift_results))
    print(f"\nDrift plots saved to {out_drift}/", file=sys.stderr)

    # --- Part 3: Ablation study ---
    print("\n" + "=" * 60, file=sys.stderr)
    print("PART 3: Ablation study (20 seeds)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    ablation_results = run_ablation(n_seeds=N_SEEDS_FAST)
    out_ablation = RESULTS_DIR / "ablation"
    save_ablation_results(ablation_results, out_ablation)
    save_ablation_plots(ablation_results, out_ablation)

    print("\n" + ablation_table(ablation_results))
    print(f"\nAblation results saved to {out_ablation}/", file=sys.stderr)


if __name__ == "__main__":
    main()
