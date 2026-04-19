# Role: body
"""Experiment 1: Stationary tool reliabilities (SPEC §7.1).

50 questions, fixed reliabilities, N seeds. Compares Bayesian, baselines,
and (optionally) LangChain agents.

Usage:
    uv run python -m experiments.run_stationary --seeds 3
    uv run python -m experiments.run_stationary --seeds 20 --include-langchain
"""

from __future__ import annotations

import argparse
import json
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
    abstention_analysis,
    calibration_plot,
    cumulative_score_plot,
    score_comparison_bar,
    tool_calls_comparison,
    tool_selection_heatmap,
)
from credence_agents.environment.benchmark import BenchmarkResult, run_benchmark
from credence_agents.environment.categories import CATEGORIES
from credence_agents.environment.questions import get_questions
from credence_agents.environment.tools import make_spec_tools, tool_config_for
from credence_agents.julia_bridge import CredenceBridge


RESULTS_DIR = Path("results")


def make_agents(spec_tools, tool_configs, bridge: CredenceBridge, include_langchain: bool = False):
    """Create all agents to benchmark. Returns list of (name, factory_fn) pairs."""
    agents = [
        ("oracle", lambda: OracleAgent(bridge=bridge, tools=list(spec_tools), tool_configs=tool_configs, category_names=CATEGORIES)),
        ("bayesian", lambda: BayesianAgent(bridge=bridge, tool_configs=tool_configs, categories=CATEGORIES)),
        ("single_best", lambda: SingleBestToolAgent(tool_idx=0)),
        ("all_tools", lambda: AllToolsAgent(num_tools=4)),
        ("random", lambda: RandomAgent(num_tools=4, seed=0)),
    ]

    if include_langchain:
        from credence_agents.agents.langchain_agent import LangChainAgent
        from credence_agents.agents.langchain_enhanced import LangChainEnhancedAgent
        agents.extend([
            ("langchain_react", lambda: LangChainAgent()),
            ("langchain_enhanced", lambda: LangChainEnhancedAgent()),
        ])

    return agents


def run_experiment(
    n_seeds: int = 20,
    include_langchain: bool = False,
) -> dict[str, list[BenchmarkResult]]:
    """Run all agents across n_seeds question orderings."""
    spec_tools = make_spec_tools()
    tool_configs = [tool_config_for(t) for t in spec_tools]
    bridge = CredenceBridge()
    agent_factories = make_agents(spec_tools, tool_configs, bridge, include_langchain)

    results: dict[str, list[BenchmarkResult]] = {}

    for seed in range(n_seeds):
        questions = get_questions(seed=seed)

        for agent_name, factory in agent_factories:
            agent = factory()
            result = run_benchmark(agent, spec_tools, questions, seed=seed)

            if agent_name not in results:
                results[agent_name] = []
            results[agent_name].append(result)

        print(f"  Seed {seed} done", file=sys.stderr)

    return results


def summary_table(results: dict[str, list[BenchmarkResult]]) -> str:
    """Format a summary table of all agents' metrics."""
    header = (
        f"{'Agent':<20} {'Score':>10} {'Accuracy':>10} {'Abstain%':>10} "
        f"{'Tools/Q':>10} {'CostEff':>10} {'ECE':>10} {'Time/Q':>12}"
    )
    lines = [header, "-" * len(header)]

    for agent_name, runs in results.items():
        scores = [total_score(r) for r in runs]
        accs = [accuracy(r) for r in runs]
        abstains = [abstention_rate(r) for r in runs]
        tools_q = [tool_calls_per_question(r) for r in runs]
        cost_effs = [cost_efficiency(r) for r in runs]
        eces = [expected_calibration_error(r) for r in runs]
        times_q = [wall_time_per_question(r) for r in runs]

        lines.append(
            f"{agent_name:<20} "
            f"{np.mean(scores):>7.1f}±{np.std(scores):>3.1f} "
            f"{np.mean(accs):>7.3f}±{np.std(accs):.3f} "
            f"{np.mean(abstains):>7.3f}±{np.std(abstains):.3f} "
            f"{np.mean(tools_q):>7.2f}±{np.std(tools_q):.2f} "
            f"{np.mean(cost_effs):>7.2f}±{np.std(cost_effs):.2f} "
            f"{np.mean(eces):>7.3f}±{np.std(eces):.3f} "
            f"{np.mean(times_q):>7.2f}±{np.std(times_q):.2f}"
        )

    return "\n".join(lines)


def save_results(results: dict[str, list[BenchmarkResult]], out_dir: Path) -> None:
    """Save results as JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for agent_name, runs in results.items():
        data[agent_name] = [
            {
                "seed": r.seed,
                "total_score": r.total_score,
                "total_reward": r.total_reward,
                "total_tool_cost": r.total_tool_cost,
                "wall_time_s": r.wall_time_s,
                "records": [
                    {
                        "question_id": rec.question_id,
                        "category": rec.category,
                        "action_type": rec.action_type,
                        "was_correct": rec.was_correct,
                        "reward": rec.reward,
                        "tool_cost": rec.tool_cost,
                        "tools_queried": list(rec.tools_queried),
                    }
                    for rec in r.records
                ],
            }
            for r in runs
        ]

    with open(out_dir / "stationary_results.json", "w") as f:
        json.dump(data, f, indent=2)


def save_plots(results: dict[str, list[BenchmarkResult]], out_dir: Path) -> None:
    """Generate and save all visualisation plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = cumulative_score_plot(results)
    fig.savefig(out_dir / "cumulative_score.png", dpi=150)
    plt.close(fig)

    fig = score_comparison_bar(results)
    fig.savefig(out_dir / "score_comparison.png", dpi=150)
    plt.close(fig)

    fig = tool_selection_heatmap(results)
    fig.savefig(out_dir / "tool_selection_heatmap.png", dpi=150)
    plt.close(fig)

    fig = tool_calls_comparison(results)
    fig.savefig(out_dir / "tool_calls.png", dpi=150)
    plt.close(fig)

    fig = abstention_analysis(results)
    fig.savefig(out_dir / "abstention_analysis.png", dpi=150)
    plt.close(fig)

    # Calibration plot for agents with belief snapshots (e.g. bayesian, oracle)
    for agent_name, runs in results.items():
        for run in runs:
            has_conf = any(
                r.belief_snapshot and "answer_posterior" in r.belief_snapshot
                for r in run.records if r.action_type == "submit"
            )
            if has_conf:
                fig = calibration_plot(run)
                fig.savefig(out_dir / f"calibration_{agent_name}_seed{run.seed}.png", dpi=150)
                plt.close(fig)
                break  # one representative per agent


def main():
    parser = argparse.ArgumentParser(description="Run stationary benchmark experiment")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds")
    parser.add_argument("--include-langchain", action="store_true", help="Include LangChain agents")
    args = parser.parse_args()

    print(f"Running stationary experiment with {args.seeds} seeds...", file=sys.stderr)
    results = run_experiment(n_seeds=args.seeds, include_langchain=args.include_langchain)

    out_dir = RESULTS_DIR / "stationary"
    save_results(results, out_dir)
    save_plots(results, out_dir)

    print("\n" + summary_table(results))
    print(f"\nResults saved to {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
