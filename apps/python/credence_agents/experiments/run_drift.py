# Role: body
"""Experiment 2: Tool reliability drift (SPEC §7.2).

Tool A's reliability degrades at question 25. The Bayesian agent with
forgetting (lambda=0.95) should detect and adapt; LangChain agents cannot.

Usage:
    uv run python -m experiments.run_drift --seeds 3
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
from credence_agents.environment.benchmark import BenchmarkResult
from credence_agents.environment.questions import get_questions
from credence_agents.environment.tools import SimulatedTool, make_spec_tools, tool_config_for
from credence_agents.environment.categories import CATEGORIES
from credence_agents.julia_bridge import CredenceBridge


RESULTS_DIR = Path("results")
DRIFT_POINT = 25


def make_degraded_tool_a(original: SimulatedTool) -> SimulatedTool:
    """Tool A after drift: reliability drops per SPEC §7.2."""
    return original._replace(
        reliability_by_category={
            "factual": 0.35,
            "numerical": 0.15,
            "recent_events": 0.30,
            "misconceptions": 0.20,
            "reasoning": 0.20,
        },
    )


def run_drift_experiment(
    n_seeds: int = 20,
) -> tuple[dict[str, list[BenchmarkResult]], dict[str, list[np.ndarray]]]:
    """Run agents with tool A degrading at question 25."""
    spec_tools = make_spec_tools()
    tool_configs = [tool_config_for(t) for t in spec_tools]
    degraded_a = make_degraded_tool_a(spec_tools[0])
    bridge = CredenceBridge()

    agent_factories = [
        ("oracle", lambda: OracleAgent(bridge=bridge, tools=list(spec_tools), tool_configs=tool_configs, category_names=CATEGORIES)),
        ("bayesian_forget", lambda: BayesianAgent(bridge=bridge, tool_configs=tool_configs, categories=CATEGORIES, forgetting=0.95,
                                                   name="bayesian_forget")),
        ("bayesian_no_forget", lambda: BayesianAgent(bridge=bridge, tool_configs=tool_configs, categories=CATEGORIES, forgetting=1.0,
                                                      name="bayesian_no_forget")),
        ("single_best", lambda: SingleBestToolAgent(tool_idx=0)),
        ("random", lambda: RandomAgent(num_tools=4, seed=0)),
    ]

    results: dict[str, list[BenchmarkResult]] = {}
    # Track Tool A learned reliability per question for Bayesian agents
    # Key: agent_name, Value: list of arrays (n_seeds x n_questions x n_categories)
    reliability_traces: dict[str, list[np.ndarray]] = {}

    for seed in range(n_seeds):
        questions = get_questions(seed=seed)
        rng = np.random.default_rng(seed)

        for agent_name, factory in agent_factories:
            agent = factory()

            # Manual benchmark loop with tool swap at DRIFT_POINT
            from credence_agents.environment.benchmark import (
                QuestionRecord,
                BenchmarkResult as BR,
            )
            from credence_agents.environment.tools import query_tool, ResponseType
            from credence_agents.inference.voi import REWARD_CORRECT, PENALTY_WRONG, REWARD_ABSTAIN
            from credence_agents.inference.decision import ActionType

            q_rng = np.random.default_rng(seed)
            records = []
            total_reward = 0.0
            total_tool_cost = 0.0
            num_tools = 4

            # Per-question reliability trace for Tool A (only for Bayesian agents)
            is_bayesian = hasattr(agent, "rel_states")
            seed_trace = []  # list of arrays, one per question

            for q_idx, question in enumerate(questions):
                # Switch tools at drift point
                current_tools = list(spec_tools)
                if q_idx >= DRIFT_POINT:
                    current_tools[0] = degraded_a

                agent.on_question_start(question.id, question.candidates, num_tools,
                                       question_text=question.text)
                used = []
                tool_responses = {}
                q_cost = 0.0

                while True:
                    action = agent.choose_action()
                    if action.action_type == ActionType.QUERY:
                        t_idx = action.tool_idx
                        resp = query_tool(current_tools[t_idx], question, q_rng)
                        cand = resp.candidate_idx if resp.response_type == ResponseType.ANSWER else None
                        q_cost += current_tools[t_idx].cost
                        used.append(t_idx)
                        tool_responses[t_idx] = cand
                        agent.on_tool_response(t_idx, cand)
                    elif action.action_type == ActionType.SUBMIT:
                        submitted = action.answer_idx
                        was_correct = submitted == question.correct_index
                        reward = REWARD_CORRECT if was_correct else PENALTY_WRONG
                        total_reward += reward
                        total_tool_cost += q_cost
                        snapshot = getattr(agent, "get_belief_snapshot", lambda: None)()
                        agent.on_question_end(was_correct)
                        records.append(QuestionRecord(
                            question.id, question.category, question.difficulty,
                            tuple(used), dict(tool_responses),
                            "submit", submitted, was_correct, reward, q_cost, snapshot,
                        ))
                        break
                    elif action.action_type == ActionType.ABSTAIN:
                        total_reward += REWARD_ABSTAIN
                        total_tool_cost += q_cost
                        snapshot = getattr(agent, "get_belief_snapshot", lambda: None)()
                        agent.on_question_end(None)
                        records.append(QuestionRecord(
                            question.id, question.category, question.difficulty,
                            tuple(used), dict(tool_responses),
                            "abstain", None, None, REWARD_ABSTAIN, q_cost, snapshot,
                        ))
                        break

                # Snapshot Tool A's learned E[r] per category after this question
                if is_bayesian:
                    er = np.array(bridge.extract_reliability_means(agent.rel_states[0]))
                    seed_trace.append(er)

            if is_bayesian:
                if agent_name not in reliability_traces:
                    reliability_traces[agent_name] = []
                reliability_traces[agent_name].append(np.array(seed_trace))

            result = BR(agent.name, seed, tuple(records),
                        total_reward - total_tool_cost, total_tool_cost, total_reward)

            if agent_name not in results:
                results[agent_name] = []
            results[agent_name].append(result)

        print(f"  Seed {seed} done", file=sys.stderr)

    return results, reliability_traces


def before_after_table(results: dict[str, list[BenchmarkResult]]) -> str:
    """Score in first half vs second half (after drift)."""
    header = f"{'Agent':<20} {'Before':>12} {'After':>12} {'Delta':>10}"
    lines = [header, "-" * len(header)]

    for agent_name, runs in results.items():
        befores = []
        afters = []
        for r in runs:
            before = sum(rec.reward - rec.tool_cost for rec in r.records[:DRIFT_POINT])
            after = sum(rec.reward - rec.tool_cost for rec in r.records[DRIFT_POINT:])
            befores.append(before)
            afters.append(after)
        mb, ma = np.mean(befores), np.mean(afters)
        lines.append(f"{agent_name:<20} {mb:>9.1f}±{np.std(befores):.1f} "
                      f"{ma:>9.1f}±{np.std(afters):.1f} {ma - mb:>+8.1f}")

    return "\n".join(lines)


def save_drift_plots(results: dict[str, list[BenchmarkResult]], out_dir: Path) -> None:
    """Cumulative score plot with drift annotation."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for agent_name, runs in results.items():
        all_cum = []
        for r in runs:
            all_cum.append(np.cumsum([rec.reward - rec.tool_cost for rec in r.records]))
        all_cum = np.array(all_cum)
        mean = all_cum.mean(axis=0)
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, label=agent_name, linewidth=2)

    ax.axvline(x=DRIFT_POINT, color="red", linestyle="--", alpha=0.7, label="Tool A degrades")
    ax.set_xlabel("Question Number")
    ax.set_ylabel("Cumulative Score")
    ax.set_title("Cumulative Score — Drift Experiment")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "drift_cumulative.png", dpi=150)
    plt.close(fig)


def save_reliability_learning_curve(
    reliability_traces: dict[str, list[np.ndarray]],
    true_before: dict[str, float],
    true_after: dict[str, float],
    out_dir: Path,
) -> None:
    """Plot how the Bayesian agent's learned Tool A reliability evolves over questions."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for agent_name, traces in reliability_traces.items():
        # traces: list of arrays, each (n_questions, n_categories)
        all_traces = np.array(traces)  # (n_seeds, n_questions, n_categories)
        mean_traces = all_traces.mean(axis=0)  # (n_questions, n_categories)
        x = np.arange(1, mean_traces.shape[0] + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        for c_idx, cat in enumerate(CATEGORIES):
            ax.plot(x, mean_traces[:, c_idx], label=cat, linewidth=2)
            # Draw true values (before and after drift)
            ax.hlines(true_before[cat], 1, DRIFT_POINT, colors=f"C{c_idx}",
                      linestyles=":", alpha=0.4)
            ax.hlines(true_after[cat], DRIFT_POINT, len(x), colors=f"C{c_idx}",
                      linestyles=":", alpha=0.4)

        ax.axvline(x=DRIFT_POINT, color="red", linestyle="--", alpha=0.7,
                   label="Tool A degrades")
        ax.set_xlabel("Question Number")
        ax.set_ylabel("Learned E[reliability] for Tool A")
        ax.set_title(f"Tool A Reliability Learning — {agent_name}")
        ax.legend(fontsize=8, ncol=2)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"reliability_curve_{agent_name}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run drift benchmark experiment")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds")
    args = parser.parse_args()

    print(f"Running drift experiment with {args.seeds} seeds...", file=sys.stderr)
    results, reliability_traces = run_drift_experiment(n_seeds=args.seeds)

    out_dir = RESULTS_DIR / "drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_drift_plots(results, out_dir)

    # Reliability learning curve for Tool A
    spec_tools = make_spec_tools()
    degraded_a = make_degraded_tool_a(spec_tools[0])
    true_before = spec_tools[0].reliability_by_category
    true_after = degraded_a.reliability_by_category
    save_reliability_learning_curve(reliability_traces, true_before, true_after, out_dir)

    print("\n" + before_after_table(results))
    print(f"\nPlots saved to {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
