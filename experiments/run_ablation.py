"""Experiment 3: Ablation studies (SPEC §7.3).

Six Bayesian agent variants with components disabled to measure contribution.

Usage:
    uv run python -m experiments.run_ablation --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from credence_agents.agents.bayesian_agent import BayesianAgent
from credence_agents.agents.common import DecisionStep
from credence_agents.analysis.metrics import accuracy, total_score, tool_calls_per_question
from credence_agents.analysis.visualisation import score_comparison_bar, tool_calls_comparison
from credence_agents.environment.benchmark import BenchmarkResult, run_benchmark
from credence_agents.environment.categories import CATEGORIES
from credence_agents.environment.questions import get_questions
from credence_agents.environment.tools import make_spec_tools, tool_config_for
from credence_agents.inference.decision import Action, ActionType
from credence_agents.inference.voi import ScoringRule
from credence_agents.julia_bridge import CredenceBridge


RESULTS_DIR = Path("results")


def _eu_submit(ans_w, scoring):
    """EU of submitting the best answer given posterior weights."""
    p_best = max(ans_w)
    return p_best * scoring.reward_correct + (1 - p_best) * scoring.penalty_wrong


# --- Ablation variants ---

class NoVOIAgent(BayesianAgent):
    """Always queries cheapest applicable tool instead of using VOI."""

    def choose_action(self) -> Action:
        assert self._answer_measure is not None
        self._step += 1

        # If no tools queried yet, pick cheapest available
        if self._available and not self._tool_responses:
            cheapest = min(self._available, key=lambda i: self.tool_configs[i].cost)
            action = Action(ActionType.QUERY, tool_idx=cheapest)
        else:
            # After one tool, submit or abstain based on EU
            ans_w = self.bridge.weights(self._answer_measure)
            eu_s = _eu_submit(ans_w, self.scoring)
            eu_a = self.scoring.reward_abstain
            if eu_s >= eu_a:
                best = int(max(range(len(ans_w)), key=lambda i: ans_w[i]))
                action = Action(ActionType.SUBMIT, answer_idx=best)
            else:
                action = Action(ActionType.ABSTAIN)

        chosen = (f"submit({action.answer_idx})" if action.action_type == ActionType.SUBMIT
                  else "abstain" if action.action_type == ActionType.ABSTAIN
                  else f"query({action.tool_idx})")
        self._trace.append(DecisionStep(
            step=self._step, eu_submit=0.0, eu_abstain=0.0,
            eu_query={}, chosen_action=chosen,
        ))
        return action


class NoCategoryAgent(BayesianAgent):
    """Treats all categories as identical (resets category belief each question)."""

    def on_question_start(self, question_id, candidates, num_tools, question_text=""):
        super().on_question_start(question_id, candidates, num_tools, question_text=question_text)
        # Reset category belief to uniform — discard any cross-question learning
        self.cat_belief = self.bridge.make_cat_belief(self._num_categories)


class NoAbstentionAgent(BayesianAgent):
    """Must always submit — abstention disabled."""

    def choose_action(self) -> Action:
        action = super().choose_action()
        if action.action_type == ActionType.ABSTAIN:
            ans_w = self.bridge.weights(self._answer_measure)
            best = int(max(range(len(ans_w)), key=lambda i: ans_w[i]))
            return Action(ActionType.SUBMIT, answer_idx=best)
        return action


class FixedReliabilityAgent(BayesianAgent):
    """No learning — uses prior reliability throughout."""

    def on_question_end(self, was_correct):
        # Skip reliability update
        pass


class SingleToolAgent(BayesianAgent):
    """Only queries one tool per question — no cross-verification."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._queried_this_q = False

    def on_question_start(self, question_id, candidates, num_tools, question_text=""):
        super().on_question_start(question_id, candidates, num_tools, question_text=question_text)
        self._queried_this_q = False

    def choose_action(self) -> Action:
        action = super().choose_action()
        if action.action_type == ActionType.QUERY:
            if self._queried_this_q:
                # Force submit/abstain instead of second query
                ans_w = self.bridge.weights(self._answer_measure)
                eu_s = _eu_submit(ans_w, self.scoring)
                eu_a = self.scoring.reward_abstain
                if eu_s >= eu_a:
                    best = int(max(range(len(ans_w)), key=lambda i: ans_w[i]))
                    return Action(ActionType.SUBMIT, answer_idx=best)
                return Action(ActionType.ABSTAIN)
            self._queried_this_q = True
        return action


def make_ablation_agents(tool_configs, bridge):
    """Create all ablation variants."""
    return [
        ("full_agent", lambda: BayesianAgent(bridge=bridge, tool_configs=tool_configs, categories=CATEGORIES, name="full_agent")),
        ("no_voi", lambda: NoVOIAgent(bridge=bridge, tool_configs=tool_configs, categories=CATEGORIES, name="no_voi")),
        ("no_category", lambda: NoCategoryAgent(bridge=bridge, tool_configs=tool_configs, categories=CATEGORIES, name="no_category")),
        ("no_abstention", lambda: NoAbstentionAgent(bridge=bridge, tool_configs=tool_configs, categories=CATEGORIES, name="no_abstention")),
        ("fixed_reliability", lambda: FixedReliabilityAgent(bridge=bridge, tool_configs=tool_configs, categories=CATEGORIES, name="fixed_reliability")),
        ("no_crossverify", lambda: SingleToolAgent(bridge=bridge, tool_configs=tool_configs, categories=CATEGORIES, name="no_crossverify")),
    ]


def run_ablation(n_seeds: int = 20) -> dict[str, list[BenchmarkResult]]:
    """Run all ablation variants."""
    spec_tools = make_spec_tools()
    tool_configs = [tool_config_for(t) for t in spec_tools]
    bridge = CredenceBridge()
    agents = make_ablation_agents(tool_configs, bridge)

    results: dict[str, list[BenchmarkResult]] = {}

    for seed in range(n_seeds):
        questions = get_questions(seed=seed)
        for name, factory in agents:
            agent = factory()
            result = run_benchmark(agent, spec_tools, questions, seed=seed)
            if name not in results:
                results[name] = []
            results[name].append(result)

        print(f"  Seed {seed} done", file=sys.stderr)

    return results


def ablation_table(results: dict[str, list[BenchmarkResult]]) -> str:
    """Format ablation comparison table."""
    header = f"{'Variant':<20} {'Score':>12} {'Accuracy':>10} {'Tools/Q':>10}"
    lines = [header, "-" * len(header)]

    full_score = np.mean([total_score(r) for r in results.get("full_agent", [])])

    for name, runs in results.items():
        scores = [total_score(r) for r in runs]
        accs = [accuracy(r) for r in runs]
        tools = [tool_calls_per_question(r) for r in runs]
        delta = np.mean(scores) - full_score

        lines.append(
            f"{name:<20} "
            f"{np.mean(scores):>7.1f}±{np.std(scores):.1f} "
            f"{np.mean(accs):>7.3f}±{np.std(accs):.3f} "
            f"{np.mean(tools):>7.2f}±{np.std(tools):.2f}"
            + (f"  ({delta:+.1f})" if name != "full_agent" else "")
        )

    return "\n".join(lines)


def save_results(results: dict[str, list[BenchmarkResult]], out_dir: Path) -> None:
    """Save ablation results as JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for agent_name, runs in results.items():
        data[agent_name] = [
            {
                "seed": r.seed,
                "total_score": r.total_score,
                "total_reward": r.total_reward,
                "total_tool_cost": r.total_tool_cost,
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

    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(data, f, indent=2)


def save_plots(results: dict[str, list[BenchmarkResult]], out_dir: Path) -> None:
    """Generate and save ablation plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = score_comparison_bar(results)
    fig.savefig(out_dir / "ablation_comparison.png")
    plt.close(fig)

    fig = tool_calls_comparison(results)
    fig.savefig(out_dir / "ablation_tool_calls.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds")
    args = parser.parse_args()

    print(f"Running ablation with {args.seeds} seeds...", file=sys.stderr)
    results = run_ablation(n_seeds=args.seeds)

    out_dir = RESULTS_DIR / "ablation"
    save_results(results, out_dir)
    save_plots(results, out_dir)

    print("\n" + ablation_table(results))
    print(f"\nResults saved to {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
