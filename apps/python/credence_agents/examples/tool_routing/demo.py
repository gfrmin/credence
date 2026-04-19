#!/usr/bin/env python3
# Role: body
"""credence-router demo: transparent tool routing via EU maximisation.

Routes multiple-choice questions to the cheapest reliable tool using
Value of Information calculations. Zero routing cost, <1ms routing latency,
learns from feedback.

Usage:
    python examples/tool_routing/demo.py [--latency-weight 0.01] [--num 10] [--seed 42]
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from credence_agents.agents.bayesian_agent import BayesianAgent
from credence_agents.environment.categories import CATEGORIES
from credence_agents.environment.questions import Question, get_questions
from credence_agents.environment.tools import ResponseType, query_tool, tool_config_for
from credence_agents.inference.voi import ScoringRule

from tools import (
    TOOL_MONETARY_COSTS,
    make_routing_tools,
)


def format_voi_trace(
    question: Question,
    tool_names: list[str],
    decision_trace: tuple,
    _answer_idx: int | None,
    was_correct: bool | None,
    monetary_cost: float,
) -> str:
    """Format a human-readable VOI trace for one question."""
    lines: list[str] = []

    # Header
    lines.append(f'  "{question.text}" [{question.category}]')

    # Show each decision step
    for step in decision_trace:
        for t_idx, net_voi in sorted(step.eu_query.items()):
            name = tool_names[t_idx]
            lines.append(f"    VOI({name:12s}) = {net_voi:+.4f}")
        if step.chosen_action.startswith("query("):
            t_idx = int(step.chosen_action[6:-1])
            lines.append(f"    -> Query {tool_names[t_idx]}")
        elif step.chosen_action.startswith("submit("):
            a_idx = int(step.chosen_action[7:-1])
            lines.append(f"    -> SUBMIT: {question.candidates[a_idx]}")

    # Outcome
    if was_correct is True:
        lines.append(f"    CORRECT. Cost: ${monetary_cost:.4f}")
    elif was_correct is False:
        correct_ans = question.candidates[question.correct_index]
        lines.append(f"    WRONG (correct: {correct_ans}). Cost: ${monetary_cost:.4f}")
    else:
        lines.append(f"    ABSTAINED. Cost: ${monetary_cost:.4f}")

    return "\n".join(lines)


def run_demo(num_questions: int = 10, latency_weight: float = 0.01, seed: int = 42) -> None:
    """Run the routing demo on a subset of the question bank."""
    # --- Setup ---
    tools = make_routing_tools(latency_weight=latency_weight)
    tool_names = [t.name for t in tools]
    tool_configs = [tool_config_for(t) for t in tools]

    scoring = ScoringRule(reward_correct=0.01, penalty_wrong=-0.005, reward_abstain=0.0)

    agent = BayesianAgent(
        tool_configs=tool_configs,
        categories=CATEGORIES,
        scoring=scoring,
        name="credence-router",
    )

    questions = get_questions(seed=seed)[:num_questions]
    rng = np.random.default_rng(seed)

    # --- Run ---
    print(f"credence-router demo: {num_questions} questions, {len(tools)} tools")
    print(f"Latency weight: ${latency_weight}/sec")
    print(f"Scoring: correct=${scoring.reward_correct}, wrong=${scoring.penalty_wrong}")
    print()

    correct_count = 0
    total_monetary_cost = 0.0
    total_effective_cost = 0.0
    tools_used_counter: Counter[str] = Counter()

    for i, question in enumerate(questions):

        def make_query_fn(q: Question, r: np.random.Generator):
            def query_fn(tool_idx: int) -> int | None:
                resp = query_tool(tools[tool_idx], q, r)
                return resp.candidate_idx if resp.response_type == ResponseType.ANSWER else None

            return query_fn

        result = agent.solve_question(
            question_text=question.text,
            candidates=question.candidates,
            category_hint=None,
            tool_query_fn=make_query_fn(question, rng),
        )

        # Track costs
        monetary_cost = sum(
            TOOL_MONETARY_COSTS[tool_names[t_idx]] for t_idx in result.tools_queried
        )
        effective_cost_q = sum(tool_configs[t_idx].cost for t_idx in result.tools_queried)

        was_correct = result.answer == question.correct_index if result.answer is not None else None
        if was_correct:
            correct_count += 1

        total_monetary_cost += monetary_cost
        total_effective_cost += effective_cost_q
        for t_idx in result.tools_queried:
            tools_used_counter[tool_names[t_idx]] += 1

        # Report outcome for reliability learning
        agent.on_question_end(was_correct)

        # Print trace
        print(
            f"Q{i + 1}: {
                format_voi_trace(
                    question,
                    tool_names,
                    result.decision_trace,
                    result.answer,
                    was_correct,
                    monetary_cost,
                )
            }"
        )
        print()

    # --- Summary ---
    print("=" * 60)
    print(f"Summary ({num_questions} questions):")
    print(
        f"  Accuracy:     {correct_count}/{num_questions} ({100 * correct_count / num_questions:.0f}%)"
    )
    print(f"  Monetary cost: ${total_monetary_cost:.4f}")
    print(f"  Effective cost: ${total_effective_cost:.4f} (incl. latency @ ${latency_weight}/s)")
    print(f"  Avg tools/q:  {sum(tools_used_counter.values()) / num_questions:.2f}")
    print(f"  Tool usage:   {dict(tools_used_counter)}")
    print()
    print("  Comparison:")
    print(
        f"  - LangGraph routing overhead: ~${0.006 * num_questions:.3f}-${0.009 * num_questions:.3f}"
    )
    print("  - credence routing overhead:  $0.000 (numpy only)")


def main() -> None:
    parser = argparse.ArgumentParser(description="credence-router demo")
    parser.add_argument("--num", type=int, default=10, help="Number of questions")
    parser.add_argument("--latency-weight", type=float, default=0.001, help="$/second latency cost")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    run_demo(num_questions=args.num, latency_weight=args.latency_weight, seed=args.seed)


if __name__ == "__main__":
    main()
