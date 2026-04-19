# Role: body
"""Benchmark metrics: score, accuracy, calibration, cost efficiency.

All functions take a BenchmarkResult (or its records) and return scalar
or structured metrics. Pure functions, no side effects.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from credence_agents.environment.benchmark import BenchmarkResult
from credence_agents.environment.tools import SimulatedTool


def total_score(result: BenchmarkResult) -> float:
    return result.total_score


def accuracy(result: BenchmarkResult) -> float:
    """Fraction of submitted answers that are correct (excluding abstentions)."""
    submitted = [r for r in result.records if r.action_type == "submit"]
    if not submitted:
        return 0.0
    return sum(1 for r in submitted if r.was_correct) / len(submitted)


def abstention_rate(result: BenchmarkResult) -> float:
    """Fraction of questions where agent abstained."""
    if not result.records:
        return 0.0
    return sum(1 for r in result.records if r.action_type == "abstain") / len(result.records)


def abstention_quality(result: BenchmarkResult) -> float:
    """Accuracy on attempted (non-abstained) questions. Same as accuracy()."""
    return accuracy(result)


def tool_calls_per_question(result: BenchmarkResult) -> float:
    """Mean number of tool queries per question."""
    if not result.records:
        return 0.0
    return sum(len(r.tools_queried) for r in result.records) / len(result.records)


def cost_efficiency(result: BenchmarkResult) -> float:
    """Total reward per point of tool cost. Inf if no tools used."""
    if result.total_tool_cost == 0:
        return float("inf") if result.total_reward > 0 else 0.0
    return result.total_reward / result.total_tool_cost


def per_category_accuracy(result: BenchmarkResult) -> dict[str, float]:
    """Accuracy broken down by question category."""
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in result.records:
        if r.action_type == "submit":
            by_cat[r.category].append(r.was_correct)
    return {cat: sum(v) / len(v) if v else 0.0 for cat, v in by_cat.items()}


def expected_calibration_error(
    result: BenchmarkResult, n_bins: int = 10,
) -> float:
    """ECE: weighted average of |bin_accuracy - bin_confidence| over bins.

    Only meaningful for agents that populate belief_snapshot with answer_posterior.
    Returns 0.0 if no confidence data is available.
    """
    confidences = []
    correct = []
    for r in result.records:
        if r.action_type != "submit":
            continue
        if r.belief_snapshot and "answer_posterior" in r.belief_snapshot:
            conf = max(r.belief_snapshot["answer_posterior"])
            confidences.append(conf)
            correct.append(r.was_correct)

    if not confidences:
        return 0.0

    confidences = np.array(confidences)
    correct = np.array(correct, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(confidences)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (count / total) * abs(bin_acc - bin_conf)

    return float(ece)


def learned_vs_true_reliability(
    agent,
    true_tools: list[SimulatedTool],
    categories: tuple[str, ...] = ("factual", "numerical", "recent_events", "misconceptions", "reasoning"),
) -> dict[str, dict[str, tuple[float, float]]]:
    """Compare agent's learned Beta means to true tool reliabilities.

    Returns: {tool_name: {category: (learned, true)}}
    """
    result = {}
    for t_idx, tool in enumerate(true_tools):
        tool_result = {}
        for c_idx, cat in enumerate(categories):
            alpha = agent.reliability_table[t_idx, c_idx, 0]
            beta = agent.reliability_table[t_idx, c_idx, 1]
            learned = alpha / (alpha + beta)
            true_r = tool.reliability_by_category.get(cat, 0.0)
            tool_result[cat] = (learned, true_r)
        result[tool.name] = tool_result
    return result


def wall_time_per_question(result: BenchmarkResult) -> float:
    """Mean wall-clock seconds per question."""
    if not result.records:
        return 0.0
    return result.wall_time_s / len(result.records)
