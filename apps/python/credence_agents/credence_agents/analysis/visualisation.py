"""Visualisation functions for benchmark results.

All functions take data and return matplotlib Figure objects. No side effects
beyond figure creation — callers decide whether to show or save.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from credence_agents.environment.benchmark import BenchmarkResult
from credence_agents.environment.categories import CATEGORIES as _DEFAULT_CATEGORIES


def set_publication_style() -> None:
    """Set matplotlib rcParams for publication-quality plots."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#cccccc",
        "lines.linewidth": 2.0,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


set_publication_style()


def cumulative_score_plot(
    results: dict[str, list[BenchmarkResult]],
) -> plt.Figure:
    """Cumulative score over questions, averaged across seeds.

    Shows learning curves: Bayesian agent should improve as it learns tool reliability.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for agent_name, runs in results.items():
        # Build cumulative score per run, then average
        all_cumulative = []
        for result in runs:
            cumulative = np.cumsum([r.reward - r.tool_cost for r in result.records])
            all_cumulative.append(cumulative)

        all_cumulative = np.array(all_cumulative)
        mean = all_cumulative.mean(axis=0)
        std = all_cumulative.std(axis=0)
        x = np.arange(1, len(mean) + 1)

        ax.plot(x, mean, label=agent_name, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Question Number")
    ax.set_ylabel("Cumulative Score")
    ax.set_title("Cumulative Score Over Questions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def tool_selection_heatmap(
    results: dict[str, list[BenchmarkResult]],
    tool_names: tuple[str, ...] = ("quick_search", "knowledge_base", "calculator", "llm_direct"),
    categories: tuple[str, ...] = _DEFAULT_CATEGORIES,
) -> plt.Figure:
    """Heatmap: fraction of times each tool was queried per category, per agent."""
    agents = list(results.keys())
    n_agents = len(agents)
    fig, axes = plt.subplots(1, n_agents, figsize=(5 * n_agents, 4), squeeze=False)

    for i, agent_name in enumerate(agents):
        ax = axes[0, i]
        # Count tool queries per category
        counts = np.zeros((len(categories), len(tool_names)))
        cat_totals = np.zeros(len(categories))

        for result in results[agent_name]:
            for rec in result.records:
                c_idx = categories.index(rec.category) if rec.category in categories else -1
                if c_idx < 0:
                    continue
                cat_totals[c_idx] += 1
                for t_idx in rec.tools_queried:
                    if t_idx < len(tool_names):
                        counts[c_idx, t_idx] += 1

        # Normalise to fractions
        with np.errstate(divide="ignore", invalid="ignore"):
            fractions = counts / cat_totals[:, None]
            fractions = np.nan_to_num(fractions)

        sns.heatmap(
            fractions, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=[t.split("_")[0] for t in tool_names],
            yticklabels=list(categories),
            vmin=0, vmax=1,
        )
        ax.set_title(agent_name)

    fig.suptitle("Tool Selection by Category", fontsize=14)
    fig.tight_layout()
    return fig


def calibration_plot(result: BenchmarkResult, n_bins: int = 10) -> plt.Figure:
    """Reliability diagram: predicted confidence vs actual accuracy."""
    confidences = []
    correct = []
    for r in result.records:
        if r.action_type != "submit":
            continue
        if r.belief_snapshot and "answer_posterior" in r.belief_snapshot:
            conf = max(r.belief_snapshot["answer_posterior"])
            confidences.append(conf)
            correct.append(float(r.was_correct))

    fig, ax = plt.subplots(figsize=(6, 6))

    if not confidences:
        ax.text(0.5, 0.5, "No confidence data", ha="center", va="center")
        return fig

    confidences = np.array(confidences)
    correct = np.array(correct)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_confs = []
    bin_accs = []
    bin_counts = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue
        bin_confs.append(confidences[mask].mean())
        bin_accs.append(correct[mask].mean())
        bin_counts.append(count)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=1.0 / n_bins * 0.8, alpha=0.7, label="Observed")
    ax.set_xlabel("Predicted Confidence")
    ax.set_ylabel("Actual Accuracy")
    ax.set_title(f"Calibration — {result.agent_name}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def reliability_learning_curve(
    agent, true_tools,
    categories_to_plot: tuple[str, ...] = ("factual", "numerical"),
) -> plt.Figure:
    """Plot learned vs true reliability for selected tool-category pairs."""
    from credence_agents.analysis.metrics import learned_vs_true_reliability

    data = learned_vs_true_reliability(agent, true_tools)
    n_tools = len(data)

    fig, axes = plt.subplots(1, n_tools, figsize=(5 * n_tools, 4), squeeze=False)

    for i, (tool_name, cats) in enumerate(data.items()):
        ax = axes[0, i]
        cat_names = []
        learned_vals = []
        true_vals = []
        for cat in categories_to_plot:
            if cat in cats:
                cat_names.append(cat)
                learned_vals.append(cats[cat][0])
                true_vals.append(cats[cat][1])

        x = np.arange(len(cat_names))
        width = 0.35
        ax.bar(x - width / 2, learned_vals, width, label="Learned", alpha=0.8)
        ax.bar(x + width / 2, true_vals, width, label="True", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cat_names, rotation=45, ha="right")
        ax.set_ylabel("Reliability")
        ax.set_title(tool_name)
        ax.legend()
        ax.set_ylim(0, 1.1)

    fig.suptitle("Learned vs True Tool Reliability", fontsize=14)
    fig.tight_layout()
    return fig


def score_comparison_bar(
    results: dict[str, list[BenchmarkResult]],
) -> plt.Figure:
    """Bar chart comparing mean total score across agents with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(results.keys())
    means = [np.mean([r.total_score for r in runs]) for runs in results.values()]
    stds = [np.std([r.total_score for r in runs]) for runs in results.values()]

    colors = sns.color_palette("husl", len(names))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_ylabel("Total Score")
    ax.set_title("Agent Score Comparison")
    ax.axhline(y=0, color="black", linewidth=0.5)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


def tool_calls_comparison(
    results: dict[str, list[BenchmarkResult]],
) -> plt.Figure:
    """Bar chart of mean tool calls per question across agents."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(results.keys())
    means = []
    for runs in results.values():
        all_counts = [len(r.tools_queried) for run in runs for r in run.records]
        means.append(np.mean(all_counts) if all_counts else 0.0)

    colors = sns.color_palette("husl", len(names))
    ax.bar(names, means, color=colors, alpha=0.85)
    ax.set_ylabel("Tool Calls per Question")
    ax.set_title("Tool Usage Efficiency")

    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


def abstention_analysis(
    results: dict[str, list[BenchmarkResult]],
) -> plt.Figure:
    """Stacked bar: submitted correct / submitted wrong / abstained for each agent."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(results.keys())
    correct_rates = []
    wrong_rates = []
    abstain_rates = []

    for runs in results.values():
        total = sum(len(run.records) for run in runs)
        if total == 0:
            correct_rates.append(0)
            wrong_rates.append(0)
            abstain_rates.append(0)
            continue
        n_correct = sum(
            1 for run in runs for r in run.records
            if r.action_type == "submit" and r.was_correct
        )
        n_wrong = sum(
            1 for run in runs for r in run.records
            if r.action_type == "submit" and not r.was_correct
        )
        n_abstain = sum(
            1 for run in runs for r in run.records
            if r.action_type == "abstain"
        )
        correct_rates.append(n_correct / total)
        wrong_rates.append(n_wrong / total)
        abstain_rates.append(n_abstain / total)

    x = np.arange(len(names))
    ax.bar(x, correct_rates, label="Correct", color="green", alpha=0.8)
    ax.bar(x, wrong_rates, bottom=correct_rates, label="Wrong", color="red", alpha=0.8)
    bottoms = [c + w for c, w in zip(correct_rates, wrong_rates)]
    ax.bar(x, abstain_rates, bottom=bottoms, label="Abstain", color="gray", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Fraction of Questions")
    ax.set_title("Outcome Breakdown")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig
