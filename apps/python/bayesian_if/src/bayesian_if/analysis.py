# Role: body
"""Post-hoc analysis of benchmark data: tool agreement, failure taxonomy, etc."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from bayesian_if.agent import GameResult, StepRecord


@dataclass
class ToolAgreement:
    """How often tools agree and who's right when they disagree."""

    total_steps_with_tools: int = 0
    all_agree: int = 0
    some_disagree: int = 0
    correct_tool_outvoted: int = 0
    all_none: int = 0


@dataclass
class FailureTaxonomy:
    """Categorization of failure modes."""

    all_tools_wrong: int = 0
    correct_tool_outvoted: int = 0
    all_tools_none: int = 0
    abstained_incorrectly: int = 0
    no_tools_queried: int = 0
    total_wrong: int = 0


def analyze_tool_agreement(results: list[GameResult]) -> ToolAgreement:
    """Analyze how often tools agree and which is right on disagreement."""
    agg = ToolAgreement()
    for game in results:
        for step in game.steps:
            recs = step.tool_recommendations
            if not recs:
                continue
            agg.total_steps_with_tools += 1
            values = list(recs.values())
            non_none = [v for v in values if v is not None]
            if not non_none:
                agg.all_none += 1
                continue
            if len(set(non_none)) == 1:
                agg.all_agree += 1
            else:
                agg.some_disagree += 1
                if step.was_correct is False:
                    agg.correct_tool_outvoted += 1
    return agg


def analyze_failures(results: list[GameResult]) -> FailureTaxonomy:
    """Classify failure modes from step records."""
    tax = FailureTaxonomy()
    for game in results:
        for step in game.steps:
            if step.was_correct is not False:
                continue
            tax.total_wrong += 1
            recs = step.tool_recommendations
            if not recs:
                tax.no_tools_queried += 1
                continue
            non_none = [v for v in recs.values() if v is not None]
            if not non_none:
                tax.all_tools_none += 1
            elif all(_rec_matches_action(step, v) is False for v in non_none):
                tax.all_tools_wrong += 1
            elif any(_rec_matches_action(step, v) for v in non_none):
                tax.correct_tool_outvoted += 1
            else:
                tax.all_tools_wrong += 1
    return tax


def analyze_per_difficulty(
    results: list[GameResult],
    specs: list[dict],
) -> dict[str, dict[str, float]]:
    """Break down performance by difficulty parameters.

    specs should be a list of dicts with keys: world_size, nb_objects, quest_length.
    Must align 1:1 with results.
    """
    by_ql: dict[int, list[float]] = {}
    for game, spec in zip(results, specs):
        ql = spec.get("quest_length", 0)
        score = game.final_score
        by_ql.setdefault(ql, []).append(float(score))

    return {
        f"ql={ql}": {
            "mean_score": sum(scores) / len(scores),
            "count": float(len(scores)),
        }
        for ql, scores in sorted(by_ql.items())
    }


def category_distribution(results: list[GameResult]) -> Counter[str | None]:
    """Count category hints across all steps."""
    counts: Counter[str | None] = Counter()
    for game in results:
        for step in game.steps:
            counts[step.category_hint] += 1
    return counts


def _rec_matches_action(step: StepRecord, rec_idx: int) -> bool | None:
    """Check if a tool's recommended index matches the chosen action."""
    if rec_idx is None:
        return None
    if rec_idx >= len(step.valid_actions):
        return False
    return step.valid_actions[rec_idx] == step.chosen_action


def format_analysis(results: list[GameResult]) -> str:
    """Format a human-readable analysis report."""
    lines: list[str] = []

    agreement = analyze_tool_agreement(results)
    lines.append("=== Tool Agreement ===")
    lines.append(f"Steps with tools: {agreement.total_steps_with_tools}")
    if agreement.total_steps_with_tools > 0:
        n = agreement.total_steps_with_tools
        lines.append(f"  All agree:    {agreement.all_agree:4d} ({agreement.all_agree / n:.1%})")
        lines.append(
            f"  Disagree:     {agreement.some_disagree:4d} ({agreement.some_disagree / n:.1%})"
        )
        lines.append(f"  All None:     {agreement.all_none:4d} ({agreement.all_none / n:.1%})")

    failures = analyze_failures(results)
    lines.append("\n=== Failure Taxonomy ===")
    lines.append(f"Total wrong steps: {failures.total_wrong}")
    if failures.total_wrong > 0:
        n = failures.total_wrong
        lines.append(f"  All tools wrong:     {failures.all_tools_wrong:4d} ({failures.all_tools_wrong / n:.1%})")
        lines.append(f"  Correct outvoted:    {failures.correct_tool_outvoted:4d} ({failures.correct_tool_outvoted / n:.1%})")
        lines.append(f"  All tools None:      {failures.all_tools_none:4d} ({failures.all_tools_none / n:.1%})")
        lines.append(f"  No tools queried:    {failures.no_tools_queried:4d} ({failures.no_tools_queried / n:.1%})")

    cats = category_distribution(results)
    lines.append("\n=== Category Distribution ===")
    for cat, count in cats.most_common():
        lines.append(f"  {cat or 'None':<16} {count:4d}")

    return "\n".join(lines)
