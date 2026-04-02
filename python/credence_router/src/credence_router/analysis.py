"""Analysis and reporting utilities for benchmark results."""

from __future__ import annotations

from credence_router.benchmark import BenchmarkResult


def per_category_accuracy(result: BenchmarkResult) -> dict[str, float]:
    """Compute accuracy per question category."""
    from collections import defaultdict

    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)

    for r in result.results:
        if r.was_correct is not None:
            total[r.category] += 1
            if r.was_correct:
                correct[r.category] += 1

    return {cat: correct[cat] / total[cat] if total[cat] > 0 else 0.0 for cat in sorted(total)}


def format_reliability_table(
    learned: dict[str, dict[str, float]],
) -> str:
    """Format learned reliability as a readable table."""
    if not learned:
        return "No reliability data"

    tools = list(learned.keys())
    categories = list(next(iter(learned.values())).keys())

    # Header
    cat_width = max(len(c) for c in categories)
    header = f"{'Tool':<16s}" + "".join(f"{c:>{cat_width + 2}s}" for c in categories)
    sep = "-" * len(header)
    lines = [header, sep]

    for tool in tools:
        vals = "".join(f"{learned[tool].get(c, 0.0):>{cat_width + 2}.2f}" for c in categories)
        lines.append(f"{tool:<16s}{vals}")

    return "\n".join(lines)


def format_learning_curve(results: list[BenchmarkResult]) -> str:
    """Show how accuracy evolves over the question sequence."""
    lines: list[str] = []
    for r in results:
        lines.append(f"\n{r.agent_name}:")
        running_correct = 0
        for i, qr in enumerate(r.results, 1):
            if qr.was_correct:
                running_correct += 1
            if i % 10 == 0 or i == len(r.results):
                lines.append(
                    f"  Q{i:3d}: {running_correct}/{i} ({100 * running_correct / i:.0f}%)"
                )
    return "\n".join(lines)
