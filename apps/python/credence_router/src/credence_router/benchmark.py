# Role: body
"""Benchmark runner: compares routing strategies on the question bank.

Measures accuracy, tool cost, routing cost, total cost, and latency
for each agent strategy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from credence_agents.environment.questions import Question, get_questions

from credence_router.answer import Answer


@dataclass(frozen=True)
class QuestionResult:
    """Result for a single question."""

    question_id: str
    category: str
    submitted: int | None
    correct_index: int
    was_correct: bool | None
    tools_used: tuple[str, ...]
    monetary_cost: float
    effective_cost: float
    wall_time: float


@dataclass
class BenchmarkResult:
    """Aggregate results for one agent across all questions."""

    agent_name: str
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        answered = [r for r in self.results if r.was_correct is not None]
        if not answered:
            return 0.0
        return sum(1 for r in answered if r.was_correct) / len(answered)

    @property
    def total_monetary_cost(self) -> float:
        return sum(r.monetary_cost for r in self.results)

    @property
    def total_effective_cost(self) -> float:
        return sum(r.effective_cost for r in self.results)

    @property
    def avg_wall_time(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.wall_time for r in self.results) / len(self.results)

    @property
    def avg_tools_per_question(self) -> float:
        if not self.results:
            return 0.0
        return sum(len(r.tools_used) for r in self.results) / len(self.results)

    @property
    def tool_usage(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for r in self.results:
            counter.update(r.tools_used)
        return dict(counter)


class SolverProtocol(Protocol):
    """Protocol for anything that can solve questions."""

    name: str

    def solve(
        self,
        question: str,
        candidates: tuple[str, ...],
        category_hint: str | None = None,
    ) -> Answer: ...

    def report_outcome(self, correct: bool) -> None: ...


def run_benchmark(
    solver: SolverProtocol,
    questions: list[Question] | None = None,
    seed: int = 42,
) -> BenchmarkResult:
    """Run a solver through the question bank and record results."""
    if questions is None:
        questions = get_questions(seed=seed)

    result = BenchmarkResult(agent_name=solver.name)

    for q in questions:
        answer = solver.solve(
            question=q.text,
            candidates=q.candidates,
            category_hint=q.category,
        )

        was_correct = answer.choice == q.correct_index if answer.choice is not None else None
        solver.report_outcome(was_correct is True)

        result.results.append(
            QuestionResult(
                question_id=q.id,
                category=q.category,
                submitted=answer.choice,
                correct_index=q.correct_index,
                was_correct=was_correct,
                tools_used=answer.tools_used,
                monetary_cost=answer.monetary_cost,
                effective_cost=answer.effective_cost,
                wall_time=answer.wall_time,
            )
        )

    return result


def format_comparison_table(results: list[BenchmarkResult]) -> str:
    """Format a comparison table of benchmark results."""
    header = (
        f"{'Agent':<22s} {'Accuracy':>8s} {'Cost$':>8s} "
        f"{'Eff.Cost$':>10s} {'Latency':>8s} {'Tools/Q':>8s}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        lines.append(
            f"{r.agent_name:<22s} {r.accuracy:>7.1%} "
            f"${r.total_monetary_cost:>6.3f} "
            f"${r.total_effective_cost:>8.3f} "
            f"{r.avg_wall_time:>6.1f}s  "
            f"{r.avg_tools_per_question:>6.2f}"
        )

    return "\n".join(lines)
