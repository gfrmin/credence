# Role: body
"""Simulated tools for testing and benchmarking without API keys.

Deterministic given a seed. Simulates tool reliability per category
by returning correct/wrong answers according to configured probabilities.

For benchmark use, provide an ``answer_key`` mapping question text to
correct candidate index so tools can simulate calibrated reliability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from credence_agents.environment.categories import CATEGORIES
from credence_router.categories import make_keyword_category_infer_fn

_CATEGORY_INFER_FN = make_keyword_category_infer_fn(CATEGORIES)


@dataclass
class SimulatedTool:
    """A simulated tool with configurable reliability and coverage per category.

    Args:
        name: Tool identifier.
        cost: Monetary cost per query ($).
        latency: Expected latency in seconds.
        reliability_by_category: P(correct | answers, category) per category name.
        coverage_by_category: P(returns an answer | category) per category name.
        answer_key: Mapping from question text to correct candidate index.
            Required for realistic benchmark simulation.
        seed: Random seed for deterministic simulation.
    """

    name: str
    cost: float
    latency: float
    reliability_by_category: dict[str, float]
    coverage_by_category: dict[str, float] = field(default_factory=dict)
    answer_key: dict[str, int] = field(default_factory=dict)
    seed: int = 42
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        # Default: full coverage where reliability > 0
        if not self.coverage_by_category:
            self.coverage_by_category = {
                cat: (1.0 if rel > 0 else 0.0) for cat, rel in self.reliability_by_category.items()
            }

    def query(self, question: str, candidates: tuple[str, ...]) -> int | None:
        """Simulate a tool query. Returns candidate index or None."""
        category = _infer_category(question)
        coverage = self.coverage_by_category.get(category, 0.0)

        if self._rng.random() >= coverage:
            return None

        reliability = self.reliability_by_category.get(category, 0.0)
        correct_idx = self.answer_key.get(question, hash(question) % len(candidates))

        if self._rng.random() < reliability:
            return correct_idx

        # Wrong answer: uniform over incorrect candidates
        wrong = [i for i in range(len(candidates)) if i != correct_idx]
        return wrong[int(self._rng.integers(len(wrong)))] if wrong else correct_idx

    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]:
        """P(returns an answer | category) for each category."""
        return np.array(
            [self.coverage_by_category.get(c, 0.0) for c in categories],
            dtype=np.float64,
        )


def _infer_category(question: str) -> str:
    """Keyword-based category inference for simulation."""
    probs = _CATEGORY_INFER_FN(question)
    return CATEGORIES[int(np.argmax(probs))]


def make_default_simulated_tools(
    seed: int = 42,
    questions: list | None = None,
) -> list[SimulatedTool]:
    """Create 4 simulated tools matching real-world API profiles.

    Args:
        seed: Random seed for deterministic simulation.
        questions: If provided, tools will use ground truth for calibrated
            reliability. Each element must have .text and .correct_index.
    """
    answer_key: dict[str, int] = {}
    if questions is not None:
        answer_key = {q.text: q.correct_index for q in questions}

    categories = ("factual", "numerical", "recent_events", "misconceptions", "reasoning")
    all_covered = {c: 1.0 for c in categories}

    return [
        SimulatedTool(
            name="calculator",
            cost=0.0,
            latency=0.001,
            reliability_by_category={
                "factual": 0.0,
                "numerical": 0.95,
                "recent_events": 0.0,
                "misconceptions": 0.0,
                "reasoning": 0.0,
            },
            coverage_by_category={
                "factual": 0.0,
                "numerical": 1.0,
                "recent_events": 0.0,
                "misconceptions": 0.0,
                "reasoning": 0.0,
            },
            answer_key=dict(answer_key),
            seed=seed,
        ),
        SimulatedTool(
            name="cheap_llm",
            cost=0.0003,
            latency=0.200,
            reliability_by_category={
                "factual": 0.60,
                "numerical": 0.45,
                "recent_events": 0.25,
                "misconceptions": 0.50,
                "reasoning": 0.55,
            },
            coverage_by_category=dict(all_covered),
            answer_key=dict(answer_key),
            seed=seed + 1,
        ),
        SimulatedTool(
            name="expert_llm",
            cost=0.001,
            latency=0.400,
            reliability_by_category={
                "factual": 0.75,
                "numerical": 0.60,
                "recent_events": 0.35,
                "misconceptions": 0.80,
                "reasoning": 0.85,
            },
            coverage_by_category=dict(all_covered),
            answer_key=dict(answer_key),
            seed=seed + 2,
        ),
        SimulatedTool(
            name="web_search",
            cost=0.005,
            latency=0.800,
            reliability_by_category={
                "factual": 0.80,
                "numerical": 0.20,
                "recent_events": 0.90,
                "misconceptions": 0.40,
                "reasoning": 0.30,
            },
            coverage_by_category=dict(all_covered),
            answer_key=dict(answer_key),
            seed=seed + 3,
        ),
    ]
