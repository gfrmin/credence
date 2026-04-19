# Role: body
"""Simple baseline routing strategies: random, always-cheapest, always-best.

These provide reference points for the credence-router benchmark.
No learning, no VOI calculations.
"""

from __future__ import annotations

import time

import numpy as np

from credence_router.answer import Answer
from credence_router.tool import Tool


class RandomSolver:
    """Randomly picks one tool per question."""

    name = "random-tool"

    def __init__(self, tools: list[Tool], seed: int = 42):
        self._tools = tools
        self._rng = np.random.default_rng(seed)

    def solve(
        self,
        question: str,
        candidates: tuple[str, ...],
        category_hint: str | None = None,
    ) -> Answer:
        t_start = time.monotonic()
        idx = int(self._rng.integers(len(self._tools)))
        tool = self._tools[idx]
        result = tool.query(question, candidates)
        wall_time = time.monotonic() - t_start

        return Answer(
            choice=result,
            choice_text=candidates[result] if result is not None else None,
            confidence=0.0,
            tools_used=(tool.name,),
            monetary_cost=tool.cost,
            effective_cost=tool.cost,
            wall_time=wall_time,
            reasoning=f"Randomly selected {tool.name}",
        )

    def report_outcome(self, correct: bool) -> None:
        pass


class AlwaysCheapestSolver:
    """Always queries the cheapest tool."""

    name = "always-cheapest"

    def __init__(self, tools: list[Tool]):
        self._tools = sorted(tools, key=lambda t: t.cost)

    def solve(
        self,
        question: str,
        candidates: tuple[str, ...],
        category_hint: str | None = None,
    ) -> Answer:
        t_start = time.monotonic()

        # Try tools in order of cost until one answers
        for tool in self._tools:
            result = tool.query(question, candidates)
            if result is not None:
                wall_time = time.monotonic() - t_start
                return Answer(
                    choice=result,
                    choice_text=candidates[result],
                    confidence=0.0,
                    tools_used=(tool.name,),
                    monetary_cost=tool.cost,
                    effective_cost=tool.cost,
                    wall_time=wall_time,
                    reasoning=f"Cheapest tool: {tool.name}",
                )

        wall_time = time.monotonic() - t_start
        return Answer(
            choice=None,
            choice_text=None,
            confidence=0.0,
            tools_used=(),
            monetary_cost=0.0,
            effective_cost=0.0,
            wall_time=wall_time,
            reasoning="No tool returned an answer",
        )

    def report_outcome(self, correct: bool) -> None:
        pass


class AlwaysBestSolver:
    """Always queries the most expensive (presumably best) tool."""

    name = "always-best"

    def __init__(self, tools: list[Tool]):
        self._tools = sorted(tools, key=lambda t: t.cost, reverse=True)

    def solve(
        self,
        question: str,
        candidates: tuple[str, ...],
        category_hint: str | None = None,
    ) -> Answer:
        t_start = time.monotonic()

        # Try tools in order of cost (most expensive first)
        for tool in self._tools:
            result = tool.query(question, candidates)
            if result is not None:
                wall_time = time.monotonic() - t_start
                return Answer(
                    choice=result,
                    choice_text=candidates[result],
                    confidence=0.0,
                    tools_used=(tool.name,),
                    monetary_cost=tool.cost,
                    effective_cost=tool.cost,
                    wall_time=wall_time,
                    reasoning=f"Best tool: {tool.name}",
                )

        wall_time = time.monotonic() - t_start
        return Answer(
            choice=None,
            choice_text=None,
            confidence=0.0,
            tools_used=(),
            monetary_cost=0.0,
            effective_cost=0.0,
            wall_time=wall_time,
            reasoning="No tool returned an answer",
        )

    def report_outcome(self, correct: bool) -> None:
        pass
