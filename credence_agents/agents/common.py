"""Shared agent interface and result types."""

from __future__ import annotations

from typing import NamedTuple, Protocol


class DecisionStep(NamedTuple):
    """One step in the agent's decision trace."""
    step: int
    eu_submit: float
    eu_abstain: float
    eu_query: dict[int, float]  # tool_idx -> net EU (VOI - cost)
    chosen_action: str          # "submit(idx)", "abstain", "query(idx)"


class AgentResult(NamedTuple):
    answer: int | None          # candidate index, or None for abstain
    tools_queried: tuple[int, ...]
    confidence: float           # P(correct) at submission time
    decision_trace: tuple[DecisionStep, ...]


class AgentInterface(Protocol):
    """High-level agent interface for solve_question style usage."""

    def solve_question(
        self,
        question_text: str,
        candidates: tuple[str, ...],
        category_hint: str | None,
        tool_query_fn: "ToolQueryFn",
    ) -> AgentResult: ...


# tool_query_fn(tool_idx) -> int | None  (candidate index or None)
ToolQueryFn = type[None]  # just a type hint placeholder; actual type is Callable[[int], int | None]
