# Role: body
"""Simulated tools with category-dependent reliability and coverage.

Each tool is a pure data structure. The query function is standalone and
deterministic given an RNG — no hidden state, no side effects.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import NamedTuple

import numpy as np

from credence_agents.environment.categories import CATEGORIES
from credence_agents.inference.voi import ToolConfig


class ResponseType(Enum):
    ANSWER = auto()
    NO_RESULT = auto()
    NOT_APPLICABLE = auto()


class ToolResponse(NamedTuple):
    candidate_idx: int | None
    response_type: ResponseType


class SimulatedTool(NamedTuple):
    name: str
    reliability_by_category: dict[str, float]
    cost: float
    coverage_by_category: dict[str, float]
    no_answer_type: ResponseType  # NO_RESULT (Tool B) or NOT_APPLICABLE (Tool C)


def query_tool(
    tool: SimulatedTool,
    question: "Question",  # forward ref to avoid circular import
    rng: np.random.Generator,
) -> ToolResponse:
    """Query a simulated tool on a question. Pure function, deterministic given rng.

    1. Check coverage — if not covered, return no_answer_type with no candidate.
    2. If covered and rng < reliability → correct answer.
    3. Otherwise → uniform random wrong answer from the 3 incorrect candidates.
    """
    category = question.category
    coverage = tool.coverage_by_category.get(category, 0.0)

    if rng.random() >= coverage:
        return ToolResponse(candidate_idx=None, response_type=tool.no_answer_type)

    reliability = tool.reliability_by_category.get(category, 0.0)
    if rng.random() < reliability:
        return ToolResponse(candidate_idx=question.correct_index, response_type=ResponseType.ANSWER)

    # Wrong answer: uniform over the 3 incorrect candidates
    wrong_indices = [i for i in range(len(question.candidates)) if i != question.correct_index]
    chosen = wrong_indices[int(rng.integers(len(wrong_indices)))]
    return ToolResponse(candidate_idx=chosen, response_type=ResponseType.ANSWER)


def make_spec_tools() -> tuple[SimulatedTool, SimulatedTool, SimulatedTool, SimulatedTool]:
    """Create the 4 tools from SPEC §2.2 with exact reliability tables."""
    all_covered = {c: 1.0 for c in CATEGORIES}

    tool_a = SimulatedTool(
        name="quick_search",
        reliability_by_category={
            "factual": 0.70, "numerical": 0.20, "recent_events": 0.65,
            "misconceptions": 0.25, "reasoning": 0.40,
        },
        cost=1.0,
        coverage_by_category=dict(all_covered),
        no_answer_type=ResponseType.NO_RESULT,
    )

    tool_b = SimulatedTool(
        name="knowledge_base",
        reliability_by_category={
            "factual": 0.92, "numerical": 0.40, "recent_events": 0.55,
            "misconceptions": 0.88, "reasoning": 0.45,
        },
        cost=2.0,
        coverage_by_category={
            "factual": 0.65, "numerical": 0.30, "recent_events": 0.35,
            "misconceptions": 0.55, "reasoning": 0.20,
        },
        no_answer_type=ResponseType.NO_RESULT,
    )

    tool_c = SimulatedTool(
        name="calculator",
        reliability_by_category={
            "factual": 0.0, "numerical": 1.0, "recent_events": 0.0,
            "misconceptions": 0.0, "reasoning": 0.0,
        },
        cost=1.0,
        coverage_by_category={
            "factual": 0.0, "numerical": 1.0, "recent_events": 0.0,
            "misconceptions": 0.0, "reasoning": 0.0,
        },
        no_answer_type=ResponseType.NOT_APPLICABLE,
    )

    tool_d = SimulatedTool(
        name="llm_direct",
        reliability_by_category={
            "factual": 0.65, "numerical": 0.50, "recent_events": 0.45,
            "misconceptions": 0.40, "reasoning": 0.72,
        },
        cost=2.0,
        coverage_by_category=dict(all_covered),
        no_answer_type=ResponseType.NO_RESULT,
    )

    return (tool_a, tool_b, tool_c, tool_d)


def tool_config_for(tool: SimulatedTool) -> ToolConfig:
    """Convert a SimulatedTool to the inference layer's ToolConfig."""
    coverage_array = np.array(
        [tool.coverage_by_category.get(c, 0.0) for c in CATEGORIES],
        dtype=np.float64,
    )
    return ToolConfig(cost=tool.cost, coverage_by_category=coverage_array)
