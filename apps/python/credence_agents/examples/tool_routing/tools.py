# Role: body
"""Tool profiles modelling real-world API costs, latencies, and reliabilities.

Four tools that mirror a realistic routing scenario:
  - calculator:  free, instant, only handles numerical questions
  - cheap_llm:   Claude Haiku — cheap, fast, decent on everything
  - expert_llm:  Claude Sonnet — pricier, slower, much better on reasoning
  - web_search:  Perplexity Sonar — expensive, slow, great on factual/recent
"""

from __future__ import annotations

from credence_agents.environment.categories import CATEGORIES
from credence_agents.environment.tools import ResponseType, SimulatedTool

# Real-world costs per short MC query (~200 input + 20 output tokens)
# Latency in seconds (used to compute effective cost)

TOOL_LATENCIES: dict[str, float] = {
    "calculator": 0.001,
    "cheap_llm": 0.200,
    "expert_llm": 0.400,
    "web_search": 0.800,
}

# Monetary cost in dollars
TOOL_MONETARY_COSTS: dict[str, float] = {
    "calculator": 0.0000,
    "cheap_llm": 0.0003,
    "expert_llm": 0.0010,
    "web_search": 0.0050,
}


def effective_cost(name: str, latency_weight: float = 0.0) -> float:
    """Compute effective cost = monetary + latency_weight * latency_seconds."""
    return TOOL_MONETARY_COSTS[name] + latency_weight * TOOL_LATENCIES[name]


def make_routing_tools(
    latency_weight: float = 0.0,
) -> tuple[SimulatedTool, SimulatedTool, SimulatedTool, SimulatedTool]:
    """Create 4 tools with real-world cost/reliability profiles.

    Args:
        latency_weight: $/second — converts latency to cost.
            E.g. 0.01 means "each second of latency costs me $0.01".
    """
    all_covered = {c: 1.0 for c in CATEGORIES}

    calculator = SimulatedTool(
        name="calculator",
        reliability_by_category={
            "factual": 0.0,
            "numerical": 0.95,
            "recent_events": 0.0,
            "misconceptions": 0.0,
            "reasoning": 0.0,
        },
        cost=effective_cost("calculator", latency_weight),
        coverage_by_category={
            "factual": 0.0,
            "numerical": 1.0,
            "recent_events": 0.0,
            "misconceptions": 0.0,
            "reasoning": 0.0,
        },
        no_answer_type=ResponseType.NOT_APPLICABLE,
    )

    cheap_llm = SimulatedTool(
        name="cheap_llm",
        reliability_by_category={
            "factual": 0.60,
            "numerical": 0.45,
            "recent_events": 0.25,
            "misconceptions": 0.50,
            "reasoning": 0.55,
        },
        cost=effective_cost("cheap_llm", latency_weight),
        coverage_by_category=dict(all_covered),
        no_answer_type=ResponseType.NO_RESULT,
    )

    expert_llm = SimulatedTool(
        name="expert_llm",
        reliability_by_category={
            "factual": 0.75,
            "numerical": 0.60,
            "recent_events": 0.35,
            "misconceptions": 0.80,
            "reasoning": 0.85,
        },
        cost=effective_cost("expert_llm", latency_weight),
        coverage_by_category=dict(all_covered),
        no_answer_type=ResponseType.NO_RESULT,
    )

    web_search = SimulatedTool(
        name="web_search",
        reliability_by_category={
            "factual": 0.80,
            "numerical": 0.20,
            "recent_events": 0.90,
            "misconceptions": 0.40,
            "reasoning": 0.30,
        },
        cost=effective_cost("web_search", latency_weight),
        coverage_by_category=dict(all_covered),
        no_answer_type=ResponseType.NO_RESULT,
    )

    return (calculator, cheap_llm, expert_llm, web_search)
