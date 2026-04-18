"""Answer dataclass returned by Router.solve()."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Answer:
    """Result of routing a question through credence-router.

    Contains the chosen answer, confidence, cost breakdown, and a full
    reasoning trace showing why each tool was or wasn't queried.
    """

    choice: int | None
    """Candidate index, or None if abstained."""

    choice_text: str | None
    """The actual answer text."""

    confidence: float
    """Posterior probability of the chosen answer being correct."""

    tools_used: tuple[str, ...]
    """Names of tools that were queried."""

    monetary_cost: float
    """Sum of tool monetary costs in dollars."""

    effective_cost: float
    """Monetary cost + latency-weighted cost."""

    wall_time: float
    """Actual elapsed time in seconds."""

    reasoning: str
    """Human-readable VOI trace."""

    answer_posterior: tuple[float, ...] = field(default_factory=tuple)
    """Full probability distribution over candidates at decision time."""

    tool_responses: tuple[tuple[int, int | None], ...] = field(default_factory=tuple)
    """Per-tool responses: ((tool_idx, candidate_idx_or_None), ...). Only tools that were queried appear."""

    decision_trace: tuple[dict, ...] = field(default_factory=tuple)
    """Machine-readable trace for analysis."""
