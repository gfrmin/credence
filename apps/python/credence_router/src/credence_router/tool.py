# Role: body
"""Tool protocol and base class for credence-router.

Tools answer multiple-choice questions. Each tool declares its monetary cost,
expected latency, and per-category coverage probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Tool(Protocol):
    """A tool that can answer multiple-choice questions."""

    @property
    def name(self) -> str: ...

    @property
    def cost(self) -> float:
        """Monetary cost per query in dollars."""
        ...

    @property
    def latency(self) -> float:
        """Expected latency in seconds."""
        ...

    def query(self, question: str, candidates: tuple[str, ...]) -> int | None:
        """Return index of best candidate, or None if can't answer."""
        ...

    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]:
        """P(returns an answer | category) for each category."""
        ...


@dataclass(frozen=True)
class SearchResult:
    """Result from a web search provider."""

    text: str
    urls: list[str] = field(default_factory=list)
    provider: str = ""
    raw: dict = field(default_factory=dict)


@runtime_checkable
class SearchTool(Protocol):
    """A tool that performs web searches and returns free-text results."""

    @property
    def name(self) -> str: ...

    @property
    def cost(self) -> float:
        """Monetary cost per query in dollars."""
        ...

    @property
    def latency(self) -> float:
        """Expected latency in seconds."""
        ...

    def search(self, query: str) -> SearchResult | None:
        """Execute a search query. Returns SearchResult or None if failed."""
        ...

    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]:
        """P(returns a useful result | category) for each category."""
        ...
