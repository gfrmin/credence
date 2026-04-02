"""Tool protocol and base class for credence-router.

Tools answer multiple-choice questions. Each tool declares its monetary cost,
expected latency, and per-category coverage probabilities.
"""

from __future__ import annotations

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
