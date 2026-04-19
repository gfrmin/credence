# Role: body
"""KeywordCategoryTool: zero-cost Tool that votes for a category on regex match."""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray


class KeywordCategoryTool:
    """Votes for a specific category when its keywords match the query.

    Used as a tool in a category-inference Router. Each instance wraps one
    regex pattern and votes for the corresponding category on match, or
    abstains (returns None). Zero cost and zero latency.
    """

    def __init__(
        self,
        category_name: str,
        pattern: re.Pattern,
        categories: tuple[str, ...],
    ):
        self._category = category_name
        self._pattern = pattern
        self._categories = categories

    @property
    def name(self) -> str:
        return f"{self._category}_keywords"

    @property
    def cost(self) -> float:
        return 0.0

    @property
    def latency(self) -> float:
        return 0.0

    def query(self, question: str, candidates: tuple[str, ...]) -> int | None:
        if self._pattern.search(question):
            try:
                return candidates.index(self._category)
            except ValueError:
                return None
        return None

    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]:
        return np.ones(len(categories), dtype=np.float64)
