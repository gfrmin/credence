"""Adapts LangChain tools for credence-router's VOI-based selection.

Routing probes are free (keyword matching only, no LLM call).
The Router treats tool selection as multiple-choice: candidates are tool names,
and each adapter votes for its tool when the question keywords match.
"""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

from credence_agents.environment.categories import CATEGORIES
from credence_router.categories import make_keyword_category_infer_fn

_CATEGORY_INFER_FN = make_keyword_category_infer_fn(CATEGORIES)

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "and",
        "but",
        "or",
        "not",
        "so",
        "if",
        "when",
        "what",
        "which",
        "who",
        "how",
        "this",
        "that",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "you",
        "your",
        "he",
        "she",
        "his",
        "her",
        "my",
        "our",
    }
)


def _extract_keywords(text: str) -> frozenset[str]:
    """Extract lowercase alphabetic tokens >= 3 chars, excluding stop words."""
    return frozenset(t for t in re.findall(r"[a-z]{3,}", text.lower()) if t not in _STOP_WORDS)


class ToolRoutingAdapter:
    """Adapts a LangChain tool for credence-router's VOI-based selection.

    The Router treats tool selection as multiple-choice: candidates are tool names,
    and each adapter votes for its tool when the question matches its coverage.
    Routing probes are free (keyword matching only, no LLM call).
    """

    def __init__(
        self,
        lc_tool,
        tool_index: int,
        num_tools: int,
        cost: float = 0.0,
        latency: float = 0.0,
    ):
        self._lc_tool = lc_tool
        self._tool_index = tool_index
        self._num_tools = num_tools
        self._cost = cost
        self._latency = latency
        desc = f"{lc_tool.name} {getattr(lc_tool, 'description', '')}"
        self._desc_keywords = _extract_keywords(desc)

    @property
    def name(self) -> str:
        return self._lc_tool.name

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def latency(self) -> float:
        return self._latency

    def query(self, question: str, candidates: tuple[str, ...]) -> int | None:
        """Vote for this tool if question keywords overlap with description.

        Returns this tool's index in the candidate list if relevant, None otherwise.
        Free computation, no LLM call.
        """
        question_kw = _extract_keywords(question)
        if question_kw & self._desc_keywords:
            return self._tool_index
        return None

    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]:
        """Estimate per-category coverage from tool description keywords.

        Uses the same keyword-based category inference as credence's environment.
        Floor at 0.1 so VOI can explore all tools initially.
        """
        desc = f"{self._lc_tool.name} {getattr(self._lc_tool, 'description', '')}"
        probs = _CATEGORY_INFER_FN(desc)
        return np.clip(probs, 0.1, 1.0).astype(np.float64)
