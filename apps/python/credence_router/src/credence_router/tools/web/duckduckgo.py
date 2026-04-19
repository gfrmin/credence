# Role: body
"""DuckDuckGo search adapter (no API key required)."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from credence_router.tool import SearchResult

log = logging.getLogger(__name__)

# DuckDuckGo strengths: broad web coverage, free, fast.
# Weaker on recent events (stale index) and synthesis (no AI layer).
_DEFAULT_COVERAGE = {
    "factual": 0.70,
    "recent_events": 0.35,
    "technical": 0.60,
    "synthesis": 0.35,
    "local": 0.55,
}


class DuckDuckGoSearchTool:
    """DuckDuckGo search wrapper implementing SearchTool protocol."""

    def __init__(self, max_results: int = 5):
        self._max_results = max_results

    @property
    def name(self) -> str:
        return "duckduckgo"

    @property
    def cost(self) -> float:
        return 0.0

    @property
    def latency(self) -> float:
        return 1.0

    def search(self, query: str) -> SearchResult | None:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=self._max_results)
        if not results:
            log.info("DuckDuckGo returned no results for: %s", query)
            return None

        text_parts = []
        urls = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            text_parts.append(f"{title}\n{body}")
            if href:
                urls.append(href)

        return SearchResult(
            text="\n\n".join(text_parts),
            urls=urls,
            provider="duckduckgo",
            raw={"results": results},
        )

    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]:
        return np.array([_DEFAULT_COVERAGE.get(c, 0.5) for c in categories])
