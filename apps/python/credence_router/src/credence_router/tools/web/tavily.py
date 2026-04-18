"""Tavily Search API adapter."""

from __future__ import annotations

import logging
import os

import httpx
import numpy as np
from numpy.typing import NDArray

from credence_router.tool import SearchResult

log = logging.getLogger(__name__)

TAVILY_API_URL = "https://api.tavily.com/search"

# Tavily strengths: research-grade extraction, good structured content.
# Moderate across categories, strongest on technical/synthesis.
_DEFAULT_COVERAGE = {
    "factual": 0.70,
    "recent_events": 0.55,
    "technical": 0.80,
    "synthesis": 0.75,
    "local": 0.35,
}


class TavilySearchTool:
    """Tavily Search API wrapper implementing SearchTool protocol."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 10.0,
        max_results: int = 5,
        search_depth: str = "basic",
    ):
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self._timeout = timeout
        self._max_results = max_results
        self._search_depth = search_depth

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def cost(self) -> float:
        return 0.004  # ~$4/1000 for basic

    @property
    def latency(self) -> float:
        return 2.0

    def search(self, query: str) -> SearchResult | None:
        resp = httpx.post(
            TAVILY_API_URL,
            json={
                "query": query,
                "max_results": self._max_results,
                "search_depth": self._search_depth,
                "api_key": self._api_key,
            },
            headers={"Content-Type": "application/json"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            log.info("Tavily returned no results for: %s", query)
            return None

        text_parts = []
        urls = []
        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            text_parts.append(f"{title}\n{content}")
            if url:
                urls.append(url)

        return SearchResult(
            text="\n\n".join(text_parts),
            urls=urls,
            provider="tavily",
            raw=data,
        )

    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]:
        return np.array([_DEFAULT_COVERAGE.get(c, 0.5) for c in categories])
