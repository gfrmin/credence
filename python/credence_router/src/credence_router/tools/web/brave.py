"""Brave Search API adapter."""

from __future__ import annotations

import logging
import os

import httpx
import numpy as np
from numpy.typing import NDArray

from credence_router.tool import SearchResult

log = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Brave strengths: broad web coverage, good for factual queries, decent for technical.
# Weaker on very recent events and synthesis.
_DEFAULT_COVERAGE = {
    "factual": 0.85,
    "recent_events": 0.50,
    "technical": 0.75,
    "synthesis": 0.40,
    "local": 0.60,
}


class BraveSearchTool:
    """Brave Search API wrapper implementing SearchTool protocol."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 10.0,
        count: int = 5,
    ):
        self._api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self._timeout = timeout
        self._count = count

    @property
    def name(self) -> str:
        return "brave"

    @property
    def cost(self) -> float:
        return 0.005  # ~$5/1000 queries on paid plan; free tier = 0

    @property
    def latency(self) -> float:
        return 1.0

    def search(self, query: str) -> SearchResult | None:
        resp = httpx.get(
            BRAVE_SEARCH_URL,
            params={"q": query, "count": self._count},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self._api_key,
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        web_results = data.get("web", {}).get("results", [])
        if not web_results:
            log.info("Brave returned no results for: %s", query)
            return None

        text_parts = []
        urls = []
        for r in web_results:
            title = r.get("title", "")
            description = r.get("description", "")
            url = r.get("url", "")
            text_parts.append(f"{title}\n{description}")
            if url:
                urls.append(url)

        return SearchResult(
            text="\n\n".join(text_parts),
            urls=urls,
            provider="brave",
            raw=data,
        )

    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]:
        return np.array([_DEFAULT_COVERAGE.get(c, 0.5) for c in categories])
