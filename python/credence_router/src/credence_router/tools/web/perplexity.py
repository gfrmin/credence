"""Perplexity Search API adapter."""

from __future__ import annotations

import os

import httpx
import numpy as np
from numpy.typing import NDArray

from credence_router.tool import SearchResult

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# Perplexity strengths: synthesised answers, excellent on recent events,
# good at combining multiple sources. Weaker on raw factual lookups.
_DEFAULT_COVERAGE = {
    "factual": 0.65,
    "recent_events": 0.90,
    "technical": 0.70,
    "synthesis": 0.85,
    "local": 0.45,
}


class PerplexitySearchTool:
    """Perplexity search API wrapper implementing SearchTool protocol."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "sonar",
        timeout: float = 15.0,
    ):
        self._api_key = api_key or os.environ.get("PERPLEXITY_API_KEY", "")
        self._model = model
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def cost(self) -> float:
        return 0.005  # ~$5/1000 requests for sonar

    @property
    def latency(self) -> float:
        return 3.0  # synthesis takes longer

    def search(self, query: str) -> SearchResult | None:
        try:
            resp = httpx.post(
                PERPLEXITY_API_URL,
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": query}],
                },
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            text = data["choices"][0]["message"]["content"]
            citations = data.get("citations", [])

            return SearchResult(
                text=text,
                urls=citations,
                provider="perplexity",
                raw=data,
            )
        except (httpx.HTTPError, KeyError, IndexError, ValueError):
            return None

    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]:
        return np.array([_DEFAULT_COVERAGE.get(c, 0.5) for c in categories])
