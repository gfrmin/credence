# Role: body
"""Perplexity tool: web search via Perplexity Sonar API.

Stronger on factual and recent events, weaker on reasoning.
Cost dominated by search retrieval ($0.005/query).
"""

from __future__ import annotations

import os
import re

import numpy as np
from numpy.typing import NDArray


class PerplexityTool:
    """Perplexity Sonar API for web-search-augmented question answering.

    Args:
        cost: Monetary cost per query (default: $0.005).
        latency: Expected latency in seconds (default: 0.8s).
        api_key: Perplexity API key (default: $PERPLEXITY_API_KEY).
        model: Perplexity model ID (default: sonar).
    """

    def __init__(
        self,
        cost: float = 0.005,
        latency: float = 0.800,
        api_key: str | None = None,
        model: str = "sonar",
    ):
        self._cost = cost
        self._latency = latency
        self._api_key = api_key or os.environ.get("PERPLEXITY_API_KEY", "")
        self._model = model

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def latency(self) -> float:
        return self._latency

    def query(self, question: str, candidates: tuple[str, ...]) -> int | None:
        """Search the web via Perplexity and answer the question."""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx package required for PerplexityTool. "
                "Install with: pip install credence-router[perplexity]"
            )

        if not self._api_key:
            raise ValueError("PERPLEXITY_API_KEY not set")

        letters = [chr(65 + i) for i in range(len(candidates))]
        options = "\n".join(f"{ltr}) {c}" for ltr, c in zip(letters, candidates))

        prompt = (
            f"Question: {question}\n\n"
            f"Options:\n{options}\n\n"
            f"Which letter is correct? Reply with ONLY the letter."
        )

        response = httpx.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 5,
            },
            timeout=30.0,
        )
        response.raise_for_status()

        text = response.json()["choices"][0]["message"]["content"].strip().upper()
        return _parse_letter(text, len(candidates))

    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]:
        """Perplexity covers all categories."""
        return np.ones(len(categories), dtype=np.float64)


def _parse_letter(text: str, n_candidates: int) -> int | None:
    """Parse a letter response (A, B, C, D) to candidate index."""
    match = re.match(r"^([A-Z])", text)
    if match:
        idx = ord(match.group(1)) - 65
        if 0 <= idx < n_candidates:
            return idx
    return None
