# Role: body
"""Claude tool: routes questions to Anthropic's Claude API.

Parameterised by model — use ClaudeTool("haiku") for cheap/fast routing
or ClaudeTool("sonnet") for higher accuracy at higher cost.
"""

from __future__ import annotations

import os
import re

import numpy as np
from numpy.typing import NDArray

# Model profiles: (cost_per_query, latency_seconds)
_MODEL_PROFILES: dict[str, tuple[float, float]] = {
    "haiku": (0.0003, 0.200),
    "sonnet": (0.001, 0.400),
}

# Model ID mapping
_MODEL_IDS: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
}


class ClaudeTool:
    """Claude API tool for answering multiple-choice questions.

    Args:
        model: "haiku" or "sonnet".
        cost: Override monetary cost (default: from model profile).
        latency: Override expected latency (default: from model profile).
        api_key: Anthropic API key (default: $ANTHROPIC_API_KEY).
    """

    def __init__(
        self,
        model: str = "haiku",
        cost: float | None = None,
        latency: float | None = None,
        api_key: str | None = None,
    ):
        if model not in _MODEL_PROFILES:
            raise ValueError(f"Unknown model: {model}. Choose from {list(_MODEL_PROFILES)}")

        profile_cost, profile_latency = _MODEL_PROFILES[model]
        self._model = model
        self._model_id = _MODEL_IDS[model]
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._name = f"claude_{model}"
        self._cost = cost if cost is not None else profile_cost
        self._latency = latency if latency is not None else profile_latency

    @property
    def name(self) -> str:
        return self._name

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def latency(self) -> float:
        return self._latency

    def query(self, question: str, candidates: tuple[str, ...]) -> int | None:
        """Ask Claude to answer a multiple-choice question."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for ClaudeTool. "
                "Install with: pip install credence-router[anthropic]"
            )

        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        letters = [chr(65 + i) for i in range(len(candidates))]
        options = "\n".join(f"{ltr}) {c}" for ltr, c in zip(letters, candidates))

        prompt = (
            f"Question: {question}\n\n"
            f"Options:\n{options}\n\n"
            f"Which letter is correct? Reply with ONLY the letter."
        )

        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model_id,
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip().upper()
        return _parse_letter(text, len(candidates))

    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]:
        """Claude covers all categories."""
        return np.ones(len(categories), dtype=np.float64)


def _parse_letter(text: str, n_candidates: int) -> int | None:
    """Parse a letter response (A, B, C, D) to candidate index."""
    match = re.match(r"^([A-Z])", text)
    if match:
        idx = ord(match.group(1)) - 65
        if 0 <= idx < n_candidates:
            return idx
    return None
