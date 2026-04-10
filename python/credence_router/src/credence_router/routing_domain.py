"""RoutingDomain: provider routing via the DSL.

The host holds one opaque state and calls three DSL functions:
  router-decide  — which provider to use
  router-observe — update beliefs from quality score
  make-router-state — create initial beliefs

All Bayesian mechanics (condition, expect, optimise) live in
the DSL (examples/router.bdsl). The host knows nothing about
BetaMeasure, MixtureMeasure, or any inference machinery.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from credence_agents.julia_bridge import CredenceBridge

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observation:
    """Outcome observation from a single routed request."""

    completed: bool
    error_type: str | None = None
    ttft_seconds: float = 0.0
    total_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    truncated: bool = False
    cost_usd: float = 0.0
    quality_score: float | None = None  # 0.0-1.0 continuous quality
    response_text: str = ""

    @property
    def useful(self) -> bool:
        return self.completed and not self.truncated and self.error_type is None


@dataclass(frozen=True)
class RouteDecision:
    """Result of a routing decision."""

    provider_idx: int
    provider_name: str
    category_weights: list[float]
    wall_time: float = 0.0


class RoutingDomain:
    """Provider routing backed by the DSL.

    The host holds one opaque DSL state and calls:
    - router-decide(state, cat_weights, costs, reward) → provider index
    - router-observe(state, provider_idx, cat_weights, quality) → new state
    """

    def __init__(
        self,
        bridge: CredenceBridge,
        provider_names: list[str],
        costs: list[float],
        categories: tuple[str, ...],
        category_infer: Callable[[str], NDArray[np.float64]],
        reward: float = 1.0,
    ):
        self._bridge = bridge
        self._provider_names = provider_names
        self._costs = bridge._make_float_vector(costs)
        self._categories = categories
        self._category_infer = category_infer
        self._reward = reward

        # Opaque DSL state — host doesn't know what's inside
        self._state = bridge.call_router(
            "make-router-state", len(provider_names), len(categories),
        )
        self._last_decision: RouteDecision | None = None
        self._cached_reliability: dict[str, dict[str, float]] = {}

    def route(self, text: str, category_hint: str | None = None) -> RouteDecision:
        """Route a query to the best provider."""
        t_start = time.monotonic()

        if category_hint and category_hint in self._categories:
            cat_weights = [0.0] * len(self._categories)
            cat_weights[self._categories.index(category_hint)] = 1.0
        else:
            cat_weights = self._category_infer(text).tolist()

        cat_w_jl = self._bridge._make_float_vector(cat_weights)

        # One DSL call — the DSL does EU maximisation
        provider_idx = int(self._bridge.call_router(
            "router-decide", self._state, cat_w_jl, self._costs, self._reward,
        ))

        decision = RouteDecision(
            provider_idx=provider_idx,
            provider_name=self._provider_names[provider_idx],
            category_weights=cat_weights,
            wall_time=time.monotonic() - t_start,
        )
        self._last_decision = decision

        # Cache reliability (safe here — we're in the sync request handler)
        try:
            self._cached_reliability = self._compute_reliability()
        except Exception:
            pass

        return decision

    def report_outcome(self, observation: Observation) -> None:
        """Update beliefs from an observed outcome."""
        if self._last_decision is None:
            return

        # Determine quality signal
        if observation.quality_score is not None:
            quality = observation.quality_score  # continuous — DSL handles it
        elif observation.useful:
            quality = 0.8  # default "good" signal
        else:
            quality = 0.2  # default "bad" signal

        cat_w_jl = self._bridge._make_float_vector(self._last_decision.category_weights)

        # One DSL call — the DSL does conditioning
        self._state = self._bridge.call_router(
            "router-observe",
            self._state,
            self._last_decision.provider_idx,
            cat_w_jl,
            float(quality),
        )

    def _compute_reliability(self) -> dict[str, dict[str, float]]:
        """Compute per-provider per-category expected reliability from DSL state."""
        result: dict[str, dict[str, float]] = {}
        _extract = self._bridge.jl.seval("(s, i, j) -> s[i+1][j+1]")  # noqa: S307
        for i, name in enumerate(self._provider_names):
            result[name] = {}
            for j, cat in enumerate(self._categories):
                try:
                    provider_beliefs = _extract(self._state, i, j)
                    mean = float(self._bridge.call_router(
                        "expected-reliability", provider_beliefs,
                    ))
                    result[name][cat] = mean
                except Exception:
                    result[name][cat] = 0.5
        return result

    @property
    def learned_reliability(self) -> dict[str, dict[str, float]]:
        """Per-provider per-category expected reliability (cached, safe for async)."""
        return self._cached_reliability

    @property
    def provider_names(self) -> list[str]:
        return list(self._provider_names)

    @property
    def categories(self) -> tuple[str, ...]:
        return self._categories

    @property
    def last_decision(self) -> RouteDecision | None:
        return self._last_decision
