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

import json
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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

        # Outcome queue: async judge appends here, route() drains synchronously.
        # This avoids calling Julia from async context (single-threaded deadlock).
        self._pending_outcomes: deque[tuple[Observation, RouteDecision]] = deque()

    def route(self, text: str, category_hint: str | None = None) -> RouteDecision:
        """Route a query to the best provider."""
        t_start = time.monotonic()

        # Drain pending outcomes (from async judge) before deciding.
        # All Julia calls happen here, in the sync request handler.
        self._drain_outcomes()

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

    def queue_outcome(self, observation: Observation) -> None:
        """Queue an outcome for processing on the next route() call.

        Thread-safe: appends to a deque. No Julia calls — safe from async context.
        """
        if self._last_decision is None:
            return
        self._pending_outcomes.append((observation, self._last_decision))

    def _drain_outcomes(self) -> None:
        """Process all queued outcomes synchronously. Called from route()."""
        while self._pending_outcomes:
            observation, decision = self._pending_outcomes.popleft()
            self._apply_outcome(observation, decision)

    def _apply_outcome(self, observation: Observation, decision: RouteDecision) -> None:
        """Update beliefs from an observed outcome. Calls Julia — must be sync."""
        if observation.quality_score is not None:
            quality = observation.quality_score
        elif observation.useful:
            quality = 0.8
        else:
            quality = 0.2

        cat_w_jl = self._bridge._make_float_vector(decision.category_weights)

        self._state = self._bridge.call_router(
            "router-observe",
            self._state,
            decision.provider_idx,
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

    # --- State persistence ---

    def save_state(self, path: str | Path) -> None:
        """Persist opaque DSL state to disk via Julia Serialization."""
        path = Path(path)
        jl = self._bridge.jl
        _serialize = jl.seval(
            '(s, p) -> open(io -> (using Serialization; serialize(io, s)), p, "w")'
        )
        _serialize(self._state, str(path))

        # Sidecar: last decision (needed to attribute queued outcomes on reload)
        if self._last_decision is not None:
            sidecar = path.with_suffix(".json")
            sidecar.write_text(json.dumps({
                "provider_idx": self._last_decision.provider_idx,
                "provider_name": self._last_decision.provider_name,
                "category_weights": self._last_decision.category_weights,
            }))
        log.info("LLM state saved to %s", path)

    def load_state(self, path: str | Path) -> None:
        """Restore opaque DSL state from disk via Julia Serialization."""
        path = Path(path)
        if not path.exists():
            return
        jl = self._bridge.jl
        _deserialize = jl.seval(
            'p -> open(io -> (using Serialization; deserialize(io)), p, "r")'
        )
        self._state = _deserialize(str(path))

        # Restore last decision from sidecar
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            data = json.loads(sidecar.read_text())
            self._last_decision = RouteDecision(
                provider_idx=data["provider_idx"],
                provider_name=data["provider_name"],
                category_weights=data["category_weights"],
            )
        log.info("LLM state restored from %s", path)
