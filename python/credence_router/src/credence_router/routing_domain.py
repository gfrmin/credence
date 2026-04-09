"""RoutingDomain: DSL-backed provider routing via BayesianSelector.

Extends BayesianSelector to route requests to the best provider.
Each provider is an "option" in the selector's vocabulary. Category
inference maps query text to a distribution over categories. The
selector's EU maximisation picks the best provider. Outcome observations
update beliefs via the Julia DSL.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from credence_agents.agents.bayesian_selector import BayesianSelector
from credence_agents.inference.voi import ScoringRule, ToolConfig
from credence_agents.julia_bridge import CredenceBridge

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observation:
    """Outcome observation from a single routed request."""

    completed: bool
    error_type: str | None = None  # rate_limit, server_error, auth, timeout
    ttft_seconds: float = 0.0
    total_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    truncated: bool = False
    cost_usd: float = 0.0
    quality_score: float | None = None  # 0.0-1.0 continuous quality (from LLM judge)
    response_text: str = ""  # buffered response for judging

    @property
    def useful(self) -> bool:
        return self.completed and not self.truncated and self.error_type is None


@dataclass(frozen=True)
class RouteDecision:
    """Result of a routing decision."""

    provider_idx: int
    provider_name: str
    category_weights: list[float]
    reasoning: str
    wall_time: float = 0.0


class RoutingDomain:
    """DSL-backed provider routing via EU maximisation.

    Wraps BayesianSelector with:
    - Named providers (not just indices)
    - Category inference from query text
    - Observation-based belief updates
    - Introspection (learned reliability per provider per category)
    """

    def __init__(
        self,
        bridge: CredenceBridge,
        providers: list[ToolConfig],
        provider_names: list[str],
        categories: tuple[str, ...],
        category_infer: Callable[[str], NDArray[np.float64]],
        scoring: ScoringRule | None = None,
    ):
        if scoring is None:
            # For routing: reward=1 for useful result, penalty for useless, no abstain concept
            scoring = ScoringRule(reward_correct=1.0, penalty_wrong=-0.5, reward_abstain=0.0)

        self._selector = BayesianSelector(
            bridge=bridge,
            option_configs=providers,
            categories=categories,
            scoring=scoring,
        )
        self._provider_names = provider_names
        self._categories = categories
        self._category_infer = category_infer
        self._last_decision: RouteDecision | None = None

    def route(self, text: str, category_hint: str | None = None) -> RouteDecision:
        """Route a query to the best provider.

        Args:
            text: the query/message text
            category_hint: optional explicit category

        Returns:
            RouteDecision with provider index, name, category weights, reasoning
        """
        t_start = time.monotonic()

        # Infer category distribution
        if category_hint and category_hint in self._categories:
            cat_weights = [0.0] * len(self._categories)
            cat_weights[self._categories.index(category_hint)] = 1.0
        else:
            cat_weights = self._category_infer(text).tolist()

        # EU maximise via DSL
        provider_idx = self._selector.select(cat_weights)
        provider_name = self._provider_names[provider_idx]

        # Build reasoning trace
        reliability = self._selector.learned_reliability
        lines = [f"Selected: {provider_name}"]
        for i, name in enumerate(self._provider_names):
            rel = reliability.get(i, [])
            marker = " <-" if i == provider_idx else ""
            lines.append(f"  {name}: rel={rel}{marker}")
        cat_str = ", ".join(
            f"{c}={w:.2f}" for c, w in zip(self._categories, cat_weights) if w > 0.01
        )
        lines.append(f"Category: [{cat_str}]")

        decision = RouteDecision(
            provider_idx=provider_idx,
            provider_name=provider_name,
            category_weights=cat_weights,
            reasoning="\n".join(lines),
            wall_time=time.monotonic() - t_start,
        )
        self._last_decision = decision
        return decision

    def report_outcome(self, observation: Observation) -> None:
        """Update beliefs from an observed outcome.

        Uses continuous quality_score when available (from LLM judge),
        falls back to binary useful signal. The DSL's update_beta_state
        accepts any float in [0, 1] for soft Beta updates.
        """
        if self._last_decision is None:
            return

        idx = self._last_decision.provider_idx

        if observation.quality_score is not None:
            signal = observation.quality_score  # continuous [0, 1]
        else:
            signal = 1.0 if observation.useful else 0.0

        self._selector.update_reliability(idx, signal)

        if observation.completed:
            self._selector.update_coverage(idx, 1.0)
        elif observation.error_type in ("timeout", "rate_limit"):
            self._selector.update_coverage(idx, 0.0)

        log.debug(
            "Updated beliefs for %s: useful=%s, latency=%.2fs, cost=$%.4f",
            self._provider_names[idx],
            observation.useful,
            observation.total_seconds,
            observation.cost_usd,
        )

    @property
    def learned_reliability(self) -> dict[str, dict[str, float]]:
        """Per-provider per-category reliability means."""
        raw = self._selector.learned_reliability
        result: dict[str, dict[str, float]] = {}
        for i, name in enumerate(self._provider_names):
            rel = raw.get(i, [0.5] * len(self._categories))
            result[name] = {cat: rel[j] for j, cat in enumerate(self._categories)}
        return result

    @property
    def provider_names(self) -> list[str]:
        return list(self._provider_names)

    @property
    def categories(self) -> tuple[str, ...]:
        return self._categories

    @property
    def last_decision(self) -> RouteDecision | None:
        return self._last_decision

    def save_state(self, path: str | Path) -> None:
        """Persist learned state to disk.

        Extracts full MixtureMeasure state (all components + log weights) from Julia.
        """
        bridge = self._selector.bridge
        rel_data = {}
        cov_data = {}
        for i, name in enumerate(self._provider_names):
            rel_data[name] = bridge.extract_mixture_state(self._selector.rel_states[i])
            cov_data[name] = bridge.extract_mixture_state(self._selector.cov_states[i])

        state = {
            "provider_names": self._provider_names,
            "categories": list(self._categories),
            "rel_states": rel_data,
            "cov_states": cov_data,
        }
        Path(path).write_text(json.dumps(state))
        log.debug("Saved RoutingDomain state to %s", path)

    def load_state(self, path: str | Path) -> None:
        """Restore learned state from disk.

        Reconstructs full Julia MixtureMeasures from saved state.
        """
        state = json.loads(Path(path).read_text())
        bridge = self._selector.bridge

        for i, name in enumerate(self._provider_names):
            if name in state.get("rel_states", {}):
                self._selector.rel_states[i] = bridge.make_rel_state_from_mixture(
                    state["rel_states"][name]
                )
            if name in state.get("cov_states", {}):
                self._selector.cov_states[i] = bridge.make_rel_state_from_mixture(
                    state["cov_states"][name]
                )

        log.info("Loaded RoutingDomain state from %s", path)
