"""RoutingDomain: provider routing via the brain server.

State is a nested ProductMeasure (providers × categories × (theta, k)).
Decide is `brain.optimise` with a `functional_per_action` spec of
LinearCombination of NestedProjections that descends to the θ leaf.
Observe is factor → condition → replace_factor on the chosen leaf.

No DSL wrappers around axiom-constrained operations. The host
orchestrates the primitives directly against the protocol handlers
in brain/server.jl.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def _ensure_brain_importable() -> Path:
    """Locate brain/ and make it importable. Returns the dir containing brain/."""
    candidates: list[Path] = []
    env = os.environ.get("CREDENCE_REPO_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.extend(Path(__file__).resolve().parents)
    candidates.append(Path("/credence"))
    for root in candidates:
        if (root / "brain" / "client.py").is_file():
            s = str(root)
            if s not in sys.path:
                sys.path.insert(0, s)
            return root
    raise RuntimeError(
        "Could not locate brain/client.py. Set CREDENCE_REPO_ROOT to the repo root."
    )


_REPO_ROOT = _ensure_brain_importable()
from brain.client import BrainClient  # noqa: E402

if TYPE_CHECKING:
    pass


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


def _make_router_state(brain: BrainClient, n_providers: int, n_categories: int) -> str:
    """Build a nested ProductMeasure state.

    Shape: providers × categories × (theta, concentration).
    Prior per leaf: Beta(1,1) ⊗ Gamma(2, 0.5). Matches the prior in
    examples/router.bdsl.
    """
    leaf = {
        "type": "product",
        "factors": [
            {"type": "beta", "alpha": 1.0, "beta": 1.0},
            {"type": "gamma", "alpha": 2.0, "beta": 0.5},
        ],
    }
    provider = {"type": "product", "factors": [leaf for _ in range(n_categories)]}
    state = {"type": "product", "factors": [provider for _ in range(n_providers)]}
    return brain.create_state(**state)


def _router_preference(
    n_providers: int,
    n_categories: int,
    cat_weights: list[float],
    costs: list[float],
    reward: float,
) -> dict:
    """functional_per_action spec for EU maximisation.

    EU(a) = reward * sum_c cat_weights[c] * E[theta_{a,c}] - costs[a]

    NestedProjection([a, c, 0]) descends providers → categories → (θ, k)
    and projects to θ. Identity() at the Beta leaf yields mean(Beta).
    """
    actions: dict[str, dict] = {}
    for a in range(n_providers):
        terms = [
            [
                reward * cat_weights[c],
                {"type": "nested_projection", "indices": [a, c, 0]},
            ]
            for c in range(n_categories)
        ]
        actions[str(a)] = {
            "type": "linear_combination",
            "terms": terms,
            "offset": -costs[a],
        }
    return {"type": "functional_per_action", "actions": actions}


class RoutingDomain:
    """Provider routing backed by the brain server.

    The host holds one state_id and orchestrates:
    - route()          → brain.optimise(state, actions, functional_per_action)
    - _apply_outcome() → factor → condition → replace_factor chain
    """

    def __init__(
        self,
        brain: BrainClient,
        provider_names: list[str],
        costs: list[float],
        categories: tuple[str, ...],
        category_infer: Callable[[str], NDArray[np.float64]],
        reward: float = 1.0,
    ):
        self._brain = brain
        self._provider_names = provider_names
        self._costs = list(costs)
        self._categories = categories
        self._category_infer = category_infer
        self._reward = reward

        self._state_id = _make_router_state(
            brain, len(provider_names), len(categories),
        )
        self._actions_spec = {
            "type": "finite",
            "values": list(range(len(provider_names))),
        }

        self._last_decision: RouteDecision | None = None
        self._cached_reliability: dict[str, dict[str, float]] = {}

        # Outcome queue: async judge appends here, route() drains synchronously
        # before deciding. Keeps brain-subprocess calls off the async path.
        self._pending_outcomes: deque[tuple[Observation, RouteDecision]] = deque()

        # Warm the reliability cache so /state returns useful data before
        # the first request lands.
        try:
            self._cached_reliability = self._compute_reliability()
        except Exception as e:
            log.debug("initial reliability warmup failed: %s", e)

    def route(self, text: str, category_hint: str | None = None) -> RouteDecision:
        """Route a query to the best provider."""
        t_start = time.monotonic()

        self._drain_outcomes()

        if category_hint and category_hint in self._categories:
            cat_weights = [0.0] * len(self._categories)
            cat_weights[self._categories.index(category_hint)] = 1.0
        else:
            cat_weights = self._category_infer(text).tolist()

        pref = _router_preference(
            len(self._provider_names),
            len(self._categories),
            cat_weights,
            self._costs,
            self._reward,
        )
        action, _eu = self._brain.optimise(self._state_id, self._actions_spec, pref)
        provider_idx = int(action) if not isinstance(action, int) else action

        decision = RouteDecision(
            provider_idx=provider_idx,
            provider_name=self._provider_names[provider_idx],
            category_weights=cat_weights,
            wall_time=time.monotonic() - t_start,
        )
        self._last_decision = decision

        try:
            self._cached_reliability = self._compute_reliability()
        except Exception as e:
            log.debug("reliability readback failed: %s", e)

        return decision

    def queue_outcome(self, observation: Observation) -> None:
        """Queue an outcome for processing on the next route() call.

        Thread-safe: appends to a deque. No brain calls — safe from async context.
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
        """Update beliefs from an observed outcome via factor → condition → replace_factor.

        NOTE: brain protocol currently supports single-path updates — one
        (provider, category) leaf per observation. We credit the argmax
        category of decision.category_weights. Distributed credit assignment
        across multiple categories needs FiringByTag / DispatchByComponent
        support in the brain protocol (see CLAUDE.md "No opaque likelihood
        functions" entry); follow-up work.
        """
        if observation.quality_score is not None:
            quality = float(observation.quality_score)
        elif observation.useful:
            quality = 0.8
        else:
            quality = 0.2

        if not decision.category_weights:
            return
        cat_idx = int(max(
            range(len(decision.category_weights)),
            key=lambda i: decision.category_weights[i],
        ))

        brain = self._brain
        provider_id = brain.factor(self._state_id, decision.provider_idx)
        category_id = brain.factor(provider_id, cat_idx)
        brain.condition(
            category_id, kernel={"type": "quality"}, observation=quality,
        )
        provider_updated = brain.replace_factor(provider_id, cat_idx, category_id)
        self._state_id = brain.replace_factor(
            self._state_id, decision.provider_idx, provider_updated,
        )

    def _compute_reliability(self) -> dict[str, dict[str, float]]:
        """Per-provider per-category E[theta] from the current ProductMeasure."""
        brain = self._brain
        theta_projection = {"type": "projection", "index": 0}
        result: dict[str, dict[str, float]] = {}
        for i, name in enumerate(self._provider_names):
            provider_id = brain.factor(self._state_id, i)
            result[name] = {}
            for j, cat in enumerate(self._categories):
                try:
                    leaf_id = brain.factor(provider_id, j)
                    mean = float(brain.expect(leaf_id, function=theta_projection))
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
        """Persist state via brain snapshot (base64). JSON sidecar holds last_decision."""
        path = Path(path)
        data_b64 = self._brain.snapshot_state(self._state_id)
        path.write_bytes(base64.b64decode(data_b64))

        if self._last_decision is not None:
            sidecar = path.with_suffix(".json")
            sidecar.write_text(json.dumps({
                "provider_idx": self._last_decision.provider_idx,
                "provider_name": self._last_decision.provider_name,
                "category_weights": self._last_decision.category_weights,
            }))
        log.info("LLM state saved to %s", path)

    def load_state(self, path: str | Path) -> None:
        """Restore state via brain snapshot."""
        path = Path(path)
        if not path.exists():
            return
        data_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        self._state_id = self._brain.restore_state(data_b64)

        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            data = json.loads(sidecar.read_text())
            self._last_decision = RouteDecision(
                provider_idx=data["provider_idx"],
                provider_name=data["provider_name"],
                category_weights=data["category_weights"],
            )
        log.info("LLM state restored from %s", path)
