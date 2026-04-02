"""Effort-to-parameters mapping for routing configuration."""

from __future__ import annotations

from dataclasses import dataclass

from credence_agents.inference.voi import ScoringRule


@dataclass(frozen=True)
class RoutingParams:
    """Routing parameters derived from a single effort value."""

    scoring: ScoringRule
    cost_scale: float
    latency_weight: float


def derive_routing_params(effort: float, reward_unit: float = 1.0) -> RoutingParams:
    """Derive consistent routing parameters from a single effort value.

    Args:
        effort: 0.0 (fast/cheap) to 1.0 (thorough/expensive).
        reward_unit: Base reward unit (scales all parameters proportionally).

    Returns:
        RoutingParams with scoring rule, cost scale, and latency weight.
    """
    effort = max(0.0, min(1.0, effort))
    U = reward_unit
    scoring = ScoringRule(
        reward_correct=U,
        penalty_wrong=-U / 2,
        reward_abstain=0.0,
    )
    cost_scale = U * 50.0 * (1.0 - effort) ** 2
    latency_weight = U * 0.05 * (1.0 - effort) ** 2
    return RoutingParams(
        scoring=scoring,
        cost_scale=cost_scale,
        latency_weight=latency_weight,
    )
