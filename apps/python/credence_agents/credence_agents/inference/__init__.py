# Role: body
"""Inference layer: types and Julia bridge.

Computation (Beta posteriors, VOI, EU-based decisions) is now handled by the
Julia Credence DSL. This package retains configuration types and post-question
feedback logic used by the Python agent and benchmark harness.
"""

from credence_agents.inference.voi import (
    PENALTY_WRONG,
    REWARD_ABSTAIN,
    REWARD_CORRECT,
    ScoringRule,
    ToolConfig,
)
from credence_agents.inference.decision import (
    Action,
    ActionType,
    compute_binary_reliability_updates,
    compute_reliability_updates,
)

__all__ = [
    "ScoringRule",
    "REWARD_CORRECT",
    "PENALTY_WRONG",
    "REWARD_ABSTAIN",
    "ToolConfig",
    "ActionType",
    "Action",
    "compute_reliability_updates",
    "compute_binary_reliability_updates",
]
