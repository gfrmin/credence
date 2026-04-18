"""Credence: Bayesian decision-theoretic agents.

Inference is handled by the Julia Credence DSL via juliacall.
This package provides the Python agent, types, and benchmark harness.
"""

# Types and configuration
from credence_agents.inference.voi import (
    ScoringRule,
    ToolConfig,
)
from credence_agents.inference.decision import (
    Action,
    ActionType,
    compute_binary_reliability_updates,
    compute_reliability_updates,
)

# Bridge
from credence_agents.julia_bridge import CredenceBridge

# Agent
from credence_agents.agents.bayesian_agent import BayesianAgent
from credence_agents.agents.common import AgentResult, DecisionStep

__all__ = [
    # Types
    "ScoringRule", "ToolConfig",
    "ActionType", "Action",
    "compute_binary_reliability_updates", "compute_reliability_updates",
    # Bridge
    "CredenceBridge",
    # Agent
    "BayesianAgent", "AgentResult", "DecisionStep",
]
