# Role: body
"""Bayesian decision-theoretic Interactive Fiction agent."""

from bayesian_if.world import Observation, StateSnapshot, World
from bayesian_if.categories import CATEGORIES, make_if_category_infer_fn
from bayesian_if.tools import IFTool, LookTool, ExamineTool, InventoryTool, LLMAdvisorTool
from bayesian_if.reward import attribute_reward
from bayesian_if.agent import IFAgent, GameResult

__all__ = [
    "Observation",
    "StateSnapshot",
    "World",
    "CATEGORIES",
    "make_if_category_infer_fn",
    "IFTool",
    "LookTool",
    "ExamineTool",
    "InventoryTool",
    "LLMAdvisorTool",
    "attribute_reward",
    "IFAgent",
    "GameResult",
]
