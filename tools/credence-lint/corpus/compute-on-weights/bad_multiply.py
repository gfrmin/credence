# Role: body
"""Multiplying a weight by a reward is a parallel decision path."""
from skin.client import SkinClient


def weighted_reward(skin: SkinClient, state_id: str, reward: float) -> float:
    return skin.weights(state_id)[0] * reward  # violation: arithmetic on weights
