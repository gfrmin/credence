# Role: body
"""Comparison-to-branch on DSL returns IS the decision — flagged."""
from skin.client import SkinClient


def pick_higher(skin: SkinClient, state_id: str) -> int:
    return 0 if skin.weights(state_id)[0] > skin.weights(state_id)[1] else 1  # violation
