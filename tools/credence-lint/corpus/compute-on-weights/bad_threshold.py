# Role: body
"""Branching on a weight value is a parallel decision mechanism."""
from skin.client import SkinClient


def should_interact(skin: SkinClient, state_id: str) -> bool:
    if skin.weights(state_id)[0] > 0.5:  # violation: compare-to-branch on weights
        return True
    return False
