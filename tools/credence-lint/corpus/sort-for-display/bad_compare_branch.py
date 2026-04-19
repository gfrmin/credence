# Role: body
"""Comparison-to-branch IS the decision — must flow through optimise."""
from skin.client import SkinClient


def pick_by_weight(skin: SkinClient, state_id: str) -> int:
    w = skin.weights(state_id)
    if w[0] > w[1]:  # violation: compare-to-branch on weights
        return 0
    return 1
