# Role: body
"""Returning weights unchanged to a caller is pure read access."""
from skin.client import SkinClient


def current_weights(skin: SkinClient, state_id: str) -> list[float]:
    return skin.weights(state_id)
