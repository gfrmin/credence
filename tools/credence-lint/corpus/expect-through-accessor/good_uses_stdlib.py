# Role: body
"""Legal: uses stdlib accessors, not structural fields."""
from skin.client import SkinClient


def show_mean(skin: SkinClient, state_id: str) -> float:
    return skin.mean(state_id)


def show_weights(skin: SkinClient, state_id: str) -> list:
    return skin.weights(state_id)
