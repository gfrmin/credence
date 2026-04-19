# Role: body
"""Percentage arithmetic without pragma — author must declare non-causation."""
from skin.client import SkinClient


def format_confidence(skin: SkinClient, state_id: str) -> str:
    w = skin.weights(state_id)
    return f"{round(w[0] * 100, 1)}%"  # violation: arithmetic on weights, no pragma
