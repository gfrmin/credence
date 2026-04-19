# Role: body
"""Percentage arithmetic on a DSL return, no pragma — flagged."""
from skin.client import SkinClient


def format_confidence(skin: SkinClient, state_id: str) -> str:
    return f"{round(skin.weights(state_id)[0] * 100, 1)}%"  # violation
