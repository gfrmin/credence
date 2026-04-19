# Role: body
"""Formatting a probability as a percentage string with pragma."""
from skin.client import SkinClient


def format_confidence(skin: SkinClient, state_id: str) -> str:
    w = skin.weights(state_id)
    # credence-lint: allow — precedent:display-arithmetic — percentage formatting for report
    return f"{round(w[0] * 100, 1)}%"
