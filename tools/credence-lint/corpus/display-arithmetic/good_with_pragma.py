# Role: body
"""Formatting a probability as a percentage — same-line pragma."""
from skin.client import SkinClient


def format_confidence(skin: SkinClient, state_id: str) -> str:
    return f"{round(skin.weights(state_id)[0] * 100, 1)}%"  # credence-lint: allow — precedent:display-arithmetic — percentage formatting for report
