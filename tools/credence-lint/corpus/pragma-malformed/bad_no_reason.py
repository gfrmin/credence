# Role: body
"""Slug present but reason missing — malformed pragma, still fails."""
from skin.client import SkinClient


def show(skin: SkinClient, state_id: str) -> str:
    return f"{round(skin.weights(state_id)[0] * 100, 1)}%"  # credence-lint: allow — precedent:display-arithmetic
