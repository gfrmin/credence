# Role: body
"""Slug present but reason missing — malformed pragma, still fails."""
from skin.client import SkinClient


def show(skin: SkinClient, state_id: str) -> str:
    w = skin.weights(state_id)
    # credence-lint: allow — precedent:display-arithmetic
    return f"{round(w[0] * 100, 1)}%"
