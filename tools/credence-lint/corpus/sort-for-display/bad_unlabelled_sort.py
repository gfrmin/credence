# Role: body
"""Sort without pragma — intent unclear, flagged."""
from skin.client import SkinClient


def maybe_display(skin: SkinClient, state_id: str) -> list[tuple[int, float]]:
    w = skin.weights(state_id)
    pairs = list(enumerate(w))
    pairs.sort(key=lambda p: p[1])  # violation: sort on weights without pragma
    return pairs
