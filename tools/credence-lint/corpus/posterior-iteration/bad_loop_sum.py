# Role: body
"""Manual loop over posterior support — should be a Functional + expect."""
from skin.client import SkinClient


def mass_above_threshold(skin: SkinClient, state_id: str, threshold: float) -> float:
    w = skin.weights(state_id)
    support = skin.support(state_id)
    total = 0.0
    for h, wi in zip(support, w):
        if h > threshold:
            total += wi  # violation: arithmetic on weights inside loop
    return total
