# Role: body
"""Aggregating probability weights in consumer code is a violation.

The rewrite is expect(m, indicator_functional), which performs the
aggregation inside the DSL with declared structure.
"""
from skin.client import SkinClient


def posterior_mass_over_threshold(skin: SkinClient, state_id: str) -> float:
    w = skin.weights(state_id)
    return sum(wi for wi in w if wi > 0.1)  # violation: arithmetic on weights
