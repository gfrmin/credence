# Role: body
"""Aggregating probability weights in consumer code is a violation.

The rewrite is expect(m, indicator_functional), which performs the
aggregation inside the DSL with declared structure.
"""
from skin.client import SkinClient


def posterior_mass(skin: SkinClient, state_id: str) -> float:
    return sum(skin.weights(state_id))  # violation: aggregator over DSL return
