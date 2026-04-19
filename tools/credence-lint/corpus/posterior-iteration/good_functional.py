# Role: body
"""The right way: declare an Indicator functional, call expect."""
from skin.client import SkinClient


def mass_above_threshold(skin: SkinClient, state_id: str, threshold: float) -> float:
    indicator = {"type": "indicator", "predicate": {"op": ">", "value": threshold}}
    return skin.expect(state_id, indicator)
