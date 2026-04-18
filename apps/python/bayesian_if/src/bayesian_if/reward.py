"""Reward attribution: bridge IF score deltas to credence's was_correct signal."""

from __future__ import annotations

from bayesian_if.world import Observation


def _state_changed(prev: Observation, new: Observation) -> bool:
    """True if the observation represents a meaningful state change.

    Compares location and inventory (structured state). Falls back to text
    comparison only when location is unavailable for both observations.
    """
    if prev.location is not None or new.location is not None:
        return prev.location != new.location or prev.inventory != new.inventory
    return prev.text != new.text or prev.inventory != new.inventory


def attribute_reward(
    score_delta: float,
    prev_obs: Observation | None = None,
    new_obs: Observation | None = None,
) -> bool | None:
    """Convert a score delta into a correctness signal for reliability updates.

    - score_delta > 0  → True  (action led to progress)
    - score_delta < 0  → False (action was harmful)
    - score_delta == 0 + intermediate_reward > 0 → True (sub-quest progress)
    - score_delta == 0 + state changed   → True  (evidence of progress)
    - score_delta == 0 + state unchanged → None  (ambiguous — many correct IF actions yield no score)
    - score_delta == 0 + no observations → None  (backward compatible)
    """
    if score_delta > 0:
        return True
    elif score_delta < 0:
        return False
    elif prev_obs is not None and new_obs is not None:
        if new_obs.intermediate_reward > 0:
            return True
        return True if _state_changed(prev_obs, new_obs) else None
    else:
        return None
