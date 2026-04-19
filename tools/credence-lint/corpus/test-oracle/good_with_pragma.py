# Role: body
"""Test oracle: hand-computed ground truth sanctioned by pragma."""
import pytest
from skin.client import SkinClient


def test_beta_mean(skin: SkinClient) -> None:
    state_id = skin.create_state(type="beta", alpha=3.0, beta=7.0)
    # credence-lint: allow — precedent:test-oracle — Beta(3,7) mean = 3/(3+7) = 0.3
    assert skin.mean(state_id) == pytest.approx(0.3)
