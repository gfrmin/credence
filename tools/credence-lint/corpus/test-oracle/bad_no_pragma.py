# Role: body
"""Test without pragma — even tests must mark the escape hatch."""
import pytest
from skin.client import SkinClient


def test_beta_mean(skin: SkinClient) -> None:
    state_id = skin.create_state(type="beta", alpha=3.0, beta=7.0)
    assert skin.mean(state_id) == pytest.approx(0.3)  # violation: comparison on DSL return, no pragma
