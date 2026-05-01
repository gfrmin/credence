# Role: tests for the (model_id, tool_name) -> Beta posterior store.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from credence_router.tool_decision.state import ToolDecisionState


class TestToolDecisionState:
    def test_default_prior_is_diffuse(self, tmp_path: Path):
        state = ToolDecisionState(path=tmp_path / "state.json")
        alpha, beta = state.get_beta("claude-sonnet-4-5", "Bash")
        assert alpha == pytest.approx(1.0)
        assert beta == pytest.approx(1.0)

    def test_update_increments_alpha_on_yes(self, tmp_path: Path):
        state = ToolDecisionState(path=tmp_path / "state.json")
        state.update("claude-sonnet-4-5", "Bash", approved=True)
        alpha, beta = state.get_beta("claude-sonnet-4-5", "Bash")
        assert alpha == pytest.approx(2.0)
        assert beta == pytest.approx(1.0)

    def test_update_increments_beta_on_no(self, tmp_path: Path):
        state = ToolDecisionState(path=tmp_path / "state.json")
        state.update("claude-sonnet-4-5", "Bash", approved=False)
        alpha, beta = state.get_beta("claude-sonnet-4-5", "Bash")
        assert alpha == pytest.approx(1.0)
        assert beta == pytest.approx(2.0)

    def test_persist_and_reload_round_trip(self, tmp_path: Path):
        path = tmp_path / "state.json"
        state = ToolDecisionState(path=path)
        state.update("haiku", "Read", approved=True)
        state.update("haiku", "Read", approved=True)
        state.update("haiku", "Read", approved=False)
        state.save()

        reloaded = ToolDecisionState(path=path)
        reloaded.load()
        alpha, beta = reloaded.get_beta("haiku", "Read")
        assert alpha == pytest.approx(3.0)
        assert beta == pytest.approx(2.0)

    def test_embedding_cache_round_trip(self, tmp_path: Path):
        state = ToolDecisionState(path=tmp_path / "state.json")
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        state.put_embedding(content_hash="abc123", embedding=vec)
        out = state.get_embedding("abc123")
        assert out is not None
        np.testing.assert_array_almost_equal(out, vec)

    def test_get_embedding_returns_none_on_miss(self, tmp_path: Path):
        state = ToolDecisionState(path=tmp_path / "state.json")
        assert state.get_embedding("nonexistent") is None

    def test_iter_observed_tools_returns_all_known_pairs(self, tmp_path: Path):
        state = ToolDecisionState(path=tmp_path / "state.json")
        state.update("haiku", "Read", approved=True)
        state.update("sonnet", "Bash", approved=False)
        pairs = sorted(state.iter_observed_pairs())
        assert pairs == [("haiku", "Read"), ("sonnet", "Bash")]
