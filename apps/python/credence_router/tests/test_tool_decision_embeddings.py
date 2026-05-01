# Role: tests for tool embedding + cache + kNN.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from credence_router.tool_decision.embeddings import (
    embed_tool,
    knn_smoothed_prior,
    tool_content_hash,
)
from credence_router.tool_decision.state import ToolDecisionState


def _stub_embed(text: str) -> np.ndarray:
    """Deterministic 4-d embedding: hash-of-text-into-buckets."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(4).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v


class TestContentHash:
    def test_same_input_same_hash(self):
        spec = {"name": "Bash", "description": "Run a shell command",
                "parameters": {"command": "string"}}
        assert tool_content_hash(spec) == tool_content_hash(spec)

    def test_different_inputs_differ(self):
        a = {"name": "Bash", "description": "Run a shell command", "parameters": {}}
        b = {"name": "Bash", "description": "Execute a shell command", "parameters": {}}
        assert tool_content_hash(a) != tool_content_hash(b)


class TestEmbedTool:
    def test_caches_on_second_call(self, tmp_path: Path):
        state = ToolDecisionState(path=tmp_path / "s.json")
        spec = {"name": "Bash", "description": "Run a shell command", "parameters": {}}
        calls = {"n": 0}

        def counting_embed(text: str) -> np.ndarray:
            calls["n"] += 1
            return _stub_embed(text)

        v1 = embed_tool(spec, state=state, embed_fn=counting_embed)
        v2 = embed_tool(spec, state=state, embed_fn=counting_embed)
        np.testing.assert_array_equal(v1, v2)
        assert calls["n"] == 1


class TestKnnSmoothedPrior:
    def test_no_neighbours_returns_diffuse_prior(self, tmp_path: Path):
        state = ToolDecisionState(path=tmp_path / "s.json")
        target = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        alpha, beta = knn_smoothed_prior(
            target_embedding=target,
            model_id="any",
            state=state,
            k=3,
        )
        assert alpha == pytest.approx(1.0)
        assert beta == pytest.approx(1.0)

    def test_single_close_neighbour_dominates(self, tmp_path: Path):
        state = ToolDecisionState(path=tmp_path / "s.json")
        # Plant a high-reliability cell and its embedding for "grep"
        for _ in range(8):
            state.update("sonnet", "grep", approved=True)
        grep_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state.put_embedding(tool_content_hash({"name": "grep"}), grep_emb)
        # Query with an embedding identical to grep's, same model
        prior_alpha, prior_beta = knn_smoothed_prior(
            target_embedding=grep_emb,
            model_id="sonnet",
            state=state,
            k=3,
            tool_name_to_hash={"grep": tool_content_hash({"name": "grep"})},
        )
        # Should pull strongly toward grep's posterior (alpha=9, beta=1)
        assert prior_alpha > prior_beta * 4.0

    def test_orthogonal_neighbour_does_not_pull(self, tmp_path: Path):
        state = ToolDecisionState(path=tmp_path / "s.json")
        for _ in range(20):
            state.update("sonnet", "grep", approved=True)
        grep_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state.put_embedding(tool_content_hash({"name": "grep"}), grep_emb)

        target = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        prior_alpha, prior_beta = knn_smoothed_prior(
            target_embedding=target,
            model_id="sonnet",
            state=state,
            k=3,
            tool_name_to_hash={"grep": tool_content_hash({"name": "grep"})},
        )
        # cosine(target, grep_emb) == 0, so weight should be 0 → diffuse
        assert prior_alpha == pytest.approx(1.0)
        assert prior_beta == pytest.approx(1.0)
