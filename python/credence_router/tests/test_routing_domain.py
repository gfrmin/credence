"""Tests for RoutingDomain and LLM proxy components."""

from __future__ import annotations

import json

import numpy as np
import pytest

from credence_router.categories import LLM_CATEGORIES, make_llm_category_infer_fn
from credence_router.routing_domain import Observation, RouteDecision
from credence_router.tools.llm.anthropic import (
    ANTHROPIC_MODELS,
    compute_cost,
    extract_user_message,
    model_cost,
    model_coverage,
    resolve_model,
)


# ---------------------------------------------------------------------------
# Observation tests
# ---------------------------------------------------------------------------


class TestObservation:
    def test_useful_when_completed(self):
        obs = Observation(completed=True)
        assert obs.useful

    def test_not_useful_when_error(self):
        obs = Observation(completed=True, error_type="rate_limit")
        assert not obs.useful

    def test_not_useful_when_truncated(self):
        obs = Observation(completed=True, truncated=True)
        assert not obs.useful

    def test_not_useful_when_not_completed(self):
        obs = Observation(completed=False)
        assert not obs.useful


# ---------------------------------------------------------------------------
# Category inference tests
# ---------------------------------------------------------------------------


class TestLLMCategoryInference:
    def test_code_query(self):
        infer = make_llm_category_infer_fn()
        dist = infer("implement a function to sort a list in Python")
        cats = dict(zip(LLM_CATEGORIES, dist))
        assert cats["code"] > cats["chat"]

    def test_reasoning_query(self):
        infer = make_llm_category_infer_fn()
        dist = infer("explain why functional programming is better for concurrency")
        cats = dict(zip(LLM_CATEGORIES, dist))
        assert cats["reasoning"] > cats["chat"]

    def test_creative_query(self):
        infer = make_llm_category_infer_fn()
        dist = infer("write a short poem about the ocean")
        cats = dict(zip(LLM_CATEGORIES, dist))
        assert cats["creative"] > cats["chat"]

    def test_factual_query(self):
        infer = make_llm_category_infer_fn()
        dist = infer("what is the capital of France")
        cats = dict(zip(LLM_CATEGORIES, dist))
        assert cats["factual"] > cats["chat"]

    def test_chat_default(self):
        infer = make_llm_category_infer_fn()
        dist = infer("hello how are you")
        cats = dict(zip(LLM_CATEGORIES, dist))
        assert cats["chat"] == max(cats.values())

    def test_returns_distribution(self):
        infer = make_llm_category_infer_fn()
        dist = infer("test query")
        assert abs(sum(dist) - 1.0) < 1e-6
        assert all(d >= 0 for d in dist)


# ---------------------------------------------------------------------------
# Anthropic provider tests
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def test_resolve_model_alias(self):
        assert resolve_model("haiku").startswith("claude-haiku")
        assert resolve_model("sonnet").startswith("claude-sonnet")
        assert resolve_model("opus").startswith("claude-opus")

    def test_resolve_model_passthrough(self):
        full = "claude-haiku-4-5-20251001"
        assert resolve_model(full) == full

    def test_model_cost_positive(self):
        for model in ANTHROPIC_MODELS:
            assert model_cost(model) > 0

    def test_model_coverage_shape(self):
        for model in ANTHROPIC_MODELS:
            cov = model_coverage(model)
            assert len(cov) == len(LLM_CATEGORIES)
            assert all(0 <= c <= 1 for c in cov)

    def test_compute_cost(self):
        cost = compute_cost("claude-haiku-4-5-20251001", 1000, 500)
        assert cost > 0
        # Haiku is cheap
        opus_cost = compute_cost("claude-opus-4-6-20250514", 1000, 500)
        assert opus_cost > cost

    def test_extract_user_message_string(self):
        body = json.dumps({
            "model": "claude-sonnet-4-6-20250514",
            "messages": [
                {"role": "user", "content": "hello world"},
            ],
        }).encode()
        assert extract_user_message(body) == "hello world"

    def test_extract_user_message_content_blocks(self):
        body = json.dumps({
            "model": "claude-sonnet-4-6-20250514",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "describe this image"},
                ]},
            ],
        }).encode()
        assert extract_user_message(body) == "describe this image"

    def test_extract_user_message_multi_turn(self):
        body = json.dumps({
            "model": "claude-sonnet-4-6-20250514",
            "messages": [
                {"role": "user", "content": "first message"},
                {"role": "assistant", "content": "response"},
                {"role": "user", "content": "second message"},
            ],
        }).encode()
        # Should extract LAST user message
        assert extract_user_message(body) == "second message"

    def test_extract_user_message_empty(self):
        body = json.dumps({"model": "test", "messages": []}).encode()
        assert extract_user_message(body) == ""


# ---------------------------------------------------------------------------
# RoutingDomain tests (requires Julia)
# ---------------------------------------------------------------------------
# These are marked with a comment so they can be skipped if Julia isn't available.
# They test the full DSL-backed routing path.

try:
    from credence_agents.julia_bridge import CredenceBridge
    from credence_agents.inference.voi import ToolConfig

    _JULIA_AVAILABLE = True
except ImportError:
    _JULIA_AVAILABLE = False


@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="Julia bridge not available")
class TestRoutingDomainWithDSL:
    """Tests that exercise the full DSL-backed routing path."""

    @pytest.fixture(scope="class")
    def bridge(self):
        return CredenceBridge()

    def _make_domain(self, bridge):
        from credence_router.routing_domain import RoutingDomain

        providers = [
            ToolConfig(cost=0.001, coverage_by_category=np.array([0.8, 0.3, 0.5, 0.7, 0.9])),
            ToolConfig(cost=0.01, coverage_by_category=np.array([0.5, 0.9, 0.8, 0.4, 0.3])),
            ToolConfig(cost=0.05, coverage_by_category=np.array([0.9, 0.9, 0.9, 0.6, 0.5])),
        ]
        names = ["cheap-fast", "medium", "expensive-good"]

        return RoutingDomain(
            bridge=bridge,
            providers=providers,
            provider_names=names,
            categories=LLM_CATEGORIES,
            category_infer=make_llm_category_infer_fn(),
        )

    def test_route_returns_decision(self, bridge):
        domain = self._make_domain(bridge)
        decision = domain.route("hello world")
        assert isinstance(decision, RouteDecision)
        assert decision.provider_name in ("cheap-fast", "medium", "expensive-good")

    def test_route_has_category_weights(self, bridge):
        domain = self._make_domain(bridge)
        decision = domain.route("implement quicksort in Python")
        assert len(decision.category_weights) == len(LLM_CATEGORIES)
        assert abs(sum(decision.category_weights) - 1.0) < 1e-6

    def test_report_outcome_updates_beliefs(self, bridge):
        domain = self._make_domain(bridge)
        rel_before = domain.learned_reliability

        domain.route("test query")
        domain.report_outcome(Observation(completed=True))

        rel_after = domain.learned_reliability
        # At least one provider's reliability should have changed
        changed = False
        for name in rel_before:
            for cat in rel_before[name]:
                if abs(rel_before[name][cat] - rel_after[name][cat]) > 0.001:
                    changed = True
        assert changed

    def test_learned_reliability_structure(self, bridge):
        domain = self._make_domain(bridge)
        rel = domain.learned_reliability
        assert set(rel.keys()) == {"cheap-fast", "medium", "expensive-good"}
        for name, cats in rel.items():
            assert set(cats.keys()) == set(LLM_CATEGORIES)
            for val in cats.values():
                assert 0.0 <= val <= 1.0
