"""Tests for RoutingDomain and LLM proxy components."""

from __future__ import annotations

import json

import pytest

from credence_router.categories import LLM_CATEGORIES, make_llm_category_infer_fn
from credence_router.routing_domain import Observation, RouteDecision
from credence_router.tools.llm.provider import (
    ALL_MODELS,
    compute_cost,
    extract_user_message,
    model_cost,
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


class TestLLMProvider:
    def test_all_models_have_specs(self):
        for name, spec in ALL_MODELS.items():
            assert spec.name == name
            assert spec.provider in ("anthropic", "openai")

    def test_model_cost_positive(self):
        for spec in ALL_MODELS.values():
            assert model_cost(spec) > 0

    def test_coverage_shape(self):
        for spec in ALL_MODELS.values():
            assert len(spec.coverage) == len(LLM_CATEGORIES)
            assert all(0 <= c <= 1 for c in spec.coverage)

    def test_compute_cost(self):
        haiku = ALL_MODELS["claude-haiku-4-5"]
        opus = ALL_MODELS["claude-opus-4-6"]
        haiku_cost = compute_cost(haiku, 1000, 500)
        opus_cost = compute_cost(opus, 1000, 500)
        assert haiku_cost > 0
        assert opus_cost > haiku_cost

    def test_cheaper_models_exist(self):
        costs = {name: model_cost(spec) for name, spec in ALL_MODELS.items()}
        cheapest = min(costs, key=costs.get)
        most_expensive = max(costs, key=costs.get)
        assert cheapest != most_expensive

    def test_extract_user_message_string(self):
        body = json.dumps({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello world"}],
        }).encode()
        assert extract_user_message(body) == "hello world"

    def test_extract_user_message_content_blocks(self):
        body = json.dumps({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe this image"},
            ]}],
        }).encode()
        assert extract_user_message(body) == "describe this image"

    def test_extract_user_message_multi_turn(self):
        body = json.dumps({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "first message"},
                {"role": "assistant", "content": "response"},
                {"role": "user", "content": "second message"},
            ],
        }).encode()
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

        return RoutingDomain(
            bridge=bridge,
            provider_names=["cheap-fast", "medium", "expensive-good"],
            costs=[0.001, 0.01, 0.05],
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
