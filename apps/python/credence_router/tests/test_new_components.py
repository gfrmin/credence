# Role: body
"""Tests for new credence-router components: KeywordCategoryTool, CoveragePrior,
RoutingParams, derive_routing_params, RouterGroup, make_keyword_category_infer_fn,
make_router_category_infer_fn.
"""

from __future__ import annotations

import re

import pytest
from numpy.testing import assert_allclose

from credence_router.config import derive_routing_params
from credence_router.group import RouterGroup
from credence_router.router import Router
from credence_router.tools.coverage_prior import CoveragePrior
from credence_router.tools.keyword_category import KeywordCategoryTool
from credence_router.tools.simulated import make_default_simulated_tools
from credence_router.categories import (
    make_keyword_category_infer_fn,
    make_router_category_infer_fn,
)


# --- KeywordCategoryTool ---


class TestKeywordCategoryTool:
    def test_properties(self):
        pat = re.compile(r"\bscience\b", re.IGNORECASE)
        tool = KeywordCategoryTool("scientific", pat, ("scientific", "general"))
        assert tool.name == "scientific_keywords"
        assert tool.cost == 0.0
        assert tool.latency == 0.0

    def test_query_matches(self):
        pat = re.compile(r"\bscience\b", re.IGNORECASE)
        tool = KeywordCategoryTool("scientific", pat, ("scientific", "general"))
        result = tool.query("Tell me about science", ("scientific", "general"))
        assert result == 0

    def test_query_no_match(self):
        pat = re.compile(r"\bscience\b", re.IGNORECASE)
        tool = KeywordCategoryTool("scientific", pat, ("scientific", "general"))
        result = tool.query("Tell me about cooking", ("scientific", "general"))
        assert result is None

    def test_query_category_not_in_candidates(self):
        pat = re.compile(r"\bscience\b", re.IGNORECASE)
        tool = KeywordCategoryTool("scientific", pat, ("scientific", "general"))
        result = tool.query("Tell me about science", ("general", "other"))
        assert result is None

    def test_coverage_all_ones(self):
        pat = re.compile(r"\bscience\b")
        tool = KeywordCategoryTool("scientific", pat, ("scientific",))
        cov = tool.coverage(("a", "b", "c"))
        assert_allclose(cov, 1.0)
        assert len(cov) == 3


# --- CoveragePrior ---


class TestCoveragePrior:
    def test_from_priors(self):
        cp = CoveragePrior.from_priors(
            ("a", "b", "c"),
            {"a": 0.9, "b": 0.1, "c": 0.5},
            strength=10.0,
        )
        assert cp.alpha[0] == pytest.approx(9.0)
        assert cp.beta[0] == pytest.approx(1.0)
        assert cp.alpha[1] == pytest.approx(1.0)
        assert cp.beta[1] == pytest.approx(9.0)

    def test_coverage(self):
        cp = CoveragePrior.from_priors(("a", "b"), {"a": 0.9, "b": 0.05}, strength=10.0)
        cov = cp.coverage()
        assert cov[0] == pytest.approx(0.9)
        assert cov[1] == pytest.approx(0.1)  # clipped from 0.05

    def test_serialisation_roundtrip(self):
        cp = CoveragePrior.from_priors(("a", "b", "c"), {"a": 0.8, "b": 0.2, "c": 0.5})
        d = cp.to_dict()
        cp2 = CoveragePrior.from_dict(d)
        assert_allclose(cp2.alpha, cp.alpha)
        assert_allclose(cp2.beta, cp.beta)

    def test_default_prior(self):
        cp = CoveragePrior.from_priors(("a", "b"), {}, strength=10.0)
        assert cp.alpha[0] == pytest.approx(5.0)  # default 0.5


# --- RoutingParams / derive_routing_params ---


class TestRoutingParams:
    def test_effort_zero(self):
        params = derive_routing_params(0.0)
        assert params.scoring.reward_correct == 1.0
        assert params.scoring.penalty_wrong == -0.5
        assert params.cost_scale == pytest.approx(50.0)
        assert params.latency_weight == pytest.approx(0.05)

    def test_effort_one(self):
        params = derive_routing_params(1.0)
        assert params.cost_scale == pytest.approx(0.0)
        assert params.latency_weight == pytest.approx(0.0)

    def test_effort_half(self):
        params = derive_routing_params(0.5)
        assert params.cost_scale == pytest.approx(50.0 * 0.25)
        assert params.latency_weight == pytest.approx(0.05 * 0.25)

    def test_clamps_effort(self):
        p1 = derive_routing_params(-1.0)
        p2 = derive_routing_params(0.0)
        assert p1.cost_scale == p2.cost_scale

        p3 = derive_routing_params(2.0)
        p4 = derive_routing_params(1.0)
        assert p3.cost_scale == p4.cost_scale

    def test_reward_unit_scales(self):
        params = derive_routing_params(0.0, reward_unit=2.0)
        assert params.scoring.reward_correct == 2.0
        assert params.cost_scale == pytest.approx(2.0 * 50.0)

    def test_frozen(self):
        params = derive_routing_params(0.5)
        with pytest.raises(AttributeError):
            params.cost_scale = 0.0  # type: ignore[misc]


# --- RouterGroup ---


class TestRouterGroup:
    @staticmethod
    def _make_group() -> RouterGroup:
        tools = make_default_simulated_tools()
        r1 = Router(tools=tools)
        r2 = Router(tools=tools)
        return RouterGroup({"first": r1, "second": r2})

    def test_getitem(self):
        group = self._make_group()
        assert isinstance(group["first"], Router)

    def test_contains(self):
        group = self._make_group()
        assert "first" in group
        assert "third" not in group

    def test_names(self):
        group = self._make_group()
        assert set(group.names) == {"first", "second"}

    def test_save_load_roundtrip(self, tmp_path):
        group = self._make_group()
        # Run some questions on one router
        for _ in range(3):
            group["first"].solve("What is 2+2?", ("3", "4", "5", "6"), category_hint="numerical")
            group["first"].report_outcome(True)

        path = tmp_path / "group.json"
        group.save_state(path)

        group2 = self._make_group()
        group2.load_state(path)

        r1 = group["first"].learned_reliability
        r2 = group2["first"].learned_reliability
        for tool_name in r1:
            for cat in r1[tool_name]:
                assert abs(r1[tool_name][cat] - r2[tool_name][cat]) < 1e-10

    def test_save_load_state_dict(self):
        group = self._make_group()
        group["second"].solve("test", ("A", "B", "C", "D"))
        group["second"].report_outcome(True)

        state = group.save_state_dict()
        group2 = self._make_group()
        group2.load_state_dict(state)

        r1 = group["second"].learned_reliability
        r2 = group2["second"].learned_reliability
        for tool_name in r1:
            for cat in r1[tool_name]:
                assert abs(r1[tool_name][cat] - r2[tool_name][cat]) < 1e-10

    def test_load_missing_file_is_noop(self, tmp_path):
        group = self._make_group()
        group.load_state(tmp_path / "nonexistent.json")  # should not raise


# --- make_keyword_category_infer_fn ---


class TestMakeKeywordCategoryInferFn:
    def test_basic_match(self):
        categories = ("scientific", "medical", "general")
        patterns = {
            "scientific": re.compile(r"\bscience\b", re.IGNORECASE),
            "medical": re.compile(r"\bhealth\b", re.IGNORECASE),
        }
        infer = make_keyword_category_infer_fn(categories, patterns, default_category="general")
        dist = infer("Tell me about science")
        assert dist[0] > dist[1]  # scientific > medical
        assert dist[0] > dist[2]  # scientific > general
        assert dist.sum() == pytest.approx(1.0)

    def test_no_match_default(self):
        categories = ("a", "b", "default")
        patterns = {"a": re.compile(r"\bfoo\b"), "b": re.compile(r"\bbar\b")}
        infer = make_keyword_category_infer_fn(categories, patterns, default_category="default")
        dist = infer("nothing matches here")
        assert dist[2] > dist[0]  # default boosted
        assert dist.sum() == pytest.approx(1.0)

    def test_no_default(self):
        categories = ("a", "b")
        patterns = {"a": re.compile(r"\bfoo\b")}
        infer = make_keyword_category_infer_fn(categories, patterns)
        dist = infer("nothing matches")
        assert_allclose(dist, 0.5)  # uniform

    def test_pattern_list(self):
        categories = ("cat_a", "cat_b")
        patterns = {
            "cat_a": [re.compile(r"\bfoo\b"), re.compile(r"\bbar\b")],
        }
        infer = make_keyword_category_infer_fn(categories, patterns, match_boost=9.0)
        dist = infer("something about foo")
        assert dist[0] > dist[1]

    def test_custom_boost(self):
        categories = ("a", "b")
        patterns = {"a": re.compile(r"\bx\b")}
        infer_low = make_keyword_category_infer_fn(categories, patterns, match_boost=2.0)
        infer_high = make_keyword_category_infer_fn(categories, patterns, match_boost=20.0)
        d_low = infer_low("x")
        d_high = infer_high("x")
        assert d_high[0] > d_low[0]  # higher boost -> more concentrated

    def test_count_matches(self):
        categories = ("combat", "exploration", "other")
        patterns = {
            "combat": re.compile(r"\b(?:attack|fight|sword)\b", re.IGNORECASE),
            "exploration": re.compile(r"\b(?:room|door|passage)\b", re.IGNORECASE),
        }
        # Boolean mode: one match = same boost regardless of count
        infer_bool = make_keyword_category_infer_fn(
            categories,
            patterns,
            match_boost=2.0,
            count_matches=False,
        )
        # Count mode: multiple matches scale the boost
        infer_count = make_keyword_category_infer_fn(
            categories,
            patterns,
            match_boost=2.0,
            count_matches=True,
        )
        text = "attack the troll with a sword then fight again"
        d_bool = infer_bool(text)
        d_count = infer_count(text)
        # Boolean mode: combat gets flat 2.0 boost
        assert d_bool[0] == pytest.approx((1.0 + 2.0) / (3.0 + 2.0))
        # Count mode: 3 combat matches → 6.0 boost → more concentrated
        assert d_count[0] > d_bool[0]
        assert d_count[0] == pytest.approx((1.0 + 6.0) / (3.0 + 6.0))
        assert d_count.sum() == pytest.approx(1.0)

    def test_count_matches_zero_hits(self):
        categories = ("a", "b")
        patterns = {"a": re.compile(r"\bfoo\b")}
        infer = make_keyword_category_infer_fn(
            categories,
            patterns,
            match_boost=5.0,
            count_matches=True,
        )
        dist = infer("nothing matches")
        assert_allclose(dist, 0.5)  # uniform — zero matches, zero boost


# --- make_router_category_infer_fn ---


class TestMakeRouterCategoryInferFn:
    def test_returns_distribution(self):
        categories = ("a", "b", "c")
        patterns = {
            "a": re.compile(r"\balpha\b"),
            "b": re.compile(r"\bbeta\b"),
        }
        infer = make_router_category_infer_fn(categories, patterns)
        dist = infer("something about alpha")
        assert len(dist) == 3
        assert dist.sum() == pytest.approx(1.0)

    def test_exposes_router(self):
        categories = ("a", "b")
        patterns = {"a": re.compile(r"\bfoo\b")}
        infer = make_router_category_infer_fn(categories, patterns)
        assert hasattr(infer, "router")
        assert isinstance(infer.router, Router)
