# Role: body
"""Tests for SearchRouter, search tool protocols, and eval harness."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from credence_router.benchmarks.search_eval import (
    ProviderResult,
    QueryData,
    RawEvalData,
    format_frontier_table,
    simulate_routing,
    sweep_preferences,
)
from credence_router.search_router import SearchRouter, SEARCH_CATEGORIES
from credence_router.tool import SearchResult, SearchTool


class FakeSearchTool:
    """Deterministic search tool for testing."""

    def __init__(
        self,
        name: str,
        cost: float = 0.005,
        latency: float = 1.0,
        coverage_map: dict[str, float] | None = None,
        always_returns: bool = True,
    ):
        self._name = name
        self._cost = cost
        self._latency = latency
        self._coverage_map = coverage_map or {c: 0.5 for c in SEARCH_CATEGORIES}
        self._always_returns = always_returns
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def latency(self) -> float:
        return self._latency

    def search(self, query: str) -> SearchResult | None:
        self.call_count += 1
        if not self._always_returns:
            return None
        return SearchResult(
            text=f"Results from {self._name} for: {query}",
            urls=[f"https://example.com/{self._name}"],
            provider=self._name,
            raw={"query": query},
        )

    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]:
        return np.array([self._coverage_map.get(c, 0.5) for c in categories])


def _make_test_tools() -> list[FakeSearchTool]:
    """Three fake providers with different strengths."""
    return [
        FakeSearchTool(
            "provider_a",
            cost=0.005,
            latency=1.0,
            coverage_map={"factual": 0.9, "recent_events": 0.3, "technical": 0.5,
                          "synthesis": 0.4, "local": 0.6},
        ),
        FakeSearchTool(
            "provider_b",
            cost=0.005,
            latency=3.0,
            coverage_map={"factual": 0.5, "recent_events": 0.9, "technical": 0.6,
                          "synthesis": 0.8, "local": 0.3},
        ),
        FakeSearchTool(
            "provider_c",
            cost=0.004,
            latency=2.0,
            coverage_map={"factual": 0.6, "recent_events": 0.5, "technical": 0.9,
                          "synthesis": 0.7, "local": 0.3},
        ),
    ]


class TestSearchToolProtocol:
    def test_fake_tool_satisfies_protocol(self):
        tool = FakeSearchTool("test")
        assert isinstance(tool, SearchTool)

    def test_search_returns_result(self):
        tool = FakeSearchTool("test")
        result = tool.search("hello")
        assert isinstance(result, SearchResult)
        assert result.provider == "test"
        assert "hello" in result.text

    def test_search_returns_none(self):
        tool = FakeSearchTool("test", always_returns=False)
        assert tool.search("hello") is None

    def test_coverage_returns_array(self):
        tool = FakeSearchTool("test")
        cov = tool.coverage(SEARCH_CATEGORIES)
        assert len(cov) == len(SEARCH_CATEGORIES)


class TestSearchRouterInit:
    def test_creates_with_tools(self):
        tools = _make_test_tools()
        router = SearchRouter(tools)
        assert router is not None

    def test_creates_with_custom_categories(self):
        tools = _make_test_tools()
        router = SearchRouter(tools, categories=("factual", "technical"))
        assert router is not None


class TestSearchRouterRoute:
    def test_returns_route_result(self):
        tools = _make_test_tools()
        router = SearchRouter(tools)
        result = router.route("What is the population of Tokyo?", category_hint="factual")
        assert result.provider in ("provider_a", "provider_b", "provider_c")
        assert result.result is not None
        assert result.wall_time > 0

    def test_returns_search_result_with_text(self):
        tools = _make_test_tools()
        router = SearchRouter(tools)
        result = router.route("test query")
        assert result.result is not None
        assert len(result.result.text) > 0

    def test_has_reasoning_trace(self):
        tools = _make_test_tools()
        router = SearchRouter(tools)
        result = router.route("test query")
        assert result.reasoning  # non-empty string


class TestSearchRouterLearning:
    def test_report_outcome_does_not_error(self):
        tools = _make_test_tools()
        router = SearchRouter(tools)
        router.route("test query")
        router.report_outcome(True)

    def test_learned_reliability_has_all_providers(self):
        tools = _make_test_tools()
        router = SearchRouter(tools)
        router.route("test query")
        router.report_outcome(True)
        reliability = router.learned_reliability
        assert set(reliability.keys()) == {"provider_a", "provider_b", "provider_c"}

    def test_reliability_values_are_probabilities(self):
        tools = _make_test_tools()
        router = SearchRouter(tools)
        for _ in range(5):
            router.route("some query")
            router.report_outcome(True)
        for provider, cats in router.learned_reliability.items():
            for cat, val in cats.items():
                assert 0.0 <= val <= 1.0, f"{provider}/{cat} = {val}"


class TestSearchRouterPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        tools = _make_test_tools()
        router = SearchRouter(tools)
        for _ in range(3):
            router.route("test")
            router.report_outcome(True)

        path = tmp_path / "state.json"
        router.save_state(path)
        assert path.exists()

        router2 = SearchRouter(tools)
        router2.load_state(path)

        # Learned state should be preserved
        rel1 = router.learned_reliability
        rel2 = router2.learned_reliability
        for provider in rel1:
            for cat in rel1[provider]:
                assert abs(rel1[provider][cat] - rel2[provider][cat]) < 0.01


# ---------------------------------------------------------------------------
# Two-phase eval tests
# ---------------------------------------------------------------------------

def _make_fake_raw_data() -> RawEvalData:
    """Build fake RawEvalData with 2 providers and 6 queries (no API calls)."""
    providers = ["fast_cheap", "slow_good"]
    costs = {"fast_cheap": 0.0, "slow_good": 0.005}
    latencies = {"fast_cheap": 1.0, "slow_good": 3.0}

    queries = []
    # 3 factual queries: slow_good is better
    for i in range(3):
        queries.append(QueryData(
            query=f"factual query {i}",
            category="factual",
            quality_criteria=["correct answer"],
            provider_results={
                "fast_cheap": ProviderResult(
                    provider="fast_cheap", text="ok answer", urls=["http://a.com"],
                    scores={"composite": 5.0, "relevance": 5, "freshness": 5,
                            "completeness": 5, "source_quality": 5},
                    wall_time=0.8,
                ),
                "slow_good": ProviderResult(
                    provider="slow_good", text="great answer", urls=["http://b.com"],
                    scores={"composite": 8.0, "relevance": 8, "freshness": 8,
                            "completeness": 8, "source_quality": 8},
                    wall_time=2.5,
                ),
            },
        ))
    # 3 recent_events queries: slow_good is much better
    for i in range(3):
        queries.append(QueryData(
            query=f"latest news {i}",
            category="recent_events",
            quality_criteria=["recent info"],
            provider_results={
                "fast_cheap": ProviderResult(
                    provider="fast_cheap", text="stale", urls=["http://a.com"],
                    scores={"composite": 2.0, "relevance": 3, "freshness": 1,
                            "completeness": 2, "source_quality": 3},
                    wall_time=0.5,
                ),
                "slow_good": ProviderResult(
                    provider="slow_good", text="fresh synthesis", urls=["http://b.com"],
                    scores={"composite": 9.0, "relevance": 9, "freshness": 9,
                            "completeness": 9, "source_quality": 9},
                    wall_time=3.0,
                ),
            },
        ))

    return RawEvalData(
        queries=queries,
        provider_names=providers,
        provider_costs=costs,
        provider_latencies=latencies,
    )


class TestRawEvalDataPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        raw = _make_fake_raw_data()
        path = tmp_path / "raw.json"
        raw.save(path)
        loaded = RawEvalData.load(path)
        assert len(loaded.queries) == len(raw.queries)
        assert loaded.provider_names == raw.provider_names
        assert loaded.provider_costs == raw.provider_costs
        for qd_orig, qd_loaded in zip(raw.queries, loaded.queries):
            assert qd_orig.query == qd_loaded.query
            assert set(qd_orig.provider_results.keys()) == set(qd_loaded.provider_results.keys())


class TestSimulateRouting:
    def test_returns_results_for_all_solvers(self):
        raw = _make_fake_raw_data()
        results = simulate_routing(raw, reward=0.25, latency_weight=0.05)
        solver_names = {r.solver_name for r in results}
        assert "credence-search" in solver_names
        assert "openclaw-fallback" in solver_names
        assert "round-robin" in solver_names
        assert "random-search" in solver_names
        assert "static-fast_cheap" in solver_names
        assert "static-slow_good" in solver_names

    def test_all_solvers_have_correct_query_count(self):
        raw = _make_fake_raw_data()
        results = simulate_routing(raw)
        for r in results:
            assert len(r.results) == 6, f"{r.solver_name} has {len(r.results)} results"

    def test_static_solver_uses_one_provider(self):
        raw = _make_fake_raw_data()
        results = simulate_routing(raw)
        for r in results:
            if r.solver_name.startswith("static-"):
                providers_used = set(qr.provider for qr in r.results)
                assert len(providers_used) == 1

    def test_fallback_always_picks_first(self):
        raw = _make_fake_raw_data()
        results = simulate_routing(raw)
        for r in results:
            if r.solver_name == "openclaw-fallback":
                for qr in r.results:
                    assert qr.provider == "fast_cheap"

    def test_effective_utility_varies_with_params(self):
        raw = _make_fake_raw_data()
        r_quality = simulate_routing(raw, reward=1.0, latency_weight=0.0)
        r_speed = simulate_routing(raw, reward=0.05, latency_weight=0.20)

        # With high reward/low latency cost, static-slow_good should have high EU
        quality_eus = {r.solver_name: r.effective_utility(1.0, 0.0) for r in r_quality}
        speed_eus = {r.solver_name: r.effective_utility(0.05, 0.20) for r in r_speed}

        # slow_good has better quality, so should win when quality dominates
        assert quality_eus["static-slow_good"] > quality_eus["static-fast_cheap"]
        # fast_cheap should do relatively better when speed matters
        # (the gap should shrink or reverse)
        quality_gap = quality_eus["static-slow_good"] - quality_eus["static-fast_cheap"]
        speed_gap = speed_eus["static-slow_good"] - speed_eus["static-fast_cheap"]
        assert speed_gap < quality_gap


class TestPreferenceSweep:
    def test_sweep_produces_frontier_points(self):
        raw = _make_fake_raw_data()
        frontier = sweep_preferences(
            raw,
            seeds=[42],
            reward_values=[0.10, 1.00],
            latency_weight_values=[0.01, 0.10],
        )
        assert len(frontier.points) == 4  # 2 × 2

    def test_frontier_points_have_all_solvers(self):
        raw = _make_fake_raw_data()
        frontier = sweep_preferences(
            raw,
            seeds=[42],
            reward_values=[0.25],
            latency_weight_values=[0.05],
        )
        assert len(frontier.points) == 1
        solver_names = {r.solver_name for r in frontier.points[0].solver_results}
        assert "credence-search" in solver_names
        assert "openclaw-fallback" in solver_names

    def test_format_frontier_table_runs(self):
        raw = _make_fake_raw_data()
        frontier = sweep_preferences(
            raw,
            seeds=[42],
            reward_values=[0.10, 1.00],
            latency_weight_values=[0.01, 0.10],
        )
        table = format_frontier_table(frontier)
        assert "PREFERENCE FRONTIER" in table
        assert "credence-search" in table
