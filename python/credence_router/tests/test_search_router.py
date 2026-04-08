"""Tests for SearchRouter and search tool protocols."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

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
