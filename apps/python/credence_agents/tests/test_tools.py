"""Tests for simulated tools: structure, determinism, rates, coverage, and bridge."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from credence_agents.environment.questions import Question
from credence_agents.environment.tools import (
    ResponseType,
    SimulatedTool,
    make_spec_tools,
    query_tool,
    tool_config_for,
)
from credence_agents.environment.categories import CATEGORIES


# --- Fixtures ---

@pytest.fixture
def spec_tools():
    return make_spec_tools()


def _make_question(correct_index: int = 0, category: str = "factual") -> Question:
    return Question(
        id="test", text="Test?",
        candidates=("A", "B", "C", "D"),
        correct_index=correct_index, category=category, difficulty="easy",
    )


# --- Structure tests ---

class TestSpecToolStructure:
    def test_four_tools(self, spec_tools):
        assert len(spec_tools) == 4

    def test_names(self, spec_tools):
        names = [t.name for t in spec_tools]
        assert names == ["quick_search", "knowledge_base", "calculator", "llm_direct"]

    def test_costs(self, spec_tools):
        costs = [t.cost for t in spec_tools]
        assert costs == [1.0, 2.0, 1.0, 2.0]

    def test_tool_a_reliability(self, spec_tools):
        r = spec_tools[0].reliability_by_category
        assert r == {"factual": 0.70, "numerical": 0.20, "recent_events": 0.65,
                      "misconceptions": 0.25, "reasoning": 0.40}

    def test_tool_b_reliability(self, spec_tools):
        r = spec_tools[1].reliability_by_category
        assert r == {"factual": 0.92, "numerical": 0.40, "recent_events": 0.55,
                      "misconceptions": 0.88, "reasoning": 0.45}

    def test_tool_c_reliability(self, spec_tools):
        r = spec_tools[2].reliability_by_category
        assert r == {"factual": 0.0, "numerical": 1.0, "recent_events": 0.0,
                      "misconceptions": 0.0, "reasoning": 0.0}

    def test_tool_d_reliability(self, spec_tools):
        r = spec_tools[3].reliability_by_category
        assert r == {"factual": 0.65, "numerical": 0.50, "recent_events": 0.45,
                      "misconceptions": 0.40, "reasoning": 0.72}

    def test_tool_b_coverage(self, spec_tools):
        c = spec_tools[1].coverage_by_category
        assert c == {"factual": 0.65, "numerical": 0.30, "recent_events": 0.35,
                      "misconceptions": 0.55, "reasoning": 0.20}

    def test_tool_c_coverage(self, spec_tools):
        c = spec_tools[2].coverage_by_category
        assert c == {"factual": 0.0, "numerical": 1.0, "recent_events": 0.0,
                      "misconceptions": 0.0, "reasoning": 0.0}

    def test_tools_a_d_full_coverage(self, spec_tools):
        for idx in (0, 3):  # A and D
            for cat in CATEGORIES:
                assert spec_tools[idx].coverage_by_category[cat] == 1.0

    def test_no_answer_types(self, spec_tools):
        assert spec_tools[1].no_answer_type == ResponseType.NO_RESULT
        assert spec_tools[2].no_answer_type == ResponseType.NOT_APPLICABLE


# --- Determinism ---

class TestDeterminism:
    def test_same_seed_same_result(self, spec_tools):
        q = _make_question()
        results_a = [query_tool(spec_tools[0], q, np.random.default_rng(99)) for _ in range(10)]
        results_b = [query_tool(spec_tools[0], q, np.random.default_rng(99)) for _ in range(10)]
        assert results_a == results_b

    def test_sequential_calls_deterministic(self, spec_tools):
        q = _make_question()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        for _ in range(20):
            assert query_tool(spec_tools[0], q, rng1) == query_tool(spec_tools[0], q, rng2)


# --- Rate tests (statistical) ---

class TestRates:
    N = 2000

    def _measure_correct_rate(self, tool, category, seed=12345):
        q = _make_question(category=category)
        rng = np.random.default_rng(seed)
        answers = [query_tool(tool, q, rng) for _ in range(self.N)]
        covered = [a for a in answers if a.response_type == ResponseType.ANSWER]
        if not covered:
            return 0.0
        correct = sum(1 for a in covered if a.candidate_idx == q.correct_index)
        return correct / len(covered)

    @pytest.mark.parametrize("cat,expected", [
        ("factual", 0.70), ("numerical", 0.20), ("recent_events", 0.65),
        ("misconceptions", 0.25), ("reasoning", 0.40),
    ])
    def test_tool_a_rates(self, spec_tools, cat, expected):
        rate = self._measure_correct_rate(spec_tools[0], cat)
        assert abs(rate - expected) < 0.05, f"Tool A {cat}: got {rate:.3f}, expected {expected}"

    @pytest.mark.parametrize("cat,expected", [
        ("factual", 0.92), ("numerical", 0.40), ("recent_events", 0.55),
        ("misconceptions", 0.88), ("reasoning", 0.45),
    ])
    def test_tool_b_rates(self, spec_tools, cat, expected):
        rate = self._measure_correct_rate(spec_tools[1], cat)
        assert abs(rate - expected) < 0.08, f"Tool B {cat}: got {rate:.3f}, expected {expected}"

    def test_tool_c_numerical_perfect(self, spec_tools):
        q = _make_question(category="numerical")
        rng = np.random.default_rng(42)
        for _ in range(100):
            resp = query_tool(spec_tools[2], q, rng)
            assert resp.response_type == ResponseType.ANSWER
            assert resp.candidate_idx == q.correct_index

    @pytest.mark.parametrize("cat,expected", [
        ("factual", 0.65), ("numerical", 0.50), ("recent_events", 0.45),
        ("misconceptions", 0.40), ("reasoning", 0.72),
    ])
    def test_tool_d_rates(self, spec_tools, cat, expected):
        rate = self._measure_correct_rate(spec_tools[3], cat)
        assert abs(rate - expected) < 0.05, f"Tool D {cat}: got {rate:.3f}, expected {expected}"


# --- Coverage tests ---

class TestCoverage:
    N = 2000

    def test_tool_b_no_result_rate(self, spec_tools):
        """Tool B returns NO_RESULT at approximately (1 - coverage) rate."""
        tool_b = spec_tools[1]
        for cat in CATEGORIES:
            q = _make_question(category=cat)
            rng = np.random.default_rng(777)
            no_results = sum(
                1 for _ in range(self.N)
                if query_tool(tool_b, q, rng).response_type == ResponseType.NO_RESULT
            )
            expected_no_result = 1.0 - tool_b.coverage_by_category[cat]
            actual = no_results / self.N
            assert abs(actual - expected_no_result) < 0.05, (
                f"Tool B {cat}: no_result rate {actual:.3f}, expected {expected_no_result}"
            )

    def test_tool_c_not_applicable_non_numerical(self, spec_tools):
        """Tool C returns NOT_APPLICABLE for all non-numerical categories."""
        tool_c = spec_tools[2]
        for cat in ("factual", "recent_events", "misconceptions", "reasoning"):
            q = _make_question(category=cat)
            rng = np.random.default_rng(42)
            for _ in range(20):
                resp = query_tool(tool_c, q, rng)
                assert resp.response_type == ResponseType.NOT_APPLICABLE
                assert resp.candidate_idx is None

    def test_tool_c_answers_numerical(self, spec_tools):
        """Tool C always returns ANSWER for numerical questions."""
        tool_c = spec_tools[2]
        q = _make_question(category="numerical")
        rng = np.random.default_rng(42)
        for _ in range(50):
            resp = query_tool(tool_c, q, rng)
            assert resp.response_type == ResponseType.ANSWER


# --- Wrong answer uniformity ---

class TestWrongAnswerDistribution:
    def test_uniform_wrong_answers(self, spec_tools):
        """When Tool A is wrong, each of the 3 wrong candidates appears roughly equally."""
        q = _make_question(correct_index=0)
        rng = np.random.default_rng(54321)
        # Force many wrong answers by using a category where reliability is low
        wrong = Counter()
        n_wrong = 0
        for _ in range(6000):
            resp = query_tool(spec_tools[0], q, rng)
            if resp.response_type == ResponseType.ANSWER and resp.candidate_idx != q.correct_index:
                wrong[resp.candidate_idx] += 1
                n_wrong += 1

        assert n_wrong > 100, "Not enough wrong answers to test distribution"
        for idx in (1, 2, 3):
            frac = wrong[idx] / n_wrong
            assert abs(frac - 1 / 3) < 0.06, (
                f"Wrong answer {idx} fraction {frac:.3f}, expected ~0.333"
            )


# --- Bridge to inference layer ---

class TestToolConfigBridge:
    def test_tool_config_cost(self, spec_tools):
        for tool in spec_tools:
            config = tool_config_for(tool)
            assert config.cost == tool.cost

    def test_tool_config_coverage_shape(self, spec_tools):
        for tool in spec_tools:
            config = tool_config_for(tool)
            assert config.coverage_by_category.shape == (len(CATEGORIES),)

    def test_tool_config_coverage_values(self, spec_tools):
        tool_c = spec_tools[2]
        config = tool_config_for(tool_c)
        for i, cat in enumerate(CATEGORIES):
            expected = tool_c.coverage_by_category.get(cat, 0.0)
            assert config.coverage_by_category[i] == expected

    def test_tool_a_full_coverage(self, spec_tools):
        config = tool_config_for(spec_tools[0])
        np.testing.assert_array_equal(config.coverage_by_category, np.ones(len(CATEGORIES)))
