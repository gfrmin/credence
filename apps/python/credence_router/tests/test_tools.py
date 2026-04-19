# Role: body
"""Tests for built-in tools."""

from __future__ import annotations


from credence_router.tools.calculator import (
    CalculatorTool,
    _extract_expression,
    _safe_ast_eval as safe_math_eval,
)
from credence_router.tools.simulated import SimulatedTool, make_default_simulated_tools


class TestCalculatorTool:
    def setup_method(self):
        self.calc = CalculatorTool()

    def test_properties(self):
        assert self.calc.name == "calculator"
        assert self.calc.cost == 0.0
        assert self.calc.latency == 0.001

    def test_coverage_numerical_only(self):
        cats = ("factual", "numerical", "reasoning")
        cov = self.calc.coverage(cats)
        assert cov[0] == 0.0  # factual
        assert cov[1] == 1.0  # numerical
        assert cov[2] == 0.0  # reasoning

    def test_percentage_calculation(self):
        result = self.calc.query(
            "What is 17% of 4,230?",
            ("718.1", "719.1", "721.1", "723.1"),
        )
        assert result == 1  # 17% of 4230 = 719.1

    def test_square_root(self):
        result = self.calc.query(
            "What is the square root of 1,764?",
            ("38", "40", "42", "44"),
        )
        assert result == 2  # sqrt(1764) = 42

    def test_power(self):
        result = self.calc.query(
            "What is 2^15?",
            ("16384", "32768", "65536", "8192"),
        )
        assert result == 1  # 2^15 = 32768

    def test_non_numerical_returns_none(self):
        result = self.calc.query(
            "What is the capital of France?",
            ("Paris", "London", "Berlin", "Madrid"),
        )
        assert result is None


class TestSafeMathEval:
    def test_basic_arithmetic(self):
        assert safe_math_eval("2 + 3") == 5.0
        assert safe_math_eval("10 * 5") == 50.0
        assert safe_math_eval("100 / 4") == 25.0

    def test_power(self):
        assert safe_math_eval("2 ** 15") == 32768.0

    def test_sqrt(self):
        result = safe_math_eval("sqrt(1764)")
        assert result is not None
        assert abs(result - 42.0) < 0.01

    def test_rejects_invalid(self):
        assert safe_math_eval("import os") is None
        assert safe_math_eval("__import__('os')") is None


class TestExtractExpression:
    def test_percentage(self):
        expr = _extract_expression("What is 17% of 4,230?")
        assert expr is not None
        result = safe_math_eval(expr)
        assert result is not None
        assert abs(result - 719.1) < 0.1

    def test_power(self):
        expr = _extract_expression("What is 2^15?")
        assert expr is not None
        result = safe_math_eval(expr)
        assert result == 32768.0


class TestSimulatedTool:
    def test_properties(self):
        tool = SimulatedTool(
            name="test",
            cost=0.001,
            latency=0.1,
            reliability_by_category={"factual": 0.8},
            seed=42,
        )
        assert tool.name == "test"
        assert tool.cost == 0.001
        assert tool.latency == 0.1

    def test_coverage_array(self):
        tool = SimulatedTool(
            name="test",
            cost=0.001,
            latency=0.1,
            reliability_by_category={"factual": 0.8, "numerical": 0.0},
            coverage_by_category={"factual": 1.0, "numerical": 0.0},
            seed=42,
        )
        cov = tool.coverage(("factual", "numerical", "reasoning"))
        assert cov[0] == 1.0
        assert cov[1] == 0.0
        assert cov[2] == 0.0

    def test_deterministic_with_same_seed(self):
        """Same seed produces same results."""
        tool1 = SimulatedTool(
            name="test",
            cost=0.001,
            latency=0.1,
            reliability_by_category={"factual": 0.8},
            coverage_by_category={"factual": 1.0},
            seed=42,
        )
        tool2 = SimulatedTool(
            name="test",
            cost=0.001,
            latency=0.1,
            reliability_by_category={"factual": 0.8},
            coverage_by_category={"factual": 1.0},
            seed=42,
        )
        q = "What is the capital of France?"
        c = ("Paris", "London", "Berlin", "Madrid")
        r1 = tool1.query(q, c)
        r2 = tool2.query(q, c)
        assert r1 == r2

    def test_make_default_tools(self):
        tools = make_default_simulated_tools()
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"calculator", "cheap_llm", "expert_llm", "web_search"}
