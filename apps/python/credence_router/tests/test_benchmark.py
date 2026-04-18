"""Tests for the benchmark runner."""

from __future__ import annotations


from credence_router.benchmark import BenchmarkResult, format_comparison_table, run_benchmark
from credence_router.questions import get_questions
from credence_router.router import Router
from credence_router.tools.simulated import make_default_simulated_tools
from credence_router.baselines.simple import AlwaysCheapestSolver, RandomSolver


class TestRunBenchmark:
    def test_runs_all_questions(self):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)
        questions = get_questions(seed=42)[:5]  # subset for speed

        result = run_benchmark(router, questions)
        assert result.agent_name == "credence-router"
        assert len(result.results) == 5

    def test_accuracy_in_range(self):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)
        questions = get_questions(seed=42)[:10]

        result = run_benchmark(router, questions)
        assert 0.0 <= result.accuracy <= 1.0

    def test_costs_non_negative(self):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)
        questions = get_questions(seed=42)[:5]

        result = run_benchmark(router, questions)
        assert result.total_monetary_cost >= 0
        assert result.total_effective_cost >= 0


class TestBaselines:
    def test_random_solver(self):
        tools = make_default_simulated_tools()
        solver = RandomSolver(tools=tools, seed=42)
        questions = get_questions(seed=42)[:5]

        result = run_benchmark(solver, questions)
        assert len(result.results) == 5

    def test_always_cheapest(self):
        tools = make_default_simulated_tools()
        solver = AlwaysCheapestSolver(tools=tools)
        questions = get_questions(seed=42)[:5]

        result = run_benchmark(solver, questions)
        assert len(result.results) == 5


class TestFormatComparison:
    def test_format_table(self):
        result = BenchmarkResult(agent_name="test")
        table = format_comparison_table([result])
        assert "test" in table
        assert "Agent" in table
