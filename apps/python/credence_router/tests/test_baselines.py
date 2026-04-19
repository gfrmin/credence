# Role: body
"""Tests for baseline solvers."""

from __future__ import annotations


from credence_router.answer import Answer
from credence_router.baselines.simple import (
    AlwaysBestSolver,
    AlwaysCheapestSolver,
    RandomSolver,
)
from credence_router.baselines.langgraph_react import LangGraphReActSolver
from credence_router.tools.simulated import make_default_simulated_tools


QUESTION = "What is the capital of France?"
CANDIDATES = ("Paris", "London", "Berlin", "Madrid")


class TestRandomSolver:
    def test_returns_answer(self):
        tools = make_default_simulated_tools()
        solver = RandomSolver(tools=tools, seed=42)
        answer = solver.solve(QUESTION, CANDIDATES)
        assert isinstance(answer, Answer)
        assert answer.monetary_cost >= 0

    def test_deterministic_with_seed(self):
        tools = make_default_simulated_tools()
        s1 = RandomSolver(tools=tools, seed=42)
        s2 = RandomSolver(tools=tools, seed=42)
        a1 = s1.solve(QUESTION, CANDIDATES)
        a2 = s2.solve(QUESTION, CANDIDATES)
        assert a1.tools_used == a2.tools_used


class TestAlwaysCheapestSolver:
    def test_returns_answer(self):
        tools = make_default_simulated_tools()
        solver = AlwaysCheapestSolver(tools=tools)
        answer = solver.solve(QUESTION, CANDIDATES)
        assert isinstance(answer, Answer)

    def test_uses_cheapest_tool(self):
        tools = make_default_simulated_tools()
        solver = AlwaysCheapestSolver(tools=tools)
        # For a factual question, calculator has 0 coverage,
        # so it should fall through to cheap_llm
        answer = solver.solve(QUESTION, CANDIDATES)
        if answer.tools_used:
            # First tool tried should be cheapest (calculator at $0)
            assert answer.tools_used[0] in ("calculator", "cheap_llm")


class TestAlwaysBestSolver:
    def test_returns_answer(self):
        tools = make_default_simulated_tools()
        solver = AlwaysBestSolver(tools=tools)
        answer = solver.solve(QUESTION, CANDIDATES)
        assert isinstance(answer, Answer)


class TestLangGraphReActSolver:
    def test_simulated_mode(self):
        """Should work without LangGraph installed (falls back to simulation)."""
        tools = make_default_simulated_tools()
        solver = LangGraphReActSolver(tools=tools)
        answer = solver.solve(QUESTION, CANDIDATES)
        assert isinstance(answer, Answer)
        assert answer.monetary_cost > 0  # routing cost included
        assert "routing" in answer.reasoning.lower()
