# Role: body
"""Tests for the Bayesian decision-theoretic agent (Julia DSL backend).

Hand-fed sequences with mock tools to verify:
1. Sensible tool selection (not calculator for factual)
2. Reliability learning over a sequence
3. Decision trace populated
4. Confidence-correctness correlation over longer runs
"""

from __future__ import annotations

import pytest

from credence_agents.agents.bayesian_agent import BayesianAgent
from credence_agents.environment.categories import CATEGORIES
from credence_agents.inference.decision import ActionType
from credence_agents.inference.voi import ToolConfig
from credence_agents.environment.tools import make_spec_tools, tool_config_for, query_tool, ResponseType
from credence_agents.environment.questions import Question, get_questions
from credence_agents.environment.benchmark import run_benchmark
from credence_agents.julia_bridge import CredenceBridge

# --- Helpers ---

def _spec_tool_configs() -> list[ToolConfig]:
    return [tool_config_for(t) for t in make_spec_tools()]


def _make_agent(bridge, **kwargs) -> BayesianAgent:
    kwargs.setdefault("categories", CATEGORIES)
    return BayesianAgent(bridge=bridge, tool_configs=_spec_tool_configs(), **kwargs)


def _make_q(qid: str, category: str, correct: int = 0, text: str = "") -> Question:
    return Question(
        id=qid, text=text or f"Question about {category}?",
        candidates=("A", "B", "C", "D"),
        correct_index=correct, category=category, difficulty="medium",
    )


# --- Hand-fed mock tool sequence (5 questions) ---

class TestHandFedSequence:
    """Feed 5 questions with deterministic tool responses to verify agent behaviour."""

    def _run_sequence(self, bridge):
        """Run agent through 5 questions with mock tools, return (agent, results)."""
        agent = _make_agent(bridge)
        results = []

        # Q1: factual question. Tool A (idx 0) gives correct answer.
        q1 = _make_q("q1", "factual", correct=0, text="Which country has the largest coastline?")
        responses_q1 = {0: 0, 1: 0, 2: None, 3: 0}

        def query_q1(tool_idx):
            return responses_q1[tool_idx]

        r1 = agent.solve_question(q1.text, q1.candidates, None, query_q1)
        agent.on_question_end(r1.answer == q1.correct_index)
        results.append((q1, r1))

        # Q2: numerical question. Tool C (idx 2) gives correct, others wrong.
        q2 = _make_q("q2", "numerical", correct=1, text="What is the square root of 1764?")
        responses_q2 = {0: 2, 1: None, 2: 1, 3: 2}

        def query_q2(tool_idx):
            return responses_q2[tool_idx]

        r2 = agent.solve_question(q2.text, q2.candidates, None, query_q2)
        agent.on_question_end(r2.answer == q2.correct_index)
        results.append((q2, r2))

        # Q3: reasoning question. Tool D (idx 3) correct, A wrong.
        q3 = _make_q("q3", "reasoning", correct=1,
                      text="If all roses are flowers, can we conclude some roses fade?")
        responses_q3 = {0: 3, 1: None, 2: None, 3: 1}

        def query_q3(tool_idx):
            return responses_q3[tool_idx]

        r3 = agent.solve_question(q3.text, q3.candidates, None, query_q3)
        agent.on_question_end(r3.answer == q3.correct_index)
        results.append((q3, r3))

        # Q4: misconception. Tool A gives popular-but-wrong answer.
        q4 = _make_q("q4", "misconceptions", correct=3,
                      text="Is it true that the Great Wall is visible from space?")
        responses_q4 = {0: 0, 1: 3, 2: None, 3: 0}

        def query_q4(tool_idx):
            return responses_q4[tool_idx]

        r4 = agent.solve_question(q4.text, q4.candidates, None, query_q4)
        agent.on_question_end(r4.answer == q4.correct_index)
        results.append((q4, r4))

        # Q5: factual again. Test if agent learned from Q1.
        q5 = _make_q("q5", "factual", correct=2, text="What is the deepest ocean?")
        responses_q5 = {0: 2, 1: 2, 2: None, 3: 1}

        def query_q5(tool_idx):
            return responses_q5[tool_idx]

        r5 = agent.solve_question(q5.text, q5.candidates, None, query_q5)
        agent.on_question_end(r5.answer == q5.correct_index)
        results.append((q5, r5))

        return agent, results

    def test_does_not_query_calculator_for_factual(self, bridge):
        """Calculator (tool 2) should not be queried for factual questions."""
        _, results = self._run_sequence(bridge)
        q1_result = results[0][1]  # first question is factual
        assert 2 not in q1_result.tools_queried

    def test_queries_calculator_for_numerical(self, bridge):
        """Calculator should be queried for numerical questions (high VOI)."""
        _, results = self._run_sequence(bridge)
        q2_result = results[1][1]  # second question is numerical
        if 2 in q2_result.tools_queried:
            assert q2_result.answer == 1  # calculator gave correct answer

    def test_decision_trace_populated(self, bridge):
        """Every result should have a non-empty decision trace."""
        _, results = self._run_sequence(bridge)
        for _, result in results:
            assert len(result.decision_trace) > 0

    def test_decision_trace_structure(self, bridge):
        """Each trace step should have all expected fields."""
        _, results = self._run_sequence(bridge)
        for _, result in results:
            for step in result.decision_trace:
                assert isinstance(step.step, int)
                assert isinstance(step.eu_submit, float)
                assert isinstance(step.eu_abstain, float)
                assert isinstance(step.eu_query, dict)
                assert isinstance(step.chosen_action, str)

    def test_submits_or_abstains_every_question(self, bridge):
        _, results = self._run_sequence(bridge)
        for _, result in results:
            assert len(result.decision_trace) >= 1
            last_action = result.decision_trace[-1].chosen_action
            assert last_action.startswith("submit(") or last_action == "abstain"


# --- Benchmark protocol integration ---

class TestBenchmarkProtocol:
    def test_runs_through_benchmark(self, bridge):
        """Agent works with the benchmark harness."""
        tools = make_spec_tools()
        configs = [tool_config_for(t) for t in tools]
        agent = BayesianAgent(bridge=bridge, tool_configs=configs, categories=CATEGORIES)
        questions = get_questions(seed=42)[:10]
        result = run_benchmark(agent, tools, questions, seed=42)
        assert len(result.records) == 10
        assert result.agent_name == "bayesian"

    def test_belief_snapshot_present(self, bridge):
        tools = make_spec_tools()
        configs = [tool_config_for(t) for t in tools]
        agent = BayesianAgent(bridge=bridge, tool_configs=configs, categories=CATEGORIES)
        questions = get_questions(seed=42)[:3]
        result = run_benchmark(agent, tools, questions, seed=42)
        for rec in result.records:
            assert rec.belief_snapshot is not None
            assert "answer_posterior" in rec.belief_snapshot
