# Role: body
"""Tests for the benchmark harness: scoring, abstention, tool costs, guards, snapshots."""

from __future__ import annotations

import pytest

from credence_agents.inference.decision import Action, ActionType
from credence_agents.inference.voi import PENALTY_WRONG, REWARD_CORRECT
from credence_agents.environment.benchmark import run_benchmark, BenchmarkResult
from credence_agents.environment.questions import Question
from credence_agents.environment.tools import make_spec_tools


# --- Test helpers ---

def _make_questions(n: int = 5) -> list[Question]:
    """Small question set: candidate 0 is correct for even indices, candidate 1 for odd."""
    return [
        Question(
            id=f"q{i}", text=f"Question {i}?",
            candidates=("A", "B", "C", "D"),
            correct_index=i % 2, category="factual", difficulty="easy",
        )
        for i in range(n)
    ]


class _AlwaysSubmitZero:
    """Mock agent that always immediately submits candidate 0."""
    name = "always_submit_0"

    def on_question_start(self, question_id, candidates, num_tools, question_text=""):
        pass

    def choose_action(self):
        return Action(ActionType.SUBMIT, answer_idx=0)

    def on_tool_response(self, tool_idx, response):
        pass

    def on_question_end(self, was_correct):
        pass


class _AlwaysAbstain:
    """Mock agent that always abstains."""
    name = "always_abstain"

    def on_question_start(self, question_id, candidates, num_tools, question_text=""):
        pass

    def choose_action(self):
        return Action(ActionType.ABSTAIN)

    def on_tool_response(self, tool_idx, response):
        pass

    def on_question_end(self, was_correct):
        pass


class _QueryThenSubmit:
    """Mock agent that queries tool 0 once, then submits candidate 0."""
    name = "query_then_submit"

    def __init__(self):
        self._queried = False

    def on_question_start(self, question_id, candidates, num_tools, question_text=""):
        self._queried = False

    def choose_action(self):
        if not self._queried:
            self._queried = True
            return Action(ActionType.QUERY, tool_idx=0)
        return Action(ActionType.SUBMIT, answer_idx=0)

    def on_tool_response(self, tool_idx, response):
        pass

    def on_question_end(self, was_correct):
        pass


class _DoubleQuerySameTool:
    """Mock agent that queries tool 0 twice — should trigger guard."""
    name = "double_query"

    def __init__(self):
        self._count = 0

    def on_question_start(self, question_id, candidates, num_tools, question_text=""):
        self._count = 0

    def choose_action(self):
        self._count += 1
        return Action(ActionType.QUERY, tool_idx=0)

    def on_tool_response(self, tool_idx, response):
        pass

    def on_question_end(self, was_correct):
        pass


class _SnapshotAgent:
    """Mock agent that provides a belief snapshot."""
    name = "snapshot_agent"

    def __init__(self):
        self._question_id = None

    def on_question_start(self, question_id, candidates, num_tools, question_text=""):
        self._question_id = question_id

    def choose_action(self):
        return Action(ActionType.SUBMIT, answer_idx=0)

    def on_tool_response(self, tool_idx, response):
        pass

    def on_question_end(self, was_correct):
        pass

    def get_belief_snapshot(self):
        return {"question_id": self._question_id, "confidence": 0.75}


# --- Tests ---

class TestScoring:
    def test_submit_correct_and_wrong(self):
        """Candidate 0 is correct for even indices, wrong for odd."""
        questions = _make_questions(5)
        tools = make_spec_tools()
        result = run_benchmark(_AlwaysSubmitZero(), tools, questions)

        # q0: correct (idx 0), q1: wrong (idx 1), q2: correct, q3: wrong, q4: correct
        expected_correct = 3
        expected_wrong = 2
        expected_reward = expected_correct * REWARD_CORRECT + expected_wrong * PENALTY_WRONG

        assert result.total_reward == expected_reward
        assert len(result.records) == 5
        assert result.agent_name == "always_submit_0"

    def test_individual_record_rewards(self):
        questions = _make_questions(4)
        tools = make_spec_tools()
        result = run_benchmark(_AlwaysSubmitZero(), tools, questions)

        # q0: correct, q1: wrong, q2: correct, q3: wrong
        assert result.records[0].was_correct is True
        assert result.records[0].reward == REWARD_CORRECT
        assert result.records[1].was_correct is False
        assert result.records[1].reward == PENALTY_WRONG


class TestAbstention:
    def test_all_abstain_zero_score(self):
        questions = _make_questions(5)
        tools = make_spec_tools()
        result = run_benchmark(_AlwaysAbstain(), tools, questions)

        assert result.total_reward == 0.0
        assert result.total_tool_cost == 0.0
        assert result.total_score == 0.0

        for rec in result.records:
            assert rec.action_type == "abstain"
            assert rec.submitted_answer is None
            assert rec.was_correct is None
            assert rec.reward == 0.0


class TestToolCostTracking:
    def test_query_then_submit_cost(self):
        questions = _make_questions(3)
        tools = make_spec_tools()
        result = run_benchmark(_QueryThenSubmit(), tools, questions)

        # Tool A costs 1.0 per query, agent queries once per question
        assert result.total_tool_cost == 3.0

        for rec in result.records:
            assert rec.tool_cost == 1.0
            assert rec.tools_queried == (0,)

    def test_total_score_includes_cost(self):
        questions = _make_questions(3)
        tools = make_spec_tools()
        result = run_benchmark(_QueryThenSubmit(), tools, questions)

        assert result.total_score == result.total_reward - result.total_tool_cost


class TestGuardRails:
    def test_double_query_raises(self):
        questions = _make_questions(1)
        tools = make_spec_tools()
        with pytest.raises(RuntimeError, match="queried tool 0 twice"):
            run_benchmark(_DoubleQuerySameTool(), tools, questions)


class TestBeliefSnapshot:
    def test_snapshot_populated(self):
        questions = _make_questions(3)
        tools = make_spec_tools()
        result = run_benchmark(_SnapshotAgent(), tools, questions)

        for rec in result.records:
            assert rec.belief_snapshot is not None
            assert "question_id" in rec.belief_snapshot
            assert rec.belief_snapshot["question_id"] == rec.question_id
            assert rec.belief_snapshot["confidence"] == 0.75

    def test_no_snapshot_is_none(self):
        """Agents without get_belief_snapshot return None."""
        questions = _make_questions(1)
        tools = make_spec_tools()
        result = run_benchmark(_AlwaysSubmitZero(), tools, questions)

        assert result.records[0].belief_snapshot is None


class TestBenchmarkResult:
    def test_seed_recorded(self):
        questions = _make_questions(1)
        tools = make_spec_tools()
        result = run_benchmark(_AlwaysSubmitZero(), tools, questions, seed=123)
        assert result.seed == 123

    def test_records_are_tuple(self):
        questions = _make_questions(2)
        tools = make_spec_tools()
        result = run_benchmark(_AlwaysSubmitZero(), tools, questions)
        assert isinstance(result.records, tuple)
