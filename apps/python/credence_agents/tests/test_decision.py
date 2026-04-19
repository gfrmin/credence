# Role: body
"""Tests for post-question reliability update logic.

Action selection and Bayesian updates are now in Julia. These tests cover
the Python-side feedback-to-correctness mapping functions.
"""

from credence_agents.inference.decision import (
    compute_binary_reliability_updates,
    compute_reliability_updates,
)


class TestComputeReliabilityUpdates:
    def test_correct_submission_tools_agree_disagree(self):
        """Correct submission: agreeing tools → True, disagreeing → False."""
        updates = compute_reliability_updates(
            submitted_answer_idx=2,
            was_correct=True,
            tool_responses={0: 2, 1: 0, 2: None},
        )
        assert updates[0] is True   # agreed
        assert updates[1] is False  # disagreed
        assert updates[2] is None   # no response

    def test_wrong_submission_tools_agree_disagree(self):
        """Wrong submission: tools that agreed with us → False, disagreed → None."""
        updates = compute_reliability_updates(
            submitted_answer_idx=1,
            was_correct=False,
            tool_responses={0: 1, 1: 3, 2: None},
        )
        assert updates[0] is False  # agreed with wrong answer
        assert updates[1] is None   # disagreed (might be right)
        assert updates[2] is None   # no response

    def test_abstained(self):
        """Abstained: all None."""
        updates = compute_reliability_updates(
            submitted_answer_idx=None,
            was_correct=None,
            tool_responses={0: 2, 1: None},
        )
        assert all(v is None for v in updates.values())

    def test_was_correct_none(self):
        """was_correct=None: all None."""
        updates = compute_reliability_updates(
            submitted_answer_idx=0,
            was_correct=None,
            tool_responses={0: 0},
        )
        assert updates[0] is None


class TestComputeBinaryReliabilityUpdates:
    def test_correct(self):
        updates = compute_binary_reliability_updates(
            tool_responses={0: 2, 1: None},
            was_correct=True,
        )
        assert updates[0] is True
        assert updates[1] is None

    def test_wrong(self):
        updates = compute_binary_reliability_updates(
            tool_responses={0: 2, 1: None},
            was_correct=False,
        )
        assert updates[0] is False
        assert updates[1] is None

    def test_none(self):
        updates = compute_binary_reliability_updates(
            tool_responses={0: 2},
            was_correct=None,
        )
        assert updates[0] is None
