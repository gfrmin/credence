"""Tests for the question bank: count, distribution, structure, uniqueness, shuffling."""

from __future__ import annotations

from collections import Counter

import pytest

from credence_agents.environment.questions import QUESTION_BANK, Question, get_questions
from credence_agents.environment.categories import CATEGORIES


class TestQuestionCount:
    def test_exactly_50(self):
        assert len(QUESTION_BANK) == 50


class TestCategoryDistribution:
    def test_distribution(self):
        counts = Counter(q.category for q in QUESTION_BANK)
        assert counts["factual"] == 15
        assert counts["numerical"] == 10
        assert counts["recent_events"] == 8
        assert counts["misconceptions"] == 7
        assert counts["reasoning"] == 10


class TestQuestionStructure:
    @pytest.mark.parametrize("question", QUESTION_BANK, ids=lambda q: q.id)
    def test_four_candidates(self, question):
        assert len(question.candidates) == 4

    @pytest.mark.parametrize("question", QUESTION_BANK, ids=lambda q: q.id)
    def test_correct_index_in_range(self, question):
        assert 0 <= question.correct_index <= 3

    @pytest.mark.parametrize("question", QUESTION_BANK, ids=lambda q: q.id)
    def test_valid_category(self, question):
        assert question.category in CATEGORIES

    @pytest.mark.parametrize("question", QUESTION_BANK, ids=lambda q: q.id)
    def test_valid_difficulty(self, question):
        assert question.difficulty in ("easy", "medium", "hard")

    @pytest.mark.parametrize("question", QUESTION_BANK, ids=lambda q: q.id)
    def test_has_text(self, question):
        assert len(question.text) > 0

    @pytest.mark.parametrize("question", QUESTION_BANK, ids=lambda q: q.id)
    def test_has_id(self, question):
        assert len(question.id) > 0


class TestUniqueness:
    def test_unique_ids(self):
        ids = [q.id for q in QUESTION_BANK]
        assert len(ids) == len(set(ids))


class TestShuffling:
    def test_seeded_shuffle_changes_order(self):
        q1 = get_questions(seed=1)
        q2 = get_questions(seed=2)
        ids1 = [q.id for q in q1]
        ids2 = [q.id for q in q2]
        assert ids1 != ids2, "Different seeds should produce different orderings"

    def test_seeded_shuffle_same_questions(self):
        q1 = get_questions(seed=1)
        q2 = get_questions(seed=2)
        assert set(q.id for q in q1) == set(q.id for q in q2)

    def test_no_seed_original_order(self):
        q = get_questions(None)
        ids = [q_.id for q_ in q]
        bank_ids = [q_.id for q_ in QUESTION_BANK]
        assert ids == bank_ids

    def test_same_seed_same_order(self):
        q1 = get_questions(seed=42)
        q2 = get_questions(seed=42)
        assert [q.id for q in q1] == [q.id for q in q2]

    def test_returns_all_50(self):
        assert len(get_questions(seed=7)) == 50
