# Role: tests for the action-selection bridge.
from __future__ import annotations

import pytest

from credence_router.tool_decision.decide import (
    Action,
    DecideInputs,
    compute_action_eus,
    decide,
)


class TestComputeActionEus:
    def test_execute_eu_uses_proposed_alpha_beta(self):
        # alpha=9, beta=1 → P(approve)=0.9; cost=0.0 → EU(execute)=0.9
        eus = compute_action_eus(
            proposed_alpha=9.0,
            proposed_beta=1.0,
            best_alt_alpha=2.0,
            best_alt_beta=2.0,
            stop_alpha=5.0,
            stop_beta=5.0,
            llm_cost=0.0,
        )
        assert eus[0] == pytest.approx(0.9)

    def test_substitute_eu_uses_best_alt(self):
        eus = compute_action_eus(
            proposed_alpha=2.0,
            proposed_beta=2.0,
            best_alt_alpha=9.0,
            best_alt_beta=1.0,
            stop_alpha=5.0,
            stop_beta=5.0,
            llm_cost=0.0,
        )
        assert eus[1] == pytest.approx(0.9)

    def test_stop_eu_uses_stop_cell(self):
        eus = compute_action_eus(
            proposed_alpha=2.0,
            proposed_beta=2.0,
            best_alt_alpha=2.0,
            best_alt_beta=2.0,
            stop_alpha=18.0,
            stop_beta=2.0,
            llm_cost=0.0,
        )
        assert eus[2] == pytest.approx(0.9)


class TestDecide:
    def test_picks_execute_when_proposal_is_strongest(self):
        inputs = DecideInputs(
            action_eus=[0.9, 0.5, 0.1, 0.0],
            voi_ask=0.0,
            ask_cost=0.05,
        )
        assert decide(inputs) is Action.EXECUTE

    def test_picks_ask_when_voi_minus_cost_dominates(self):
        inputs = DecideInputs(
            action_eus=[0.3, 0.2, 0.1, 0.0],
            voi_ask=0.6,
            ask_cost=0.05,
        )
        assert decide(inputs) is Action.ASK
