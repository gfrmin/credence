"""Action types and post-question reliability update logic.

Action selection and Bayesian updates are now handled by the Julia Credence DSL.
This module retains the action/state types and the pure-Python logic for mapping
post-question feedback to per-tool correctness labels.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import NamedTuple


class ActionType(Enum):
    SUBMIT = auto()
    ABSTAIN = auto()
    QUERY = auto()


class Action(NamedTuple):
    action_type: ActionType
    tool_idx: int | None = None      # for QUERY
    answer_idx: int | None = None    # for SUBMIT
    eu: float = 0.0


def compute_reliability_updates(
    submitted_answer_idx: int | None,
    was_correct: bool | None,
    tool_responses: dict[int, int | None],
) -> dict[int, bool | None]:
    """Determine per-tool correctness from post-question feedback.

    Returns: {tool_idx: True (correct), False (wrong), or None (unknown)}.

    Logic:
    - Abstained (submitted_answer_idx is None): no ground truth -> all None
    - Correct submission: tool agreed -> True, disagreed -> False
    - Wrong submission: tool agreed with our wrong answer -> False,
                        tool disagreed -> None (might have given different wrong answer)
    - Tools that returned None: always None (no reliability signal)
    """
    if submitted_answer_idx is None or was_correct is None:
        return {t: None for t in tool_responses}

    updates: dict[int, bool | None] = {}
    for tool_idx, resp in tool_responses.items():
        if resp is None:
            updates[tool_idx] = None
        elif was_correct:
            updates[tool_idx] = (resp == submitted_answer_idx)
        else:
            # Wrong submission: tools that agreed with us were wrong.
            # Tools that disagreed: we don't know if they were right.
            updates[tool_idx] = False if resp == submitted_answer_idx else None

    return updates


def compute_binary_reliability_updates(
    tool_responses: dict[int, int | None],
    was_correct: bool | None,
) -> dict[int, bool | None]:
    """Determine per-tool correctness from binary (correct/wrong) feedback.

    Simpler than compute_reliability_updates: no candidate-agreement logic.
    Each tool that returned a response inherits the overall outcome; tools
    that returned None get None (no reliability signal).
    """
    if was_correct is None:
        return {t: None for t in tool_responses}
    return {
        t: (was_correct if resp is not None else None)
        for t, resp in tool_responses.items()
    }
