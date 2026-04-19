# Role: body
"""Scoring rules and tool configuration types.

Computation (VOI, EU) is now handled by the Julia Credence DSL.
This module retains the configuration types used throughout the Python codebase.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class ScoringRule(NamedTuple):
    reward_correct: float = 10.0
    penalty_wrong: float = -5.0
    reward_abstain: float = 0.0


_DEFAULT_SCORING = ScoringRule()

# Backward-compatible module-level constants
REWARD_CORRECT = _DEFAULT_SCORING.reward_correct
PENALTY_WRONG = _DEFAULT_SCORING.penalty_wrong
REWARD_ABSTAIN = _DEFAULT_SCORING.reward_abstain


class ToolConfig(NamedTuple):
    cost: float
    coverage_by_category: NDArray[np.float64]  # shape (num_categories,)
