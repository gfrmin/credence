# Role: body
"""Baseline routing strategies for comparison."""

from credence_router.baselines.simple import (
    AlwaysBestSolver,
    AlwaysCheapestSolver,
    RandomSolver,
)

__all__ = ["AlwaysBestSolver", "AlwaysCheapestSolver", "RandomSolver"]
