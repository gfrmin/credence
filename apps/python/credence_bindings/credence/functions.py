# Role: body
"""Standalone wrappers for axiom-constrained and host-boundary functions."""

from __future__ import annotations

from typing import Callable

from credence.measure import Measure
from credence.kernel import Kernel


def condition(prior: Measure, kernel: Kernel, observation) -> Measure:
    """Bayesian inversion: prior x kernel x observation -> posterior."""
    return prior.condition(kernel, observation)


def expect(measure: Measure, f: Callable) -> float:
    """Integration against a measure: E_m[f]."""
    return measure.expect(f)


def push(measure: Measure, kernel: Kernel) -> Measure:
    """Pushforward: Measure(H) x Kernel(H,T) -> Measure(T)."""
    return measure.push(kernel)


def density(kernel: Kernel, h, o) -> float:
    """Log-density of the kernel at a point: log P(o|h)."""
    return kernel.density(h, o)


def draw(measure: Measure):
    """Draw a sample from a measure. The ONLY source of randomness."""
    return measure.draw()
