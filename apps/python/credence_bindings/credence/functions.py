# Role: body
"""Standalone wrappers for axiom-constrained and host-boundary functions."""

from __future__ import annotations

from typing import Callable

from credence.prevision import Prevision
from credence.kernel import Kernel


def condition(prior: Prevision, kernel: Kernel, observation) -> Prevision:
    """Bayesian inversion: prior x kernel x observation -> posterior."""
    return prior.condition(kernel, observation)


def expect(prevision: Prevision, f: Callable) -> float:
    """Integration against a prevision: E_p[f]."""
    return prevision.expect(f)


def push(prevision: Prevision, kernel: Kernel) -> Prevision:
    """Pushforward: Prevision(H) x Kernel(H,T) -> Prevision(T)."""
    return prevision.push(kernel)


def density(kernel: Kernel, h, o) -> float:
    """Log-density of the kernel at a point: log P(o|h)."""
    return kernel.density(h, o)


def draw(prevision: Prevision):
    """Draw a sample from a prevision. The ONLY source of randomness."""
    return prevision.draw()
