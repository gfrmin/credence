# Role: body
"""Prevision — a coherent linear functional (de Finetti 1974)."""

from __future__ import annotations

import math
from typing import Callable

from credence._bridge import _get_bridge
from credence.space import Space
from credence.kernel import Kernel


class Prevision:
    """Prevision. Wraps a Julia Prevision or Measure object."""

    __slots__ = ("_jl",)

    def __init__(self, jl_obj):
        self._jl = jl_obj

    # ── Named constructors ──

    @staticmethod
    def uniform(space: Space) -> Prevision:
        b = _get_bridge()
        jl = b.jl
        is_finite = bool(jl.seval("x -> x isa Finite")(space._jl))
        if is_finite:
            return Prevision(jl.CategoricalMeasure(space._jl))
        is_interval = bool(jl.seval("x -> x isa Interval")(space._jl))
        if is_interval:
            return Prevision(jl.BetaPrevision(1.0, 1.0))
        raise ValueError("uniform requires a Finite or [0,1] Interval space")

    @staticmethod
    def categorical(space: Space, weights: list[float]) -> Prevision:
        b = _get_bridge()
        logw = b.make_float_vector([math.log(max(w, 1e-300)) for w in weights])
        return Prevision(b.jl.CategoricalMeasure(space._jl, logw))

    @staticmethod
    def beta(alpha: float, beta: float) -> Prevision:
        return Prevision(_get_bridge().jl.BetaPrevision(float(alpha), float(beta)))

    @staticmethod
    def gaussian(mu: float, sigma: float) -> Prevision:
        return Prevision(_get_bridge().jl.GaussianPrevision(float(mu), float(sigma)))

    @staticmethod
    def dirichlet(alphas: list[float]) -> Prevision:
        b = _get_bridge()
        alpha_jl = b.make_float_vector(alphas)
        return Prevision(b.jl.DirichletPrevision(alpha_jl))

    @staticmethod
    def product(factors: list[Prevision]) -> Prevision:
        b = _get_bridge()
        jl_factors = b.make_jl_vector("Any", [f._jl for f in factors])
        return Prevision(b.jl.ProductPrevision(jl_factors))

    @staticmethod
    def mixture(
        components: list[Prevision], weights: list[float]
    ) -> Prevision:
        b = _get_bridge()
        jl_components = b.make_jl_vector("Any", [c._jl for c in components])
        log_wts = b.make_float_vector([math.log(max(w, 1e-300)) for w in weights])
        return Prevision(b.jl.MixturePrevision(jl_components, log_wts))

    # ── Query methods ──

    def weights(self) -> list[float]:
        b = _get_bridge()
        w = b.jl.weights(self._jl)
        return [float(w[i]) for i in range(len(w))]

    def mean(self) -> float:
        return float(_get_bridge().jl.mean(self._jl))

    def variance(self) -> float:
        return float(_get_bridge().jl.variance(self._jl))

    def support(self) -> list:
        b = _get_bridge()
        s = b.jl.seval("x -> x isa Credence.CategoricalMeasure ? Credence.support(x.space) : error(\"support requires CategoricalMeasure\")")(self._jl)
        return [s[i] for i in range(len(s))]

    # ── Axiom-constrained operations (convenience methods) ──

    def condition(self, kernel: Kernel, observation) -> Prevision:
        return Prevision(_get_bridge().jl.condition(self._jl, kernel._jl, observation))

    def expect(self, f: Callable) -> float:
        bridge = _get_bridge()
        return float(bridge.jl.expect(self._jl, bridge.wrap_callable(f)))

    def push(self, kernel: Kernel) -> Prevision:
        return Prevision(_get_bridge().jl.push_measure(self._jl, kernel._jl))

    # ── Host boundary operations ──

    def draw(self):
        return _get_bridge().jl.draw(self._jl)

    # ── Mixture maintenance ──

    def prune(self, threshold: float = -20.0) -> Prevision:
        return Prevision(_get_bridge().jl.prune(self._jl, threshold=threshold))

    def truncate(self, max_components: int = 10) -> Prevision:
        return Prevision(
            _get_bridge().jl.Credence.truncate(self._jl, max_components=max_components)
        )

    def __repr__(self) -> str:
        return f"Prevision({self._jl})"
