"""Measure — a probability distribution over a space."""

from __future__ import annotations

import math
from typing import Callable

from credence._bridge import _get_bridge
from credence.space import Space
from credence.kernel import Kernel


class Measure:
    """Probability measure. Wraps a Julia Measure object."""

    __slots__ = ("_jl",)

    def __init__(self, jl_obj):
        self._jl = jl_obj

    # ── Named constructors ──

    @staticmethod
    def uniform(space: Space) -> Measure:
        b = _get_bridge()
        jl = b.jl
        # juliacall.seval is the standard Julia interop API
        is_finite = bool(jl.seval("x -> x isa Finite")(space._jl))
        if is_finite:
            return Measure(jl.CategoricalMeasure(space._jl))
        is_interval = bool(jl.seval("x -> x isa Interval")(space._jl))
        if is_interval:
            return Measure(jl.BetaMeasure(space._jl, 1.0, 1.0))
        raise ValueError("uniform requires a Finite or [0,1] Interval space")

    @staticmethod
    def categorical(space: Space, weights: list[float]) -> Measure:
        b = _get_bridge()
        logw = b.make_float_vector([math.log(max(w, 1e-300)) for w in weights])
        return Measure(b.jl.CategoricalMeasure(space._jl, logw))

    @staticmethod
    def beta(alpha: float, beta: float) -> Measure:
        return Measure(_get_bridge().jl.BetaMeasure(float(alpha), float(beta)))

    @staticmethod
    def gaussian(mu: float, sigma: float) -> Measure:
        b = _get_bridge()
        space = b.jl.Euclidean(1)
        return Measure(b.jl.GaussianMeasure(space, float(mu), float(sigma)))

    @staticmethod
    def dirichlet(categories: Space, alphas: list[float]) -> Measure:
        b = _get_bridge()
        k = len(alphas)
        simplex = b.jl.Simplex(k)
        alpha_jl = b.make_float_vector(alphas)
        return Measure(b.jl.DirichletMeasure(simplex, categories._jl, alpha_jl))

    @staticmethod
    def product(factors: list[Measure]) -> Measure:
        b = _get_bridge()
        jl_factors = b.make_measure_vector([f._jl for f in factors])
        return Measure(b.jl.ProductMeasure(jl_factors))

    @staticmethod
    def mixture(
        components: list[Measure], weights: list[float]
    ) -> Measure:
        b = _get_bridge()
        jl = b.jl
        jl_components = b.make_measure_vector([c._jl for c in components])
        log_wts = b.make_float_vector([math.log(max(w, 1e-300)) for w in weights])
        # juliacall.seval is the standard Julia interop API
        space = jl.seval("x -> x.space")(components[0]._jl)
        return Measure(jl.MixtureMeasure(space, jl_components, log_wts))

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
        # juliacall.seval is the standard Julia interop API
        s = b.jl.support(b.jl.seval("x -> x.space")(self._jl))
        return [s[i] for i in range(len(s))]

    # ── Axiom-constrained operations (convenience methods) ──

    def condition(self, kernel: Kernel, observation) -> Measure:
        return Measure(_get_bridge().jl.condition(self._jl, kernel._jl, observation))

    def expect(self, f: Callable) -> float:
        bridge = _get_bridge()
        return float(bridge.jl.expect(self._jl, bridge.wrap_callable(f)))

    def push(self, kernel: Kernel) -> Measure:
        return Measure(_get_bridge().jl.push_measure(self._jl, kernel._jl))

    # ── Host boundary operations ──

    def draw(self):
        return _get_bridge().jl.draw(self._jl)

    def optimise(self, actions: Space, pref: Callable):
        bridge = _get_bridge()
        return bridge.jl.optimise(self._jl, actions._jl, bridge.wrap_callable(pref))

    def value(self, actions: Space, pref: Callable) -> float:
        return float(_get_bridge().jl.value(self._jl, actions._jl, pref))

    # ── Mixture maintenance ──

    def prune(self, threshold: float = -20.0) -> Measure:
        return Measure(_get_bridge().jl.prune(self._jl, threshold=threshold))

    def truncate(self, max_components: int = 10) -> Measure:
        # juliacall.seval is the standard Julia interop API
        return Measure(
            _get_bridge().jl.Credence.truncate(self._jl, max_components=max_components)
        )

    def __repr__(self) -> str:
        return f"Measure({self._jl})"
