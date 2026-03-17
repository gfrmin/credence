"""Kernel — a conditional distribution between two spaces."""

from __future__ import annotations

from typing import Any, Callable

from credence._bridge import _get_bridge
from credence.space import Space


class Kernel:
    """Kernel(source, target, log_density=...) or Kernel(jl_obj=...)."""

    __slots__ = ("_jl",)

    def __init__(
        self,
        source: Space | None = None,
        target: Space | None = None,
        *,
        log_density: Callable[[Any, Any], float] | None = None,
        jl_obj=None,
    ):
        if jl_obj is not None:
            self._jl = jl_obj
            return

        if source is None or target is None or log_density is None:
            raise ValueError("Kernel requires source, target, and log_density")

        b = _get_bridge()
        jl = b.jl

        py_ld = log_density

        # juliacall.seval is the standard Julia interop API
        # Wrap Python callables into Julia Functions via seval closures
        # because Julia's Kernel constructor requires ::Function arguments
        target_jl = target._jl
        is_finite = bool(jl.seval("x -> x isa Finite")(target_jl))

        if is_finite:
            tgt_vals = target.support()

            def py_gen(h):
                log_probs = [float(py_ld(h, o)) for o in tgt_vals]
                logw = b.make_float_vector(log_probs)
                return jl.CategoricalMeasure(target_jl, logw)

            def py_log_dens(h, o):
                return float(py_ld(h, o))

            # Create Julia Function wrappers around Python callables
            # Float64() ensures Julia gets native Float64, not Py objects
            make_kernel = jl.seval(
                "(src, tgt, py_gen, py_ld) -> Kernel(src, tgt, h -> py_gen(h), (h, o) -> pyconvert(Float64, py_ld(h, o)))"
            )
            self._jl = make_kernel(source._jl, target_jl, py_gen, py_log_dens)
        else:
            def py_gen(h):
                return lambda o: float(py_ld(h, o))

            def py_log_dens(h, o):
                return float(py_ld(h, o))

            make_kernel = jl.seval(
                "(src, tgt, py_gen, py_ld) -> Kernel(src, tgt, h -> py_gen(h), (h, o) -> pyconvert(Float64, py_ld(h, o)))"
            )
            self._jl = make_kernel(source._jl, target_jl, py_gen, py_log_dens)

    @property
    def source(self) -> Space:
        return Space(_get_bridge().jl.kernel_source(self._jl))

    @property
    def target(self) -> Space:
        return Space(_get_bridge().jl.kernel_target(self._jl))

    def density(self, h, o) -> float:
        return float(_get_bridge().jl.density(self._jl, h, o))

    def __repr__(self) -> str:
        return f"Kernel({self._jl})"
