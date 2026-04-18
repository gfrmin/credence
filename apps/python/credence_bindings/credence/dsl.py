"""DSL interop — run_dsl and load_dsl."""

from __future__ import annotations

from typing import Any

from credence._bridge import _get_bridge
from credence.space import Space
from credence.measure import Measure
from credence.kernel import Kernel


def _maybe_wrap(jl_obj) -> Any:
    """Auto-wrap Julia objects into Python wrapper types."""
    b = _get_bridge()
    jl = b.jl
    # juliacall.seval is the standard Julia interop API
    if bool(jl.seval("x -> x isa Measure")(jl_obj)):
        return Measure(jl_obj)
    if bool(jl.seval("x -> x isa Space")(jl_obj)):
        return Space(jl_obj)
    if bool(jl.seval("x -> x isa Kernel")(jl_obj)):
        return Kernel(jl_obj=jl_obj)
    return jl_obj


def run_dsl(source: str) -> Any:
    """Run DSL source, return the final result (auto-wrapped)."""
    b = _get_bridge()
    result = b.jl.run_dsl(source)
    return _maybe_wrap(result)


def load_dsl(source: str) -> dict[str, Any]:
    """Load DSL source, return environment as a Python dict.

    Keys are strings. Values are auto-wrapped: Julia Measures become
    Measure, Spaces become Space, Kernels become Kernel, everything
    else passes through.
    """
    b = _get_bridge()
    jl = b.jl
    env = jl.load_dsl(source)

    # juliacall.seval is the standard Julia interop API
    keys = jl.seval("d -> [string(k) for k in keys(d)]")(env)
    result = {}
    for i in range(len(keys)):
        key = str(keys[i])
        val = jl.seval("(d, k) -> d[Symbol(k)]")(env, key)
        result[key] = _maybe_wrap(val)
    return result
