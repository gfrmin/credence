# Role: body
"""Space — a set of possibilities. Wraps Julia Space objects."""

from __future__ import annotations

from credence._bridge import _get_bridge


class Space:
    """A set of possibilities. Wraps a Julia Space object."""

    __slots__ = ("_jl",)

    def __init__(self, jl_obj):
        self._jl = jl_obj

    @staticmethod
    def finite(values: list) -> Space:
        b = _get_bridge()
        jl_vals = b.make_float_vector(values)
        return Space(b.jl.Finite(jl_vals))

    @staticmethod
    def interval(lo: float, hi: float) -> Space:
        return Space(_get_bridge().jl.Interval(float(lo), float(hi)))

    @staticmethod
    def product(*spaces: Space) -> Space:
        b = _get_bridge()
        jl_spaces = b.make_jl_vector("Space", [s._jl for s in spaces])
        return Space(b.jl.ProductSpace(jl_spaces))

    @staticmethod
    def euclidean(n: int) -> Space:
        return Space(_get_bridge().jl.Euclidean(int(n)))

    @staticmethod
    def positive_reals() -> Space:
        return Space(_get_bridge().jl.PositiveReals())

    @staticmethod
    def simplex(k: int) -> Space:
        return Space(_get_bridge().jl.Simplex(int(k)))

    def support(self) -> list:
        b = _get_bridge()
        s = b.jl.support(self._jl)
        return [s[i] for i in range(len(s))]

    def __repr__(self) -> str:
        return f"Space({self._jl})"
