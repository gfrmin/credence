"""Internal Julia bridge — singleton that lazy-loads Julia and Credence."""

from __future__ import annotations

from pathlib import Path


class _Bridge:
    _instance: _Bridge | None = None

    def __new__(cls) -> _Bridge:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._jl = None  # type: ignore[attr-defined]
        return cls._instance

    def _ensure_loaded(self) -> None:
        if self._jl is not None:
            return
        from juliacall import Main as jl

        # juliacall.seval is the standard Julia interop API (not Python eval)
        src_path = Path(__file__).resolve().parent.parent.parent / "src"
        jl.seval(f'push!(LOAD_PATH, "{src_path}")')
        jl.seval("using Credence")
        self._jl = jl

    @property
    def jl(self):
        self._ensure_loaded()
        return self._jl

    def make_float_vector(self, values):
        """Convert Python iterable of numbers to Julia Float64 vector."""
        literal = "Float64[" + ", ".join(str(float(v)) for v in values) + "]"
        return self.jl.seval(literal)

    def make_jl_vector(self, type_name: str, items):
        """Build a typed Julia vector from Python list of Julia objects."""
        jl = self.jl
        vec = jl.seval(f"{type_name}[]")
        for item in items:
            jl.push_b(vec, item)
        return vec

    def make_measure_vector(self, items):
        """Build a Julia Measure[] vector."""
        return self.make_jl_vector("Measure", items)


def _get_bridge() -> _Bridge:
    return _Bridge()
