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

        # Resolve Credence src path — monorepo layout first, then ~/git/credence
        # fallback for standalone installs. Matches credence_agents/julia_bridge.py.
        monorepo = Path(__file__).resolve().parents[3]
        candidate = monorepo / "src"
        if candidate.exists():
            src_path = candidate
        else:
            fallback = Path.home() / "git" / "credence" / "src"
            if not fallback.exists():
                raise FileNotFoundError(
                    f"Cannot find credence/src/. Expected at {candidate} or {fallback}."
                )
            src_path = fallback

        # juliacall.seval is the standard Julia interop API (not Python eval)
        jl.seval(f'push!(LOAD_PATH, "{src_path}")')
        jl.seval("using Credence")
        # Adapter that wraps a Python callable as a Julia Function. Needed
        # because Credence.expect / Credence.optimise declare `f::Function`
        # and Python lambdas arrive as `Py`, not `Function`.
        self._py_to_julia_fn = jl.seval("pf -> (x -> pf(x))")
        self._jl = jl

    @property
    def jl(self):
        self._ensure_loaded()
        return self._jl

    def wrap_callable(self, f):
        """Wrap a Python callable as a Julia Function for dispatch."""
        self._ensure_loaded()
        return self._py_to_julia_fn(f)

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
