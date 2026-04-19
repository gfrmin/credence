# Role: body
"""Internal Julia bridge — singleton that lazy-loads Julia and Credence."""

from __future__ import annotations

import os
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

        # Resolve Credence repo root via (1) CREDENCE_SRC env var (explicit user
        # intent) or (2) monorepo layout. The env var is interpreted as the repo
        # root, not the src/ directory — we append src/ ourselves so a stable
        # marker file (src/Credence.jl) can be checked.
        env_root = os.environ.get("CREDENCE_SRC")
        monorepo_root = Path(__file__).resolve().parents[4]
        if env_root is not None:
            repo_root = Path(env_root)
            origin = f"CREDENCE_SRC={env_root}"
        elif (monorepo_root / "src" / "Credence.jl").exists():
            repo_root = monorepo_root
            origin = f"monorepo layout at {monorepo_root}"
        else:
            raise FileNotFoundError(
                "Cannot locate the Credence repo root. Set the CREDENCE_SRC env "
                "var to the repo root (e.g. /home/user/git/credence), or run "
                f"from a monorepo checkout where {monorepo_root}/src/Credence.jl exists."
            )

        src_path = repo_root / "src"
        if not (src_path / "Credence.jl").exists():
            raise FileNotFoundError(
                f"{origin} does not contain src/Credence.jl — expected the repo "
                f"root (e.g. /home/user/git/credence), not a subdirectory. "
                f"Looked for {src_path / 'Credence.jl'}."
            )

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
