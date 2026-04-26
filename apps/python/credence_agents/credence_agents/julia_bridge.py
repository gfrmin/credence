# Role: body
"""Bridge to Julia Credence DSL via juliacall.

Single module that loads Julia, the Credence module, and the BDSL agent
specification. Provides Python wrappers for DSL functions (agent-step,
update-on-response, answer-kernel) and host helpers (update_beta_state,
marginalize_betas, initial_rel_state, initial_cov_state).

All inference runs in Julia. Python keeps host concerns: tool queries,
benchmark loop, persistence, LangChain comparison.
"""

from __future__ import annotations

from pathlib import Path


class CredenceBridge:
    """Lazy-loading bridge to Julia Credence DSL."""

    def __init__(
        self,
        dsl_path: str | Path | None = None,
        credence_src: str | Path | None = None,
    ):
        self._jl = None
        self._env = None
        self._router_env = None
        self._dsl_path = str(Path(dsl_path).resolve()) if dsl_path else None
        self._credence_src = str(Path(credence_src).resolve()) if credence_src else None

    def _init(self):
        """Load Julia, Credence module, and BDSL. Called once on first use."""
        from juliacall import Main as jl

        # Resolve paths
        dsl_path = self._dsl_path
        credence_src = self._credence_src
        if dsl_path is None:
            # Monorepo: apps/python/credence_agents/credence_agents/julia_bridge.py
            # → 5 levels up to reach repo root.
            monorepo = Path(__file__).resolve().parents[4]
            candidate = monorepo / "examples" / "credence_agent.bdsl"
            if candidate.exists():
                dsl_path = str(candidate)
            else:
                # Fallback to ~/git/credence for standalone installs
                fallback = Path.home() / "git" / "credence" / "examples" / "credence_agent.bdsl"
                if fallback.exists():
                    dsl_path = str(fallback)
                else:
                    raise FileNotFoundError(
                        "Cannot find credence_agent.bdsl. Pass dsl_path= explicitly."
                    )
        if credence_src is None:
            monorepo = Path(__file__).resolve().parents[4]
            candidate = monorepo / "src"
            if candidate.exists():
                credence_src = str(candidate)
            else:
                fallback = Path.home() / "git" / "credence" / "src"
                if fallback.exists():
                    credence_src = str(fallback)
                else:
                    raise FileNotFoundError(
                        "Cannot find credence/src/. Pass credence_src= explicitly."
                    )

        # Load Credence module
        jl_code = f'push!(LOAD_PATH, "{credence_src}")'
        jl.seval(jl_code)  # noqa: S307 — trusted path string, not user input
        jl.seval("using Credence")  # noqa: S307

        # Load BDSL
        read_code = f'read("{dsl_path}", String)'
        bdsl_source = jl.seval(read_code)  # noqa: S307
        self._env = jl.load_dsl(bdsl_source)
        self._jl = jl

    @property
    def jl(self):
        if self._jl is None:
            self._init()
        return self._jl

    @property
    def env(self):
        if self._env is None:
            self._init()
        return self._env

    @property
    def router_env(self):
        """Load the router DSL (examples/router.bdsl) on first use."""
        if self._router_env is None:
            jl = self.jl  # ensures Julia + Credence are loaded
            monorepo = Path(__file__).resolve().parent.parent.parent.parent
            router_path = monorepo / "examples" / "router.bdsl"
            if not router_path.exists():
                router_path = Path.home() / "git" / "credence" / "examples" / "router.bdsl"
            if not router_path.exists():
                raise FileNotFoundError("Cannot find router.bdsl")
            source = jl.seval(f'read("{router_path}", String)')  # noqa: S307
            self._router_env = jl.load_dsl(source)
        return self._router_env

    def call_router(self, fn_name: str, *args):
        """Call a function from router.bdsl by name."""
        fn = self.router_env[self.jl.Symbol(fn_name)]
        return fn(*args)

    def _make_float_vector(self, values):
        """Convert Python iterable of numbers to Julia Float64 vector."""
        jl = self.jl
        literal = "Float64[" + ", ".join(str(float(v)) for v in values) + "]"
        return jl.seval(literal)  # noqa: S307 — numeric literal only

    # --- DSL function calls ---

    def agent_step(self, answer_measure, rel_measures, costs,
                   cov_probs, submit_val, abstain_val, penalty_wrong):
        """Call DSL agent-step. Returns (action_type, action_arg) as Python ints."""
        jl = self.jl
        fn = self.env[jl.Symbol("agent-step")]
        cov_jl = self._make_float_vector(cov_probs)
        result = fn(
            answer_measure, rel_measures, costs, cov_jl,
            float(submit_val), float(abstain_val), float(penalty_wrong),
        )
        # PythonCall uses 0-based indexing for Julia arrays/lists
        return (int(result[0]), int(result[1]))

    def update_on_response(self, answer_measure, kernel, response):
        """Call DSL update-on-response (condition). Returns updated prevision."""
        fn = self.env[self.jl.Symbol("update-on-response")]
        return fn(answer_measure, kernel, float(response))

    def answer_kernel(self, rel_measure, n_answers):
        """Call DSL answer-kernel. Returns Kernel."""
        fn = self.env[self.jl.Symbol("answer-kernel")]
        return fn(rel_measure, float(n_answers))

    # --- Host helper calls ---

    def initial_rel_state(self, n_categories):
        return self.jl.initial_rel_state(int(n_categories))

    def initial_cov_state(self, n_categories, coverage):
        """Create initial coverage state from coverage vector."""
        cov_jl = self._make_float_vector(coverage)
        return self.jl.initial_cov_state(int(n_categories), cov_jl)

    def marginalize_betas(self, state, cat_weights):
        """Marginalize MixturePrevision of ProductPrevisions to MixturePrevision of Betas."""
        cat_w_jl = self._make_float_vector(cat_weights)
        return self.jl.marginalize_betas(state, cat_w_jl)

    def update_beta_state(self, state, cat_belief, obs):
        """Update per-tool Beta state. Returns (new_state, new_cat_belief)."""
        result = self.jl.update_beta_state(state, cat_belief, float(obs))
        # Julia returns a Tuple; PythonCall uses 0-based indexing
        return (result[0], result[1])

    # --- Type constructors ---

    def make_answer_measure(self, n_answers):
        """Uniform CategoricalMeasure over n answers (0..n-1)."""
        jl = self.jl
        answers = self._make_float_vector(range(n_answers))
        return jl.CategoricalMeasure(jl.Finite(answers))

    def make_cat_belief(self, n_categories):
        """Uniform CategoricalMeasure over categories (0..n-1)."""
        jl = self.jl
        cats = self._make_float_vector(range(n_categories))
        return jl.CategoricalMeasure(jl.Finite(cats))

    def make_warm_rel_state(self, n_categories, alpha=7.0, beta=3.0):
        """Create a rel_state with Beta(alpha, beta) per category (for warm-starting)."""
        jl = self.jl
        beta_strs = [f"BetaPrevision({float(alpha)}, {float(beta)})"] * n_categories
        code = (
            "let factors = Prevision[" + ", ".join(beta_strs) + "]; "
            "prod = ProductPrevision(factors); "
            "MixturePrevision(Prevision[prod], Float64[0.0]) end"
        )
        return jl.seval(code)  # noqa: S307 — trusted numeric literals

    def make_oracle_rel_state(self, reliabilities):
        """Create a rel_state with known reliabilities (for OracleAgent).

        reliabilities: list of floats, one per category (true reliability values).
        Returns MixturePrevision wrapping a single ProductPrevision of tight Betas.
        """
        jl = self.jl
        n_pseudo = 100.0
        beta_strs = []
        for r in reliabilities:
            alpha = max(0.01, n_pseudo * r)
            beta = max(0.01, n_pseudo * (1.0 - r))
            beta_strs.append(f"BetaPrevision({alpha}, {beta})")
        code = (
            "let factors = Prevision[" + ", ".join(beta_strs) + "]; "
            "prod = ProductPrevision(factors); "
            "MixturePrevision(Prevision[prod], Float64[0.0]) end"
        )
        return jl.seval(code)  # noqa: S307 — trusted numeric literals

    # --- Prevision accessors ---

    def weights(self, measure):
        """Extract probability weights as Python list."""
        w = self.jl.weights(measure)
        return [float(w[i]) for i in range(len(w))]

    def expect_identity(self, measure):
        """Compute E_measure[x] (identity function) as Python float."""
        return float(self.jl.expect(measure, self.jl.seval("r -> r")))  # noqa: S307

    def mean(self, measure):
        """Extract mean of a BetaPrevision."""
        return float(self.jl.mean(measure))

    def extract_mixture_state(self, rel_state) -> dict:
        """Extract full MixturePrevision state for persistence.

        Returns dict with 'log_weights' and 'components' (list of list of (alpha, beta)).
        """
        jl = self.jl
        code = """function _extract_full(state)
            lw = state.log_weights
            comps = []
            for prod in state.components
                push!(comps, [(f.alpha, f.beta) for f in prod.factors])
            end
            (lw, comps)
        end"""
        fn = jl.seval(code)  # noqa: S307
        result = fn(rel_state)
        log_weights = [float(result[0][i]) for i in range(len(result[0]))]
        components = []
        for ci in range(len(result[1])):
            comp = result[1][ci]
            components.append([(float(comp[j][0]), float(comp[j][1])) for j in range(len(comp))])
        return {"log_weights": log_weights, "components": components}

    def make_rel_state_from_mixture(self, state_dict: dict):
        """Reconstruct a MixturePrevision from saved state.

        Inverse of extract_mixture_state.
        """
        jl = self.jl
        components = state_dict["components"]
        log_weights = state_dict["log_weights"]

        comp_strs = []
        for comp in components:
            beta_strs = [f"BetaPrevision({a}, {b})" for a, b in comp]
            comp_strs.append("ProductPrevision(Prevision[" + ", ".join(beta_strs) + "])")

        lw_str = "Float64[" + ", ".join(str(w) for w in log_weights) + "]"
        code = (
            "let comps = Prevision[" + ", ".join(comp_strs) + "]; "
            f"MixturePrevision(comps, {lw_str}) end"
        )
        return jl.seval(code)  # noqa: S307

    def extract_reliability_means(self, rel_state):
        """Extract per-category mean reliability from a rel_state.

        Returns list of floats, one per category.
        """
        jl = self.jl
        means = jl.extract_reliability_means(rel_state)
        return [float(means[i]) for i in range(len(means))]
