# Posture 4 — Move 8 design doc: Python bindings speak Prevision

## 0. Final-state alignment

Move 8 carries Prevision vocabulary through all four Python packages (`credence_bindings`, `credence_agents`, `credence_router`, `bayesian_if`). After this move, no Python code constructs or references `BetaMeasure`, `ProductMeasure`, `MixtureMeasure`, or `GaussianMeasure` — the Measure vocabulary exists only in `src/ontology.jl` (where it is a declared view over Prevision) and in the skin server's backward-compatible `build_measure` (kept for Product sub-factors, design doc §5.1 of Move 7). The Python bindings class is renamed from `Measure` to `Prevision`. The skin wire protocol type strings (`"beta"`, `"gamma"`, `"categorical"`, etc.) are unchanged — Move 7's `build_prevision` already accepts them.

Transient state: `CategoricalMeasure` survives in the skin's `build_prevision` (design doc Move 7 §5.1) and in the bindings' `Prevision.categorical()` constructor (which calls `jl.CategoricalMeasure`). This is principled, not drift — CategoricalMeasure binds carrier space, and CategoricalPrevision stores only log_weights.

## 1. Purpose

Rename the Python-side `Measure` class and all Measure-vocabulary Julia interop strings to Prevision vocabulary, completing the Python layer's migration to the de Finettian framing. After this move, all four layers — DSL core, Julia hosts, skin server, Python callers — speak Prevision.

## 2. Files touched

### credence_bindings (MODIFY)

- `credence/__init__.py:5,11` — `Measure` import/export → `Prevision`; no backward-compat alias (§5.1 review decision)
- `credence/measure.py` → **RENAME to `credence/prevision.py`** — class `Measure` → `Prevision`; all return types `-> Measure` → `-> Prevision`; constructor bodies updated:
  - Line 31: `jl.CategoricalMeasure(space._jl)` — **unchanged** (CategoricalMeasure stays, §0)
  - Line 34: `jl.BetaMeasure(space._jl, 1.0, 1.0)` → `jl.BetaPrevision(1.0, 1.0)`
  - Line 41: `jl.CategoricalMeasure(space._jl, logw)` — **unchanged**
  - Line 45: `jl.BetaMeasure(float(alpha), float(beta))` → `jl.BetaPrevision(...)`
  - Line 51: `jl.GaussianMeasure(space, float(mu), float(sigma))` → `jl.GaussianPrevision(...)`
  - Line 59: `jl.DirichletMeasure(simplex, categories._jl, alpha_jl)` → `jl.DirichletPrevision(alpha_jl)` (DirichletPrevision takes only alpha; categories are kernel-side per Move 7)
  - Line 65: `jl.ProductMeasure(jl_factors)` — **unchanged** (ProductMeasure stays, design doc Move 7 §5.1)
  - Line 77: `jl.MixtureMeasure(space, jl_components, log_wts)` → `jl.MixturePrevision(jl_components, log_wts)` (no space arg)
  - Lines 95, 117, 120: `.space` field access — remove or replace (Prevision types have no `.space`)
- `credence/functions.py:8,12,17,22,32` — type hints `Measure` → `Prevision`; import path updated
- `credence/dsl.py:10,19–20` — import path `credence.measure` → `credence.prevision`; Julia type-check `x isa Measure` → `x isa Prevision`; wrapper `Measure(jl_obj)` → `Prevision(jl_obj)`
- `credence/_bridge.py:83–85` — `make_measure_vector` → `make_prevision_vector`; Julia type `"Measure"` → `"Any"` (MixturePrevision components are `Any[]`)

### credence_bindings tests (MODIFY)

- `tests/test_bindings.py` — all `Measure.xxx()` calls → `Prevision.xxx()`; test names updated; `isinstance(env["prior"], Measure)` → `Prevision`

### credence_agents (MODIFY)

- `credence_agents/julia_bridge.py:154,167,170,173,176,181–185,193,202–206,222,226,249,259–265` — all `BetaMeasure(...)` in seval strings → `BetaPrevision(...)`; `ProductMeasure(factors)` → `ProductPrevision(factors)`; `MixtureMeasure(space, comps, lw)` → `MixturePrevision(comps, lw)`; `Measure[...]` vector literals → `Any[...]`; `CategoricalMeasure(...)` — **unchanged** (categorical stays as Measure)
- `credence_agents/agents/bayesian_agent.py:64` — comment "Julia CategoricalMeasure" → "Julia CategoricalMeasure (Prevision-backed)" or similar
- `credence_agents/agents/bayesian_selector.py` — comments only

### credence_router (MODIFY)

- `credence_router/routing_domain.py:4,274` — comments only. Wire protocol type strings (`"beta"`, `"gamma"`, `"product"`) are **unchanged** — they match `build_prevision`'s dispatch keys.

### bayesian_if (MODIFY — if any)

- `bayesian_if/agent.py` — no Measure references. Comments only if any mention Measure vocabulary.

### skin (NO CHANGE)

- `apps/skin/client.py` — no changes. The `create_state(type="beta", ...)` dict keys match `build_prevision` already.
- `apps/skin/server.jl` — no changes (Move 7 completed).
- `apps/skin/test_skin.py` — no changes. Wire protocol strings are stable.

## 3. Behaviour preserved

All Python integration tests (`uv run pytest apps/python/`) pass before and after. The credence-proxy Docker image rebuilds and `curl /ready` passes.

Numerical invariants: the Python bindings construct the same Julia objects with the same parameters. `BetaPrevision(α, β)` has the same `alpha`/`beta` fields as `BetaMeasure(α, β).prevision`. `MixturePrevision(comps, lw)` normalises log-weights identically to `MixtureMeasure(space, comps, lw)`. No numerical values change.

The Move 0 fixture capture at `test/fixtures/posture-3-capture/` is Julia-side only; Python tests do not reference it.

## 4. Worked end-to-end example

### Before (current):

```python
from credence import Measure, Space, condition

s = Space.interval(0, 1)
prior = Measure.beta(1.0, 1.0)        # calls jl.BetaMeasure(1.0, 1.0)
print(prior.mean())                     # calls jl.mean(beta_measure) → 0.5
posterior = prior.condition(kernel, 1.0) # calls jl.condition(beta_measure, k, 1.0) → BetaMeasure
print(posterior.mean())                  # 0.667
```

### After (Move 8):

```python
from credence import Prevision, Space, condition

prior = Prevision.beta(1.0, 1.0)       # calls jl.BetaPrevision(1.0, 1.0)
print(prior.mean())                     # calls jl.mean(beta_prevision) → stdlib → expect(p, Identity()) → 0.5
posterior = prior.condition(kernel, 1.0) # calls jl.condition(beta_prevision, k, 1.0) → BetaPrevision (Move 7 substrate)
print(posterior.mean())                  # 0.667
```

Trace through the Julia layer for `prior.mean()`:
1. Python `prior.mean()` → `float(_get_bridge().jl.mean(self._jl))`
2. Julia `mean(p::BetaPrevision)` → `src/stdlib.jl:22`: `expect(p, Identity())`
3. Julia `expect(p::BetaPrevision, ::Identity)` → `src/ontology.jl:583`: `p.alpha / (p.alpha + p.beta)`
4. Returns `0.5`

### julia_bridge seval migration:

```python
# Before (credence_agents/julia_bridge.py:181–185):
beta_strs = [f"BetaMeasure({float(alpha)}, {float(beta)})"] * n_categories
code = (
    "let factors = Measure[" + ", ".join(beta_strs) + "]; "
    "prod = ProductMeasure(factors); "
    "MixtureMeasure(prod.space, Measure[prod], Float64[0.0]) end"
)

# After:
beta_strs = [f"BetaPrevision({float(alpha)}, {float(beta)})"] * n_categories
code = (
    "let factors = Any[" + ", ".join(beta_strs) + "]; "
    "prod = ProductPrevision(factors); "
    "MixturePrevision(Any[prod], Float64[0.0]) end"
)
```

### Dirichlet constructor change:

```python
# Before: DirichletMeasure(simplex, categories, alpha)
Measure(b.jl.DirichletMeasure(simplex, categories._jl, alpha_jl))

# After: DirichletPrevision(alpha) — no space/categories args
Prevision(b.jl.DirichletPrevision(alpha_jl))
```

The `categories` Space argument moves to kernel construction time (the `Categorical(cats)` likelihood family carries it). The `Prevision.dirichlet()` constructor signature changes from `(categories: Space, alphas: list[float])` to `(alphas: list[float])`.

## 5. Open design questions

### 5.1 Backward-compatible `Measure` alias

The `credence` Python package currently exports `Measure`. External callers (if any exist outside this repo) would break on rename. **My prior:** add `Measure = Prevision` alias in `__init__.py` for one release cycle, with a deprecation warning on first use.

**Review decision: hard rename, no alias.** `Measure` and `Prevision` are architecturally different things (Measure carries carrier-binding, Prevision doesn't); aliasing them flattens a distinction six moves have been making. A user reading `Measure.beta(1, 1)` under the alias thinks they're constructing a Measure when they're constructing a Prevision. All consumers are internal (`credence_agents`, `credence_router`, `bayesian_if`); no external migration window needed.

### 5.2 `Prevision.dirichlet()` signature change

`DirichletMeasure(simplex, categories, alpha)` takes three args; `DirichletPrevision(alpha)` takes one. The Python constructor `Prevision.dirichlet(categories, alphas)` currently passes `categories` to Julia. After migration, `categories` is a kernel-side concern (the `Categorical(cats)` likelihood family). Options:

- **(a) Drop `categories` from the Python constructor.** `Prevision.dirichlet(alphas)` — clean, matches the Julia type. Callers that need categories provide them at kernel construction. Breaking change for anyone calling `Measure.dirichlet(categories, alphas)`.
- **(b) Accept `categories` and warn.** Accept but ignore, with a deprecation warning. Soft migration.
- **(c) Keep `categories` and construct a `DirichletMeasure` behind the scenes.** Anti-pattern — the whole point of Move 8 is to stop constructing Measures.

**My prior:** option (a). The `credence_bindings` package is internal to this repo; there are no external callers. The one call site (`julia_bridge.py` doesn't call `Measure.dirichlet` — it constructs directly via seval). The `test_bindings.py` call is the only consumer; update it.

**Review decision: approved.** Constructor argument is architecturally wrong (categories belong at kernel construction time per Move 5/7).

### 5.3 `ProductMeasure` vs `ProductPrevision` in julia_bridge seval strings

The julia_bridge constructs `ProductMeasure(Measure[...])` via seval strings. `ProductPrevision` exists (`src/prevision.jl`) and takes `Any[]` factors. Options:

- **(a) Migrate to `ProductPrevision(Any[...])`.** Consistent with the Prevision vocabulary. `ProductPrevision.factors` is `Any[]`, matching the existing `Any[]` pattern for MixturePrevision.
- **(b) Keep `ProductMeasure` in seval strings.** ProductMeasure stays in the frozen layer alongside CategoricalMeasure per Move 7 §5.1.

**My prior:** option (a). The julia_bridge doesn't need ProductMeasure's space-binding capability — it's constructing opaque Julia objects for `condition`/`expect` dispatch. `ProductPrevision` is the lighter-weight type. Move 7 §5.1 keeps `ProductMeasure` in the *skin* because the skin needs `build_measure` for Product sub-factors; the julia_bridge constructs directly via seval, so it can use the Prevision type.

**Review decision: approved, conditional on verifying no downstream `.space` access.** Grep julia_bridge return values for `.space` accesses before implementing. ProductMeasure's components carry their own spaces; ProductPrevision's don't — the question is whether anything downstream needs them.

## 6. Risk + mitigation

**Risk 1: `Prevision.uniform(interval_space)` dispatch.** Currently `Measure.uniform` creates `BetaMeasure(space, 1, 1)` for intervals. `BetaPrevision(1.0, 1.0)` has no space field. The space argument is vestigial — `BetaPrevision` always represents a distribution on `[0,1]`. Mitigation: the constructor drops the space argument silently. Test: `Prevision.uniform(Space.interval(0, 1)).mean() == 0.5`.

**Risk 2: `Prevision.mixture()` no longer takes a space argument.** `MixtureMeasure(space, comps, lw)` required a space; `MixturePrevision(comps, lw)` does not. The current `Measure.mixture()` extracts `.space` from the first component (`jl.seval("x -> x.space")(components[0]._jl)`). After migration, this line is deleted. Mitigation: the constructor simply passes components and log-weights. Test: existing mixture tests in `test_bindings.py`.

**Risk 3: `extract_mixture_state` in julia_bridge reads `.factors` on ProductMeasure components.** After migration to `ProductPrevision`, the field is still `.factors` (same name on both types). Mitigation: grep for all `.factors` accesses; verify `ProductPrevision` exposes the same field name.

**Risk 4: `Prevision.support()` reads `.space` field.** `Prevision.support()` does `jl.seval("x -> x.space")(self._jl)` then `jl.support(space)`. Prevision types have no `.space` field. Mitigation: add `support()` method that dispatches through Julia's `support` function directly, or remove if unused. Check callers; if only CategoricalMeasure (which stays as Measure) needs it, the method can delegate to the Julia-level `support(m::CategoricalMeasure)` which knows its space.

## 7. Verification cadence

End-of-PR:
1. `uv run pytest apps/python/credence_bindings/tests/` — bindings tests
2. `uv run pytest apps/python/credence_agents/tests/` — agent tests
3. `uv run pytest apps/python/credence_router/tests/ --ignore=apps/python/credence_router/tests/test_live.py` — router tests (excluding live API tests)
4. `uv run pytest apps/python/bayesian_if/tests/` — IF tests (if they exist and are runnable without external deps)
5. `python apps/skin/test_skin.py` — skin smoke (should pass unchanged; no server changes)
6. `python tools/credence-lint/credence_lint.py check apps/ && python tools/credence-lint/credence_lint.py test` — lint
7. `docker build -t credence-proxy . && docker run --rm credence-proxy curl -s localhost:8080/ready` — proxy smoke (if Docker available)

## 8. de Finettian discipline self-audit

1. **Is every numerical query in this move routed through `expect`?** Yes. Move 8 changes Python-side type names and constructor calls, not query paths. The Julia-side `mean(p::Prevision) = expect(p, Identity())` pipeline (Move 5 + Move 7 substrate) is already in place. No new Python-side functions return probabilistic properties without calling `expect`.

2. **Does this move hold a Prevision inside a Measure, or a Measure inside a Prevision?** No. The direction of wrapping is eliminated: `MixturePrevision` components are `Any[]` (not `Measure[]`); `ProductPrevision` factors are `Any[]`. The two principled Measure survivors (`CategoricalMeasure` in `Prevision.categorical()`, `ProductMeasure` in the skin's `build_prevision`) hold Previsions inside Measures — the direction being retired — but these are documented exceptions (Move 7 §5.1), not new introductions.

3. **Does this move introduce an opaque closure where a declared structure would fit?** No. The Python bindings' `expect(f: Callable)` already wraps the callable via `bridge.wrap_callable(f)` for Julia dispatch. Move 8 does not change this; the closure is the existing `OpaqueClosure` fallback, not a new introduction.

4. **Does this move add a `getproperty` override on any Prevision subtype?** No.

---

## Commit sequence

1. **Rename `measure.py` → `prevision.py`, class `Measure` → `Prevision`, update `__init__.py`/`functions.py`/`dsl.py`/`_bridge.py`.** Add `Measure = Prevision` backward-compat alias. Update constructor bodies to call Prevision Julia types. Update `test_bindings.py`.
2. **Migrate `credence_agents/julia_bridge.py` seval strings.** `BetaMeasure` → `BetaPrevision`, `ProductMeasure` → `ProductPrevision`, `MixtureMeasure` → `MixturePrevision`, `Measure[...]` → `Any[...]`. Update comments in `bayesian_agent.py`, `bayesian_selector.py`.
3. **Update `credence_router/routing_domain.py` comments.** Wire protocol strings unchanged.
4. **Update `bayesian_if` comments if any.** Run full test suite.
