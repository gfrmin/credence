# Phase 2 design doc — Family-BMA (complexity prior on a family axis)

> Seven-section template (`docs/collapse-towers/DESIGN-DOC-TEMPLATE.md`). Master plan:
> `docs/collapse-towers/master-plan.md`. Precedents: `docs/precedents.md`.

> **Implementation outcome (two deviations from the design below — landed, full suite 40/40 green):**
> 1. **Exact conjugate predictives added (approved scope expansion).** Family reweighting needs the
>    per-family *marginal likelihood*, and Gaussian/NormalGamma had no closed form — only the generic
>    Monte-Carlo `_predictive_ll(::Measure)` fallback (approximate + non-deterministic), which the
>    nominal kernel's error-stub correctly refused. Added exact closed forms in `ontology.jl`:
>    `_predictive_ll(::GaussianMeasure, NormalNormal)` = `N(obs|μ₀,√(σ₀²+σ²))`, and
>    `_predictive_ll(::NormalGammaMeasure, NormalGammaLikelihood)` = Student-t, plus a machine-precision
>    Lanczos `_loggamma` (the core is stdlib-only — no SpecialFunctions; this is a special-function
>    eval like `log`/`exp`, not a Bayesian approximation). Non-conjugate kernels still fall back to the
>    generic path. This also closes the latent VOI-over-Gaussian MC gap.
> 2. **The class-1 mixture-condition dedup was REVERTED — deferred to `measure-as-view`.** Collapsing
>    `condition(MixtureMeasure)` to a facade over `condition(m.prevision)` is **unsafe**: it passes
>    Prevision components where consumers expect Measures and drops per-component carrier-space context
>    (a `CategoricalPrevision`/`ProductPrevision` can't be `wrap_in_measure`d without its `Finite`
>    space). It broke `test_flat_mixture` + `test_host`; reverted to the Measure-level loop. So Phase 2
>    adds the per-component routing **only** to `condition(MixturePrevision)` (which Family-BMA uses);
>    the mixture-condition dedup joins the other facade dedups in the `measure-as-view` arc. (§2 below
>    still describes the attempted facade — kept for the reasoning; the NOTE in `ontology.jl` records why.)

## 1. Purpose

Phase 2 as scoped in the master plan: a **posterior over likelihood families** for one leaf, built
as a `MixturePrevision` whose components are the per-family conjugate priors (over the *same*
observation space), prior-weighted by Phase 1's `complexity_logprior` on the family index, and
conditioned by the *existing* chain-rule marginal-likelihood reweighting. This makes "which family
generates this leaf?" an inference (BMA), replacing the hand-declared single `:family` choice — and
the agent **averages over the posterior, never selects a family** (`average-not-collapse`).

It is the structural twin of structure-BMA (`src/structure_bma.jl`): there the mixture ranges over
parent-sets with a `complexity_logprior(|parents|; …)` prior; here it ranges over families with a
`complexity_logprior(family_index_length; …)` prior. The one substrate gap (traced in §4) is that
`condition(MixturePrevision)` does not resolve per-component families for heterogeneous prior types;
closing it also collapses the `condition(MixtureMeasure)` duplication (the "assess duplication"
finding — class 1 only; classes 2+3 are the separate `measure-as-view` arc).

## 2. Files touched

- **`src/family_bma.jl`** — *new*. The builder + readout, in `structure_bma.jl`'s spirit (compositions
  over existing Tier-1 objects; no new frozen type, no new axiom-constrained function):
  - `FamilyBMA` descriptor (the candidate specs: per family a `(LikelihoodFamily, prior::Prevision)`
    pair + a declared `L_family` length; the shared observation `Space`; `λ_family`).
  - `build_family_prior(model) -> MixturePrevision` — components = the candidate priors (heterogeneous
    `Prevision[]`); weights = `complexity_logprior(L_family_i; λ=λ_family)` (uniform default).
  - `_family_kernel(model)` — a `DispatchByComponent(classify)` kernel; `classify(c)` dispatches on the
    component's **prevision type** → that family's `LikelihoodFamily`. Nominal source + error-stub
    `generate`/`log_density` (the rho_latent pattern, `test_rho_latent.jl:37-40`); `target` = the
    shared obs space.
  - `family_observe(model, mixture, obs) = condition(mixture, _family_kernel(model), obs)`.
  - `family_posterior(mixture) = weights(mixture)` — the readout IS the mixture; carries the
    `# credence-lint: allow — precedent:average-not-collapse — …` pragma.
  - Construction guards: **commensurability** (all candidate families score the same obs `Space`);
    **distinct prevision types** (so `classify` is unambiguous — see §5).
- **`src/ontology.jl`** — *modify*:
  - `condition(p::MixturePrevision, k, obs)` (`:1610`): add **gated per-component family resolution**
    — when `k.likelihood_family isa DispatchByComponent`, resolve `fam = _resolve_likelihood_family(
    k.likelihood_family, comp)` and pass `_with_resolved_family(k, fam)` to that component's
    `_predictive_ll`/`condition`; otherwise pass `k` through unchanged (FiringByTag stays cell-resolved).
  - `condition(m::MixtureMeasure, k, obs)` (`:1630`): collapse to a **thin facade** delegating to the
    Prevision-level (`wrap_in_measure(condition(m.prevision, k, obs))`), retiring the duplicated loop.
  - `include("family_bma.jl")` after `structure_bma.jl`; `export` the builder surface.
- **`src/Credence.jl`** — *modify*: re-export the `FamilyBMA` surface (mirroring the `StructureBMA`
  export line at `:61`).
- **`test/test_family_bma.jl`** — *new* (see §7).

## 3. Behaviour preserved

Tolerance classes:

- **Regression — bit-exact (`==`), capture-before-refactor.** The gated-resolution + facade-delegation
  edits to the two mixture-`condition` methods must leave existing mixture conditioning identical:
  - structure-BMA (`test_structure_bma.jl`): FiringByTag is *not* `DispatchByComponent` ⇒ ungated ⇒
    `k` passes through ⇒ `SparseStructurePrevision` cell-level self-resolution unchanged.
  - rho-latent (`test_rho_latent.jl`): `DispatchByComponent` over `LabelledCategoricalPrevision` ⇒
    now resolved at the mixture level to the `GroupNoisyChannel` leaf, then `condition(
    LabelledCategoricalPrevision, …)`'s own `_resolve_likelihood_family` returns that leaf unchanged
    (**idempotent** re-resolution) ⇒ bit-identical. Pinned by capturing the L=1/L>1 posteriors pre-edit.
  - product-BMA (`test_product_bma_routing.jl`): uses `condition(ProductMeasure/ProductPrevision)`, a
    *different* method — untouched.
  - mixture core (`test_core.jl` TEST 53 DispatchByComponent) and `MixtureMeasure` conditioning:
    capture canonical posteriors pre-edit, assert `==` post-edit (the facade must equal the old loop).
- **New capability — directional (no thresholds).** Family recovery: more posterior mass on the true
  family; the gap *widens* with data. Singleton candidate set: **bit-exact** reduction to the
  fixed-family posterior (`==`).

## 4. Worked end-to-end example (mandatory — dual residency)

**Inputs.** Two candidate families over ℝ (`obs ∈ Euclidean(1)`): `NormalNormal(σ=1.0)` with prior
`GaussianPrevision(μ0=0.0, σ0=2.0)`, and `NormalGammaLikelihood()` with prior
`NormalGammaPrevision(κ0=1, μ0=0, α0=2, β0=2)`. Uniform family prior (`λ_family=0`). `obs = 1.5`.

**Construction (owner: `family_bma.jl`).** `build_family_prior` →
`MixturePrevision(Prevision[GaussianPrevision(0,2), NormalGammaPrevision(1,0,2,2)], [0.0, 0.0])`
(weights from `complexity_logprior(L; λ=0)` = equal). `_family_kernel` →
`Kernel(nominal, Euclidean(1), stub, stub; likelihood_family = DispatchByComponent(classify))`,
`classify(c::GaussianPrevision)=NormalNormal(1.0)`, `classify(c::NormalGammaPrevision)=NormalGammaLikelihood()`.

**`family_observe` → `condition(p::MixturePrevision, k, 1.5)` (owner: `ontology.jl:1610`).**
`k.likelihood_family isa DispatchByComponent` ⇒ gated path ON. Per component:
- `comp₁ = GaussianPrevision(0,2)`: `fam₁ = _resolve_likelihood_family(DispatchByComponent, comp₁) =
  NormalNormal(1.0)` (owner: `kernels.jl:323`); `k₁ = _with_resolved_family(k, fam₁)`.
  - `pred_ll₁ = _predictive_ll(comp₁, k₁, 1.5)` → `GaussianMeasure` NormalNormal predictive (owner:
    `ontology.jl:1560`→GaussianMeasure path) = log N(1.5 | 0, √(2²+1²)).
  - `conditioned₁ = condition(comp₁, k₁, 1.5)` → `maybe_conjugate(GaussianPrevision, k₁=NormalNormal)`
    matches ⇒ closed-form Gaussian update (owner: `conjugate.jl` Gaussian-Normal). **Note:** the
    conjugate path uses only `k₁.likelihood_family` + obs, *not* `k.source`/`target` (verified at
    `ontology.jl:1292-1296`) — which is why the nominal kernel source is sound.
- `comp₂ = NormalGammaPrevision(…)`: `fam₂ = NormalGammaLikelihood()`; `k₂`.
  - `pred_ll₂` → Student-t predictive (owner: NormalGamma `_predictive_ll`).
  - `conditioned₂ = condition(comp₂, k₂, 1.5)` → `maybe_conjugate(NormalGammaPrevision, NormalGammaLikelihood)`
    matches ⇒ NormalGamma update (owner: `conjugate.jl`).
- `new_log_weights = [0.0 + pred_ll₁, 0.0 + pred_ll₂]`; result `MixturePrevision([conditioned₁,
  conditioned₂], new_log_weights)`.

**Readout (owner: `family_bma.jl`).** `family_posterior(result) = weights(result)` → e.g.
`[0.46, 0.54]` — the **posterior over families**. The agent carries this and marginalises in any
decision (`argmax_a expect(result, u_a)`); there is no `argmax_m weights` step (`average-not-collapse`).

**Dual residency.** After this phase `condition(MixtureMeasure)` is a thin facade; the **authoritative**
reweighting loop lives once, at `condition(MixturePrevision)`. The Measure path delegates via
`m.prevision` and re-wraps with `wrap_in_measure`.

## 5. Open design questions

1. **`classify` by prevision-type + a distinctness guard (a designed tripwire, not a limitation) — ACCEPTED.**
   `classify` dispatches on each component's *prevision type*, so v1 requires the candidate priors to
   have **distinct prevision types** (the builder errors on duplicates). Classify-by-type is a *shortcut*
   that conflates a family with the prevision type it produces; the genuinely-correct design is a
   labelled component (cf. `LabelledCategoricalPrevision.label`). So the distinct-type guard is a
   **tripwire** — the moment it fires is the moment the labelled wrapper is earned, and not before; the
   right place to spend that complexity later, not now. **Verified mechanism (corrected from the first
   draft):** the `:family` trio `bernoulli`/`soft`/`weighted` (and `flat`) are *all* conjugate to
   `BetaPrevision` (`conjugate.jl:12-23`), so the guard excludes them directly via the **shared prior
   type** — not via commensurability. (Their observations *also* differ — scalar `{0,1}` vs the `(r,w)` /
   `(outcome,w)` tuples — a secondary reinforcement.) The guard's only *false* exclusion would be a pair
   of genuinely-different families sharing **both** a prior type and an obs space — rare, and exactly the
   case that earns the wrapper. **Deferred.**
2. **Nominal kernel source + error-stub — ACCEPTED.** The conjugate condition reads only the prevision,
   the resolved `likelihood_family`, and the obs — never `k.source`/`target` (verified
   `ontology.jl:1292-1296`). The safety is **complete, not partial:** conjugate is correct, and the
   *non-conjugate* branch falls through to the measure-level condition, which routes through the kernel's
   `log_density` — the **error-stub** — so it fails loud. There is no silent third path. **Added guard
   (cheap, earlier-and-clearer than waiting for the stub to fire):** the builder validates at
   construction that each `(prior, family)` candidate is **conjugate-recognised**
   (`maybe_conjugate(prior, kernel-with-that-family) !== nothing`), so a non-conjugate candidate errors
   at `build_family_*`, not on the first observation. Mirrors the commensurability + distinct-type
   construction guards.
3. **`L_family` / `λ_family` default = uniform — ACCEPTED, with the meaning of `L_family` pinned.**
   Mirrors structure-BMA's `p_edge=0.5` default. The marginal likelihood (the evidence integral) already
   performs Occam over parameter *dimension*, so a non-uniform family prior is a *second, additive*
   penalty. **`L_family` must encode the *specification length of the family*** — the bits to name and
   define the family itself — and **explicitly NOT a proxy for its parameter count** (already priced by
   the evidence); conflating the two penalises flexibility twice. **Honesty note:** at the uniform
   default the family-axis `complexity_logprior` is a *no-op* — structural-and-available, not
   load-bearing. The no-op default must not later be "fixed" into a parameter-count penalty — that *is*
   the double-count. Pin this meaning in the `L_family` field docstring.
4. **Resolution scope = `condition` only for v1 — ACCEPTED, and the deferral is PINNED BY A TEST.**
   Phase 2 delivers the family *posterior* (carry + condition); only `condition(MixturePrevision)` needs
   the gated resolution. Using a family mixture as a *decision belief* (VOI/`predictive_prob` over the
   whole mixture) would also need gating in `log_predictive(::MixturePrevision)` (`:1587`), which today
   passes the kernel **unresolved**. The deferral is **safe but incidentally so:** for a family mixture
   the heterogeneous components (`GaussianPrevision`/`NormalGammaPrevision`) don't self-resolve,
   `maybe_conjugate` won't match an unresolved `DispatchByComponent`, and the path falls through to the
   nominal kernel's **error-stub `log_density` — so a family-mixture predictive ERRORS today.** Because
   that safety is incidental, **a test asserts `log_predictive`/`predictive_prob` over a family mixture
   raises** — guarding the deferral by assertion, not assumption. (Without it, a future self-resolving
   predictive path would silently make family-mixture predictive "work" by collapsing/misresolving the
   family belief — the `average-not-collapse`-class failure this phase exists to prevent, landing with no
   test red.)

## 6. Risk + mitigation

- **Gated resolution silently changes an existing mixture posterior.**
  - *Blast radius:* `test_structure_bma`, `test_rho_latent`, `test_product_bma_routing`, `test_core`
    (TEST 53/55), any `MixtureMeasure` conditioning test.
  - *Pre-emptive grep (run before code):* `grep -rn "condition(.*Mixture" src/ test/ apps/` and
    `grep -rn "DispatchByComponent\|FiringByTag" test/` — list each mixture-conditioning site and its
    disposition (gated vs pass-through). FiringByTag sites must be pass-through (ungated); only
    DispatchByComponent sites take the new path.
  - *Mitigation:* capture canonical posteriors PRE-edit for the four regression suites; assert `==`
    post-edit (capture-before-refactor). rho-latent is the sharp case (DispatchByComponent re-resolution
    must be idempotent) — pin its L=1 and L>1 posteriors byte-for-byte.
- **`condition(MixtureMeasure)` facade drifts from the old loop.**
  - *Mitigation:* a test conditions a `MixtureMeasure` and asserts the facade result `==` a captured
    pre-refactor value; `wrap_in_measure∘condition(.prevision)` must reproduce the space + components.
- **Heterogeneous `Prevision[]` mixture trips the `untyped-mixture-construction` lint.**
  - *Mitigation:* the family mixture is *genuinely* heterogeneous (distinct families) — construct with
    an explicit `Prevision[...]` typed literal (not `Any[]`); this is the precedent's sanctioned form,
    documented at the construction site.
- **`average-not-collapse` drift.** *Mitigation:* a test asserts `family_posterior` returns the full
  weight vector (length = #candidates), not a scalar/index; no `select_family` function exists.
- **The error-stub `log_density` is a single load-bearing backstop for TWO decisions (Q2 + Q4).** It
  underwrites both the nominal-source soundness (the non-conjugate `condition` fallback) and the
  predictive-deferral safety (the unresolved family-mixture `log_predictive` fallback). *Mitigation:* it
  must be a genuine `error(...)` — **not** a warning, **not** a NaN-returning stub — and the tests
  exercise it on **both** paths (a family-mixture `log_predictive` attempt; and the construction
  conjugate-recognised guard, which forecloses the non-conjugate `condition` path before the stub fires).
  One correct stub secures both decisions.

## 7. Verification cadence

End of Phase-2 code (from repo root):
```
julia test/test_family_bma.jl
julia test/test_structure_bma.jl
julia test/test_rho_latent.jl
julia test/test_product_bma_routing.jl
julia test/test_core.jl
julia test/test_family_registry.jl
```
Then the full `test/test_*.jl` suite green, the lint corpus self-test + `check apps/`, and **stop and
report** at the phase boundary. Skin smoke is **optional** for Phase 2 (no JSON-RPC verb added; the
mixture-condition change is engine-internal — no wire-spec emits a family-BMA kernel yet).

`test_family_bma.jl` (repo `check(name, cond, detail)` idiom; no `using Test`):
- Generative recovery (directional): data from the true family ⇒ its posterior weight rises and the
  gap to the rival *widens* with more data (no threshold).
- Degenerate reduction: a **singleton** candidate set conditions **bit-identically** (`==`) to the
  plain conjugate posterior for that family (the `test_decide_with_voi` degenerate style).
- rho-latent / structure-BMA / mixture-core regressions pinned `==` to pre-refactor captures.
- Commensurability guard: candidates over different obs spaces ⇒ **error** at construction (no fallback).
- Distinct-type guard: two candidates with the same prevision type ⇒ **error** at construction (the
  `bernoulli`/`soft`/`weighted` trio — all `BetaPrevision` — is the concrete case).
- Conjugate-recognised guard (Q2): a candidate `(prior, family)` that `maybe_conjugate` does not
  recognise ⇒ **error** at construction.
- Deferral guard (Q4): `log_predictive`/`predictive_prob` over a family mixture **raises** today
  (exercises the error-stub on the predictive path) — pins the deferral by assertion.
- `average-not-collapse`: `family_posterior` returns the mixture weights, not an `argmax`; the
  comparison oracle carries the `test-oracle` pragma.
