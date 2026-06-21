# Design doc — the linear-Gaussian (Bayesian-linear-regression) conjugate

> Engine response to `docs/rssfeed-linear-gaussian-conjugate-request.md`. Adds one
> domain-agnostic conjugate pair so a *joint* linear-Gaussian update happens inside
> `condition`, in closed form. No RSS vocabulary in the engine.
>
> **Positioning: this is a decouple Move-2 family extension.** Move 2 (rssfeed MAUT
> ranker enablement; `docs/decouple/move-2-design.md`) established the `FAMILY_REGISTRY`
> reflected by the BDSL `:family` surface (`:normal/:soft/:weighted`) and the thin-brain
> consumption pattern (ship BDSL as `dsl_sources`, drive via `call_dsl`, beliefs cross
> back as `{type, params}` specs — the `read_params` verb was deliberately deferred there
> in favour of that round-trip). `linear-gaussian` is the *joint* generalisation of
> Move 2's per-weight `:family normal`: it conditions all active weights together, fixing
> the independent-update double-counting `maut_demo.bdsl` currently has.

## Why

A consumer (rssfeed) ranks with a parametric linear model: an article's score is
`aᵀw` over feature weights `w`, each a Gaussian belief; one engagement is one noisy
measurement of the sum, `y ~ N(aᵀw, σ²)`. The engine's only Gaussian conjugate is
scalar `(GaussianPrevision, NormalNormal)` — a direct measurement of a *single*
weight. Updating each active weight independently on the same scalar double-counts
co-occurring features and cannot explain away. The correct model is Bayesian linear
regression with a **joint** conjugate update (the Kalman measurement update).

Today the consumer hand-rolls that update with `+ - * /` on raw means/variances in
its BDSL model, because BDSL can't read a belief's variance. That arithmetic, living
outside `src/` and feeding decisions, is an Invariant-1 violation. Granting the
conjugate moves it back inside `condition`, where it belongs.

## The decision: exact, not diagonal mean-field

The request asks for a *diagonal* (mean-field / assumed-density) update that discards
the off-diagonal covariance the joint update induces. **We implement the exact
full-covariance update instead.** The ontology is always exact — exactness is not a
knob to weigh. Reasons:

1. **The off-diagonal covariance is the explaining-away signal.** The exact posterior
   `Σ' = Σ − (Σa)(Σa)ᵀ/s` induces anti-correlation between co-firing weights — the
   memory of "these competed to explain the evidence." Discarding it re-creates a
   cross-observation version of the very double-counting the request is fixing —
   worst in the consumer's stated worst case (a feed and its sole author always
   co-occur).
2. **Exact is cheap.** Closed form, O(d³) with d ≈ 5–8 active features. No performance
   problem exists, so the constitution's "don't substitute an approximation for cheap
   exact inference" applies with no escape hatch.
3. **`condition` stays exact.** A diagonal update would bake mean-field into an
   axiom-constrained function. If the consumer chooses to persist only per-feature
   marginals, that lossy projection is *its* explicit choice in *its* storage layer —
   the constitutionally correct home for an approximation.
4. **Zero cost to the consumer today.** The diagonal `vᵢ' = vᵢ − (aᵢvᵢ)²/s` is exactly
   `diag(Σ')`; after a single update the per-feature `{mu, sigma}` marginals are
   identical either way. The consumer reads the same numbers; the engine additionally
   retains `Σ'` off-diagonal for any future batched/correlated use.

## Surface

New multivariate Gaussian **Prevision** (a product of independent scalars cannot hold a
dense `Σ'`, so a genuine type is required — `prevision-not-measure`):

```julia
struct MvGaussianPrevision <: Prevision   # src/prevision.jl, module Previsions
    mu::Vector{Float64}
    Sigma::Matrix{Float64}                 # dense covariance
end
```

New leaf likelihood family (carries the per-observation coefficient vector + noise):

```julia
struct LinearGaussian <: LeafFamily        # src/kernels.jl
    coeffs::Vector{Float64}                 # a  (the article's feature values)
    sigma_obs::Float64                      # σ
end
```

Conjugate pair `(MvGaussianPrevision, LinearGaussian) → MvGaussianPrevision` —
**conditioning a genuine multivariate belief, returning a genuine multivariate
posterior.** This sidesteps the awkward "joint likelihood over a `ProductMeasure`"
path: `ProductMeasure` conditioning stays per-factor-independent as it is; the consumer
builds an `MvGaussianPrevision` (diagonal on step 1) from its stored marginals.

Exact update (`src/conjugate.jl`):

```
Σa = Σ·a ;  ŷ = aᵀm ;  s = σ² + aᵀΣa ;  k = Σa/s
m' = m + k·(y − ŷ) ;  Σ' = Σ − (Σa)(Σa)ᵀ/s
```

Read surface: `mean(p) = mu`, `variance(p) = diag(Σ)`, `marginal(p, i) =
GaussianPrevision(mu[i], √Σ[i,i])` (the exact per-feature marginal the consumer
persists), `draw(p)` via Cholesky, `expect(p, Identity) = mu`.

Wire (`apps/skin/server.jl`), mirroring `gaussian_known_var`:

```json
{ "type": "mv_gaussian", "mu": [...], "sigma": [[...],[...]] }   // prevision spec (Σ as rows)
{ "type": "linear_gaussian", "coeffs": [...], "variance": 1.0 }  // kernel spec
```

`params(p::MvGaussianPrevision) = (type=:mv_gaussian, mu=copy(mu), sigma=rows(Σ))` —
`Σ` emitted as a vector-of-rows so it round-trips through JSON (a bare Julia matrix
flattens column-major on the wire, losing shape).

**BDSL `:family` surface (the thin-brain authoring path).** `linear-gaussian`
self-registers in `FAMILY_REGISTRY` like every Move-2 family, so the consumer's pure
BDSL declares the kernel as data and the engine supplies the math:

```lisp
(define observe
  (lambda (weights xs y sigma)              ; weights: a joint MvGaussian belief
    (condition weights                       ; xs: the article's feature values (= a)
      (kernel (space :euclidean d) (space :euclidean 1)
              (lambda (w) (lambda (o) o))    ; placeholder generator; conjugate drives condition
              :family linear-gaussian xs sigma)
      y)))
```

The one generalisation Move 2 didn't need: `linear-gaussian`'s coefficients are a
**runtime vector** (`xs`, per article), whereas every prior family took 0–1 literal
scalars. So `_build_family` (`src/eval.jl`) now **evaluates** each `:family` argument in
the environment (a literal `0.5` evaluates to itself — backward-compatible) instead of
reading literal numbers. Read-back is the Move-2 `call_dsl` belief-spec round-trip:
`observe` returns the conditioned `MvGaussianMeasure`, which `serialize_value → params`
emits as `{type:mv_gaussian, mu, sigma}`; the consumer marginalises client-side. No new
wire method (the deferred `read_params` verb stays deferred — rssfeed is a stateless
thin body that persists its own marginals).

The JSON skin kernel spec `{"type":"linear_gaussian", …}` is *also* wired (for the
stateful `create_state`+`condition` wire pattern), but the thin-body path above is the
decouple-preferred one.

## Out of scope (v1)

- **Unknown σ² (Normal-Gamma regression).** Known σ² only.
- **Batched design matrix.** One `(a, y)` per call; observations folded one at a time.
- **Full `MvNormal` `expect` over arbitrary functionals.** `Identity` + marginals cover
  the consumer's need; richer functional dispatch (e.g. the linear score `aᵀw` as a
  `LinearCombination`) can accrete when a caller needs it.

## Resolved decisions

- **Read-back over the wire — RESOLVED: the Move-2 `call_dsl` belief-spec round-trip.**
  Earlier draft framed this as open; it is settled by `docs/decouple/move-2-design.md`.
  The consumer's `observe` returns the conditioned belief and `serialize_value → params`
  emits a readable `{type:mv_gaussian, mu, sigma}` spec (the scalar `mean` wire method and
  the opaque `snapshot` blob are *not* the consumer's path). No new wire method — the
  deferred `read_params` verb stays deferred.
- **BDSL authoring surface — RESOLVED: `:family linear-gaussian` (extend `:family` to
  evaluate args).** Chosen over a new `(linear-gaussian …)` builtin because the
  frozen-type-constructor rule keeps the kernel surface to `(kernel … :family …)`, and
  over a wire read method because that would add a second consumption surface against the
  one-surface decouple commitment. The `:family`-arg evaluation generalisation is the
  minimal change that admits the runtime coeffs vector.

## Open design questions

- **Should `marginal(p, indices)` return a `ProductMeasure` of the selected marginals**
  (dropping cross-covariance, an explicit consumer-side projection) rather than only the
  per-index `GaussianPrevision`? Deferred until a caller needs the multi-index form; the
  per-index accessor is exact and sufficient for the persist-marginals use.
- **Should the skin accept a diagonal convenience spec** (`"sigma": [v1, v2, ...]` as a
  diagonal) so the consumer need not send a full matrix on step 1? Minor ergonomics;
  left out of v1 (the consumer sends a diagonal matrix).

## Verification

- `test/test_linear_gaussian_conjugate.jl` — the conjugate at the prevision level: the
  confident-vs-diffuse explaining-away check, exact posterior `mu`/`Σ` against an
  independent Kalman oracle at rtol 1e-12, the off-diagonal-is-nonzero exactness
  assertion, the `diag(Σ') == diagonal-form` zero-cost identity, `condition≡update`,
  error paths, two-step folding.
- `test/test_linear_gaussian_dsl.jl` — the thin-brain BDSL path end-to-end: a pure
  `observe` conditions a joint `MvGaussian` via `:family linear-gaussian xs sigma` (runtime
  coeffs), asserts the posterior against the oracle, checks the `params` belief-spec
  read-back, and confirms `:family normal 0.5` (literal args) still parses.
- `test/test_prevision_params.jl` — `mv_gaussian` `params` round-trip (Σ as rows).
- `test/test_family_registry.jl` — updated for the evaluated-args `:family` surface +
  `linear-gaussian` in the roster.
