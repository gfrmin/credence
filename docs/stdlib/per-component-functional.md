# `FiringChoice` — per-component dispatch over a mixture

> Stdlib addition resolving issue #39. Tier-1 `TestFunction` subtype; behaviour
> constrained by the axioms (it is an `expect` integrand), interface negotiable.

## Problem

Six sites in `apps/` computed a weighted mixture expectation by hand-iterating
the components:

```julia
acc = 0.0
for (j, comp) in enumerate(belief.components)
    rec_j  = compiled_kernels[j].evaluate(features, …)   # the program's recommendation
    θ_j    = mean(comp)                                   # the component's Beta mean
    acc   += weights(belief)[j] * (rec_j == target ? θ_j : 1 - θ_j)
end
```

Each carried a `precedent:posterior-iteration` lint pragma. The arithmetic
`Σ_j w_j · g_j(θ_j)` belongs inside `expect` (Invariant 1, topological face); the
loop only existed because the **sub-functional `g_j` differs per component**.

`expect(mixture, φ)` already computes `Σ_j w_j · expect(component_j, φ)`
(`src/ontology.jl`), but applies the *same* `φ` to every component. There was no
Functional that selects a sub-functional per component.

## Key observation: every site is binary

In all six sites the per-component choice is binary — the component's program
either recommends the target outcome or it doesn't — and the per-component value
is an **affine function of the component's mean**:

| site | when it fires | when it doesn't |
|------|---------------|-----------------|
| grid_world `p_enemy` | `θ` | `1-θ` |
| grid_world EU | `5-10θ` | `-5+10θ` |
| email per-action `P` | `θ` | `(1-θ)/(\|A\|-1)` |
| skin per-label `P` | `θ` | `(1-θ)/(n-1)` |
| skin EU (n=2) | `r₁θ+r₂(1-θ)` | `r₁(1-θ)+r₂θ` |
| rss score | `θ` | `0.5` (constant) |

Because `when_fires`/`when_not` can be **arbitrary `TestFunction`s**, a single
binary subtype covers all of them — `1-θ`, `(1-θ)/(n-1)`, an affine EU
contribution, or a constant are all expressible as `LinearCombination`s
(`1-θ = LinearCombination([(-1.0, Identity())], 1.0)`; a constant `c =
LinearCombination(Tuple{Float64,TestFunction}[], c)`).

## The type

```julia
struct FiringChoice <: TestFunction
    fired::Vector{Bool}        # one flag per mixture component
    when_fires::TestFunction
    when_not::TestFunction
end
```

```julia
expect(p::MixturePrevision, fc::FiringChoice) =
    Σ_i weights(p)[i] · expect(p.components[i], fc.fired[i] ? fc.when_fires : fc.when_not)
```

Defined on `MixturePrevision` and `MixtureMeasure` (length of `fired` checked at
dispatch, as `Tabular` checks its length); a non-mixture prevision errors — it is
inherently a mixture-level integrand, with no point-wise `apply`. A
`LinearCombination` branch is expanded by linearity in `_firing_branch` before it
reaches a leaf component, sidestepping the pre-existing dispatch ambiguity between
`expect(::leaf, ::TestFunction)` (apply-fallback) and `expect(::Prevision,
::LinearCombination)` (the mixture-level path that previously kept leaves from
ever seeing a `LinearCombination`).

The caller precomputes `fired` by evaluating each component's compiled kernel at
the query features — a forward program evaluation that builds declarative data,
not weight arithmetic. The weight arithmetic lives entirely in `expect`.

## Why this shape (over a general per-component vector)

A more general subtype carrying `branches::Vector{TestFunction}` (one arbitrary
sub-functional per component) was considered. `FiringChoice` was chosen because:

1. **It names the pattern (Invariant 2).** The constitution already establishes a
   declarative vocabulary for per-component routing in mixtures — kernel-side
   `FiringByTag(fires, when_fires, when_not)` — and instructs adding a declarative
   subtype before reaching for the opaque `DispatchByComponent`/`Vector` escape
   hatch. `FiringChoice` is the exact `expect`-side dual: same "some fired, some
   didn't" structure, self-documenting in the type.
2. **It is sufficient.** All six sites are binary; a general vector is never
   exercised, and `when_fires`/`when_not` being full `TestFunction`s already
   absorbs every affine/constant variation the sites need.

The general `Vector{TestFunction}` form is the natural escape hatch if a genuine
≥3-way per-component dispatch ever appears (the analogue of `DispatchByComponent`);
it is deliberately **not** shipped now.

## Exact parity

`when_fires = Identity()` reads each leaf's mean via the closed-form
`expect(::BetaPrevision, ::Identity) = α/(α+β)`. The grid_world and skin sites
already used the closed-form mean, so their rewrites are numerically exact; the
email site previously used `expect(tbm, identity)` (64-point quadrature of the
identity), so its rewrite is *more* precise — the difference is quadrature error,
and the email suite (behavioural) is unaffected. Parity is covered by the existing
suites plus an explicit old-vs-new block in
`test/test_per_component_functional.jl` for the untested skin `handle_eu_interact`.

## Open design questions

- **Name.** `FiringChoice` mirrors `FiringByTag` but the "tag" half doesn't
  carry over (firing here is a precomputed per-position flag, not a `Set{Int}` of
  tags membership-tested against the component). `PerComponentFiring` /
  `MixtureFiring` were alternatives.
- **`fired::Vector{Bool}` vs `Set{Int}`.** A `Vector{Bool}` is length-checkable
  against the component count and reads as obviously per-component; `Set{Int}` of
  positions would mirror `FiringByTag.fires` more literally but is sparser. The
  `Vector{Bool}` form was chosen for the length check and readability.
- **A `Constant(c)` `TestFunction`.** The empty-`LinearCombination` idiom for a
  constant works but is slightly opaque; a named `Constant` could be added later
  if constants proliferate. Deferred.
- **`apps/julia/rss`.** Reference-only post-decouple (its test rots on master);
  its loop keeps the `posterior-iteration` pragma (re-documented, #39 linkage
  dropped) and collapses identically if the directory is revived.
