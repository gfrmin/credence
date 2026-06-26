# complexity.jl — the single structural complexity log-prior (SPEC §1.3).
# collapse-towers Phase 1. Tested by test/test_complexity.jl.

"""
    complexity_logprior(L; λ, offset = 0.0) -> Float64

The structural complexity log-prior `−λ·L + offset` — the operational form of SPEC §1.3's
description-length weighting (`P(program) = 2^{-|program|}`, "each symbol costs 1 bit").

- `L` — a structural description length (node count, parent count, …).
- `λ` — the per-axis bits-per-unit weight. **PER-AXIS, never shared:** the program axis is pinned
  to `log(2)` by §1.3; the edge-inclusion axis is `log((1−p_edge)/p_edge)`. A single shared `λ`
  contradicts the axiom (`0 ≠ ln 2`).
- `offset` — the per-axis normalising constant. It is shared across the structures of one axis, so
  it cancels under a mixture's renormalisation; it exists so an axis can be expressed as one prior
  over one description length (e.g. the edge axis over `L = |parents|`).

`λ = 0` ⇒ uniform. Instances (collapse-towers): the program node-count prior
(`enumeration.jl`/`agent_state.jl`, `L = g.complexity + p.complexity` as the two-part MDL code,
`λ = log(2)`), the structure-BMA edge prior (`structure_bma.jl`, `L = |parents|`,
`λ = log((1−p_edge)/p_edge)`, `offset = n_features·log(1−p_edge)`), and the Family-BMA family
index (Phase 2).
"""
complexity_logprior(L::Real; λ::Real, offset::Real = 0.0) = -λ * L + offset
