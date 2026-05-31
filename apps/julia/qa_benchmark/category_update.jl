# Role: brain-side application
"""
    category_update.jl — reliability learning under inferred category uncertainty.

Paper 1, B2c (`docs/paper1/move-2c-design.md`). Tool reliability `θ_{t,c}`
is per (tool, category). When a question's category is uncertain (soft
posterior π from the classifier), both the decision and the update
marginalise over π:

- `marginal_reliability(row, π)` — the category-marginalised reliability
  belief for the decision side: a `MixturePrevision` over the per-category
  Betas weighted by π, fed to the existing `eu`/`voi`/`expect`.
- `update_reliability(row, π, outcome)` — the resource-rational update:
  credit *every* category by its posterior weight via the
  `WeightedBernoulli` conjugate `condition`
  (`α_{t,c} += π_c·correct, β_{t,c} += π_c·wrong`). Uses the whole
  posterior (not MAP); reduces *exactly* to the unit-count update when π
  is one-hot.

`row` is a tool's reliability row `[θ_{t,c} for c]::Vector{BetaPrevision}`;
`π` is the category posterior (same length). Pure functions — no embedding
dependency; the host supplies π from the classifier (wired in B3/B4).
"""

using Credence
using Credence: BetaPrevision

# Reliability observation kernel declaring the fractional family. Source =
# reliability ∈ [0,1]; target = the binary correctness outcome. The
# conjugate path is registry-driven; generate/log_density are placeholders.
const WEIGHTED_RELIABILITY_KERNEL = Kernel(
    Interval(0.0, 1.0),
    Finite([0, 1]),
    r -> error("WEIGHTED_RELIABILITY_KERNEL.generate not exercised (conjugate path)"),
    (r, o) -> 0.0;
    likelihood_family = WeightedBernoulli(),
)

"""
    update_reliability(row, π, outcome) -> Vector{BetaPrevision}

Full-posterior-weighted reliability update. `outcome ∈ {0,1}` (1 = tool
correct). Each category's Beta is conditioned on the same outcome with
weight `π_c`, via `WeightedBernoulli`. One-hot π reduces exactly to the
unit-count update.
"""
function update_reliability(row::Vector{BetaPrevision}, π::Vector{Float64}, outcome)
    length(row) == length(π) ||
        error("update_reliability: row length $(length(row)) ≠ π length $(length(π))")
    o = Float64(outcome)
    BetaPrevision[
        condition(row[c], WEIGHTED_RELIABILITY_KERNEL, (o, π[c])) for c in eachindex(row)
    ]
end

"""
    marginal_reliability(row, π) -> MixturePrevision

Category-marginalised reliability belief for the decision side: the
mixture `Σ_c π_c · Beta(θ_{t,c})`. Zero-weight categories are dropped (a
category with `π_c = 0` contributes nothing). Fed to `eu`/`voi`/`expect`.
"""
function marginal_reliability(row::Vector{BetaPrevision}, π::Vector{Float64})
    length(row) == length(π) ||
        error("marginal_reliability: row length $(length(row)) ≠ π length $(length(π))")
    keep = π .> 0.0
    any(keep) || error("marginal_reliability: π has no positive mass")
    MixturePrevision(row[keep], log.(π[keep]))
end
