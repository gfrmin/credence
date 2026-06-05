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
- `update_reliability(row, π, outcome)` — *soft-credit* (B2c): credit every
  category by the classifier weight `π_c` via the `WeightedBernoulli`
  conjugate `condition` (`α_{t,c} += π_c·correct, β_{t,c} += π_c·wrong`).
  `π` is the *prior* for the credit step — it ignores the outcome's
  likelihood. Reduces *exactly* to the unit-count update when π is one-hot.
- `post_update(row, π, outcome)` — *posterior-weighted credit* (issue #111,
  the deployed default): credit by the one-step category posterior
  `ρ_c ∝ π_c·ℓ_{t,c}(outcome)` instead of the prior `π_c`, so a correct
  answer credits the category where the tool is reliable, not the
  misclassified one. Both the category posterior `ρ`
  (`posterior_credit_weights`) and the per-category Beta updates go through
  `condition` — no host arithmetic. Strictly dominates soft (which discards
  the likelihood); reduces to soft, hence to the unit-count update, when π
  is one-hot.

DECLARED APPROXIMATION (no-silent-approximation rule). The exact joint
latent-category update does NOT factorise over tools: each question's
category is a latent shared across every tool queried on it and the
decision, so the exact posterior over `{θ_{t,c}}` is a mixture over global
category assignments (~K^#questions) — intractable. `soft` and `post` are
both mean-field / assumed-density projections; they differ only in the
credit weight. `post`'s category posterior `ρ` is *exact one-step*, but the
per-category reliability collapse (the fractional pseudo-count update)
projects the exact 2-component mixture
`ρ_c·Beta(α+o,β+1-o) + (1-ρ_c)·Beta(α,β)` back onto a single Beta. The
principled extension — retain k mixture components and choose k by EU-max
over accuracy-vs-compute (metareasoned fidelity) — is out of scope here.

`row` is a tool's reliability row `[θ_{t,c} for c]::Vector{BetaPrevision}`;
`π` is the classifier's category posterior (same length). Pure functions —
no embedding dependency; the host supplies π from the classifier.
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

Soft-credit reliability update (B2c). `outcome ∈ {0,1}` (1 = tool correct).
Each category's Beta is conditioned on the same outcome with weight `π_c`
(the classifier prior — the outcome likelihood is NOT used), via
`WeightedBernoulli`. One-hot π reduces exactly to the unit-count update.
`post_update` is the deployed default (issue #111); pass these weights as a
posterior `ρ` and this becomes the posterior-weighted update.
"""
function update_reliability(row::Vector{BetaPrevision}, π::Vector{Float64}, outcome)
    length(row) == length(π) ||
        error("update_reliability: row length $(length(row)) ≠ π length $(length(π))")
    o = Float64(outcome)
    BetaPrevision[
        condition(row[c], WEIGHTED_RELIABILITY_KERNEL, (o, π[c])) for c in eachindex(row)
    ]
end

# Category-belief kernel for the posterior credit step. Source = category
# index space (0-based, aligned to `CATEGORIES`); target = the binary
# correctness outcome. For category `c` the per-outcome log-likelihood uses
# the queried tool's predictive correctness `m_c = E[θ_{t,c}]` (`o=1`) or
# `1 - m_c` (`o=0`). Conditioning the classifier prior π on the observed
# outcome through this kernel is the exact one-step category posterior —
# the Bayes step lives in `condition`, not in host arithmetic.
# `rel_means[Int(c)+1]` maps the 0-based space value to the 1-based vector.
function category_belief_kernel(rel_means::Vector{Float64})
    Kernel(
        Finite(collect(0:length(rel_means) - 1)),
        Finite([0, 1]),
        c -> error("category_belief_kernel.generate not exercised (categorical condition path)"),
        (c, o) -> (Float64(o) == 1.0 ? log(rel_means[Int(c) + 1]) :
                                       log(1.0 - rel_means[Int(c) + 1]));
        likelihood_family = Flat(),
    )
end

"""
    posterior_credit_weights(row, π, outcome) -> Vector{Float64}

The exact one-step category posterior `ρ` given the observed `outcome ∈
{0,1}` from a tool whose per-category reliability beliefs are `row`:
`ρ_c ∝ π_c · ℓ_{t,c}(outcome)`, computed by conditioning the categorical
prior π (via `condition`) — never by host-side Bayes. The predictive
likelihood `ℓ_{t,c}` reads `mean(row[c])` (the sanctioned accessor); for
Beta–Bernoulli the predictive correctness probability IS the mean, so this
likelihood is exact. One-hot π returns one-hot ρ.
"""
function posterior_credit_weights(row::Vector{BetaPrevision}, π::Vector{Float64}, outcome)
    length(row) == length(π) ||
        error("posterior_credit_weights: row length $(length(row)) ≠ π length $(length(π))")
    means = Float64[mean(row[c]) for c in eachindex(row)]
    prior = CategoricalMeasure(Finite(collect(0:length(π) - 1)), log.(π))
    post = condition(prior, category_belief_kernel(means), Float64(outcome))
    weights(post)
end

"""
    post_update(row, π, outcome) -> Vector{BetaPrevision}

Posterior-weighted reliability update (issue #111, the deployed default).
Credits each category by the one-step category posterior
`ρ_c ∝ π_c·ℓ_{t,c}(outcome)` instead of the classifier prior `π_c` — see the
file-level docstring for the declared-approximation statement (both soft and
post are mean-field projections of the intractable joint; post is the better
projection). Reduces to `update_reliability` (hence to the unit-count update)
when π is one-hot.
"""
function post_update(row::Vector{BetaPrevision}, π::Vector{Float64}, outcome)
    ρ = posterior_credit_weights(row, π, outcome)
    update_reliability(row, ρ, outcome)
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
