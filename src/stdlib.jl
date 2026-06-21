# stdlib.jl — Derived functions and convenience accessors
#
# These are compositions of the axiom-constrained functions with
# ordinary computation. They are convenience, not capability.
# Their interfaces are negotiable and will evolve.

# ── WeightsDomainError: informative error for continuous Previsions ──

struct WeightsDomainError <: Exception
    message::String
end

Base.showerror(io::IO, e::WeightsDomainError) = print(io, "WeightsDomainError: ", e.message)

const _WEIGHTS_DOMAIN_MSG =
    "weights is defined only for finite-support Previsions; " *
    "for continuous Previsions, use probability(p, e::Event) with a declared Event " *
    "to obtain a measure of an event's mass, or expect(p, f) for an integrated functional."

# ── Stdlib one-liners over expect ──

mean(p::Prevision) = expect(p, Identity())

function variance(p::Prevision)
    μ = mean(p)
    expect(p, CenteredSquare(μ))
end

variance(p::BetaPrevision) = p.alpha * p.beta / ((p.alpha + p.beta)^2 * (p.alpha + p.beta + 1))
variance(p::GaussianPrevision) = p.sigma^2
# Per-coordinate marginal variances diag(Σ) — exact (a Gaussian's marginals are
# read off the covariance diagonal). Cross-covariance is in `p.Sigma`.
variance(p::MvGaussianPrevision) = [p.Sigma[i, i] for i in 1:length(p.mu)]

# The exact i-th marginal of a multivariate Gaussian is the scalar Gaussian
# N(μᵢ, Σᵢᵢ) — what a consumer persisting per-feature {mu, sigma} reads back.
# Marginalisation drops cross-covariance by construction; that projection is the
# consumer's explicit choice (the joint Σ stays available on `p`).
marginal(p::MvGaussianPrevision, i::Int) =
    GaussianPrevision(p.mu[i], sqrt(p.Sigma[i, i]))

probability(p::Prevision, e::Event) = expect(p, Indicator(e))

function weights(p::CategoricalPrevision)
    lw = p.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./ sum(w)
end

function weights(p::MixturePrevision)
    lw = p.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./ sum(w)
end

weights(p::ParticlePrevision) = begin
    lw = p.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./ sum(w)
end

weights(p::QuadraturePrevision) = begin
    lw = p.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./ sum(w)
end

weights(p::BetaPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))
weights(p::TaggedBetaPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))
weights(p::GaussianPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))
weights(p::MvGaussianPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))
weights(p::GammaPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))
weights(p::DirichletPrevision) = p.alpha ./ sum(p.alpha)
weights(p::NormalGammaPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))

function marginal(p::MixturePrevision, indices::Vector{Int})
    w = weights(p)
    marginal_logw = Float64[]
    for i in indices
        push!(marginal_logw, p.log_weights[i])
    end
    MixturePrevision([p.components[i] for i in indices], marginal_logw)
end

"""
    with_components(p::MixturePrevision, components) -> MixturePrevision

Re-key a mixture: keep `p`'s weights verbatim, replace its per-component beliefs.
The dual of `marginal` (which selects components+weights by index) — this preserves
the weights and swaps the components, one-for-one. The structure posterior is
carried in log-space unchanged (no normalisation assumption is touched). Used to
build a per-context decision view from a structure-BMA belief.
"""
function with_components(p::MixturePrevision, components)
    length(components) == length(p.components) ||
        error("with_components: $(length(components)) components for a $(length(p.components))-way mixture")
    MixturePrevision(collect(components), copy(p.log_weights))
end

# ── Typed decision stdlib: argmax / EVPI over a functional-per-action preference ──
#
# These are the typed-Functional encoding of the ONE decision mechanism
# (argmax_a expect(belief, φ_a)); the `.bdsl` `optimise`/`value`/`voi`/`net-voi`
# (src/stdlib.bdsl) are the lambda-pref surface of the same ops. This Julia form
# exists because a typed preference keeps EU closed-form (Identity / Projection /
# LinearCombination dispatch) instead of forcing the opaque-closure quadrature a
# `(lambda (h) (pref h a))` would — the Invariant-2 reason. The skin's
# `functional_per_action` handler (apps/skin/server.jl `handle_optimise`/
# `handle_value`) mirrors this loop over a JSON spec; this is the in-process
# canonical home for in-Julia callers (the credence-pi feature brain). `actions`
# fixes a deterministic iteration order and strict `>` makes the first action win
# a tie — no host-dependent tie-break.
#
# `fpa` maps each action to a `Functional`; `eu(belief, fpa[a]) = expect(...)`.
# Asserted exact against a hand oracle in test/test_typed_decision.jl.

eu(belief, φ) = Float64(expect(belief, φ))

function optimise(belief, actions, fpa::AbstractDict)
    best_a = nothing
    best_eu = -Inf
    for a in actions
        e = eu(belief, fpa[a])
        if e > best_eu
            best_eu = e
            best_a = a
        end
    end
    best_a === nothing && error("optimise: empty action set")
    best_a
end

function value(belief, actions, fpa::AbstractDict)
    best_eu = -Inf
    for a in actions
        best_eu = max(best_eu, eu(belief, fpa[a]))
    end
    best_eu === -Inf && error("value: empty action set")
    best_eu
end

# Marginal predictive P(obs), marginalised over the belief through the kernel.
predictive_prob(belief, k::Kernel, obs) = exp(_predictive_ll(belief, k, obs))

# voi: EVPI of one observation from `k` for the decision over `actions`/`fpa`.
#   voi = Σ_o P(o)·value(condition(belief,k,o)) − value(belief)
# with P(o) the marginal predictive, renormalised over `possible_obs`. Conditioning
# routes through the axiom-constrained `condition` (Invariant 1); no belief is
# modified outside it.
function voi(belief, k::Kernel, actions, fpa::AbstractDict, possible_obs)
    base = value(belief, actions, fpa)
    preds = [predictive_prob(belief, k, o) for o in possible_obs]
    total = sum(preds)
    total <= 0.0 && return 0.0
    posterior_val = 0.0
    for (i, o) in enumerate(possible_obs)
        posterior_val += (preds[i] / total) * value(condition(belief, k, o), actions, fpa)
    end
    posterior_val - base
end

# net-voi: VOI minus the cost of observing — the `ask`-gate EU.
net_voi(belief, k::Kernel, actions, fpa::AbstractDict, possible_obs, cost) =
    voi(belief, k, actions, fpa, possible_obs) - cost
