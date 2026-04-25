# conjugate.jl — Conjugate registry (Move 4)
#
# `maybe_conjugate(p::Prevision, k::Kernel)` dispatches on the prior type
# and returns a `ConjugatePrevision` when a closed-form update exists.
# `update(cp::ConjugatePrevision{Prior, Likelihood}, obs)` applies that
# update.
#
# Included inside module Ontology by ontology.jl.

# ── Pair: (BetaPrevision, BetaBernoulli) ──

function maybe_conjugate(p::BetaPrevision, k::Kernel)
    if k.likelihood_family isa BetaBernoulli
        return ConjugatePrevision(p, k.likelihood_family)
    elseif k.likelihood_family isa Flat
        return ConjugatePrevision(p, k.likelihood_family)
    end
    nothing
end

function update(cp::ConjugatePrevision{BetaPrevision, BetaBernoulli}, obs)
    if obs == 1 || obs == 1.0 || obs === true
        ConjugatePrevision(BetaPrevision(cp.prior.alpha + 1.0, cp.prior.beta), cp.likelihood)
    elseif obs == 0 || obs == 0.0 || obs === false
        ConjugatePrevision(BetaPrevision(cp.prior.alpha, cp.prior.beta + 1.0), cp.likelihood)
    else
        error("BetaBernoulli update: obs must be ∈ {0, 1, true, false}, got $obs")
    end
end

function update(cp::ConjugatePrevision{BetaPrevision, Flat}, obs)
    cp  # no-op
end

# ── Pair: (GaussianPrevision, NormalNormal) ──

function maybe_conjugate(p::GaussianPrevision, k::Kernel)
    if k.likelihood_family isa NormalNormal
        return ConjugatePrevision(p, k.likelihood_family)
    elseif k.params !== nothing && haskey(k.params, :sigma_obs)
        sigma_obs = k.params[:sigma_obs]::Float64
        return ConjugatePrevision(p, NormalNormal(sigma_obs))
    end
    nothing
end

function update(cp::ConjugatePrevision{GaussianPrevision, NormalNormal}, obs)
    sigma_obs = cp.likelihood.sigma_obs
    τ_prior = 1.0 / cp.prior.sigma^2
    τ_obs = 1.0 / sigma_obs^2
    τ_post = τ_prior + τ_obs
    μ_post = (τ_prior * cp.prior.mu + τ_obs * Float64(obs)) / τ_post
    σ_post = 1.0 / sqrt(τ_post)
    ConjugatePrevision(GaussianPrevision(μ_post, σ_post), cp.likelihood)
end

# ── Pair: (DirichletPrevision, Categorical) ──

function maybe_conjugate(p::DirichletPrevision, k::Kernel)
    if k.likelihood_family isa Categorical
        return ConjugatePrevision(p, k.likelihood_family)
    end
    nothing
end

function update(cp::ConjugatePrevision{DirichletPrevision, Categorical{T}}, obs) where T
    idx = findfirst(==(obs), cp.likelihood.categories.values)
    idx !== nothing || error("observation $obs not in categories $(cp.likelihood.categories.values)")
    new_alpha = copy(cp.prior.alpha)
    new_alpha[idx] += 1.0
    ConjugatePrevision(DirichletPrevision(new_alpha), cp.likelihood)
end

# ── Pair: (NormalGammaPrevision, NormalGammaLikelihood) ──

function maybe_conjugate(p::NormalGammaPrevision, k::Kernel)
    if k.likelihood_family isa NormalGammaLikelihood
        return ConjugatePrevision(p, k.likelihood_family)
    elseif k.params !== nothing && haskey(k.params, :normal_gamma)
        return ConjugatePrevision(p, NormalGammaLikelihood())
    end
    nothing
end

function update(cp::ConjugatePrevision{NormalGammaPrevision, NormalGammaLikelihood}, obs)
    r = Float64(obs)
    κ, μ, α, β = cp.prior.κ, cp.prior.μ, cp.prior.α, cp.prior.β
    κₙ = κ + 1.0
    μₙ = (κ * μ + r) / κₙ
    αₙ = α + 0.5
    βₙ = β + κ * (r - μ)^2 / (2.0 * κₙ)
    ConjugatePrevision(NormalGammaPrevision(κₙ, μₙ, αₙ, βₙ), cp.likelihood)
end

# ── Pair: (GammaPrevision, Exponential) ──

function maybe_conjugate(p::GammaPrevision, k::Kernel)
    if k.likelihood_family isa Exponential
        return ConjugatePrevision(p, k.likelihood_family)
    end
    nothing
end

function update(cp::ConjugatePrevision{GammaPrevision, Exponential}, obs)
    r = Float64(obs)
    r > 0 || error("Exponential observations must be positive, got $r")
    ConjugatePrevision(GammaPrevision(cp.prior.alpha + 1.0, cp.prior.beta + r), cp.likelihood)
end
