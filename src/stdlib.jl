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

weights(p::EnumerationPrevision) = begin
    lw = p.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./ sum(w)
end

weights(p::BetaPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))
weights(p::TaggedBetaPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))
weights(p::GaussianPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))
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
