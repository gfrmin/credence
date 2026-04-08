"""
    ConditionalProbabilityDistribution (CPD)

Represents P(V | parents) for a discrete variable V.
Thin wrapper around credence's DirichletMeasure, preserving the mutable
update!/predict/rand interface for compatibility with bayesian-julia's
agent loop while delegating all inference to credence.

Mathematical foundations:
- Prior: θ ~ Dirichlet(α)
- Likelihood: observations ~ Categorical(θ)
- Posterior: θ ~ Dirichlet(α + counts)
- Predictive: P(V=v | data) = (α_v + count_v) / (Σα + n)
"""

"""
    DirichletCategorical

Mutable wrapper around credence's immutable DirichletMeasure.
The measure field holds the current posterior (prior alpha merged with counts).
The prior_alpha field stores the original prior for loglikelihood reconstruction.

Fields:
- domain::Vector              # Possible values for this variable
- measure::DirichletMeasure   # Credence type: posterior over simplex
- kernel::Kernel              # Categorical kernel (Simplex → Finite) built once
- prior_alpha::Vector{Float64} # Original prior for loglikelihood computation
"""
mutable struct DirichletCategorical
    domain::Vector              # Possible values: [v₁, v₂, ..., vₖ]
    measure::DirichletMeasure   # Credence: holds posterior alpha (prior + counts merged)
    kernel::Kernel              # Categorical kernel for condition/draw
    prior_alpha::Vector{Float64} # Original prior alpha for loglikelihood

    function DirichletCategorical(domain::Vector, alpha::Vector{Float64})
        @assert length(domain) == length(alpha) "domain and alpha must have same length"
        @assert all(alpha .> 0) "alpha must be positive (Dirichlet support)"

        k = length(domain)
        cats = Finite(domain)
        measure = DirichletMeasure(Simplex(k), cats, copy(alpha))
        kernel = Kernel(Simplex(k), cats,
            θ -> (o -> begin; idx = findfirst(==(o), cats.values); log(θ[idx]); end),
            (θ, o) -> begin; idx = findfirst(==(o), cats.values); log(θ[idx]); end)
        new(domain, measure, kernel, copy(alpha))
    end
end

"""
    DirichletCategorical(domain, alpha_scalar)

Constructor: uniform Dirichlet prior with concentration alpha_scalar.
"""
function DirichletCategorical(domain::Vector, alpha_scalar::Float64)
    alpha = fill(alpha_scalar, length(domain))
    DirichletCategorical(domain, alpha)
end

# Backward-compatible property access: cpd.alpha and cpd.counts
# Tests access these fields directly
function Base.getproperty(cpd::DirichletCategorical, name::Symbol)
    if name === :alpha
        return cpd.prior_alpha
    elseif name === :counts
        return round.(Int, getfield(cpd, :measure).alpha .- getfield(cpd, :prior_alpha))
    else
        return getfield(cpd, name)
    end
end

function Base.setproperty!(cpd::DirichletCategorical, name::Symbol, value)
    if name === :measure || name === :prior_alpha
        setfield!(cpd, name, value)
    elseif name === :counts
        # Reconstruct measure from prior_alpha + new counts
        new_alpha = getfield(cpd, :prior_alpha) .+ Float64.(value)
        domain = getfield(cpd, :domain)
        setfield!(cpd, :measure, DirichletMeasure(Simplex(length(domain)), Finite(domain), new_alpha))
    else
        setfield!(cpd, name, value)
    end
end

"""
    update!(cpd::DirichletCategorical, observation)

Update posterior with an observation.
Delegates to credence's condition().
"""
function update!(cpd::DirichletCategorical, observation)
    idx = findfirst(v -> v == observation, cpd.domain)
    if isnothing(idx)
        @warn "Observation $observation not in domain $(cpd.domain)"
        return
    end
    cpd.measure = condition(cpd.measure, cpd.kernel, observation)
end

"""
    predict(cpd::DirichletCategorical) → Vector{Float64}

Compute posterior predictive distribution P(V=v | data).
Returns normalized probabilities for each value in domain.

Formula: P(V=vᵢ | data) = αᵢ / Σα (where α includes prior + counts)
"""
function predict(cpd::DirichletCategorical)::Vector{Float64}
    return weights(cpd.measure)
end

"""
    rand(cpd::DirichletCategorical) → value

Sample from posterior predictive distribution.
Uses credence's push_measure for posterior predictive sampling:
draws directly from the predictive (one draw, weights α/Σα).
"""
function Random.rand(cpd::DirichletCategorical)
    # Posterior predictive: single draw from categorical with weights α/Σα
    pred = push_measure(cpd.measure, cpd.kernel)
    return draw(pred)
end

"""
    loglikelihood(cpd::DirichletCategorical) → Float64

Compute log marginal likelihood log P(data | prior).
Uses credence's log_predictive in a sequential loop.

Since Dirichlet-Categorical is exchangeable, observation order doesn't matter —
the marginal likelihood is invariant to ordering. We iterate over counts
synthetically (any ordering gives the same result).
"""
function loglikelihood(cpd::DirichletCategorical)::Float64
    counts = _counts(cpd)
    count_sum = sum(counts)
    if count_sum == 0
        return 0.0
    end

    # Reconstruct prior measure and iterate log_predictive + condition
    log_ml = 0.0
    current = DirichletMeasure(Simplex(length(cpd.domain)), Finite(cpd.domain), copy(cpd.prior_alpha))
    k = cpd.kernel

    for i in eachindex(cpd.domain)
        for _ in 1:counts[i]
            obs = cpd.domain[i]
            log_ml += log_predictive(current, k, obs)
            current = condition(current, k, obs)
        end
    end
    return log_ml
end

"""
    entropy(cpd::DirichletCategorical) → Float64

Compute Shannon entropy of posterior predictive distribution.
H[V | data] = -Σᵢ P(V=vᵢ) log P(V=vᵢ)
"""
function entropy(cpd::DirichletCategorical)::Float64
    probs = weights(cpd.measure)
    return -sum(probs .* log.(probs .+ 1e-10))
end

"""
    mode(cpd::DirichletCategorical) → value

Return the mode (maximum a posteriori estimate) of posterior.
"""
function mode(cpd::DirichletCategorical)
    max_idx = argmax(cpd.measure.alpha)
    return cpd.domain[max_idx]
end

"""
    copy(cpd::DirichletCategorical) → DirichletCategorical

Create a deep copy of the CPD.
"""
function Base.copy(cpd::DirichletCategorical)
    new_cpd = DirichletCategorical(copy(cpd.domain), copy(cpd.prior_alpha))
    # Set the measure to match current posterior
    new_cpd.measure = DirichletMeasure(Simplex(length(cpd.domain)), Finite(copy(cpd.domain)), copy(cpd.measure.alpha))
    return new_cpd
end

"""
    reset!(cpd::DirichletCategorical)

Reset to prior (clear all observations).
"""
function reset!(cpd::DirichletCategorical)
    cpd.measure = DirichletMeasure(Simplex(length(cpd.domain)), Finite(cpd.domain), copy(cpd.prior_alpha))
end

export DirichletCategorical, update!, predict, entropy, mode
