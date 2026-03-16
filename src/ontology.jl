"""
    ontology.jl — Three types. Axiom-constrained functions.

Types (frozen):
    Space   — a set of possibilities
    Measure — a probability distribution over a space
    Kernel  — a conditional distribution between two spaces

Axiom-constrained functions (behaviour frozen, interface negotiable):
    condition — Bayesian inversion
    expect    — integration against a measure
    push_     — composition of measure with kernel (push is reserved in Julia)
    density   — kernel density at a point
"""
module Ontology

export Space, Finite, Interval, ProductSpace, Simplex, Euclidean, PositiveReals, support
export Measure, CategoricalMeasure, BetaMeasure, GaussianMeasure, DirichletMeasure, ProductMeasure, MixtureMeasure
export Kernel, kernel_source, kernel_target, kernel_generate
export condition, expect, push_measure, density
export draw, optimise, value
export weights, mean, variance, log_density_at, prune, truncate

# ================================================================
# TYPE 1: Space
# ================================================================

abstract type Space end

struct Finite{T} <: Space
    values::Vector{T}
end

struct Interval <: Space
    lo::Float64
    hi::Float64
end

struct ProductSpace <: Space
    factors::Vector{Space}
end

struct Simplex <: Space
    k::Int  # Δ^(k-1): vectors of length k, non-negative, summing to 1
end

struct Euclidean <: Space
    dim::Int
end

struct PositiveReals <: Space end

support(s::Finite) = s.values

# ================================================================
# TYPE 2: Measure
# ================================================================

abstract type Measure end

# ── Categorical: finite discrete ──

struct CategoricalMeasure{T} <: Measure
    space::Finite{T}
    logw::Vector{Float64}

    function CategoricalMeasure{T}(space::Finite{T}, logw::Vector{Float64}) where T
        length(space.values) == length(logw) || error("space and weights must match")
        # Handle all -Inf (impossible)
        if all(lw -> lw == -Inf, logw)
            error("measure has zero total mass — all hypotheses impossible")
        end
        max_lw = maximum(logw)
        log_total = max_lw + log(sum(exp.(logw .- max_lw)))
        new{T}(space, logw .- log_total)
    end
end

CategoricalMeasure(s::Finite{T}, logw::Vector{Float64}) where T = CategoricalMeasure{T}(s, logw)
CategoricalMeasure(s::Finite{T}) where T = CategoricalMeasure{T}(s, fill(0.0, length(s.values)))

function weights(m::CategoricalMeasure)
    max_lw = maximum(m.logw)
    w = exp.(m.logw .- max_lw)
    w ./ sum(w)
end

# ── Beta: continuous on [0,1] ──

struct BetaMeasure <: Measure
    space::Interval
    alpha::Float64
    beta::Float64

    function BetaMeasure(space::Interval, alpha::Float64, beta::Float64)
        space.lo == 0.0 && space.hi == 1.0 || error("BetaMeasure requires [0,1]")
        alpha > 0 && beta > 0 || error("alpha and beta must be positive")
        new(space, alpha, beta)
    end
end

BetaMeasure(α::Float64, β::Float64) = BetaMeasure(Interval(0.0, 1.0), α, β)
BetaMeasure() = BetaMeasure(1.0, 1.0)

mean(m::BetaMeasure) = m.alpha / (m.alpha + m.beta)
variance(m::BetaMeasure) = m.alpha * m.beta / ((m.alpha + m.beta)^2 * (m.alpha + m.beta + 1))

# ── Gaussian: continuous on interval ──

struct GaussianMeasure <: Measure
    space::Euclidean
    mu::Float64
    sigma::Float64
end

mean(m::GaussianMeasure) = m.mu
variance(m::GaussianMeasure) = m.sigma^2

# ── Dirichlet: distribution over the probability simplex ──

struct DirichletMeasure <: Measure
    space::Simplex
    categories::Finite     # the category labels the probability vectors are about
    alpha::Vector{Float64} # concentration parameters, length k

    function DirichletMeasure(space::Simplex, categories::Finite, alpha::Vector{Float64})
        space.k == length(categories.values) == length(alpha) ||
            error("simplex dimension, categories, and alpha must all have the same length")
        all(a -> a > 0, alpha) || error("all alpha must be positive")
        new(space, categories, alpha)
    end
end

weights(m::DirichletMeasure) = m.alpha ./ sum(m.alpha)
mean(m::DirichletMeasure) = weights(m)

# ── Product: independent joint ──

struct ProductMeasure <: Measure
    space::ProductSpace
    factors::Vector{Measure}

    function ProductMeasure(space::ProductSpace, factors::Vector{<:Measure})
        length(space.factors) == length(factors) || error("space and factors must match")
        new(space, Vector{Measure}(factors))
    end
end

ProductMeasure(factors::Vector{<:Measure}) =
    ProductMeasure(ProductSpace(Space[f.space for f in factors]), factors)

# ── Mixture: weighted combination ──

struct MixtureMeasure <: Measure
    space::Space
    components::Vector{Measure}
    log_weights::Vector{Float64}

    function MixtureMeasure(space::Space, components::Vector{<:Measure}, log_weights::Vector{Float64})
        length(components) == length(log_weights) || error("components and weights must match")
        length(components) > 0 || error("mixture must have at least one component")
        max_lw = maximum(log_weights)
        log_total = max_lw + log(sum(exp.(log_weights .- max_lw)))
        new(space, Vector{Measure}(components), log_weights .- log_total)
    end
end

MixtureMeasure(components::Vector{<:Measure}, log_weights::Vector{Float64}) =
    MixtureMeasure(components[1].space, components, log_weights)

function weights(m::MixtureMeasure)
    max_lw = maximum(m.log_weights)
    w = exp.(m.log_weights .- max_lw)
    w ./ sum(w)
end

# ================================================================
# TYPE 3: Kernel
# ================================================================

struct Kernel
    source::Space       # H
    target::Space       # O
    generate::Function  # h → distribution spec (for push)
    log_density::Function  # (h, o) → log p(o|h) (for condition)
end

kernel_source(k::Kernel) = k.source
kernel_target(k::Kernel) = k.target

# ================================================================
# AXIOM-CONSTRAINED FUNCTION: density
# ================================================================
# The kernel's log-density at a point.
# density(kernel, h, o) = log P(o | h)

density(k::Kernel, h, o) = k.log_density(h, o)

# ================================================================
# AXIOM-CONSTRAINED FUNCTION: expect
# ================================================================
# Integration against a measure: E_m[f]
# This is what a measure IS.

function expect(m::CategoricalMeasure, f)
    w = weights(m)
    sum(w[i] * f(m.space.values[i]) for i in eachindex(w))
end

function expect(m::BetaMeasure, f; n::Int=64)
    grid = range(1/(2n), 1-1/(2n), length=n)
    logw = [log_density_at(m, x) for x in grid]
    max_lw = maximum(logw)
    w = exp.(logw .- max_lw)
    w ./= sum(w)
    sum(w[i] * f(grid[i]) for i in eachindex(w))
end

function expect(m::GaussianMeasure, f; n::Int=64)
    lo = m.mu - 4 * m.sigma
    hi = m.mu + 4 * m.sigma
    grid = range(lo, hi, length=n)
    logw = [-0.5 * ((x - m.mu) / m.sigma)^2 for x in grid]
    max_lw = maximum(logw)
    w = exp.(logw .- max_lw)
    w ./= sum(w)
    sum(w[i] * f(grid[i]) for i in eachindex(w))
end

function expect(m::DirichletMeasure, f; n_samples::Int=1000)
    total = 0.0
    for _ in 1:n_samples
        θ = draw(m)
        total += f(θ)
    end
    total / n_samples
end

function expect(m::ProductMeasure, f; n_samples::Int=1000)
    total = 0.0
    for _ in 1:n_samples
        total += f(draw(m))
    end
    total / n_samples
end

function expect(m::MixtureMeasure, f)
    w = weights(m)
    sum(w[i] * expect(m.components[i], f) for i in eachindex(w))
end

# ================================================================
# AXIOM-CONSTRAINED FUNCTION: condition
# ================================================================
# Bayesian inversion: prior × kernel × observation → posterior
# P(h|o) ∝ P(o|h) · P(h)
# The ONE learning mechanism.

function condition(m::CategoricalMeasure{T}, k::Kernel, observation) where T
    new_logw = copy(m.logw)
    for i in eachindex(m.space.values)
        ll = density(k, m.space.values[i], observation)
        !isnan(ll) || error("density returned NaN for hypothesis $i")
        new_logw[i] += ll
    end
    CategoricalMeasure{T}(m.space, new_logw)
end

function condition(m::BetaMeasure, k::Kernel, observation)
    # Conjugate path: if target is binary and density is Bernoulli-shaped
    if observation == 1 || observation == 1.0 || observation == true
        BetaMeasure(m.space, m.alpha + 1.0, m.beta)
    elseif observation == 0 || observation == 0.0 || observation == false
        BetaMeasure(m.space, m.alpha, m.beta + 1.0)
    else
        # Non-conjugate: fall back to grid
        _condition_by_grid(m, k, observation)
    end
end

function condition(m::GaussianMeasure, k::Kernel, observation)
    _condition_by_grid(m, k, observation)
end

function condition(m::DirichletMeasure, k::Kernel, observation)
    k.source isa Simplex && k.target isa Finite ||
        error("DirichletMeasure condition requires a Categorical kernel (Simplex → Finite)")

    idx = findfirst(==(observation), m.categories.values)
    idx !== nothing || error("observation $observation not in categories $(m.categories.values)")

    new_alpha = copy(m.alpha)
    new_alpha[idx] += 1.0
    DirichletMeasure(m.space, m.categories, new_alpha)
end

function _condition_by_grid(m::BetaMeasure, k::Kernel, observation; n::Int=64)
    grid = collect(range(m.space.lo + 1e-10, m.space.hi - 1e-10, length=n))
    logw = Float64[]
    for (i, h) in enumerate(grid)
        lp = log_density_at(m, h)
        ll = density(k, h, observation)
        !isnan(ll) || error("density returned NaN for hypothesis $i")
        push!(logw, lp + ll)
    end
    CategoricalMeasure(Finite(grid), logw)
end

function _condition_by_grid(m::GaussianMeasure, k::Kernel, observation; n::Int=64)
    lo = m.mu - 4 * m.sigma
    hi = m.mu + 4 * m.sigma
    grid = collect(range(lo, hi, length=n))
    logw = Float64[]
    for (i, h) in enumerate(grid)
        lp = log_density_at(m, h)
        ll = density(k, h, observation)
        !isnan(ll) || error("density returned NaN for hypothesis $i")
        push!(logw, lp + ll)
    end
    CategoricalMeasure(Finite(grid), logw)
end

log_density_at(m::BetaMeasure, x) = (m.alpha - 1) * log(x) + (m.beta - 1) * log(1 - x)
log_density_at(m::GaussianMeasure, x) = -0.5 * ((x - m.mu) / m.sigma)^2
log_density_at(m::DirichletMeasure, x) = sum((m.alpha[i] - 1) * log(x[i]) for i in eachindex(m.alpha))
log_density_at(m::ProductMeasure, x) =
    sum(log_density_at(m.factors[i], x[i]) for i in eachindex(m.factors))

function log_density_at(m::MixtureMeasure, x)
    terms = [m.log_weights[i] + log_density_at(m.components[i], x) for i in eachindex(m.components)]
    max_t = maximum(terms)
    max_t + log(sum(exp(t - max_t) for t in terms))
end

# ── MixtureMeasure condition: condition each component (theorem) ──

function _predictive_ll(m::CategoricalMeasure, k::Kernel, obs)
    w = weights(m)
    total = sum(w[i] * exp(k.log_density(m.space.values[i], obs)) for i in eachindex(w))
    log(max(total, 1e-300))
end

function _predictive_ll(m::BetaMeasure, k::Kernel, obs)
    val = expect(m, h -> exp(k.log_density(h, obs)))
    log(max(val, 1e-300))
end

function _predictive_ll(m::Measure, k::Kernel, obs; n_samples::Int=200)
    total = 0.0
    for _ in 1:n_samples
        total += exp(k.log_density(draw(m), obs))
    end
    log(max(total / n_samples, 1e-300))
end

function condition(m::MixtureMeasure, k::Kernel, obs)
    new_components = Measure[]
    new_log_wts = Float64[]
    for (i, comp) in enumerate(m.components)
        pred_ll = _predictive_ll(comp, k, obs)
        conditioned = condition(comp, k, obs)
        base_lw = m.log_weights[i] + pred_ll
        if conditioned isa MixtureMeasure
            for (j, sub) in enumerate(conditioned.components)
                push!(new_components, sub)
                push!(new_log_wts, base_lw + conditioned.log_weights[j])
            end
        else
            push!(new_components, conditioned)
            push!(new_log_wts, base_lw)
        end
    end
    MixtureMeasure(m.space, new_components, new_log_wts)
end

# ── General condition fallback: importance sampling ──

function condition(m::Measure, k::Kernel, obs; n_particles::Int=1000)
    samples = [draw(m) for _ in 1:n_particles]
    log_weights = Float64[k.log_density(s, obs) for s in samples]
    CategoricalMeasure(Finite(samples), log_weights)
end

# ================================================================
# AXIOM-CONSTRAINED FUNCTION: push_measure
# ================================================================
# Pushforward: Measure(H) × Kernel(H,T) → Measure(T)
# The induced distribution on the target space.
# expect is push to ℝ.

function push_measure(m::CategoricalMeasure, k::Kernel)
    # For each hypothesis, the kernel generates a distribution on T.
    # The pushforward is the mixture: Σ_i w_i · kernel(h_i)
    # For finite target: compute weight of each target value
    tgt = k.target
    if tgt isa Finite
        w = weights(m)
        tgt_logw = fill(-Inf, length(tgt.values))
        for (i, h) in enumerate(m.space.values)
            for (j, o) in enumerate(tgt.values)
                d = density(k, h, o)
                if d > -Inf
                    contrib = log(w[i]) + d
                    if tgt_logw[j] == -Inf
                        tgt_logw[j] = contrib
                    else
                        # log-sum-exp
                        mx = max(tgt_logw[j], contrib)
                        tgt_logw[j] = mx + log(exp(tgt_logw[j] - mx) + exp(contrib - mx))
                    end
                end
            end
        end
        return CategoricalMeasure(tgt, tgt_logw)
    else
        error("push_measure to non-finite target spaces not yet implemented")
    end
end

function push_measure(m::DirichletMeasure, k::Kernel)
    # Any kernel Simplex → Finite is a categorical draw: each simplex
    # point IS a categorical distribution. The pushforward is the
    # posterior predictive: P(o_j) = E_Dir[θ_j] = alpha_j / sum(alpha).
    k.source isa Simplex || error("push_measure from DirichletMeasure requires Simplex source kernel")
    tgt = k.target
    tgt isa Finite || error("push_measure to non-Finite target not implemented for DirichletMeasure")
    w = weights(m)
    logw = [log(max(wi, 1e-300)) for wi in w]
    CategoricalMeasure(tgt, logw)
end

# ================================================================
# HOST OPERATION: draw
# ================================================================
# Crosses the boundary from mathematical object to actual value.
# This is the ONLY source of randomness in the system.
# Exported from the ontology module for Julia host callers.
# NOT added to the DSL's default_env — the DSL is pure.

function draw(m::CategoricalMeasure)
    w = weights(m)
    r = rand()
    cumw = 0.0
    for i in eachindex(w)
        cumw += w[i]
        r < cumw && return m.space.values[i]
    end
    m.space.values[end]
end

function draw(m::BetaMeasure)
    # Beta via Gamma ratio: X/(X+Y) where X ~ Gamma(α), Y ~ Gamma(β)
    x = _draw_gamma(m.alpha)
    y = _draw_gamma(m.beta)
    x / (x + y)
end

function draw(m::GaussianMeasure)
    m.mu + m.sigma * randn()
end

# ── Gamma sampling (Marsaglia-Tsang) for Dirichlet draw ──

function _draw_gamma(alpha::Float64)
    if alpha == 1.0
        return -log(rand())  # Gamma(1,1) = Exponential(1)
    elseif alpha < 1.0
        # Ahrens-Dieter: Gamma(α) = Gamma(α+1) · U^(1/α)
        return _draw_gamma(alpha + 1.0) * rand()^(1.0 / alpha)
    else
        # Marsaglia-Tsang for α > 1
        d = alpha - 1.0/3.0
        c = 1.0 / sqrt(9.0 * d)
        while true
            x = randn()
            v = (1.0 + c * x)^3
            v > 0 || continue
            u = rand()
            if u < 1.0 - 0.0331 * x^4 || log(u) < 0.5 * x^2 + d * (1.0 - v + log(v))
                return d * v
            end
        end
    end
end

function draw(m::DirichletMeasure)
    g = [_draw_gamma(a) for a in m.alpha]
    g ./= sum(g)
    g
end

draw(m::ProductMeasure) = Any[draw(f) for f in m.factors]

function draw(m::MixtureMeasure)
    w = weights(m)
    r = rand()
    cumw = 0.0
    for i in eachindex(w)
        cumw += w[i]
        r < cumw && return draw(m.components[i])
    end
    draw(m.components[end])
end

# ── Mixture maintenance ──

function prune(m::MixtureMeasure; threshold::Float64=-20.0)
    max_lw = maximum(m.log_weights)
    keep = [i for i in eachindex(m.log_weights) if m.log_weights[i] - max_lw > threshold]
    length(keep) == length(m.components) && return m
    MixtureMeasure(m.space, m.components[keep], m.log_weights[keep])
end

function truncate(m::MixtureMeasure; max_components::Int=10)
    length(m.components) <= max_components && return m
    perm = sortperm(m.log_weights, rev=true)
    keep = perm[1:min(max_components, length(perm))]
    MixtureMeasure(m.space, m.components[keep], m.log_weights[keep])
end

# ================================================================
# HOST CONVENIENCE: optimise, value
# ================================================================
# EU maximisation for Julia callers. Uses expect (the ONE
# integration implementation). The DSL's stdlib optimise is
# written in the DSL; this is the Julia-level equivalent.
# Both use expect — they cannot diverge on integration.

function optimise(m::Measure, actions::Finite, pref)
    best_a = first(actions.values)
    best_eu = expect(m, h -> pref(h, best_a))
    for a in Iterators.drop(actions.values, 1)
        eu = expect(m, h -> pref(h, a))
        if eu > best_eu; best_eu = eu; best_a = a; end
    end
    best_a
end

function value(m::Measure, actions::Finite, pref)
    best_eu = expect(m, h -> pref(h, first(actions.values)))
    for a in Iterators.drop(actions.values, 1)
        eu = expect(m, h -> pref(h, a))
        if eu > best_eu; best_eu = eu; end
    end
    best_eu
end

end # module Ontology
