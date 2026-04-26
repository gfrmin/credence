"""
    ontology.jl — Module shell + Measure facades + axiom-constrained functions.

After the Move 5 split, concept-files hold their own concerns:
    spaces.jl    — Space types + BOOLEAN_SPACE
    kernels.jl   — Kernel, LikelihoodFamily hierarchy, density
    events.jl    — Event hierarchy + indicator_kernel witnesses
    conjugate.jl — maybe_conjugate / update registry
    stdlib.jl    — mean, variance, probability, weights, marginal

What remains here: the module namespace (`module Ontology`), imports,
exports, FrozenVectorView, Measure facades (nine types preserving the
consumer API as views over Prevision), logsumexp, wrap_in_measure,
expect methods (Measure-level forwarding + Prevision-primary dispatch),
condition, push_measure, draw, prune, truncate, log_marginal.
"""
module Ontology

# ── Move 2: unify Functional hierarchy onto Previsions.TestFunction ──
import ..Previsions: Prevision
import ..Previsions: TestFunction, Identity, Projection, NestedProjection,
                     Tabular, LinearCombination, OpaqueClosure, expect
import ..Previsions: BetaPrevision, TaggedBetaPrevision, GaussianPrevision, GammaPrevision, CategoricalPrevision, DirichletPrevision, NormalGammaPrevision, ProductPrevision, MixturePrevision
import ..Previsions: ExchangeablePrevision, decompose
import ..Previsions: ParticlePrevision, QuadraturePrevision
import ..Previsions: ConditionalPrevision
import ..Previsions: condition
import ..Previsions: ConjugatePrevision, maybe_conjugate, update, _dispatch_path
import ..Previsions: CenteredPower, CenteredSquare
import ..Previsions: Indicator, apply

export Space, Finite, Interval, ProductSpace, Simplex, Euclidean, PositiveReals, support
export Measure, CategoricalMeasure, BetaMeasure, TaggedBetaMeasure, GaussianMeasure, GammaMeasure, ExponentialMeasure, DirichletMeasure, NormalGammaMeasure, EnumerationMeasure, ProductMeasure, MixtureMeasure
export Kernel, FactorSelector, kernel_source, kernel_target, kernel_params
export LikelihoodFamily, LeafFamily, PushOnly, BetaBernoulli, Flat, FiringByTag, DispatchByComponent, DepthCapExceeded
export NormalNormal, Categorical, NormalGammaLikelihood, Exponential
export Event, TagSet, FeatureEquals, FeatureInterval, Conjunction, Disjunction, Complement
export indicator_kernel, feature_value, BOOLEAN_SPACE
export Functional, Identity, Projection, NestedProjection, Tabular, LinearCombination, OpaqueClosure
export factor, replace_factor
export condition, expect, push_measure, density, log_predictive, log_marginal, wrap_in_measure
export ConjugatePrevision, maybe_conjugate, update
export draw
export weights, mean, variance, log_density_at, prune, truncate, logsumexp
export FrozenVectorView
export WeightsDomainError, probability, marginal, CenteredPower, CenteredSquare

# ================================================================
# FrozenVectorView{T}
# ================================================================

"""
    FrozenVectorView{T}(inner::Vector{T})

Read-only wrapper around `inner`. Delegates the read surface
(`getindex`, `length`, `iterate`, `eachindex`, `firstindex`,
`lastindex`, `size`, `axes`, `collect`, `isempty`) to `inner`. All
mutation operations (`push!`, `setindex!`, `append!`, `pop!`,
`resize!`, `empty!`, `deleteat!`, `insert!`) throw a Move-2-attributed
error pointing at the Prevision-level mutation APIs.

Used by the Measure-level `getproperty` shields on `MixtureMeasure`,
`ProductMeasure`, `TaggedBetaMeasure` from Move 2 Phase 4 onward, to
make shield-returned vectors self-describing as "read-only, fresh per
access; use push_component! / replace_component! for mutation."
"""
struct FrozenVectorView{T}
    inner::Vector{T}
end

Base.getindex(v::FrozenVectorView, i::Int) = v.inner[i]
Base.getindex(v::FrozenVectorView, idx) = v.inner[idx]
Base.length(v::FrozenVectorView) = length(v.inner)
Base.size(v::FrozenVectorView) = size(v.inner)
Base.axes(v::FrozenVectorView) = axes(v.inner)
Base.iterate(v::FrozenVectorView) = iterate(v.inner)
Base.iterate(v::FrozenVectorView, state) = iterate(v.inner, state)
Base.eachindex(v::FrozenVectorView) = eachindex(v.inner)
Base.firstindex(v::FrozenVectorView) = firstindex(v.inner)
Base.lastindex(v::FrozenVectorView) = lastindex(v.inner)
Base.eltype(::Type{FrozenVectorView{T}}) where T = T
Base.isempty(v::FrozenVectorView) = isempty(v.inner)
Base.collect(v::FrozenVectorView) = copy(v.inner)
Base.show(io::IO, v::FrozenVectorView) = print(io, "FrozenVectorView(", v.inner, ")")

const _FROZEN_ERR = "FrozenVectorView is read-only (shield-reconstructed fresh per access). " *
    "Use push_component!(::MixturePrevision, ...) or replace_component!(::MixturePrevision, ...) " *
    "at the Prevision level. See docs/posture-4/move-2-design.md §5.1."

Base.push!(v::FrozenVectorView, ::Any...) = error(_FROZEN_ERR)
Base.setindex!(v::FrozenVectorView, ::Any, ::Any...) = error(_FROZEN_ERR)
Base.append!(v::FrozenVectorView, ::Any...) = error(_FROZEN_ERR)
Base.pop!(v::FrozenVectorView, ::Any...) = error(_FROZEN_ERR)
Base.resize!(v::FrozenVectorView, ::Any...) = error(_FROZEN_ERR)
Base.empty!(v::FrozenVectorView) = error(_FROZEN_ERR)
Base.deleteat!(v::FrozenVectorView, ::Any...) = error(_FROZEN_ERR)
Base.insert!(v::FrozenVectorView, ::Any...) = error(_FROZEN_ERR)

# ================================================================
# TYPE 1: Space (extracted to spaces.jl)
# ================================================================
include("spaces.jl")

# ================================================================
# TYPE 3: Kernel (extracted to kernels.jl)
# ================================================================
include("kernels.jl")

# ================================================================
# TYPE 2: Measure — compatibility facades wrapping Prevision
# ================================================================

abstract type Measure end

# ── Categorical: finite discrete ──

struct CategoricalMeasure{T} <: Measure
    prevision::Prevision
    space::Finite{T}

    function CategoricalMeasure{T}(space::Finite{T}, log_weights::Vector{Float64}) where T
        length(space.values) == length(log_weights) || error("space and weights must match")
        new{T}(CategoricalPrevision(log_weights), space)
    end

    function CategoricalMeasure{T}(space::Finite{T}, p::Prevision) where T
        new{T}(p, space)
    end
end

CategoricalMeasure(s::Finite{T}, log_weights::Vector{Float64}) where T = CategoricalMeasure{T}(s, log_weights)
CategoricalMeasure(s::Finite{T}) where T = CategoricalMeasure{T}(s, fill(0.0, length(s.values)))
CategoricalMeasure(s::Finite{T}, p::Prevision) where T = CategoricalMeasure{T}(s, p)

function Base.getproperty(m::CategoricalMeasure, s::Symbol)
    if s === :logw
        return getfield(m, :prevision).log_weights
    else
        return getfield(m, s)
    end
end

Base.propertynames(::CategoricalMeasure) = (:logw, :space, :prevision)

function weights(m::CategoricalMeasure)
    max_lw = maximum(m.logw)
    w = exp.(m.logw .- max_lw)
    w ./ sum(w)
end

# ── Beta: continuous on [0,1] ──

struct BetaMeasure <: Measure
    prevision::BetaPrevision
    space::Interval

    function BetaMeasure(space::Interval, alpha::Float64, beta::Float64)
        space.lo == 0.0 && space.hi == 1.0 || error("BetaMeasure requires [0,1]")
        new(BetaPrevision(alpha, beta), space)
    end
end

BetaMeasure(α::Float64, β::Float64) = BetaMeasure(Interval(0.0, 1.0), α, β)
BetaMeasure() = BetaMeasure(1.0, 1.0)

function Base.getproperty(m::BetaMeasure, s::Symbol)
    if s === :alpha
        return getfield(m, :prevision).alpha
    elseif s === :beta
        return getfield(m, :prevision).beta
    else
        return getfield(m, s)
    end
end

Base.propertynames(::BetaMeasure) = (:alpha, :beta, :space, :prevision)

# ── TaggedBeta: program-indexed Beta for per-component kernel dispatch ──

struct TaggedBetaMeasure <: Measure
    prevision::TaggedBetaPrevision
    space::Interval

    function TaggedBetaMeasure(space::Interval, tag::Int, beta::BetaMeasure)
        new(TaggedBetaPrevision(tag, getfield(beta, :prevision)), space)
    end
    function TaggedBetaMeasure(space::Interval, tag::Int, beta::BetaPrevision)
        new(TaggedBetaPrevision(tag, beta), space)
    end
end

function Base.getproperty(m::TaggedBetaMeasure, s::Symbol)
    if s === :tag
        return getfield(m, :prevision).tag
    elseif s === :beta
        return wrap_in_measure(getfield(m, :prevision).beta)
    else
        return getfield(m, s)
    end
end

Base.propertynames(::TaggedBetaMeasure) = (:tag, :beta, :space, :prevision)

mean(m::BetaMeasure) = m.alpha / (m.alpha + m.beta)
variance(m::BetaMeasure) = m.alpha * m.beta / ((m.alpha + m.beta)^2 * (m.alpha + m.beta + 1))
mean(m::TaggedBetaMeasure) = mean(m.beta)
variance(m::TaggedBetaMeasure) = variance(m.beta)

(::Type{TaggedBetaPrevision})(tag::Int, beta::BetaMeasure) =
    TaggedBetaPrevision(tag, getfield(beta, :prevision))

# ── Gaussian: continuous on interval ──

struct GaussianMeasure <: Measure
    prevision::GaussianPrevision
    space::Euclidean

    function GaussianMeasure(space::Euclidean, mu::Float64, sigma::Float64)
        new(GaussianPrevision(mu, sigma), space)
    end
end

function Base.getproperty(m::GaussianMeasure, s::Symbol)
    if s === :mu || s === :sigma
        return getproperty(getfield(m, :prevision), s)
    else
        return getfield(m, s)
    end
end

Base.propertynames(::GaussianMeasure) = (:mu, :sigma, :space, :prevision)

mean(m::GaussianMeasure) = m.mu
variance(m::GaussianMeasure) = m.sigma^2

# ── Dirichlet: distribution over the probability simplex ──

struct DirichletMeasure <: Measure
    prevision::DirichletPrevision
    space::Simplex
    categories::Finite

    function DirichletMeasure(space::Simplex, categories::Finite, alpha::Vector{Float64})
        space.k == length(categories.values) == length(alpha) ||
            error("simplex dimension, categories, and alpha must all have the same length")
        new(DirichletPrevision(alpha), space, categories)
    end
end

function Base.getproperty(m::DirichletMeasure, s::Symbol)
    if s === :alpha
        return getproperty(getfield(m, :prevision), s)
    else
        return getfield(m, s)
    end
end

Base.propertynames(::DirichletMeasure) = (:alpha, :space, :categories, :prevision)

weights(m::DirichletMeasure) = m.alpha ./ sum(m.alpha)
mean(m::DirichletMeasure) = weights(m)

# ── Gamma: continuous on PositiveReals ──

struct GammaMeasure <: Measure
    prevision::GammaPrevision
    space::PositiveReals

    function GammaMeasure(space::PositiveReals, alpha::Float64, beta::Float64)
        new(GammaPrevision(alpha, beta), space)
    end
end

GammaMeasure(α::Float64, β::Float64) = GammaMeasure(PositiveReals(), α, β)

function Base.getproperty(m::GammaMeasure, s::Symbol)
    if s === :alpha || s === :beta
        return getproperty(getfield(m, :prevision), s)
    else
        return getfield(m, s)
    end
end

Base.propertynames(::GammaMeasure) = (:alpha, :beta, :space, :prevision)

mean(m::GammaMeasure) = m.alpha / m.beta

ExponentialMeasure(rate::Float64) = GammaMeasure(1.0, rate)

# ── Normal-Gamma: conjugate prior for Normal with unknown mean and variance ──

struct NormalGammaMeasure <: Measure
    prevision::NormalGammaPrevision
    space::ProductSpace

    function NormalGammaMeasure(space::ProductSpace, κ::Float64, μ::Float64, α::Float64, β::Float64)
        new(NormalGammaPrevision(κ, μ, α, β), space)
    end
end

NormalGammaMeasure(κ::Float64, μ::Float64, α::Float64, β::Float64) =
    NormalGammaMeasure(ProductSpace(Space[Euclidean(1), PositiveReals()]), κ, μ, α, β)

function Base.getproperty(m::NormalGammaMeasure, s::Symbol)
    if s === :κ || s === :μ || s === :α || s === :β
        return getproperty(getfield(m, :prevision), s)
    else
        return getfield(m, s)
    end
end

Base.propertynames(::NormalGammaMeasure) = (:κ, :μ, :α, :β, :space, :prevision)

mean(m::NormalGammaMeasure) = m.μ

# ── Enumeration: carrier-binding Measure over CategoricalPrevision ──

struct EnumerationMeasure{T} <: Measure
    prevision::CategoricalPrevision
    carrier::Vector{T}
    space::Finite{T}

    function EnumerationMeasure{T}(prevision::CategoricalPrevision, carrier::Vector{T}, space::Finite{T}) where {T}
        length(carrier) == length(prevision.log_weights) || error("carrier and weights must match")
        length(carrier) == length(space.values) || error("carrier and space must match")
        new{T}(prevision, carrier, space)
    end
end

function expect(m::EnumerationMeasure, f::Function)
    lw = m.prevision.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./= sum(w)
    sum(w[i] * f(m.carrier[i]) for i in eachindex(w))
end

expect(m::EnumerationMeasure, tf::TestFunction) = expect(m, x -> apply(tf, x))

# ── Product: independent joint ──

struct ProductMeasure <: Measure
    prevision::ProductPrevision
    space::ProductSpace

    function ProductMeasure(space::ProductSpace, factors::Vector{<:Measure})
        length(space.factors) == length(factors) || error("space and factors must match")
        previsions = Prevision[f.prevision for f in factors]
        new(ProductPrevision(previsions), space)
    end

    ProductMeasure(prevision::ProductPrevision, space::ProductSpace) = new(prevision, space)
end

ProductMeasure(factors::Vector{<:Measure}) =
    ProductMeasure(ProductSpace(Space[f.space for f in factors]), factors)

function Base.getproperty(m::ProductMeasure, s::Symbol)
    if s === :factors
        p = getfield(m, :prevision)
        spaces = getfield(m, :space).factors
        return FrozenVectorView(Measure[wrap_in_measure(p.factors[i], spaces[i]) for i in eachindex(p.factors)])
    else
        return getfield(m, s)
    end
end

Base.propertynames(::ProductMeasure) = (:factors, :space, :prevision)

factor(m::ProductMeasure, i::Int) = m.factors[i]

function replace_factor(m::ProductMeasure, i::Int, new_factor::Measure)
    new_factors = Measure[f for f in m.factors]
    new_factors[i] = new_factor
    ProductMeasure(new_factors)
end

# ── Mixture: weighted combination ──

struct MixtureMeasure <: Measure
    prevision::MixturePrevision
    space::Space

    function MixtureMeasure(space::Space, components::Vector{<:Measure}, log_weights::Vector{Float64})
        previsions = Prevision[c.prevision for c in components]
        new(MixturePrevision(previsions, log_weights), space)
    end

    MixtureMeasure(space::Space, prevision::MixturePrevision) = new(prevision, space)
end

MixtureMeasure(components::Vector{<:Measure}, log_weights::Vector{Float64}) =
    MixtureMeasure(components[1].space, components, log_weights)

function Base.getproperty(m::MixtureMeasure, s::Symbol)
    if s === :components
        p = getfield(m, :prevision)
        sp = getfield(m, :space)
        return FrozenVectorView(Measure[wrap_in_measure(c, sp) for c in p.components])
    elseif s === :log_weights
        return getproperty(getfield(m, :prevision), :log_weights)
    else
        return getfield(m, s)
    end
end

Base.propertynames(::MixtureMeasure) = (:components, :log_weights, :space, :prevision)

function weights(m::MixtureMeasure)
    max_lw = maximum(m.log_weights)
    w = exp.(m.log_weights .- max_lw)
    w ./ sum(w)
end

# ── Utility: numerically stable log-sum-exp ──

function logsumexp(xs::AbstractVector{<:Real})::Float64
    isempty(xs) && return -Inf
    m = maximum(xs)
    m == -Inf && return -Inf
    m + log(sum(exp(x - m) for x in xs))
end

# ── wrap_in_measure: Prevision → Measure reconstruction ──

"""
    wrap_in_measure(p::Prevision) → Measure

Reconstruct the canonical Measure wrapper for `p`. Called by the
Measure-level `getproperty` shields during shield-reconstruction of
`:components` / `:factors` / `:beta` reads.
"""
function wrap_in_measure end

wrap_in_measure(p::BetaPrevision) = BetaMeasure(Interval(0.0, 1.0), p.alpha, p.beta)

function wrap_in_measure(p::TaggedBetaPrevision)
    beta_measure = p.beta isa BetaPrevision ? wrap_in_measure(p.beta) : p.beta
    TaggedBetaMeasure(Interval(0.0, 1.0), p.tag, beta_measure)
end

wrap_in_measure(p::GaussianPrevision) = GaussianMeasure(Euclidean(1), p.mu, p.sigma)
wrap_in_measure(p::GammaPrevision) = GammaMeasure(PositiveReals(), p.alpha, p.beta)

function wrap_in_measure(p::ProductPrevision)
    factors_as_measures = Measure[wrap_in_measure(f) for f in p.factors]
    ProductMeasure(factors_as_measures)
end

function wrap_in_measure(p::MixturePrevision)
    components_as_measures = Measure[wrap_in_measure(c) for c in p.components]
    MixtureMeasure(components_as_measures, copy(p.log_weights))
end

wrap_in_measure(p::CategoricalPrevision) = error(
    "wrap_in_measure(::CategoricalPrevision) requires carrier space context " *
    "(Finite{T}); use wrap_in_measure(p, space) or construct CategoricalMeasure(space, p) " *
    "explicitly. See docs/posture-4/move-2-design.md §6 (wrap_in_measure coverage)."
)

wrap_in_measure(p::Prevision, ::Space) = wrap_in_measure(p)
wrap_in_measure(p::CategoricalPrevision, space::Finite) = CategoricalMeasure(space, p)

function wrap_in_measure(p::ProductPrevision, space::ProductSpace)
    factors_as_measures = Measure[wrap_in_measure(p.factors[i], space.factors[i]) for i in eachindex(p.factors)]
    ProductMeasure(factors_as_measures)
end

function wrap_in_measure(p::MixturePrevision, space::Space)
    components_as_measures = Measure[wrap_in_measure(c, space) for c in p.components]
    MixtureMeasure(components_as_measures, copy(p.log_weights))
end
wrap_in_measure(p::DirichletPrevision, space::Simplex) = error(
    "wrap_in_measure(::DirichletPrevision, ::Simplex) also requires categories::Finite; " *
    "construct DirichletMeasure explicitly."
)
wrap_in_measure(p::DirichletPrevision) = error(
    "wrap_in_measure(::DirichletPrevision) requires Simplex+Finite space context; " *
    "construct DirichletMeasure(space, categories, p.alpha) explicitly."
)
wrap_in_measure(p::ParticlePrevision) =
    CategoricalMeasure(Finite(p.samples), p)

wrap_in_measure(p::NormalGammaPrevision) =
    NormalGammaMeasure(p.κ, p.μ, p.α, p.β)

# ================================================================
# TYPE 4: Event (extracted to events.jl)
# ================================================================
include("events.jl")

# ================================================================
# Functional: alias for Previsions.TestFunction
# ================================================================
const Functional = TestFunction

# ================================================================
# AXIOM-CONSTRAINED FUNCTION: expect
# ================================================================

function expect(m::CategoricalMeasure, f::Function)
    w = weights(m)
    sum(w[i] * f(m.space.values[i]) for i in eachindex(w))
end

function expect(m::BetaMeasure, f::Function; n::Int=64)
    grid = range(1/(2n), 1-1/(2n), length=n)
    logw = [log_density_at(m, x) for x in grid]
    max_lw = maximum(logw)
    w = exp.(logw .- max_lw)
    w ./= sum(w)
    sum(w[i] * f(grid[i]) for i in eachindex(w))
end

expect(m::TaggedBetaMeasure, f::Function; kwargs...) = expect(m.beta, f; kwargs...)

function expect(m::GaussianMeasure, f::Function; n::Int=64)
    lo = m.mu - 4 * m.sigma
    hi = m.mu + 4 * m.sigma
    grid = range(lo, hi, length=n)
    logw = [-0.5 * ((x - m.mu) / m.sigma)^2 for x in grid]
    max_lw = maximum(logw)
    w = exp.(logw .- max_lw)
    w ./= sum(w)
    sum(w[i] * f(grid[i]) for i in eachindex(w))
end

function expect(m::GammaMeasure, f::Function; n::Int=64)
    μ = m.alpha / m.beta
    σ = sqrt(m.alpha) / m.beta
    lo = max(1e-10, μ - 4σ)
    hi = μ + 6σ
    grid = range(lo, hi, length=n)
    logw = [log_density_at(m, x) for x in grid]
    max_lw = maximum(logw)
    w = exp.(logw .- max_lw)
    w ./= sum(w)
    sum(w[i] * f(grid[i]) for i in eachindex(w))
end

function expect(m::DirichletMeasure, f::Function; n_samples::Int=1000)
    total = 0.0
    for _ in 1:n_samples
        θ = draw(m)
        total += f(θ)
    end
    total / n_samples
end

function expect(m::NormalGammaMeasure, f::Function; n_samples::Int=1000)
    total = 0.0
    for _ in 1:n_samples
        total += f(draw(m))
    end
    total / n_samples
end

function expect(m::ProductMeasure, f::Function; n_samples::Int=1000)
    total = 0.0
    for _ in 1:n_samples
        total += f(draw(m))
    end
    total / n_samples
end

function expect(m::MixtureMeasure, f::Function)
    w = weights(m)
    sum(w[i] * expect(m.components[i], f) for i in eachindex(w))
end

# ── Dispatch: closed-form cases on leaf measures ──
expect(m::BetaMeasure, ::Identity)     = m.alpha / (m.alpha + m.beta)
expect(m::TaggedBetaMeasure, ::Identity) = expect(m.beta, Identity())
expect(m::GammaMeasure, ::Identity)    = m.alpha / m.beta
expect(m::GaussianMeasure, ::Identity) = m.mu

function expect(m::CategoricalMeasure, ::Identity)
    w = weights(m)
    sum(w[i] * m.space.values[i] for i in eachindex(w))
end

# ── Dispatch: structural decomposition on ProductMeasure ──
expect(m::ProductMeasure, p::Projection) = expect(m.factors[p.index], Identity())

function expect(m::ProductMeasure, np::NestedProjection)
    length(np.indices) >= 1 || error("NestedProjection requires at least one index")
    if length(np.indices) == 1
        expect(m.factors[np.indices[1]], Identity())
    else
        expect(m.factors[np.indices[1]], NestedProjection(np.indices[2:end]))
    end
end

# ── Dispatch: CategoricalMeasure projection (vector-valued atoms) ──
function expect(m::CategoricalMeasure, p::Projection)
    w = weights(m)
    sum(w[i] * m.space.values[i][p.index] for i in eachindex(w))
end

function expect(m::CategoricalMeasure, np::NestedProjection)
    length(np.indices) >= 1 || error("NestedProjection requires at least one index")
    w = weights(m)
    total = 0.0
    for i in eachindex(w)
        v = m.space.values[i]
        for idx in np.indices
            v = v[idx]
        end
        total += w[i] * v
    end
    total
end

expect(m::CategoricalMeasure, t::Tabular) =
    sum(weights(m)[i] * t.values[i] for i in eachindex(t.values))

# ── Dispatch: algebraic combination (linearity of expectation) ──
function expect(m::Measure, lc::LinearCombination)
    total = lc.offset
    for (c, f) in lc.terms
        total += c * expect(m, f)
    end
    total
end

# ── Dispatch: mixture recursion for any Functional ──
function expect(m::MixtureMeasure, φ::Functional)
    w = weights(m)
    sum(w[i] * expect(m.components[i], φ) for i in eachindex(w))
end

expect(m::Measure, o::OpaqueClosure; kwargs...) = expect(m, o.f; kwargs...)
expect(m::MixtureMeasure, o::OpaqueClosure; kwargs...) = expect(m, o.f; kwargs...)

# ── Prevision-primary expect methods ──
# Closed-form Identity on scalar Prevision types.
expect(p::BetaPrevision, ::Identity) = p.alpha / (p.alpha + p.beta)
expect(p::TaggedBetaPrevision, ::Identity) = expect(p.beta, Identity())
expect(p::GaussianPrevision, ::Identity) = p.mu
expect(p::GammaPrevision, ::Identity) = p.alpha / p.beta
expect(p::DirichletPrevision, ::Identity) = p.alpha ./ sum(p.alpha)
expect(p::NormalGammaPrevision, ::Identity) = p.μ

# General-function expect on scalar Previsions via quadrature/delegation.
function expect(p::BetaPrevision, f::Function; n::Int=64)
    grid = range(1/(2n), 1-1/(2n), length=n)
    logw = [(p.alpha - 1) * log(x) + (p.beta - 1) * log(1 - x) for x in grid]
    max_lw = maximum(logw)
    w = exp.(logw .- max_lw)
    w ./= sum(w)
    sum(w[i] * f(grid[i]) for i in eachindex(w))
end

expect(p::TaggedBetaPrevision, f::Function; kwargs...) = expect(p.beta, f; kwargs...)

function expect(p::GaussianPrevision, f::Function; n::Int=64)
    lo = p.mu - 4 * p.sigma
    hi = p.mu + 4 * p.sigma
    grid = range(lo, hi, length=n)
    logw = [-0.5 * ((x - p.mu) / p.sigma)^2 for x in grid]
    max_lw = maximum(logw)
    w = exp.(logw .- max_lw)
    w ./= sum(w)
    sum(w[i] * f(grid[i]) for i in eachindex(w))
end

function expect(p::GammaPrevision, f::Function; n::Int=64)
    μ = p.alpha / p.beta
    σ = sqrt(p.alpha) / p.beta
    lo = max(1e-10, μ - 4σ)
    hi = μ + 6σ
    grid = range(lo, hi, length=n)
    logw = [(p.alpha - 1) * log(x) - p.beta * x for x in grid]
    max_lw = maximum(logw)
    w = exp.(logw .- max_lw)
    w ./= sum(w)
    sum(w[i] * f(grid[i]) for i in eachindex(w))
end

# Particle/Quadrature/Enumeration carry their own support data.
function expect(p::ParticlePrevision, f::Function)
    lw = p.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./= sum(w)
    sum(w[i] * f(p.samples[i]) for i in eachindex(w))
end

function expect(p::QuadraturePrevision, f::Function)
    lw = p.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./= sum(w)
    sum(w[i] * f(p.grid[i]) for i in eachindex(w))
end

# MixturePrevision: linearity of expectation.
function expect(p::MixturePrevision, f::Function)
    lw = p.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./= sum(w)
    sum(w[i] * expect(p.components[i], f) for i in eachindex(w))
end

# Functional dispatch on Prevision types.
function expect(p::MixturePrevision, φ::TestFunction)
    lw = p.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./= sum(w)
    sum(w[i] * expect(p.components[i], φ) for i in eachindex(w))
end

# Generic TestFunction dispatch on scalar Previsions: apply then integrate.
expect(p::BetaPrevision, tf::TestFunction; kwargs...) = expect(p, x -> apply(tf, x); kwargs...)
expect(p::TaggedBetaPrevision, tf::TestFunction; kwargs...) = expect(p.beta, tf; kwargs...)
expect(p::GaussianPrevision, tf::TestFunction; kwargs...) = expect(p, x -> apply(tf, x); kwargs...)
expect(p::GammaPrevision, tf::TestFunction; kwargs...) = expect(p, x -> apply(tf, x); kwargs...)
expect(p::ParticlePrevision, tf::TestFunction) = expect(p, x -> apply(tf, x))
expect(p::QuadraturePrevision, tf::TestFunction) = expect(p, x -> apply(tf, x))

# Prevision-level OpaqueClosure unwrapping.
expect(p::Prevision, o::OpaqueClosure; kwargs...) = expect(p, o.f; kwargs...)

# LinearCombination on Prevision: linearity of expectation.
function expect(p::Prevision, lc::LinearCombination)
    total = lc.offset
    for (c, f) in lc.terms
        total += c * expect(p, f)
    end
    total
end

# Measure-level probability via indicator kernel. The generic path
# integrates the indicator over the measure's support; specialised
# methods handle component-level events (TagSet on MixtureMeasure).
function probability(m::Measure, e::Event)
    k = indicator_kernel(e)
    expect(m, h -> k.log_density(h, true) == 0.0 ? 1.0 : 0.0)
end

function probability(m::MixtureMeasure, e::TagSet)
    w = weights(m)
    total = 0.0
    for (i, comp) in enumerate(m.components)
        if comp isa TaggedBetaMeasure && comp.tag in e.tags
            total += w[i]
        end
    end
    total
end

# ================================================================
# Conjugate registry (extracted to conjugate.jl)
# ================================================================
include("conjugate.jl")

# ================================================================
# AXIOM-CONSTRAINED FUNCTION: condition
# ================================================================

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
    cp = maybe_conjugate(m.prevision, k)
    if cp !== nothing
        updated = update(cp, observation).prior
        return BetaMeasure(m.space, updated.alpha, updated.beta)
    end
    if k.source isa Interval && k.target isa Finite && length(k.target.values) == 2
        if observation == 1 || observation == 1.0 || observation == true
            return BetaMeasure(m.space, m.alpha + 1.0, m.beta)
        elseif observation == 0 || observation == 0.0 || observation == false
            return BetaMeasure(m.space, m.alpha, m.beta + 1.0)
        end
    end
    _condition_by_grid(m, k, observation)
end

function condition(m::TaggedBetaMeasure, k::Kernel, observation)
    fam = _resolve_likelihood_family(k.likelihood_family, m)
    resolved_k = _with_resolved_family(k, fam)
    k.log_density(m, observation)
    new_beta = condition(m.beta, resolved_k, observation)
    TaggedBetaMeasure(m.space, m.tag, new_beta)
end

"""
    _conjugacy_prevision(component) -> Prevision

Return the prevision Move 4's `maybe_conjugate` registry keys on for a
given mixture component.
"""
_conjugacy_prevision(m::TaggedBetaMeasure) = getfield(m, :prevision).beta
_conjugacy_prevision(m::Measure) = m.prevision
_conjugacy_prevision(p::TaggedBetaPrevision) = p.beta
# Identity for bare Previsions (GaussianPrevision, GammaPrevision, etc.) that
# are themselves the conjugacy-keyed type — no unwrapping needed.
_conjugacy_prevision(p::Prevision) = p

function condition(m::GaussianMeasure, k::Kernel, observation)
    cp = maybe_conjugate(m.prevision, k)
    if cp !== nothing
        updated = update(cp, observation).prior
        return GaussianMeasure(m.space, updated.mu, updated.sigma)
    end
    _condition_by_grid(m, k, observation)
end

function condition(m::DirichletMeasure, k::Kernel, observation)
    cp = maybe_conjugate(m.prevision, k)
    if cp !== nothing
        updated = update(cp, observation).prior
        return DirichletMeasure(m.space, m.categories, updated.alpha)
    end
    k.source isa Simplex && k.target isa Finite ||
        error("DirichletMeasure condition requires a Categorical kernel (Simplex → Finite)")
    synthetic_cp = ConjugatePrevision(m.prevision, Categorical(m.categories))
    updated = update(synthetic_cp, observation).prior
    DirichletMeasure(m.space, m.categories, updated.alpha)
end

function condition(m::NormalGammaMeasure, k::Kernel, observation)
    cp = maybe_conjugate(m.prevision, k)
    if cp !== nothing
        updated = update(cp, observation).prior
        return NormalGammaMeasure(m.space, updated.κ, updated.μ, updated.α, updated.β)
    end
    _condition_particle(m, k, observation)
end

function condition(m::GammaMeasure, k::Kernel, observation)
    cp = maybe_conjugate(m.prevision, k)
    if cp !== nothing
        updated = update(cp, observation).prior
        return GammaMeasure(m.space, updated.alpha, updated.beta)
    end
    _condition_particle(m, k, observation)
end

# ── Prevision-level scalar condition methods (Move 7: Move 5 completion) ──

function condition(p::BetaPrevision, k::Kernel, observation)
    cp = maybe_conjugate(p, k)
    if cp !== nothing
        return update(cp, observation).prior
    end
    conditioned = condition(wrap_in_measure(p), k, observation)
    conditioned.prevision
end

function condition(p::TaggedBetaPrevision, k::Kernel, observation)
    fam = _resolve_likelihood_family(k.likelihood_family, p)
    resolved_k = _with_resolved_family(k, fam)
    k.log_density(p, observation)
    new_beta = condition(p.beta, resolved_k, observation)
    TaggedBetaPrevision(p.tag, new_beta)
end

function condition(p::GaussianPrevision, k::Kernel, observation)
    cp = maybe_conjugate(p, k)
    if cp !== nothing
        return update(cp, observation).prior
    end
    conditioned = condition(wrap_in_measure(p), k, observation)
    conditioned.prevision
end

function condition(p::GammaPrevision, k::Kernel, observation)
    cp = maybe_conjugate(p, k)
    if cp !== nothing
        return update(cp, observation).prior
    end
    conditioned = condition(wrap_in_measure(p), k, observation)
    conditioned.prevision
end

function condition(p::DirichletPrevision, k::Kernel, observation)
    cp = maybe_conjugate(p, k)
    if cp !== nothing
        return update(cp, observation).prior
    end
    error("condition(::DirichletPrevision, ...) non-conjugate fallback requires " *
          "categories/space context; use DirichletMeasure for non-conjugate kernels")
end

function condition(p::NormalGammaPrevision, k::Kernel, observation)
    cp = maybe_conjugate(p, k)
    if cp !== nothing
        return update(cp, observation).prior
    end
    error("condition(::NormalGammaPrevision, ...) non-conjugate fallback requires " *
          "space context; use NormalGammaMeasure for non-conjugate kernels")
end

function condition(p::ProductPrevision, k::Kernel, obs; kwargs...)
    conditioned = condition(wrap_in_measure(p), k, obs; kwargs...)
    conditioned isa MixtureMeasure ? conditioned.prevision : conditioned.prevision
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
    qp = QuadraturePrevision(grid, logw)
    CategoricalMeasure(Finite(qp.grid), qp)
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
    qp = QuadraturePrevision(grid, logw)
    CategoricalMeasure(Finite(qp.grid), qp)
end

log_density_at(m::BetaMeasure, x) = (m.alpha - 1) * log(x) + (m.beta - 1) * log(1 - x)
log_density_at(m::GammaMeasure, x) = (m.alpha - 1) * log(x) - m.beta * x
log_density_at(m::TaggedBetaMeasure, x) = log_density_at(m.beta, x)
log_density_at(m::GaussianMeasure, x) = -0.5 * ((x - m.mu) / m.sigma)^2
log_density_at(m::DirichletMeasure, x) = sum((m.alpha[i] - 1) * log(x[i]) for i in eachindex(m.alpha))

function log_density_at(m::NormalGammaMeasure, x)
    μ_val, σ² = x[1], x[2]
    σ² > 0 || return -Inf
    log_ig = m.α * log(m.β) - _log_gamma(m.α) - (m.α + 1.0) * log(σ²) - m.β / σ²
    log_n = -0.5 * log(2π * σ² / m.κ) - m.κ * (μ_val - m.μ)^2 / (2.0 * σ²)
    log_ig + log_n
end

function _log_gamma(x::Float64)
    if x < 0.5
        return log(π / sin(π * x)) - _log_gamma(1.0 - x)
    end
    x -= 1.0
    g = 7.0
    coeffs = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
              771.32342877765313, -176.61502916214059, 12.507343278686905,
              -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    t = x + g + 0.5
    s = coeffs[1]
    for i in 2:length(coeffs)
        s += coeffs[i] / (x + Float64(i - 1))
    end
    0.5 * log(2π) + (x + 0.5) * log(t) - t + log(s)
end

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

function _predictive_ll(m::TaggedBetaMeasure, k::Kernel, obs)
    k.log_density(m, obs)
end

function _predictive_ll(m::MixtureMeasure, k::Kernel, obs)
    w = weights(m)
    total = sum(w[i] * exp(_predictive_ll(m.components[i], k, obs)) for i in eachindex(w))
    log(max(total, 1e-300))
end

function _predictive_ll(m::Measure, k::Kernel, obs; n_samples::Int=200)
    total = 0.0
    for _ in 1:n_samples
        total += exp(k.log_density(draw(m), obs))
    end
    log(max(total / n_samples, 1e-300))
end

# ── Prevision-level _predictive_ll (Move 7: supports MixturePrevision condition) ──

_predictive_ll(p::BetaPrevision, k::Kernel, obs) =
    _predictive_ll(wrap_in_measure(p), k, obs)

_predictive_ll(p::TaggedBetaPrevision, k::Kernel, obs) =
    k.log_density(p, obs)

_predictive_ll(p::GaussianPrevision, k::Kernel, obs) =
    _predictive_ll(wrap_in_measure(p), k, obs)

_predictive_ll(p::GammaPrevision, k::Kernel, obs) =
    _predictive_ll(wrap_in_measure(p), k, obs)

function _predictive_ll(p::Prevision, k::Kernel, obs)
    _predictive_ll(wrap_in_measure(p), k, obs)
end

# ── log_predictive: log P(obs | beliefs) — single observation ──

function log_predictive(m::Measure, k::Kernel, obs)
    pred = expect(m, h -> exp(density(k, h, obs)))
    log(max(pred, 1e-300))
end

function log_predictive(p::Prevision, k::Kernel, obs)
    log_predictive(wrap_in_measure(p), k, obs)
end

function log_predictive(p::DirichletPrevision, k::Kernel, obs)
    lf = k.likelihood_family
    lf isa Categorical || error("DirichletPrevision log_predictive requires Categorical kernel, got $(typeof(lf))")
    idx = findfirst(==(obs), lf.categories.values)
    idx !== nothing || error("observation $obs not in categories")
    log(p.alpha[idx] / sum(p.alpha))
end

function log_predictive(m::DirichletMeasure, k::Kernel, obs)
    idx = findfirst(==(obs), m.categories.values)
    idx !== nothing || error("observation $obs not in categories")
    log(m.alpha[idx] / sum(m.alpha))
end

function log_predictive(m::CategoricalMeasure, k::Kernel, obs)
    _predictive_ll(m, k, obs)
end

function condition(p::MixturePrevision, k::Kernel, obs)
    new_components = Prevision[]
    new_log_wts = Float64[]
    for (i, comp) in enumerate(p.components)
        pred_ll = _predictive_ll(comp, k, obs)
        conditioned = condition(comp, k, obs)
        base_lw = p.log_weights[i] + pred_ll
        if conditioned isa MixturePrevision
            for (j, sub) in enumerate(conditioned.components)
                push!(new_components, sub)
                push!(new_log_wts, base_lw + conditioned.log_weights[j])
            end
        else
            push!(new_components, conditioned)
            push!(new_log_wts, base_lw)
        end
    end
    MixturePrevision(new_components, new_log_wts)
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

function _dispatch_path(p::MixturePrevision, k::Kernel)
    for comp in p.components
        fam = _resolve_likelihood_family(k.likelihood_family, comp)
        resolved_k = _with_resolved_family(k, fam)
        _dispatch_path(_conjugacy_prevision(comp), resolved_k) === :conjugate || return :mixed
    end
    :conjugate
end

_dispatch_path(m::MixtureMeasure, k::Kernel) = _dispatch_path(m.prevision, k)

function _dispatch_path(p::Prevision, k::Kernel)
    maybe_conjugate(p, k) !== nothing ? :conjugate : :particle
end

factor(p::ProductPrevision, i::Int) = p.factors[i]

function replace_factor(p::ProductPrevision, i::Int, new_factor)
    new_factors = [f for f in p.factors]
    new_factors[i] = new_factor
    ProductPrevision(new_factors)
end

function decompose(p::ExchangeablePrevision)
    p.prior_on_components isa DirichletPrevision ||
        error("decompose: only Dirichlet priors supported at Move 5 (§5.1 R3 scoping); got $(typeof(p.prior_on_components))")
    p.component_space isa Finite ||
        error("decompose: only Finite component_space supported at Move 5; got $(typeof(p.component_space))")

    α = p.prior_on_components.alpha
    total = sum(α)
    cats = p.component_space.values
    k = length(cats)

    k == length(α) ||
        error("decompose: component_space has $k categories but Dirichlet prior has $(length(α)) components")

    components = Prevision[]
    log_weights = Float64[]
    for i in 1:k
        log_w_per_cat = fill(-Inf, k)
        log_w_per_cat[i] = 0.0
        push!(components, CategoricalPrevision(log_w_per_cat))
        push!(log_weights, log(α[i] / total))
    end
    MixturePrevision(components, log_weights)
end

# ── ProductMeasure condition ──

function condition(m::ProductMeasure, k::Kernel, obs; kwargs...)
    fs = k.factor_selector
    fs === nothing && return _condition_particle(m, k, obs; kwargs...)

    d = fs.discrete_index
    cat = m.factors[d]::CategoricalMeasure
    cat_w = weights(cat)

    new_components = Measure[]
    new_log_wts = Float64[]

    for (ci, c) in enumerate(cat.space.values)
        active_indices = fs.active(c)
        length(active_indices) == 1 || error("multiple active factors not yet supported")
        ai = active_indices[1]
        factor_ai = m.factors[ai]

        restricted_ld = (h_i, o) -> begin
            full_h = Any[0.0 for _ in m.factors]
            full_h[d] = c
            full_h[ai] = h_i
            k.log_density(full_h, o)
        end
        restricted_k = Kernel(factor_ai.space, k.target,
            _ -> error("generate not used in condition"), restricted_ld;
            likelihood_family = k.likelihood_family)

        conditioned = condition(factor_ai, restricted_k, obs)
        pred_ll = _predictive_ll(factor_ai, restricted_k, obs)

        new_factors = Measure[f for f in m.factors]
        point_mass_lw = fill(-Inf, length(cat.space.values))
        point_mass_lw[ci] = 0.0
        new_factors[d] = CategoricalMeasure(cat.space, point_mass_lw)
        new_factors[ai] = conditioned

        push!(new_components, ProductMeasure(new_factors))
        push!(new_log_wts, log(max(cat_w[ci], 1e-300)) + pred_ll)
    end

    MixtureMeasure(m.space, new_components, new_log_wts)
end

function _condition_particle(m::Measure, k::Kernel, obs; n_particles::Int=1000, seed::Int=0)
    samples = [draw(m) for _ in 1:n_particles]
    log_weights = Float64[k.log_density(s, obs) for s in samples]
    pp = ParticlePrevision(samples, log_weights, seed)
    CategoricalMeasure(Finite(pp.samples), pp)
end

function condition(m::Measure, k::Kernel, obs; n_particles::Int=1000)
    _condition_particle(m, k, obs; n_particles=n_particles)
end

# ── condition — sibling form taking an Event directly ──

function condition(m::Measure, e::Event)
    condition(m, indicator_kernel(e), true)
end

function condition(m::MixtureMeasure, e::TagSet)
    new_p = condition(m.prevision, e)
    MixtureMeasure(m.space, new_p)
end

function condition(p::MixturePrevision, e::TagSet)
    new_lws = copy(p.log_weights)
    for (i, comp) in enumerate(p.components)
        if comp isa TaggedBetaPrevision
            if !(comp.tag in e.tags)
                new_lws[i] = -Inf
            end
        else
            error("condition(::MixturePrevision, ::TagSet): expected TaggedBetaPrevision; got $(typeof(comp)) at index $i")
        end
    end
    MixturePrevision(p.components, new_lws)
end

# ================================================================
# AXIOM-CONSTRAINED FUNCTION: push_measure
# ================================================================

function push_measure(m::CategoricalMeasure, k::Kernel)
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

function push_measure(m::MixtureMeasure, k::Kernel)
    tgt = k.target
    tgt isa Finite || error("push_measure to non-Finite target not implemented for MixtureMeasure")
    w = weights(m)
    tgt_logw = fill(-Inf, length(tgt.values))
    for (i, comp) in enumerate(m.components)
        for (j, o) in enumerate(tgt.values)
            p = expect(comp, h -> exp(density(k, h, o)))
            p > 0 || continue
            contrib = log(w[i]) + log(p)
            if tgt_logw[j] == -Inf
                tgt_logw[j] = contrib
            else
                mx = max(tgt_logw[j], contrib)
                tgt_logw[j] = mx + log(exp(tgt_logw[j] - mx) + exp(contrib - mx))
            end
        end
    end
    CategoricalMeasure(tgt, tgt_logw)
end

function push_measure(m::DirichletMeasure, k::Kernel)
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
    x = _draw_gamma(m.alpha)
    y = _draw_gamma(m.beta)
    x / (x + y)
end

draw(m::TaggedBetaMeasure) = draw(m.beta)

draw(m::GammaMeasure) = _draw_gamma(m.alpha) / m.beta

function draw(m::GaussianMeasure)
    m.mu + m.sigma * randn()
end

function _draw_gamma(alpha::Float64)
    if alpha == 1.0
        return -log(rand())
    elseif alpha < 1.0
        return _draw_gamma(alpha + 1.0) * rand()^(1.0 / alpha)
    else
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

function draw(m::NormalGammaMeasure)
    g = _draw_gamma(m.α)
    σ² = m.β / g
    μ_s = m.μ + sqrt(σ² / m.κ) * randn()
    (μ_s, σ²)
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

function draw(p::BetaPrevision)
    x = _draw_gamma(p.alpha)
    y = _draw_gamma(p.beta)
    x / (x + y)
end

draw(p::TaggedBetaPrevision) = draw(p.beta)

draw(p::GaussianPrevision) = p.mu + p.sigma * randn()

draw(p::GammaPrevision) = _draw_gamma(p.alpha) / p.beta

function draw(p::DirichletPrevision)
    g = [_draw_gamma(a) for a in p.alpha]
    g ./= sum(g)
    g
end

function draw(p::NormalGammaPrevision)
    g = _draw_gamma(p.α)
    σ² = p.β / g
    μ_s = p.μ + sqrt(σ² / p.κ) * randn()
    (μ_s, σ²)
end

draw(p::ProductPrevision) = Any[draw(f) for f in p.factors]

function draw(p::MixturePrevision)
    w = weights(p)
    r = rand()
    cumw = 0.0
    for i in eachindex(w)
        cumw += w[i]
        r < cumw && return draw(p.components[i])
    end
    draw(p.components[end])
end

# ── Mixture maintenance ──

function prune(m::MixtureMeasure; threshold::Float64=-20.0)
    max_lw = maximum(m.log_weights)
    keep = [i for i in eachindex(m.log_weights) if m.log_weights[i] - max_lw > threshold]
    length(keep) == length(m.components) && return m
    MixtureMeasure(m.space, m.components[keep], m.log_weights[keep])
end

function prune(p::MixturePrevision; threshold::Float64=-20.0)
    max_lw = maximum(p.log_weights)
    keep = [i for i in eachindex(p.log_weights) if p.log_weights[i] - max_lw > threshold]
    length(keep) == length(p.components) && return p
    MixturePrevision(p.components[keep], p.log_weights[keep])
end

function truncate(m::MixtureMeasure; max_components::Int=10)
    length(m.components) <= max_components && return m
    perm = sortperm(m.log_weights, rev=true)
    keep = perm[1:min(max_components, length(perm))]
    MixtureMeasure(m.space, m.components[keep], m.log_weights[keep])
end

function truncate(p::MixturePrevision; max_components::Int=10)
    length(p.components) <= max_components && return p
    perm = sortperm(p.log_weights, rev=true)
    keep = perm[1:min(max_components, length(perm))]
    MixturePrevision(p.components[keep], p.log_weights[keep])
end

# ================================================================
# log_marginal — Dirichlet-Multinomial marginal likelihood
# ================================================================

function log_marginal(m::DirichletMeasure, counts::Vector{Int})
    α = m.alpha
    length(α) == length(counts) || error("alpha and counts must have same length")
    α_sum = sum(α)
    n_sum = sum(counts)
    score = _log_gamma(α_sum) - _log_gamma(α_sum + n_sum)
    for i in eachindex(α)
        score += _log_gamma(α[i] + counts[i]) - _log_gamma(α[i])
    end
    score
end

# ================================================================
# Stdlib (extracted to stdlib.jl)
# ================================================================
include("stdlib.jl")

end # module Ontology
