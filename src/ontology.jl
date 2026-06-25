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

using LinearAlgebra: SymTridiagonal, eigen, dot, cholesky, Symmetric

# ── Move 2: unify Functional hierarchy onto Previsions.TestFunction ──
import ..Previsions: Prevision
import ..Previsions: TestFunction, Identity, Projection, NestedProjection,
                     Tabular, LinearCombination, OpaqueClosure, FiringChoice, expect
import ..Previsions: BetaPrevision, TaggedBetaPrevision, SparseStructurePrevision, GaussianPrevision, TruncatedGaussianPrevision, MvGaussianPrevision, GammaPrevision, CategoricalPrevision, DirichletPrevision, NormalGammaPrevision, ProductPrevision, MixturePrevision, LabelledCategoricalPrevision
import ..Previsions: ExchangeablePrevision, decompose
import ..Previsions: ParticlePrevision, QuadraturePrevision
import ..Previsions: ConditionalPrevision
import ..Previsions: condition
import ..Previsions: params
import ..Previsions: ConjugatePrevision, maybe_conjugate, update, _dispatch_path
import ..Previsions: CenteredPower, CenteredSquare, GeometricTail
import ..Previsions: Indicator, apply

export Space, Finite, Interval, ProductSpace, Simplex, Euclidean, PositiveReals, support
export Measure, CategoricalMeasure, BetaMeasure, TaggedBetaMeasure, GaussianMeasure, MvGaussianMeasure, GammaMeasure, ExponentialMeasure, DirichletMeasure, NormalGammaMeasure, EnumerationMeasure, ProductMeasure, MixtureMeasure
export Kernel, FactorSelector, kernel_source, kernel_target, kernel_params
export LikelihoodFamily, LeafFamily, PushOnly, BetaBernoulli, WeightedBernoulli, SoftBernoulli, Flat, FiringByTag, DispatchByComponent, DepthCapExceeded
export FAMILY_REGISTRY, register_family!
export NormalNormal, LinearGaussian, Categorical, NormalGammaLikelihood, Exponential, Poisson
export GroupNoisyChannel, group_noisy_channel_logdensity
export LogisticReaction, logistic_reaction_logdensity
export Event, TagSet, FeatureEquals, FeatureInterval, Conjunction, Disjunction, Complement
export indicator_kernel, feature_value, BOOLEAN_SPACE
export Functional, Identity, Projection, NestedProjection, Tabular, LinearCombination, OpaqueClosure
export factor, replace_factor
export SparseStructurePrevision, cell_at
export condition, expect, push_measure, density, log_predictive, log_marginal, wrap_in_measure
export ConjugatePrevision, maybe_conjugate, update
export draw
export weights, mean, variance, log_density_at, prune, truncate, logsumexp
export FrozenVectorView
export WeightsDomainError, probability, marginal, marginalise, CenteredPower, CenteredSquare, GeometricTail

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

# Serialization protocol (see prevision.jl): a Measure facade serializes via the
# Prevision it wraps. Leaf facades (Beta/Gaussian/Gamma/Dirichlet/Categorical…)
# all hold a `:prevision` field; this generic method forwards to it. Facades
# without one (e.g. EnumerationMeasure) fall through to a clear MethodError.
params(m::Measure) = params(getfield(m, :prevision))

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

# ── Multivariate Gaussian: dense-covariance belief over Euclidean(d) ──

struct MvGaussianMeasure <: Measure
    prevision::MvGaussianPrevision
    space::Euclidean

    function MvGaussianMeasure(space::Euclidean, prevision::MvGaussianPrevision)
        space.dim == length(prevision.mu) ||
            error("MvGaussianMeasure: space dim $(space.dim) ≠ length(mu) $(length(prevision.mu))")
        new(prevision, space)
    end
end

MvGaussianMeasure(mu::Vector{Float64}, Sigma::Matrix{Float64}) =
    MvGaussianMeasure(Euclidean(length(mu)), MvGaussianPrevision(mu, Sigma))

function Base.getproperty(m::MvGaussianMeasure, s::Symbol)
    if s === :mu || s === :Sigma
        return getproperty(getfield(m, :prevision), s)
    else
        return getfield(m, s)
    end
end

Base.propertynames(::MvGaussianMeasure) = (:mu, :Sigma, :space, :prevision)

# `mean` is the vector μ; `variance` is the per-coordinate marginal variances
# diag(Σ) — the exact marginals (a Gaussian's marginals are read off Σ's
# diagonal). Cross-covariance lives in `m.Sigma`.
mean(m::MvGaussianMeasure) = copy(m.mu)
variance(m::MvGaussianMeasure) = [m.Sigma[i, i] for i in 1:m.space.dim]

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
wrap_in_measure(p::MvGaussianPrevision) = MvGaussianMeasure(Euclidean(length(p.mu)), p)
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

function expect(m::BetaMeasure, f::Function; n::Int=32)
    # ── Gauss-Jacobi quadrature, hand-rolled via Golub-Welsch ─────────
    #
    # Replaces a uniform-grid Riemann sum (O(1/n²) error) that left
    # voi computations in the BDSL stdlib at ~1e-4 cumulative error —
    # see apps/credence-pi/SPEC.md and the step-2 stop-and-report.
    # Polynomials in θ up to degree 2n−1 are now captured exactly by
    # the quadrature rule; the only residual error is FP rounding in
    # the weighted sum (typically <1e-13 for well-conditioned cases).
    #
    # This is a Pass-1-sufficient numerical fix. The cleaner Pass-2-
    # or-later answer is to surface typed Functionals at the BDSL
    # surface so DSL-constructed closures dispatch through
    # expect(::BetaMeasure, ::TestFunction) — Identity, Linear-
    # Combination — and reach the closed-form fast paths without
    # quadrature at all. Until that lands, this method is the path
    # DSL `lambda` expressions take.
    #
    # ── Recurrence and parameter mapping ──
    #
    # Three-term recurrence for monic Jacobi polynomials with weight
    # (1 − x)^a (1 + x)^b on [−1, 1]:
    #     P_{k+1}(x) = (x − α_k) P_k(x) − β_k P_{k−1}(x)
    # with
    #     α_0 = (b − a) / (a + b + 2)
    #     α_k = (b² − a²) / ((2k + a + b)(2k + a + b + 2))            k ≥ 1
    #     β_k = 4k(k+a)(k+b)(k+a+b) /
    #             ((2k + a + b)² · ((2k + a + b)² − 1))               k ≥ 1
    # (Stoer & Bulirsch, "Introduction to Numerical Analysis", §3.6,
    # Eq. 3.6.21–3.6.23; equivalently DLMF §18.9.)
    #
    # Golub-Welsch: the symmetric tridiagonal Jacobi matrix J_n with
    # diagonal [α_0, …, α_{n−1}] and off-diagonal [√β_1, …, √β_{n−1}]
    # has eigenvalues = quadrature nodes (in [−1, 1]) and squared
    # first-row eigenvector components ∝ quadrature weights.
    # Normalising Σwᵢ = 1 lets us skip the explicit moment β_0.
    #
    # Beta(α_β, β_β) on [0, 1] maps to Jacobi weight on [−1, 1] via
    # θ = (1 + x) / 2:
    #     ∫₀¹ f(θ) θ^(α_β−1) (1−θ)^(β_β−1) dθ
    #       ∝ ∫₋₁¹ f((1+x)/2) (1+x)^(α_β−1) (1−x)^(β_β−1) dx
    # so set Jacobi parameters
    #     a = β_β − 1
    #     b = α_β − 1
    # and evaluate f at θᵢ = (1 + xᵢ) / 2.
    #
    # ── Choice of n ──
    #
    # n=32 captures polynomials of degree ≤ 63. Pass 1's pass1-pref is
    # degree 1; n=16 (degree ≤ 31) would be adequate. n=32 is comfort
    # margin against unanticipated future preference functions, not
    # necessity — the cost difference is roughly ~1000 vs ~3000 ops on
    # a tridiagonal eigendecomposition, both negligible compared to
    # BDSL evaluation overhead.
    a = m.beta - 1.0
    b = m.alpha - 1.0
    diag = Vector{Float64}(undef, n)
    offdiag = Vector{Float64}(undef, n - 1)
    diag[1] = (b - a) / (a + b + 2)
    for k in 1:n-1
        s = 2k + a + b
        diag[k + 1] = (b * b - a * a) / (s * (s + 2))
        β_k = 4 * k * (k + a) * (k + b) * (k + a + b) / (s * s * ((s * s) - 1))
        offdiag[k] = sqrt(β_k)
    end
    F = eigen(SymTridiagonal(diag, offdiag))
    nodes = F.values                                      # eigenvalues ∈ [−1, 1]
    w = (@view F.vectors[1, :]) .^ 2                      # ∝ quadrature weights
    s_w = sum(w)
    sum(w[i] * f((1.0 + nodes[i]) / 2) for i in eachindex(w)) / s_w
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

# Closed-form geometric-tail mean E_Beta[ρ/(1−ρ)] = α/(β−1) (β>1): the exact
# posterior-predictive expected number of remaining steps under geometric
# continuation (GeometricTail). Reaches the fast path with no quadrature; β>1
# holds for any Beta built from the Beta(≥1,≥1) priors a continuation belief uses.
function expect(m::BetaMeasure, ::GeometricTail)
    m.beta > 1.0 || error("GeometricTail diverges for β ≤ 1 (got β=$(m.beta)); the continuation posterior must have β > 1")
    m.alpha / (m.beta - 1.0)
end
expect(m::TaggedBetaMeasure, ::GeometricTail) = expect(m.beta, GeometricTail())

# Exact Beta moment E_Beta[(θ−μ)^n], closed form via binomial expansion into raw
# moments m_j = E[θ^j] = ∏_{i=0}^{j-1}(α+i)/(α+β+i). μ=0 gives the raw moment E[θ^n]
# (the integrated claim-EU needs E[θ²] = CenteredPower{2}(0.0)); μ=mean gives the
# central moment (variance at n=2). Reaches the fast path with no quadrature, so the
# integrated decision is exact, not the generic `expect(::Beta, ::TestFunction)` MC.
function _beta_central_moment(α::Float64, β::Float64, n::Int, μ::Float64)
    n >= 0 || error("CenteredPower exponent must be ≥ 0, got $n")
    raw = Vector{Float64}(undef, n + 1)          # raw[j+1] = E[θ^j]
    raw[1] = 1.0
    for j in 1:n
        raw[j+1] = raw[j] * (α + (j - 1)) / (α + β + (j - 1))
    end
    s = 0.0
    for j in 0:n
        s += binomial(n, j) * (-μ)^(n - j) * raw[j+1]
    end
    s
end
expect(m::BetaMeasure, cp::CenteredPower{n}) where n = _beta_central_moment(m.alpha, m.beta, n, cp.μ)
expect(m::TaggedBetaMeasure, cp::CenteredPower) = expect(m.beta, cp)

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

# Per-component dispatch (Measure side). Mirror of the MixturePrevision method;
# more specific than the generic `(::MixtureMeasure, ::Functional)` above.
function expect(m::MixtureMeasure, fc::FiringChoice)
    length(fc.fired) == length(m.components) ||
        error("FiringChoice has $(length(fc.fired)) flags but mixture has $(length(m.components)) components")
    w = weights(m)
    sum(w[i] * expect(m.components[i], fc.fired[i] ? fc.when_fires : fc.when_not)
        for i in eachindex(w))
end

expect(m::Measure, o::OpaqueClosure; kwargs...) = expect(m, o.f; kwargs...)
expect(m::MixtureMeasure, o::OpaqueClosure; kwargs...) = expect(m, o.f; kwargs...)

# ── Prevision-primary expect methods ──
# Closed-form Identity on scalar Prevision types.
expect(p::BetaPrevision, ::Identity) = p.alpha / (p.alpha + p.beta)
expect(p::TaggedBetaPrevision, ::Identity) = expect(p.beta, Identity())

# A labelled categorical integrates a POSITIONAL functional against its inner categorical
# (the label is opaque to the functional). `Tabular` — the per-action utility vector lookup
# ships — is positional, so `optimise`/`expect` over a MixturePrevision of these integrates
# the latent for free: Σ_label w_label · Σ_i P(i|label)·u[i] = Σ_i P(i)·u[i] (the V-marginal EU,
# no explicit collapse). Non-positional functionals (Identity, by-value Function) need the
# carrier space and are read at the CategoricalMeasure level, not here.
expect(p::LabelledCategoricalPrevision, t::Tabular) =
    sum(weights(p.categorical)[i] * t.values[i] for i in eachindex(t.values))

# Exact geometric-tail mean on Prevision leaves (mirrors the BetaMeasure form):
# E_Beta[ρ/(1−ρ)] = α/(β−1). This is the closed form the brain's continuation
# belief reaches (MixturePrevision → TaggedBetaPrevision forwarder → here),
# bypassing the generic quadrature `expect(::BetaPrevision, ::TestFunction)`.
function expect(p::BetaPrevision, ::GeometricTail)
    p.beta > 1.0 || error("GeometricTail diverges for β ≤ 1 (got β=$(p.beta)); the continuation posterior must have β > 1")
    p.alpha / (p.beta - 1.0)
end

# Exact Beta moment on Prevision leaves (mirrors the BetaMeasure form): the integrated
# claim-EU `optimise{include,withhold}` reaches this through a `beta` create_state.
expect(p::BetaPrevision, cp::CenteredPower{n}) where n = _beta_central_moment(p.alpha, p.beta, n, cp.μ)
expect(p::TaggedBetaPrevision, cp::CenteredPower) = expect(p.beta, cp)
expect(p::GaussianPrevision, ::Identity) = p.mu
expect(p::MvGaussianPrevision, ::Identity) = copy(p.mu)
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

# Sequential conditioning on a quadrature posterior. The support is already fixed by the FIRST
# (continuous → quadrature) condition; a further observation only reweights mass WITHIN that
# support, so we multiply the new kernel's likelihood into the existing grid — no re-gridding.
# This is what lets a continuous latent absorb MULTIPLE non-conjugate observations (e.g. a
# Gaussian elicitation THEN a logistic reaction) and stay on the engine's one grid. Conjugate
# latents never reach here — they update in closed form and stay parametric; quadrature is the
# fallback, and once a latent is on a grid it stays on that grid for subsequent evidence.
function condition(p::QuadraturePrevision, k::Kernel, observation)
    logw = similar(p.log_weights)
    for i in eachindex(p.grid)
        ll = density(k, p.grid[i], observation)
        !isnan(ll) || error("density returned NaN conditioning a QuadraturePrevision")
        logw[i] = p.log_weights[i] + ll
    end
    QuadraturePrevision(copy(p.grid), logw)
end

# Marginal likelihood ∫ p(x)·P(obs|x) dx over the quadrature grid — the `condition` verb's
# log_marginal for the SECOND (and later) observation. A direct quadrature on the existing grid,
# mirroring `log_predictive(::TruncatedGaussianPrevision, …)`; the generic `Prevision` fallback
# routes through `wrap_in_measure`, which a QuadraturePrevision (no carrier Space) has no method for.
function log_predictive(p::QuadraturePrevision, k::Kernel, obs)
    val = expect(p, h -> exp(density(k, h, obs)))
    log(max(val, 1e-300))
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

# Disambiguate LinearCombination on the scalar leaf Previsions that also carry a generic
# `expect(::X, ::TestFunction)` quadrature fallback: both that and the decomposition above
# match `(BetaPrevision, LinearCombination)` equally. Route to the decomposition so each
# term reaches its closed form (the integrated claim-EU stays exact, not quadrature).
expect(p::BetaPrevision, lc::LinearCombination)      = invoke(expect, Tuple{Prevision, LinearCombination}, p, lc)
expect(p::TaggedBetaPrevision, lc::LinearCombination) = invoke(expect, Tuple{Prevision, LinearCombination}, p, lc)
expect(p::GaussianPrevision, lc::LinearCombination)  = invoke(expect, Tuple{Prevision, LinearCombination}, p, lc)
expect(p::GammaPrevision, lc::LinearCombination)     = invoke(expect, Tuple{Prevision, LinearCombination}, p, lc)

# Disambiguate MixturePrevision × LinearCombination — both `expect(::MixturePrevision,
# ::TestFunction)` and `expect(::Prevision, ::LinearCombination)` match. Expand by
# linearity, delegating each sub-functional back to the mixture recursion (exact,
# closed-form on scalar leaves). Exercised by the credence-pi feature brain's
# per-context EU readout (LinearCombination over Identity on a mixture of cells).
function expect(p::MixturePrevision, lc::LinearCombination)
    total = lc.offset
    for (c, f) in lc.terms
        total += c * expect(p, f)
    end
    total
end

# Evaluate one FiringChoice branch on a single mixture component. A
# LinearCombination branch is expanded by linearity *here* (mirroring the
# mixture-level LinearCombination method) so a leaf prevision/measure never
# receives a LinearCombination directly — which would be ambiguous with the
# leaf's generic `(::leaf, ::TestFunction)` apply-fallback.
_firing_branch(p, f::TestFunction) = expect(p, f)
_firing_branch(p, lc::LinearCombination) =
    lc.offset + (isempty(lc.terms) ? 0.0 : sum(c * _firing_branch(p, f) for (c, f) in lc.terms))

# Per-component dispatch over a mixture: apply `when_fires` to the components
# where `fired[i]`, `when_not` to the rest, recombine by mixture weight. The
# Functional-side dual of kernel-side FiringByTag. More specific than the
# generic `expect(::MixturePrevision, ::TestFunction)`, so it wins dispatch.
# Asserted by test/test_per_component_functional.jl.
function expect(p::MixturePrevision, fc::FiringChoice)
    length(fc.fired) == length(p.components) ||
        error("FiringChoice has $(length(fc.fired)) flags but mixture has $(length(p.components)) components")
    lw = p.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./= sum(w)
    sum(w[i] * _firing_branch(p.components[i], fc.fired[i] ? fc.when_fires : fc.when_not)
        for i in eachindex(w))
end

# FiringChoice is inherently mixture-level (per-component dispatch). Calling it
# on a non-mixture prevision is a usage error, not a fallback.
expect(::Prevision, ::FiringChoice) =
    error("FiringChoice requires a MixturePrevision (per-component dispatch over mixture components)")

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

# Measure-facade for the exact LinearGaussian conjugate. Without this the generic
# particle fallback (condition(::Measure, …)) would shadow the conjugate when the
# DSL passes a Measure (it speaks Measures, not Previsions). No grid/particle
# fallback for a dense-covariance state — an unrecognised kernel is an error.
function condition(m::MvGaussianMeasure, k::Kernel, observation)
    cp = maybe_conjugate(m.prevision, k)
    cp !== nothing || error(
        "condition(::MvGaussianMeasure, ...) supports only the LinearGaussian conjugate; " *
        "got likelihood_family $(typeof(k.likelihood_family)).")
    updated = update(cp, observation).prior
    return MvGaussianMeasure(m.space, updated)
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

# Group-likelihood reweight of a labelled categorical: resolve the per-component leaf family
# (a `DispatchByComponent` closure reads `p.label`), then add its per-position categorical
# log-density to each V-position's log-weight. The new `CategoricalPrevision` re-normalises —
# the component-conditional posterior P(V | label); the mixture's reweight by this component's
# document marginal likelihood is `_predictive_ll` below. `_resolve_likelihood_family` reaching
# a non-leaf (a routing family with no leaf) throws, which is correct: a labelled categorical
# is only conditioned through a per-component router.
function condition(p::LabelledCategoricalPrevision, k::Kernel, observation)
    fam = _resolve_likelihood_family(k.likelihood_family, p)
    lw = p.categorical.log_weights
    new_lw = [lw[i] + categorical_logdensity(fam, i, observation) for i in eachindex(lw)]
    LabelledCategoricalPrevision(p.label, CategoricalPrevision(new_lw))
end

function condition(p::GaussianPrevision, k::Kernel, observation)
    cp = maybe_conjugate(p, k)
    if cp !== nothing
        return update(cp, observation).prior
    end
    conditioned = condition(wrap_in_measure(p), k, observation)
    conditioned.prevision
end

# A multivariate Gaussian is conditioned only through its LinearGaussian
# conjugate (the exact Kalman update). There is no generic grid/particle
# fallback for the dense-covariance state — an unrecognised kernel is a
# construction error, surfaced with remediation (mirrors NormalGamma).
function condition(p::MvGaussianPrevision, k::Kernel, observation)
    cp = maybe_conjugate(p, k)
    cp !== nothing && return update(cp, observation).prior
    error("condition(::MvGaussianPrevision, ...) supports only the LinearGaussian " *
          "conjugate; got likelihood_family $(typeof(k.likelihood_family)). Construct " *
          "the kernel with likelihood_family = LinearGaussian(coeffs, sigma_obs).")
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

# ── TruncatedGaussianPrevision: a continuous bounded prior, integrated over [lo,hi] engine-side ──
# The truncation is the model's SUPPORT (not a grid); the engine chooses the quadrature internally.
log_density_at(p::TruncatedGaussianPrevision, x) = -0.5 * ((x - p.mu) / p.sigma)^2

# The engine's quadrature grid over the support [lo,hi] — the MIDPOINT rule (O(h²), no endpoint
# over-counting). This grid is the engine's internal computation, invisible to the declared model.
_trunc_grid(p::TruncatedGaussianPrevision, n::Int) =
    [p.lo + (k - 0.5) * (p.hi - p.lo) / n for k in 1:n]

# Conditioning is always non-conjugate (truncation breaks Normal-Normal) → quadrature over [lo,hi].
function condition(p::TruncatedGaussianPrevision, k::Kernel, observation; n::Int = 64)
    grid = _trunc_grid(p, n)
    logw = Float64[]
    for h in grid
        ll = density(k, h, observation)
        !isnan(ll) || error("density returned NaN conditioning a TruncatedGaussianPrevision")
        push!(logw, log_density_at(p, h) + ll)
    end
    QuadraturePrevision(grid, logw)
end

# Expectations over the (unconditioned) truncated prior: quadrature on [lo,hi].
function expect(p::TruncatedGaussianPrevision, f::Function; n::Int = 64)
    grid = _trunc_grid(p, n)
    lw = [log_density_at(p, x) for x in grid]
    mx = maximum(lw)
    w = exp.(lw .- mx)
    w ./= sum(w)
    sum(w[i] * f(grid[i]) for i in eachindex(grid))
end
expect(p::TruncatedGaussianPrevision, tf::TestFunction; kwargs...) = expect(p, x -> apply(tf, x); kwargs...)

# Marginal likelihood ∫ p(x)·P(obs|x) dx over [lo,hi] (the `condition` verb's log_marginal) — a
# direct quadrature, mirroring `_predictive_ll(::BetaMeasure)`; no wrap-in-measure needed.
function log_predictive(p::TruncatedGaussianPrevision, k::Kernel, obs)
    val = expect(p, h -> exp(density(k, h, obs)))
    log(max(val, 1e-300))
end
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

# Exact product predictive for a per-factor-routed kernel (Move 3). The factors
# are independent, so the joint log-predictive is the SUM of per-factor
# log-predictives; a factor whose resolved family is Flat contributes a no-op
# observation (log-predictive 0). The sum therefore collapses to exactly the
# firing factor's predictive — the structure marginal-likelihood term the
# BMA layer (an enclosing MixturePrevision) reweights by. Non-routed product
# kernels keep the generic (sampling) behaviour, unchanged.
# See docs/credence-pi-pass-2/move-3-design.md.
function _predictive_ll(m::ProductMeasure, k::Kernel, obs)
    if k.factor_selector === nothing &&
       (k.likelihood_family isa FiringByTag || k.likelihood_family isa DispatchByComponent)
        total = 0.0
        for f in m.factors
            fam = _resolve_likelihood_family(k.likelihood_family, f)
            fam isa Flat && continue
            total += _predictive_ll(f, _with_resolved_family(k, fam), obs)
        end
        return total
    end
    invoke(_predictive_ll, Tuple{Measure, Kernel, Any}, m, k, obs)
end

# ── Prevision-level _predictive_ll (Move 7: supports MixturePrevision condition) ──

_predictive_ll(p::BetaPrevision, k::Kernel, obs) =
    _predictive_ll(wrap_in_measure(p), k, obs)

_predictive_ll(p::TaggedBetaPrevision, k::Kernel, obs) =
    k.log_density(p, obs)

# Document marginal likelihood of a labelled categorical at its own label:
# log Σ_V P(V | label) · exp(per-position density). The `log_weights` are normalised at
# construction, so this is exactly log P(obs | label) — the term `condition(::MixturePrevision)`
# uses to reweight the latent (the ρ-learning across documents).
function _predictive_ll(p::LabelledCategoricalPrevision, k::Kernel, obs)
    fam = _resolve_likelihood_family(k.likelihood_family, p)
    lw = p.categorical.log_weights
    logsumexp([lw[i] + categorical_logdensity(fam, i, obs) for i in eachindex(lw)])
end

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

# A mixture's marginal likelihood = logsumexp_i(log_weight_i + _predictive_ll(comp_i, k, obs)),
# routing per-component (each component resolves its own LikelihoodFamily — e.g. a
# `DispatchByComponent` group kernel reads each `LabelledCategoricalPrevision`'s label). The
# generic `Prevision` fallback above would `wrap_in_measure` and call the kernel's
# component-agnostic `log_density`, which a per-component router leaves as a stub — so this
# specific method is what makes `condition`/`log_predictive` work on a routed mixture.
function log_predictive(p::MixturePrevision, k::Kernel, obs)
    lw = p.log_weights
    logsumexp([lw[i] + _predictive_ll(p.components[i], k, obs) for i in eachindex(lw)])
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
    if fs === nothing
        # Per-factor-routed product conditioning (Move 3 — credence-pi feature
        # brain). A FiringByTag / DispatchByComponent kernel routes the
        # observation to each factor INDEPENDENTLY by the factor's own tag: the
        # firing factor updates via its conjugate, non-firing factors resolve
        # to Flat (the registered no-op, conjugate.jl) and pass through
        # unchanged. Cells of a structure are independent parameters (a
        # product), NOT competing hypotheses (a mixture), so we return a
        # ProductMeasure — leaving an enclosing structure-BMA MixturePrevision
        # un-flattened. See docs/credence-pi-pass-2/move-3-design.md.
        if k.likelihood_family isa FiringByTag || k.likelihood_family isa DispatchByComponent
            return ProductMeasure(Measure[condition(f, k, obs) for f in m.factors])
        end
        return _condition_particle(m, k, obs; kwargs...)
    end

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

# Multivariate normal sample: μ + L·z with L the lower Cholesky factor of Σ
# (Σ = L·Lᵀ) and z ~ N(0, I). `draw` is the host-side randomness boundary.
draw(p::MvGaussianPrevision) = p.mu .+ cholesky(Symmetric(p.Sigma)).L * randn(length(p.mu))

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
# Sparse structure-BMA cell store (exact execution-layer backend)
# ================================================================
include("sparse_structure.jl")

# ================================================================
# Stdlib (extracted to stdlib.jl)
# ================================================================
include("stdlib.jl")

# ================================================================
# Structure-BMA builder + observe + readout (lifted from the credence-pi app brain —
# decouple Move 3). Compositions over the above (SparseStructurePrevision / condition /
# with_components); no new frozen type, no new axiom-constrained function.
# ================================================================
include("structure_bma.jl")
export StructureBMA, build_structure_model, build_structure_prior,
       build_structure_prior_dense, structure_observe, structure_observe_soft,
       belief_at_context, context_from_features, structure_firing_tags,
       structure_decision_kernel, reconstruct_structure_prior_from_data

include("routing.jl")
export RoutingState, EmissionBelief, LatencyBelief, route, route_eu, escalation_next,
       posterior_accuracy, route_outcome!, decode_correctness, latency_at,
       route_decide, escalate_decide, routing_belief_readout, reconstruct_latency_from_data,
       reconstruct_routing_tops_from_data, _ctx_key

end # module Ontology
