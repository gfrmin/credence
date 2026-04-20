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

# ── Move 2: unify Functional hierarchy onto Previsions.TestFunction ──
# The `Functional` abstract type and its concrete subtypes (`Identity`,
# `Projection`, `NestedProjection`, `Tabular`, `LinearCombination`,
# `OpaqueClosure`) were previously declared in this file. Move 2 replaces
# them with imports from `Previsions` plus `const Functional = TestFunction`.
# The re-exports below (line continuing `Functional, Identity, ...`) now
# resolve via alias; `Ontology.Identity === Previsions.Identity` is `true`,
# so `using .Previsions, .Ontology` in Credence is unambiguous.
import ..Previsions: TestFunction, Identity, Projection, NestedProjection,
                     Tabular, LinearCombination, OpaqueClosure, expect
import ..Previsions: BetaPrevision, TaggedBetaPrevision, GaussianPrevision, GammaPrevision, CategoricalPrevision, DirichletPrevision, NormalGammaPrevision, ProductPrevision, MixturePrevision

export Space, Finite, Interval, ProductSpace, Simplex, Euclidean, PositiveReals, support
export Measure, CategoricalMeasure, BetaMeasure, TaggedBetaMeasure, GaussianMeasure, GammaMeasure, ExponentialMeasure, DirichletMeasure, NormalGammaMeasure, ProductMeasure, MixtureMeasure
export Kernel, FactorSelector, kernel_source, kernel_target, kernel_params
export LikelihoodFamily, LeafFamily, PushOnly, BetaBernoulli, Flat, FiringByTag, DispatchByComponent, DepthCapExceeded
export Event, TagSet, FeatureEquals, FeatureInterval, Conjunction, Disjunction, Complement
export indicator_kernel, feature_value, BOOLEAN_SPACE
export Functional, Identity, Projection, NestedProjection, Tabular, LinearCombination, OpaqueClosure
export factor, replace_factor
export condition, expect, push_measure, density, log_predictive, log_marginal
export draw
export weights, mean, variance, log_density_at, prune, truncate, logsumexp

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
#
# Move 3: wraps CategoricalPrevision; `m.logw` forwards via the shield.
# Normalisation of logw happens inside CategoricalPrevision's constructor.
# The Vector returned by `m.logw` is by reference — shared-reference
# contract in play; see test/test_prevision_unit.jl :: test_shared_
# reference_contract (lands with MixtureMeasure).

struct CategoricalMeasure{T} <: Measure
    prevision::CategoricalPrevision
    space::Finite{T}

    function CategoricalMeasure{T}(space::Finite{T}, logw::Vector{Float64}) where T
        length(space.values) == length(logw) || error("space and weights must match")
        new{T}(CategoricalPrevision(logw), space)
    end
end

CategoricalMeasure(s::Finite{T}, logw::Vector{Float64}) where T = CategoricalMeasure{T}(s, logw)
CategoricalMeasure(s::Finite{T}) where T = CategoricalMeasure{T}(s, fill(0.0, length(s.values)))

function Base.getproperty(m::CategoricalMeasure, s::Symbol)
    if s === :logw
        return getproperty(getfield(m, :prevision), s)
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
#
# Move 3: `BetaMeasure` is now a thin wrapper around `BetaPrevision`. The
# `Base.getproperty` shield below forwards `m.alpha` and `m.beta` reads
# to the underlying prevision so existing consumer code (m.alpha /
# (m.alpha + m.beta), etc.) works unchanged. The shield MUST return
# values by direct dereference, not copies or wrappers — see the
# shared-reference contract in test_prevision_unit.jl
# (test_shared_reference_contract). Breaking that invariant silently
# corrupts the push!(state.belief.components, ...) pattern in
# apps/skin/server.jl:549,552.

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

# `getproperty` shield: see shared-reference contract in
# test/test_prevision_unit.jl (test_shared_reference_contract).
# Do NOT defensively copy.
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
#
# Move 3: TaggedBetaMeasure wraps a TaggedBetaPrevision carrying the tag
# and the underlying BetaMeasure. Consumer code `m.tag` and `m.beta`
# reads forward via the shield; `m.beta.alpha` walks through the
# TaggedBetaMeasure shield → BetaMeasure → BetaMeasure's own shield →
# BetaPrevision.alpha. Two shield hops; both transparent.

struct TaggedBetaMeasure <: Measure
    prevision::TaggedBetaPrevision
    space::Interval

    function TaggedBetaMeasure(space::Interval, tag::Int, beta::BetaMeasure)
        new(TaggedBetaPrevision(tag, beta), space)
    end
end

function Base.getproperty(m::TaggedBetaMeasure, s::Symbol)
    if s === :tag || s === :beta
        return getproperty(getfield(m, :prevision), s)
    else
        return getfield(m, s)
    end
end

Base.propertynames(::TaggedBetaMeasure) = (:tag, :beta, :space, :prevision)

mean(m::BetaMeasure) = m.alpha / (m.alpha + m.beta)
variance(m::BetaMeasure) = m.alpha * m.beta / ((m.alpha + m.beta)^2 * (m.alpha + m.beta + 1))
mean(m::TaggedBetaMeasure) = mean(m.beta)
variance(m::TaggedBetaMeasure) = variance(m.beta)

# ── Gaussian: continuous on interval ──
#
# Move 3: wraps GaussianPrevision; `m.mu` and `m.sigma` forward via
# the getproperty shield. See the shared-reference contract in
# test/test_prevision_unit.jl :: test_shared_reference_contract
# (lands with MixtureMeasure). Do NOT defensively copy.

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

# Move 3: wraps DirichletPrevision; `m.alpha` forwards via the shield
# (Vector returned by reference — shared-reference contract).

struct DirichletMeasure <: Measure
    prevision::DirichletPrevision
    space::Simplex
    categories::Finite     # the category labels the probability vectors are about

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
#
# Move 3: wraps GammaPrevision; `m.alpha` and `m.beta` forward via
# the getproperty shield. Do NOT defensively copy; see contract test.

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

# Exponential is Gamma(1, rate) — convenience constructor, not a separate type.
ExponentialMeasure(rate::Float64) = GammaMeasure(1.0, rate)

# ── Normal-Gamma: conjugate prior for Normal with unknown mean and variance ──
#
# Move 3: wraps NormalGammaPrevision; scalar hyperparameters forward via
# the shield.

struct NormalGammaMeasure <: Measure
    prevision::NormalGammaPrevision
    space::ProductSpace          # Euclidean(1) × PositiveReals

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

# ── Product: independent joint ──
#
# Move 3: wraps ProductPrevision; `m.factors` forwards via the shield
# (Vector returned by reference — shared-reference contract applies).
# Consumer code that uses `m.factors[i]` or `replace_factor(m, i, f)`
# continues to work; the latter already constructs a new ProductMeasure,
# so no in-place mutation concern. Direct `push!(m.factors, ...)` is not
# a pattern in the current codebase but would be supported by the
# shared-reference contract if it appeared.

struct ProductMeasure <: Measure
    prevision::ProductPrevision
    space::ProductSpace

    function ProductMeasure(space::ProductSpace, factors::Vector{<:Measure})
        length(space.factors) == length(factors) || error("space and factors must match")
        new(ProductPrevision(Vector{Measure}(factors)), space)
    end
end

ProductMeasure(factors::Vector{<:Measure}) =
    ProductMeasure(ProductSpace(Space[f.space for f in factors]), factors)

function Base.getproperty(m::ProductMeasure, s::Symbol)
    if s === :factors
        return getproperty(getfield(m, :prevision), s)
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

# Move 3: wraps MixturePrevision. Shield forwards :components and
# :log_weights — BOTH returned by reference. See the shared-reference
# contract in test/test_prevision_unit.jl :: test_shared_reference_contract
# and docs/posture-3/move-3-design.md §3 and R4. The contract is
# load-bearing for apps/skin/server.jl:549,552 which push! through the
# shield. Do NOT defensively copy.

struct MixtureMeasure <: Measure
    prevision::MixturePrevision
    space::Space

    function MixtureMeasure(space::Space, components::Vector{<:Measure}, log_weights::Vector{Float64})
        new(MixturePrevision(Vector{Measure}(components), log_weights), space)
    end
end

MixtureMeasure(components::Vector{<:Measure}, log_weights::Vector{Float64}) =
    MixtureMeasure(components[1].space, components, log_weights)

function Base.getproperty(m::MixtureMeasure, s::Symbol)
    if s === :components || s === :log_weights
        return getproperty(getfield(m, :prevision), s)
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

# ================================================================
# TYPE 3: Kernel
# ================================================================

struct FactorSelector
    discrete_index::Int       # which factor is the discrete selector
    active::Function          # selector_value → Vector{Int} of factors to condition
end

# ─────────────────────────────────────────────────────────────────────
# LikelihoodFamily: declared per-θ algebraic form of a kernel's likelihood
# ─────────────────────────────────────────────────────────────────────
# A Kernel's log_density closure is opaque to the type system. Dispatch
# needs to know the likelihood's algebraic form (Beta-Bernoulli, flat, …)
# to pick the right computation — closed-form conjugate, no-op for flat,
# etc. Declaring the family at construction is the same principle as
# Functional for expect.
#
# Every Kernel must declare a likelihood_family — it is a required kwarg
# at construction. Kernels used only with push/expect (not condition)
# declare PushOnly(); condition() errors loudly if called on one.
abstract type LikelihoodFamily end

# Leaf families terminate the FiringByTag / DispatchByComponent routing
# chain. Only these may appear in a condition() dispatch's final step.
abstract type LeafFamily <: LikelihoodFamily end

# p(obs|θ) = θ^obs · (1−θ)^(1−obs). Conjugate with Beta priors.
struct BetaBernoulli <: LeafFamily end

# p(obs|θ) constant in θ. The observation provides no evidence about θ.
# Bayesian posterior = prior.
struct Flat <: LeafFamily end

# Sentinel for kernels whose condition path does not dispatch on family —
# either push/expect-only, or condition through space-shape dispatch
# (e.g. Gaussian→Gaussian conjugate, Categorical→Categorical tabular).
# `condition(::TaggedBetaMeasure, …)` with a PushOnly kernel raises a
# clear error, since TaggedBetaMeasure conditioning requires a declared
# leaf or routing family.
struct PushOnly <: LikelihoodFamily end

# Fire/not-fire routing by component tag. Tags in `fires` use
# `when_fires`; all other tags use `when_not`. This is the declarative
# capture of the dominant per-component routing pattern in program-space
# mixtures — "some predicates fire on this observation, some don't".
# Prefer this over DispatchByComponent when the routing is tag-based.
#
# The branch fields are LeafFamily-typed so declared nesting is
# statically impossible. The condition() dispatch loop still guards
# against runtime misuse from DispatchByComponent returning a router.
struct FiringByTag <: LikelihoodFamily
    fires::Set{Int}
    when_fires::LeafFamily
    when_not::LeafFamily
end

# Explicit escape hatch — the LikelihoodFamily analogue of OpaqueClosure.
# Takes a `classify(m) → LikelihoodFamily` closure called at dispatch
# time. The closure must eventually unwrap to a leaf family. Returning
# another DispatchByComponent is caught by the depth cap in condition().
# Reach for this only when no declarative subtype (FiringByTag, …) fits.
struct DispatchByComponent <: LikelihoodFamily
    classify::Function  # Measure → LikelihoodFamily
end

struct Kernel
    source::Space       # H
    target::Space       # O
    generate::Function  # h → distribution spec (for push)
    log_density::Function  # (h, o) → log p(o|h) (for condition)
    factor_selector::Union{Nothing, FactorSelector}
    params::Union{Nothing, Dict{Symbol,Any}}
    likelihood_family::LikelihoodFamily
end

# Canonical constructor: likelihood_family is a required kwarg. Optional
# factor_selector / params default to nothing.
Kernel(source::Space, target::Space, gen::Function, ld::Function;
       factor_selector::Union{Nothing, FactorSelector}=nothing,
       params::Union{Nothing, Dict{Symbol,Any}}=nothing,
       likelihood_family::LikelihoodFamily) =
    Kernel(source, target, gen, ld, factor_selector, params, likelihood_family)

kernel_source(k::Kernel) = k.source
kernel_target(k::Kernel) = k.target
kernel_params(k::Kernel) = k.params

# ================================================================
# TYPE 4: Event — first-class declared structure
# ================================================================
# An Event is a declared proposition about the state of a Space.
# Events are bearers of probability in the de Finettian sense:
# P(A) is defined directly, not derived from a measure on subsets.
#
# Every Event constructor witnesses a computable `indicator_kernel`
# into a declared Boolean Space. That witness is the mechanical
# bridge between the event layer and the kernel layer: when
# `condition(m, e::Event)` is invoked (sibling form), it expands to
# `condition(m, indicator_kernel(e), true)` — Di Lavore–Román–
# Sobociński Prop. 4.9 applied in one line.
#
# Declared structure per Invariant 2: no opaque predicate closures
# at the axiom layer. Every Event carries the data its kernel needs
# in a typed field, not in a captured lambda.

abstract type Event end

"""
    TagSet(space, tags)

Event stating that a mixture component's tag lies in a declared finite
set of `Int`s. Peer of `FiringByTag` on the expect / posterior side:
declarative tag-based dispatch, no opaque closures.

Applicable to mixtures whose components carry `Int` tags (e.g.
`TaggedBetaMeasure`). The indicator kernel uses `_tag_of(component)`
to read the tag at dispatch time; add a `_tag_of` method for any new
tag-bearing Measure type.
"""
struct TagSet <: Event
    space::Space
    tags::Set{Int}
end

"""
    FeatureEquals(space, feature, value)

Deterministic equality event on a declared feature of hypotheses in
`space`. The indicator kernel queries `feature_value(h, feature)` —
add a method per hypothesis type when needed.

Valid only for discrete features; continuous equality is a
measure-zero event and must go through a disintegration primitive
(not yet implemented) — this constructor does not guard that case at
construction time; the dispatch is undefined on measure-zero events.
"""
struct FeatureEquals{T} <: Event
    space::Space
    feature::Symbol
    value::T
end

"""
    FeatureInterval(space, feature, lo, hi)

Event stating a declared continuous feature lies in the closed
interval [lo, hi]. `feature_value(h, feature)` must return a real
number for hypotheses `h` drawn from `space`.
"""
struct FeatureInterval <: Event
    space::Space
    feature::Symbol
    lo::Float64
    hi::Float64
end

"""
    Conjunction(left, right)

Event that holds iff both `left` and `right` hold. Operands must
share the same Space (checked at `indicator_kernel` construction).
"""
struct Conjunction <: Event
    left::Event
    right::Event
end

"""
    Disjunction(left, right)

Event that holds iff `left` or `right` holds.
"""
struct Disjunction <: Event
    left::Event
    right::Event
end

"""
    Complement(inner)

Event that holds iff `inner` does not.
"""
struct Complement <: Event
    inner::Event
end

# ── Boolean Space — shared target for all indicator kernels ──
const BOOLEAN_SPACE = Finite([false, true])

# ── Tag accessor: declared-structure dispatch, not an opaque closure ──
# Extend with one-line methods for any new tag-bearing Measure type.
_tag_of(m::TaggedBetaMeasure) = m.tag

# ── Feature accessor: method dispatch is the registry ──
# Domains extend with their hypothesis types.
"""
    feature_value(h, name::Symbol)

Extract the named feature from hypothesis `h`. Default: NamedTuple
index / struct field access. Override for domain-specific hypothesis
types via method dispatch.
"""
feature_value(h::NamedTuple, name::Symbol) = h[name]
feature_value(h, name::Symbol) = getfield(h, name)

# ── indicator_kernel: the mechanical bridge ──
# For each Event, produce a Kernel from the event's Space to
# BOOLEAN_SPACE whose log_density is 0 when the event holds and -Inf
# otherwise. `likelihood_family = Flat()` because the indicator does
# not depend on the Beta parameter of a tagged component — it is a
# structural predicate on the component's tag. Under Flat dispatch,
# `condition(::TaggedBetaMeasure, k, _)` returns the component
# unchanged; the effective work is done by `_predictive_ll` on the
# mixture, which propagates the log_density as a weight multiplier.

"""
    indicator_kernel(event) → Kernel

Witness that an Event is expressible as a declared indicator kernel
into BOOLEAN_SPACE. Used internally by `condition(::Measure, ::Event)`.
Also the mechanical proof that Invariant 2 is preserved: events reach
the axiom layer only through declared kernels.
"""
function indicator_kernel(e::TagSet)
    ld = (h, o) -> begin
        holds = _tag_of(h) in e.tags
        (o === true && holds) || (o === false && !holds) ? 0.0 : -Inf
    end
    gen = h -> begin
        holds = _tag_of(h) in e.tags
        CategoricalMeasure(
            BOOLEAN_SPACE,
            [holds ? -Inf : 0.0,   # logw at false
             holds ? 0.0 : -Inf],  # logw at true
        )
    end
    Kernel(e.space, BOOLEAN_SPACE, gen, ld; likelihood_family = Flat())
end

function indicator_kernel(e::FeatureEquals{T}) where T
    ld = (h, o) -> begin
        holds = feature_value(h, e.feature) == e.value
        (o === true && holds) || (o === false && !holds) ? 0.0 : -Inf
    end
    gen = h -> begin
        holds = feature_value(h, e.feature) == e.value
        CategoricalMeasure(
            BOOLEAN_SPACE,
            [holds ? -Inf : 0.0, holds ? 0.0 : -Inf],
        )
    end
    Kernel(e.space, BOOLEAN_SPACE, gen, ld; likelihood_family = Flat())
end

function indicator_kernel(e::FeatureInterval)
    ld = (h, o) -> begin
        v = feature_value(h, e.feature)
        holds = e.lo <= v <= e.hi
        (o === true && holds) || (o === false && !holds) ? 0.0 : -Inf
    end
    gen = h -> begin
        v = feature_value(h, e.feature)
        holds = e.lo <= v <= e.hi
        CategoricalMeasure(
            BOOLEAN_SPACE,
            [holds ? -Inf : 0.0, holds ? 0.0 : -Inf],
        )
    end
    Kernel(e.space, BOOLEAN_SPACE, gen, ld; likelihood_family = Flat())
end

# Boolean algebra — compose indicator log-densities. The result is a
# single Kernel whose log_density reads the two operand indicators at
# observation `true` and combines them.
function indicator_kernel(e::Conjunction)
    kl = indicator_kernel(e.left)
    kr = indicator_kernel(e.right)
    kl.source === kr.source ||
        error("Conjunction: operands must share the same Space instance")
    ld = (h, o) -> begin
        holds = kl.log_density(h, true) == 0.0 && kr.log_density(h, true) == 0.0
        (o === true && holds) || (o === false && !holds) ? 0.0 : -Inf
    end
    gen = h -> error("indicator_kernel(Conjunction).generate not used in condition")
    Kernel(kl.source, BOOLEAN_SPACE, gen, ld; likelihood_family = Flat())
end

function indicator_kernel(e::Disjunction)
    kl = indicator_kernel(e.left)
    kr = indicator_kernel(e.right)
    kl.source === kr.source ||
        error("Disjunction: operands must share the same Space instance")
    ld = (h, o) -> begin
        holds = kl.log_density(h, true) == 0.0 || kr.log_density(h, true) == 0.0
        (o === true && holds) || (o === false && !holds) ? 0.0 : -Inf
    end
    gen = h -> error("indicator_kernel(Disjunction).generate not used in condition")
    Kernel(kl.source, BOOLEAN_SPACE, gen, ld; likelihood_family = Flat())
end

function indicator_kernel(e::Complement)
    ki = indicator_kernel(e.inner)
    ld = (h, o) -> begin
        holds = ki.log_density(h, true) != 0.0  # inner did NOT hold
        (o === true && holds) || (o === false && !holds) ? 0.0 : -Inf
    end
    gen = h -> error("indicator_kernel(Complement).generate not used in condition")
    Kernel(ki.source, BOOLEAN_SPACE, gen, ld; likelihood_family = Flat())
end

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

# ================================================================
# Functional: alias for Previsions.TestFunction
# ================================================================
# The `Functional` abstract type and its concrete subtypes are declared
# in `src/prevision.jl`'s `Previsions` module (imported at the top of
# this file). `const Functional = TestFunction` preserves the legacy
# name for existing consumers; the types are identical via alias
# (`Functional === TestFunction`, `Identity === Previsions.Identity`,
# etc.). A future cleanup pass collapses the aliases; the Posture 3
# master plan scopes that as post-Move-8 work.
#
# The `expect` dispatch methods below attach to `Previsions.expect`
# (imported as `expect`); the Functional and TestFunction method
# signatures resolve to the same types.

const Functional = TestFunction

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

# NestedProjection on CategoricalMeasure: descent reached a particle-cloud
# leaf. Remaining indices select nested elements within each atom.
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

# Resolves dispatch ambiguity between expect(::MixtureMeasure, ::Functional)
# and expect(::Measure, ::OpaqueClosure).
expect(m::MixtureMeasure, o::OpaqueClosure; kwargs...) = expect(m, o.f; kwargs...)

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
    # Conjugate path: Beta-Bernoulli requires Interval → Finite with exactly 2 outcomes
    if k.source isa Interval && k.target isa Finite && length(k.target.values) == 2
        if observation == 1 || observation == 1.0 || observation == true
            BetaMeasure(m.space, m.alpha + 1.0, m.beta)
        elseif observation == 0 || observation == 0.0 || observation == false
            BetaMeasure(m.space, m.alpha, m.beta + 1.0)
        else
            _condition_by_grid(m, k, observation)
        end
    else
        _condition_by_grid(m, k, observation)
    end
end

struct DepthCapExceeded <: Exception
    msg::String
end
Base.showerror(io::IO, e::DepthCapExceeded) = print(io, "DepthCapExceeded: ", e.msg)

function condition(m::TaggedBetaMeasure, k::Kernel, observation)
    # See LikelihoodFamily declaration for the routing vocabulary.
    fam = k.likelihood_family
    fam isa PushOnly && error(
        "condition called on a push-only kernel (likelihood_family = PushOnly()). " *
        "Declare a leaf family (BetaBernoulli, Flat, or via FiringByTag/DispatchByComponent) " *
        "at Kernel construction.")
    for _ in 1:8  # depth cap — catches accidental self-referential DispatchByComponent
        if fam isa FiringByTag
            fam = m.tag in fam.fires ? fam.when_fires : fam.when_not
        elseif fam isa DispatchByComponent
            fam = fam.classify(m)
        else
            break
        end
    end
    (fam isa FiringByTag || fam isa DispatchByComponent) &&
        throw(DepthCapExceeded(
            "LikelihoodFamily unwrap did not reach a leaf within depth cap (got $(typeof(fam)))"))

    # Call for side effects: domain kernels populate per-tag caches in
    # k.params. Return value is consumed later by _predictive_ll at the
    # mixture level, not by dispatch here.
    k.log_density(m, observation)

    if fam isa BetaBernoulli
        if observation == 1 || observation == 1.0 || observation == true
            TaggedBetaMeasure(m.space, m.tag, BetaMeasure(m.beta.space, m.beta.alpha + 1.0, m.beta.beta))
        else
            TaggedBetaMeasure(m.space, m.tag, BetaMeasure(m.beta.space, m.beta.alpha, m.beta.beta + 1.0))
        end
    elseif fam isa Flat
        m
    else
        error("condition(::TaggedBetaMeasure, k, obs): unsupported likelihood_family $(typeof(fam))")
    end
end

function condition(m::GaussianMeasure, k::Kernel, observation)
    # Conjugate path: Normal-Normal requires Euclidean → Euclidean with declared σ_obs
    if k.source isa Euclidean && k.target isa Euclidean &&
       k.params !== nothing && haskey(k.params, :sigma_obs)
        sigma_obs = k.params[:sigma_obs]::Float64
        tau_prior = 1.0 / m.sigma^2
        tau_obs = 1.0 / sigma_obs^2
        tau_post = tau_prior + tau_obs
        mu_post = (tau_prior * m.mu + tau_obs * observation) / tau_post
        sigma_post = 1.0 / sqrt(tau_post)
        return GaussianMeasure(m.space, mu_post, sigma_post)
    end
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

function condition(m::NormalGammaMeasure, k::Kernel, observation)
    # Conjugate fast-path: Normal likelihood with Normal-Gamma prior
    if k.params !== nothing && haskey(k.params, :normal_gamma)
        r = Float64(observation)
        κₙ = m.κ + 1.0
        μₙ = (m.κ * m.μ + r) / κₙ
        αₙ = m.α + 0.5
        βₙ = m.β + m.κ * (r - m.μ)^2 / (2.0 * κₙ)
        return NormalGammaMeasure(m.space, κₙ, μₙ, αₙ, βₙ)
    end
    # Non-conjugate: importance sampling fallback
    condition(m::Measure, k, observation)
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
log_density_at(m::GammaMeasure, x) = (m.alpha - 1) * log(x) - m.beta * x
log_density_at(m::TaggedBetaMeasure, x) = log_density_at(m.beta, x)
log_density_at(m::GaussianMeasure, x) = -0.5 * ((x - m.mu) / m.sigma)^2
log_density_at(m::DirichletMeasure, x) = sum((m.alpha[i] - 1) * log(x[i]) for i in eachindex(m.alpha))

function log_density_at(m::NormalGammaMeasure, x)
    # x = (μ_val, σ²_val) — joint density of Normal-Gamma
    μ_val, σ² = x[1], x[2]
    σ² > 0 || return -Inf
    # InvGamma(α, β) density for σ²
    log_ig = m.α * log(m.β) - _log_gamma(m.α) - (m.α + 1.0) * log(σ²) - m.β / σ²
    # Normal(μ, σ²/κ) density for μ_val
    log_n = -0.5 * log(2π * σ² / m.κ) - m.κ * (μ_val - m.μ)^2 / (2.0 * σ²)
    log_ig + log_n
end

# Log-Gamma via Stirling approximation (stdlib only, no SpecialFunctions)
function _log_gamma(x::Float64)
    # Lanczos approximation for log(Γ(x))
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

# ── log_predictive: log P(obs | beliefs) — single observation ──

function log_predictive(m::Measure, k::Kernel, obs)
    pred = expect(m, h -> exp(density(k, h, obs)))
    log(max(pred, 1e-300))
end

function log_predictive(m::DirichletMeasure, k::Kernel, obs)
    # Conjugate fast-path: posterior predictive = α_obs / Σα
    idx = findfirst(==(obs), m.categories.values)
    idx !== nothing || error("observation $obs not in categories")
    log(m.alpha[idx] / sum(m.alpha))
end

function log_predictive(m::CategoricalMeasure, k::Kernel, obs)
    _predictive_ll(m, k, obs)
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

# ── ProductMeasure condition: structured factored update ──

function condition(m::ProductMeasure, k::Kernel, obs; kwargs...)
    fs = k.factor_selector
    fs === nothing && return _condition_product_fallback(m, k, obs; kwargs...)

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

        # Restricted kernel: fix discrete at c, project onto active factor
        restricted_ld = (h_i, o) -> begin
            full_h = Any[0.0 for _ in m.factors]
            full_h[d] = c
            full_h[ai] = h_i
            k.log_density(full_h, o)
        end
        restricted_k = Kernel(factor_ai.space, k.target,
            _ -> error("generate not used in condition"), restricted_ld;
            likelihood_family = k.likelihood_family)

        # Condition — hits conjugate fast-paths (e.g. Beta + Bernoulli)
        conditioned = condition(factor_ai, restricted_k, obs)

        # Predictive likelihood for weighting
        pred_ll = _predictive_ll(factor_ai, restricted_k, obs)

        # New ProductMeasure: discrete → point mass, active → conditioned, rest unchanged
        new_factors = Measure[f for f in m.factors]
        new_factors[d] = CategoricalMeasure(Finite([c]))
        new_factors[ai] = conditioned

        push!(new_components, ProductMeasure(new_factors))
        push!(new_log_wts, log(max(cat_w[ci], 1e-300)) + pred_ll)
    end

    MixtureMeasure(m.space, new_components, new_log_wts)
end

function _condition_product_fallback(m::ProductMeasure, k::Kernel, obs; n_particles::Int=1000)
    samples = [draw(m) for _ in 1:n_particles]
    log_weights = Float64[k.log_density(s, obs) for s in samples]
    CategoricalMeasure(Finite(samples), log_weights)
end

# ── General condition fallback: importance sampling ──

function condition(m::Measure, k::Kernel, obs; n_particles::Int=1000)
    samples = [draw(m) for _ in 1:n_particles]
    log_weights = Float64[k.log_density(s, obs) for s in samples]
    CategoricalMeasure(Finite(samples), log_weights)
end

# ── condition — sibling form taking an Event directly ──
#
# Provably equivalent to conditioning on `indicator_kernel(e)` at
# observation `true`, for deterministic events. Di Lavore–Román–
# Sobociński Proposition 4.9: Pearl and Jeffrey coincide on
# deterministic observations, both equal Bayesian inversion at the
# observed point. The parametric form `condition(m, k, obs)` stays
# the primary signature for existing consumers; this form is the
# natural idiom for new code whose conditioning object is an event
# rather than a kernel-observation pair.
function condition(m::Measure, e::Event)
    condition(m, indicator_kernel(e), true)
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

function push_measure(m::MixtureMeasure, k::Kernel)
    # Pushforward of a mixture: Σ_i w_i · push(component_i, kernel)
    # For finite targets, integrate kernel density over each component via expect.
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

draw(m::TaggedBetaMeasure) = draw(m.beta)

draw(m::GammaMeasure) = _draw_gamma(m.alpha) / m.beta

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

function draw(m::NormalGammaMeasure)
    # Sample σ² ~ InvGamma(α, β) = 1/Gamma(α, 1/β) * β  →  β / Gamma(α)
    # InvGamma: if X ~ Gamma(α, 1), then β/X ~ InvGamma(α, β)
    g = _draw_gamma(m.α)
    σ² = m.β / g
    # Sample μ ~ Normal(μ, σ²/κ)
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
# log_marginal — Dirichlet-Multinomial marginal likelihood
# ================================================================
# log P(data | Dir(α)) = log B(α + n) / B(α)
# O(K) from sufficient statistics — important for structure learning
# which evaluates many candidate structures.

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

end # module Ontology
