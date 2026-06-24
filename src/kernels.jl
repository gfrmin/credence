# kernels.jl — Kernel type + LikelihoodFamily hierarchy
#
# A Kernel is a conditional distribution between two spaces.
# The type is frozen; the LikelihoodFamily roster is not.
#
# Included inside module Ontology by ontology.jl.

struct FactorSelector
    discrete_index::Int       # which factor is the discrete selector
    active::Function          # selector_value → Vector{Int} of factors to condition
end

# ─────────────────────────────────────────────────────────────────────
# LikelihoodFamily: declared per-θ algebraic form of a kernel's likelihood
# ─────────────────────────────────────────────────────────────────────
abstract type LikelihoodFamily end
abstract type LeafFamily <: LikelihoodFamily end

struct BetaBernoulli <: LeafFamily end
# Fractional / soft-evidence Bernoulli. The observation is a pair
# `(outcome, weight)` with `weight ∈ [0,1]` (e.g. a category posterior
# π_c); the conjugate update credits `weight` pseudo-counts:
# `α += weight·outcome, β += weight·(1-outcome)`. The strict unit-count
# `BetaBernoulli` above is left untouched (it still errors on non-{0,1}).
# Rationale + worked example: docs/paper1/move-2c-design.md.
struct WeightedBernoulli <: LeafFamily end
# Virtual / soft EVIDENCE on a latent Bernoulli. The observation is a pair
# `(r, w)` of likelihoods `r = P(evidence | θ-event true)`,
# `w = P(evidence | θ-event false)` — Pearl's λ-message, not an outcome. The
# exact posterior under `L(θ) = r·θ + w·(1−θ)` is a 2-component Beta mixture;
# the conjugate `update` here is its MEAN-EXACT ADF collapse:
#   π = r·θ̄ / (r·θ̄ + w·(1−θ̄)),   α += π,  β += (1−π)    (θ̄ = α/(α+β))
# which reproduces the exact posterior mean (α+π)/(α+β+1) — so any decision
# reading E[θ] is exact; only the belief's variance is approximated. The
# predictive marginal `r·E[θ] + w·(1−E[θ])` (the BMA structure-reweight term)
# is supplied by the kernel's `log_density`. Reduces EXACTLY to BetaBernoulli
# at `(r,w) = (1,0)` (a hard 1) and `(0,1)` (a hard 0). Distinct from
# WeightedBernoulli, which tempers a KNOWN outcome by a weight; SoftBernoulli
# carries the likelihood of an indirect signal and decodes per-cell.
struct SoftBernoulli <: LeafFamily end
struct Flat <: LeafFamily end
struct PushOnly <: LikelihoodFamily end

struct NormalNormal <: LeafFamily
    sigma_obs::Float64
end

# Linear-Gaussian (Bayesian-linear-regression) likelihood: one scalar
# observation `y ~ N(aᵀw, σ²)` of a linear combination of a multivariate
# Gaussian state `w`. `coeffs` is the per-observation coefficient vector `a`
# (the article's feature values); `sigma_obs` is σ. Conjugate to a dense
# `MvGaussianPrevision` prior via the exact Kalman measurement update
# (src/conjugate.jl). Unlike NormalNormal's single scalar, the coefficient
# vector is observation-specific and variable-length, so this family is built
# per-observation (skin kernel spec / in-Julia), NOT via the fixed-arity
# `FAMILY_REGISTRY`. See docs/linear-gaussian-conjugate.md.
struct LinearGaussian <: LeafFamily
    coeffs::Vector{Float64}
    sigma_obs::Float64
end

struct Categorical{T} <: LeafFamily
    categories::Finite{T}
end

struct NormalGammaLikelihood <: LeafFamily end
struct Exponential <: LeafFamily end
# Count likelihood: obs ~ Poisson(λ), conjugate to a Gamma(α, β) prior on the
# rate λ. One observation t updates Gamma(α, β) → Gamma(α + t, β + 1), so after
# n counts with sum S the posterior is Gamma(α + S, β + n) and its mean
# (α + S)/(β + n) = E[λ] is the posterior expected count — read through
# `expect(::GammaPrevision, Identity)`. The conjugate dual of Exponential
# (which models positive-real durations); Poisson models the integer counts.
struct Poisson <: LeafFamily end

# Group-noisy-channel likelihood: a DOCUMENT of `m` correlated chunk-extractions of one
# source, read as evidence about a categorical hypothesis `V` (which candidate is the truth,
# or NONE). The m chunks share the document's content via a binary document latent
# `D_d ∈ {reliable, noise}` with `P(reliable) = r_d = ρ·covariate`; given `D_d` the chunks are
# conditionally independent. Marginalising `D_d` analytically (no hypothesis growth) gives the
# exact per-document group-likelihood
#     P(reports | V=j) = r_d·1{all m chunks reported j} + (1−r_d)·(1/A)^m
# — a reliable document reports the truth in every chunk; a noise document reports each chunk
# uniformly over the `A` alternatives. This is the EXACT correlated-evidence model the §4.2
# "tempering" (raising each chunk's log-likelihood to a power) crudely approximated:
# corroborating chunks of ONE document move the posterior LESS than the same number of
# INDEPENDENT documents (likelihood-ratio `1 + r_d·Aᵐ/(1−r_d)` ≪ the independent
# `[1 + r_d·A/(1−r_d)]ᵐ`), with no β-knob. At `m=1` it reduces bit-for-bit to the single-obs
# noisy channel (match `r_d+(1−r_d)/A`, miss `(1−r_d)/A`). NONE and any non-reported candidate
# are explained only by the noise channel (all_match false ⇒ reliable term vanishes),
# recovering the "the truth is not among the retrieved ⇒ every report is a misread" semantics.
#
# `covariate` is the document's ρ-free reliability factor (authority·subject·time); `rho` is
# the shared extractor reliability ρ (a scalar here; carried + marginalised as a latent across
# documents in the ρ-mixture — the ρ-latent commit); `n_alternatives` is A. The chunk reports
# are the OBSERVATION (a vector of reported candidate atoms), not family data.
struct GroupNoisyChannel <: LeafFamily
    covariate::Float64
    rho::Float64
    n_alternatives::Int
end

# Per-hypothesis group log-likelihood: hypothesis `v` (a candidate atom, or the NONE atom)
# against an observed document `reports` (a vector of reported candidate atoms). This is the
# `log_density` a group-noisy-channel Kernel forwards to; `condition(::CategoricalMeasure, …)`
# sums it into each candidate's log-weight.
function group_noisy_channel_logdensity(fam::GroupNoisyChannel, v, reports)
    r_d = fam.rho * fam.covariate
    (0.0 <= r_d <= 1.0) ||
        error("group-noisy-channel: r_d = ρ·covariate = $r_d ∉ [0,1] (ρ=$(fam.rho), " *
              "covariate=$(fam.covariate)) — the consumer must keep ρ·authority·subject·time ≤ 1")
    A = fam.n_alternatives
    m = length(reports)
    reliable = all(==(v), reports) ? r_d : 0.0       # a reliable doc reports the truth in every chunk
    noise = (1.0 - r_d) / Float64(A)^m               # a noise doc reports each chunk uniform over A
    log(max(reliable + noise, 1e-300))               # m=1 ⇒ (1−r_d)/A, the single-obs miss exactly
end

# Per-position categorical log-density of a LEAF FAMILY — the routing target a
# `LabelledCategoricalPrevision` calls after `_resolve_likelihood_family` picks the
# per-component family. Position `i` is the 1-based hypothesis index; `obs` is the
# observation. Keeps `LabelledCategoricalPrevision` domain-agnostic: it dispatches here on
# whatever leaf the routing closure returns, and each family that emits a categorical
# likelihood (today: GroupNoisyChannel) provides one method.
categorical_logdensity(fam::GroupNoisyChannel, i::Int, obs) =
    group_noisy_channel_logdensity(fam, i, obs)

# Fail loud (with remediation) if a routing closure yields a family with no per-position
# categorical density — a misconfigured kernel, caught at condition rather than mis-routed.
categorical_logdensity(fam::LikelihoodFamily, i::Int, obs) = error(
    "LabelledCategoricalPrevision needs a per-component family with a categorical per-position " *
    "density (e.g. GroupNoisyChannel); got $(typeof(fam)). Check the routing kernel's " *
    "DispatchByComponent closure.")

struct FiringByTag <: LikelihoodFamily
    fires::Set{Int}
    when_fires::LeafFamily
    when_not::LeafFamily
end

struct DispatchByComponent <: LikelihoodFamily
    classify::Function
end

# ─────────────────────────────────────────────────────────────────────
# Family registry: the keyword surface the BDSL `:family` reflects.
# ─────────────────────────────────────────────────────────────────────
# Maps a BDSL family keyword → (constructor, arity), where `arity` is the
# number of trailing numeric arguments the `(kernel … :family <kw> <args…>)`
# form consumes. Families self-register below, so the BDSL surface tracks the
# roster automatically — adding a family needs no edit to the eval parser.
# The constructor receives exactly `arity` Float64 args (splatted by eval).
const FAMILY_REGISTRY = Dict{Symbol, Tuple{Function, Int}}()

function register_family!(keyword::Symbol, constructor::Function, arity::Int = 0)
    FAMILY_REGISTRY[keyword] = (constructor, arity)
    return nothing
end

register_family!(:bernoulli, () -> BetaBernoulli(), 0)
register_family!(:flat,      () -> Flat(), 0)
register_family!(:soft,      () -> SoftBernoulli(), 0)
register_family!(:weighted,  () -> WeightedBernoulli(), 0)
register_family!(:normal,    (sigma) -> NormalNormal(Float64(sigma)), 1)
# Two args: `xs` (the per-observation coefficient vector — a runtime list, hence
# the `:family` arg-evaluation generalisation) and `sigma` (observation noise).
register_family!(Symbol("linear-gaussian"),
                 (xs, sigma) -> LinearGaussian(collect(Float64, xs), Float64(sigma)), 2)
# Three scalar args: the document's ρ-free covariate, the extractor reliability ρ, and the
# alternative count A. The chunk reports are the observation, supplied at `condition` time.
register_family!(Symbol("group-noisy-channel"),
                 (covariate, rho, n_alt) ->
                     GroupNoisyChannel(Float64(covariate), Float64(rho), Int(round(n_alt))), 3)

struct Kernel
    source::Space
    target::Space
    generate::Function
    log_density::Function
    factor_selector::Union{Nothing, FactorSelector}
    params::Union{Nothing, Dict{Symbol,Any}}
    likelihood_family::LikelihoodFamily
end

Kernel(source::Space, target::Space, gen::Function, ld::Function;
       factor_selector::Union{Nothing, FactorSelector}=nothing,
       params::Union{Nothing, Dict{Symbol,Any}}=nothing,
       likelihood_family::LikelihoodFamily) =
    Kernel(source, target, gen, ld, factor_selector, params, likelihood_family)

kernel_source(k::Kernel) = k.source
kernel_target(k::Kernel) = k.target
kernel_params(k::Kernel) = k.params

density(k::Kernel, h, o) = k.log_density(h, o)

# ── LikelihoodFamily routing helpers ──

struct DepthCapExceeded <: Exception
    msg::String
end
Base.showerror(io::IO, e::DepthCapExceeded) = print(io, "DepthCapExceeded: ", e.msg)

function _resolve_likelihood_family(fam::LikelihoodFamily, component)
    fam isa PushOnly && error(
        "condition called on a push-only kernel (likelihood_family = PushOnly()). " *
        "Declare a leaf family (BetaBernoulli, Flat, or via FiringByTag/DispatchByComponent) " *
        "at Kernel construction.")
    for _ in 1:8
        if fam isa FiringByTag
            fam = component.tag in fam.fires ? fam.when_fires : fam.when_not
        elseif fam isa DispatchByComponent
            fam = fam.classify(component)
        else
            break
        end
    end
    (fam isa FiringByTag || fam isa DispatchByComponent) &&
        throw(DepthCapExceeded(
            "LikelihoodFamily unwrap did not reach a leaf within depth cap (got $(typeof(fam)))"))
    fam
end

function _with_resolved_family(k::Kernel, fam::LikelihoodFamily)
    Kernel(k.source, k.target, k.generate, k.log_density;
           factor_selector = k.factor_selector,
           params = k.params,
           likelihood_family = fam)
end
