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

# The group-noisy-channel as a CONTINUOUS-ρ likelihood: ρ is the INTEGRATED reliability latent
# (a Beta), NOT a kernel parameter. P(reports | v, ρ) is LINEAR in ρ — `a + ρ·b` with a v-independent
# noise floor a = 1/Aᵐ and a slope b = covariate·(1{all reports == v} − 1/Aᵐ). The engine carries ρ
# analytically: a Beta prior × this linear factor stays a polynomial-in-ρ × Beta, so the V-marginal
# is an exact sum of Beta moments (no ρ-grid). This is the family the `RhoCategoricalPrevision`
# condition path dispatches on; the old `GroupNoisyChannel(covariate, ρ, A)` baked ρ into the kernel
# (the discretisation antipattern — ρ then had to be a label grid).
struct RhoGroupChannel <: LeafFamily
    covariate::Float64
    n_alternatives::Int
end

# The linear-in-ρ factor (a, b) of P(reports | v, ρ) = a + ρ·b for the candidate atom value `v`
# against an observed document `reports` (a vector of reported candidate atoms; all chunks of one
# document share the reliability ρ, so they correlate). `a` is the noise floor (v-independent);
# `b` is positive iff the document corroborates `v` in EVERY chunk.
function rho_group_channel_factor(fam::RhoGroupChannel, v, reports)
    m = length(reports)
    a = 1.0 / Float64(fam.n_alternatives)^m
    b = fam.covariate * ((all(==(v), reports) ? 1.0 : 0.0) - a)
    (a, b)
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

# Logistic-reaction likelihood: a binary reaction `react ∈ {0,1}` to a latent value `x`, under a
# choice model with a CONTINUOUS temperature τ marginalised out:
#     P(react=1 | x) = ∫ σ((sign·x − threshold)/τ) · N(τ; μ, σ) dτ over τ∈[lo,hi],  σ(z)=1/(1+e^{−z})
# τ is a continuous noise scale (a truncated Gaussian on [lo,hi]); the model declares only its
# parameters (μ, σ, lo, hi). The τ-integral has no closed form, so the engine integrates it by an
# INTERNAL quadrature (the grid is the engine's, invisible to the model — "write the model strictly,
# compute it approximately"). The latent `x` itself is then conditioned through the engine's own
# non-conjugate quadrature (`_condition_by_grid` on a GaussianPrevision); the body declares a
# continuous `gaussian` x-prior, never a grid. Replaces the old τ-grid form (a discretisation baked
# into the declared kernel).
struct LogisticReaction <: LeafFamily
    sign::Float64
    threshold::Float64
    tau_mu::Float64
    tau_sigma::Float64
    tau_lo::Float64
    tau_hi::Float64
end

# Engine-internal quadrature of τ ~ TruncatedNormal(μ, σ; [lo,hi]) for the logistic choice model:
# P(react = 1) = E_τ[ 1 / (1 + exp(-g/τ)) ] over the feature g = sign·(latent feature) - threshold.
# Midpoint rule, weights re-normalised over the truncation so the mixture is a proper marginal.
# Shared by LogisticReaction (g over a scalar latent) and MarginReaction (g over a linear functional
# of a multivariate latent) — one choice model, two feature maps.
function _tau_marginal_p1(g::Float64, tau_mu, tau_sigma, tau_lo, tau_hi; n_tau::Int = 32)
    step = (tau_hi - tau_lo) / n_tau
    p1 = 0.0
    z = 0.0
    for k in 1:n_tau
        τ = tau_lo + (k - 0.5) * step
        w = exp(-0.5 * ((τ - tau_mu) / tau_sigma)^2)
        z += w
        p1 += w / (1.0 + exp(-g / τ))
    end
    p1 / z
end

_react_logdensity(p1::Float64, react) =
    (react == 1 || react == 1.0) ? log(max(p1, 1e-300)) : log(max(1.0 - p1, 1e-300))

function logistic_reaction_logdensity(fam::LogisticReaction, x, react; n_tau::Int = 32)
    g = fam.sign * x - fam.threshold
    p1 = _tau_marginal_p1(g, fam.tau_mu, fam.tau_sigma, fam.tau_lo, fam.tau_hi; n_tau = n_tau)
    _react_logdensity(p1, react)
end

# A binary reaction to a CONTINUOUS MULTIVARIATE latent x, under the same τ-marginalised logistic
# choice model applied to a LINEAR FUNCTIONAL of the latent: margin = coeffsᵀx - offset. This is the
# kernel that couples utility latents in a joint fold (§7.1) — coeffs = eⱼ recovers a single-latent
# reaction; a multi-term coeffs is a narrative margin reaction. The engine integrates τ internally;
# the declared model carries only (coeffs, offset, sign, threshold, τ-prior) — no grid.
struct MarginReaction <: LeafFamily
    coeffs::Vector{Float64}
    offset::Float64
    sign::Float64
    threshold::Float64
    tau_mu::Float64
    tau_sigma::Float64
    tau_lo::Float64
    tau_hi::Float64
end

function margin_reaction_logdensity(fam::MarginReaction, x, react; n_tau::Int = 32)
    margin = sum(fam.coeffs[i] * x[i] for i in eachindex(fam.coeffs)) - fam.offset
    g = fam.sign * margin - fam.threshold
    p1 = _tau_marginal_p1(g, fam.tau_mu, fam.tau_sigma, fam.tau_lo, fam.tau_hi; n_tau = n_tau)
    _react_logdensity(p1, react)
end

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
# Six scalar args: sign, threshold, and the continuous-τ truncated-Gaussian (μ, σ, lo, hi). The
# engine integrates τ internally — no τ-grid in the declared model.
register_family!(Symbol("logistic-reaction"),
                 (sign, threshold, tmu, tsig, tlo, thi) ->
                     LogisticReaction(Float64(sign), Float64(threshold), Float64(tmu),
                                      Float64(tsig), Float64(tlo), Float64(thi)), 6)
# Eight args: `coeffs` (the linear-functional vector over the multivariate latent — a runtime list,
# like linear-gaussian's `xs`), offset, sign, threshold, and the continuous-τ truncated-Gaussian
# (μ, σ, lo, hi). The margin coeffsᵀx - offset couples the latents; the engine integrates τ.
register_family!(Symbol("margin-reaction"),
                 (coeffs, offset, sign, threshold, tmu, tsig, tlo, thi) ->
                     MarginReaction(collect(Float64, coeffs), Float64(offset), Float64(sign),
                                    Float64(threshold), Float64(tmu), Float64(tsig),
                                    Float64(tlo), Float64(thi)), 8)

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
