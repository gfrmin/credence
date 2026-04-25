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
struct Flat <: LeafFamily end
struct PushOnly <: LikelihoodFamily end

struct NormalNormal <: LeafFamily
    sigma_obs::Float64
end

struct Categorical{T} <: LeafFamily
    categories::Finite{T}
end

struct NormalGammaLikelihood <: LeafFamily end
struct Exponential <: LeafFamily end

struct FiringByTag <: LikelihoodFamily
    fires::Set{Int}
    when_fires::LeafFamily
    when_not::LeafFamily
end

struct DispatchByComponent <: LikelihoodFamily
    classify::Function
end

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
