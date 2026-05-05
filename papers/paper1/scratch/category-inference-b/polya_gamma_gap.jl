# polya_gamma_gap.jl — scratch evidence for §3 of the B2a design doc:
# what would option (b-PG) require that the existing primitives do not
# provide?
#
# This file is INTENTIONALLY non-running. Its job is to spell out the
# additions to src/ that a Pólya-Gamma multinomial logistic implementation
# would need, in concrete signatures, so the design doc's cost claim for
# (b-PG) is not waved through.
#
# Background. The discriminative multinomial logistic
#     P(C = c_k | embedding e) = exp(β_k · e) / Σ_j exp(β_j · e)
# has no closed-form conjugate prior over the coefficients β_k under a
# Gaussian prior. Pólya-Gamma augmentation (Polson, Scott, Windle 2013)
# introduces auxiliary PG variables that, conditional on observations,
# make the conditional posterior of β Gaussian (conjugate). The full
# inference is a Gibbs sampler alternating between PG draws and β
# Gaussian updates.
#
# What's required to express this in the existing DSL primitives is laid
# out in the missing-pieces block below. See src/conjugate.jl for the
# pattern each new entry would follow.
#
# DO NOT RUN. The function bodies are stubs that error.

# ──────────────────────────────────────────────────────────────────────
# Missing piece 1: a new LeafFamily for the Pólya-Gamma binary logistic.
#
# Would land in src/kernels.jl alongside `BetaBernoulli`, `Categorical`,
# `NormalNormal`, etc. The LikelihoodFamily hierarchy is part of the
# constitutional declared-structure surface (Invariant 2): every kernel
# carries one at construction, and condition() routes on it. Adding one
# is non-cosmetic — it is a public type added to a frozen-ish hierarchy.
# ──────────────────────────────────────────────────────────────────────

# struct PolyaGammaBinary <: LeafFamily end
#
# # For multinomial-via-stick-breaking, an additional family carrying the
# # category index that this binary stage decides:
# struct PolyaGammaStick <: LeafFamily
#     stage::Int       # which stick-break stage (1 .. K-1)
# end

# ──────────────────────────────────────────────────────────────────────
# Missing piece 2: a coefficient-vector Prevision.
#
# β_k for category k is a coefficient vector ∈ R^d, with a Gaussian prior
# β_k ~ N(0, σ²_β · I). The existing GaussianPrevision is scalar
# (mu, sigma). For PG-conjugate updates we'd need either:
#
#   (a) a MultivariateGaussianPrevision (new type — ~200 lines, including
#       Cholesky-form storage to keep the conjugate update O(d²) rather
#       than O(d³) per observation),
#   (b) a ProductPrevision of d scalar GaussianPrevisions (works
#       structurally but does NOT capture the off-diagonal posterior
#       covariance that the PG-Gaussian update produces — the resulting
#       posterior is not a product of independents).
#
# (b) discards information about coefficient correlation, so the only
# faithful option is (a). MultivariateGaussianPrevision is what would
# need to land first; PG layers on top.
# ──────────────────────────────────────────────────────────────────────

# struct MultivariateGaussianPrevision <: Prevision
#     mu::Vector{Float64}
#     L::Matrix{Float64}    # lower-triangular Cholesky factor of Σ
# end

# ──────────────────────────────────────────────────────────────────────
# Missing piece 3: the conjugate-pair registry entry.
#
# A new method in src/conjugate.jl:
#
#   function maybe_conjugate(p::MultivariateGaussianPrevision, k::Kernel)
#       k.likelihood_family isa PolyaGammaBinary && return ConjugatePrevision(p, k.likelihood_family)
#       nothing
#   end
#
#   function update(cp::ConjugatePrevision{MultivariateGaussianPrevision, PolyaGammaBinary}, obs)
#       # obs is (e::Vector{Float64}, y::Bool) — the embedding and the
#       # binary outcome at this stage. The PG-augmented update:
#       #   sample ω ~ PG(1, β_old · e)
#       #   posterior precision Λ_new = Λ_old + ω · e · e'
#       #   posterior mean      μ_new  = Λ_new^{-1} · (Λ_old · μ_old + (y - 0.5) · e)
#       # Note this is *not* deterministic — the ω draw makes update()
#       # itself stochastic, breaking the closed-form contract that other
#       # ConjugatePrevision pairs hold to. So the registry entry is
#       # honest only if it returns a posterior in expectation over ω,
#       # which is not closed-form. The honest path is a Gibbs-augmented
#       # ParticlePrevision over the joint (β, ω) — but that means the
#       # registry entry above is misleading and the model lives outside
#       # the conjugate fast path entirely.
#       error("not implemented")
#   end
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# Missing piece 4: PG sampler.
#
# The Pólya-Gamma distribution PG(b, c) has no standard library
# implementation in Julia. PolyaGammaHybridSamplers.jl exists (3rd-party,
# ~500 lines, depends on Distributions.jl). Adding it is a Project.toml
# dependency change and either pulling or vendoring the algorithm. The
# master plan §5 says "DSL primitive additions — none expected"; a new
# external dependency that ships in src/ (because Prevision-level update
# methods need the sampler) is a primitive addition by another name.
# ──────────────────────────────────────────────────────────────────────

# function _sample_polya_gamma(b::Float64, c::Float64; rng=Random.default_rng())
#     error("PolyaGammaHybridSamplers.jl required (~500 lines or a dep)")
# end

# ──────────────────────────────────────────────────────────────────────
# Missing piece 5: multinomial composition.
#
# Stick-breaking turns a K-class problem into K-1 binary PG problems with
# K-1 coefficient vectors. Conceptually: each stage decides "is the class
# c_k or one of c_{k+1..K}?". The K-1 stages are conditionally
# independent given β_1..β_{K-1}, so this composes — but the composition
# would either:
#
#   (a) live as a MixturePrevision with K-1 components, each a
#       MultivariateGaussianPrevision conditioned on a `PolyaGammaStick`
#       kernel, or
#   (b) live as a new `StickBreakingPrevision` type if (a)'s arithmetic
#       gets clumsy.
#
# Either way: more new types, or more registry entries with the existing
# MixturePrevision.
# ──────────────────────────────────────────────────────────────────────

println(stderr,
    "polya_gamma_gap.jl is documentation-only; the function bodies are " *
    "intentionally stubs. See gaussian_nb_prototype.jl for the working " *
    "(b-NB) version, and the design doc §3 for the comparative analysis.")
