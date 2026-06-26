# family_bma.jl — Family-BMA: a posterior over likelihood families for one leaf. collapse-towers
# Phase 2. The complexity log-prior (`complexity.jl`) pointed at a *family* index; the existing
# chain-rule reweighting in `condition(MixturePrevision)` does the Occam averaging. No new frozen
# type, no new axiom-constrained function — a mixture of per-family conjugate priors + a
# `DispatchByComponent` kernel, conditioned through the canalised path (the structural twin of
# structure-BMA, which ranges over parent-sets instead of families).
#
# The agent AVERAGES over the family posterior (`argmax_a expect(mixture, u_a)`); it never selects a
# family (`average-not-collapse`).

"""
    FamilyCandidate(family, prior, obs_space, L)

One candidate likelihood family for a Family-BMA leaf:
- `family::LikelihoodFamily` — the leaf family (e.g. `NormalNormal`, `NormalGammaLikelihood`).
- `prior::Prevision` — its honest within-family conjugate prior, so cross-family marginal
  likelihoods are commensurable. Candidates must have **distinct prevision types** (`classify`
  dispatches by type; a same-type pair needs a labelled-component wrapper — deferred).
- `obs_space::Space` — the observation space this family scores, declared honestly. All candidates
  must share it (the commensurability requirement).
- `L::Float64` — the family's **specification length**: the bits to *name and define the family
  itself*, fed to `complexity_logprior`. It is **NOT** a proxy for parameter count — the marginal
  likelihood (the evidence integral) already prices parameter dimension via Occam, so an `L` that
  tracks parameter count double-counts flexibility.
"""
struct FamilyCandidate
    family::LikelihoodFamily
    prior::Prevision
    obs_space::Space
    L::Float64
end

"""
    FamilyBMA — the Family-BMA descriptor (structural-analysis representation; carries no beliefs).
    `kernel` is the derived per-component routing kernel.
"""
struct FamilyBMA
    candidates::Vector{FamilyCandidate}
    obs_space::Space
    λ_family::Float64
    kernel::Kernel
end

# The error-stubs are the SINGLE backstop securing two decisions (Phase 2 design §5 Q2 + Q4): the
# nominal-source soundness (the non-conjugate `condition` fallback) and the predictive deferral (the
# unresolved family-mixture `log_predictive` fallback). They must `error` — loud, never a warning or
# a NaN-returning stub.
_family_stub_generate(h) = error("family-BMA kernel: `generate` is unused — the kernel routes per " *
                                 "component via DispatchByComponent; reaching it is a bug.")
_family_stub_logdensity(h, o) = error("family-BMA kernel: `log_density` is unused on the live " *
                                      "(conjugate) path; reaching it means an unresolved family " *
                                      "kernel hit a non-conjugate / mixture-predictive path — a bug.")

# The per-component routing kernel: a `DispatchByComponent` that maps each component to its candidate
# family by PREVISION TYPE. The source is nominal — the candidate families have different latent
# spaces, and the conjugate `condition` reads only `likelihood_family` + obs, never `k.source`
# (`ontology.jl:1292-1296`). `target` = the shared obs space.
function _family_kernel(candidates::Vector{FamilyCandidate}, obs_space::Space)
    type_to_family = Dict{DataType, LikelihoodFamily}(typeof(c.prior) => c.family for c in candidates)
    classify(comp) = get(type_to_family, typeof(comp)) do
        error("family-BMA classify: component $(typeof(comp)) matches no candidate family")
    end
    Kernel(obs_space, obs_space, _family_stub_generate, _family_stub_logdensity;
           likelihood_family = DispatchByComponent(classify))
end

"""
    build_family_model(candidates; λ_family = 0.0) -> FamilyBMA

Validate the candidate set and build the per-component routing kernel. Three loud construction
guards, no fallback: **commensurability** (all candidates share an obs space), **distinct prevision
types** (so `classify`-by-type is unambiguous), and **conjugate-recognised** (each `(prior, family)`
is a registered conjugate pair — fails here, not on the first observation).
"""
function build_family_model(candidates::Vector{FamilyCandidate}; λ_family::Float64 = 0.0)
    isempty(candidates) && error("build_family_model: need ≥ 1 candidate family")
    obs_space = candidates[1].obs_space
    for c in candidates
        c.obs_space == obs_space ||
            error("family-BMA commensurability: all candidates must score the same observation " *
                  "space; got $(c.obs_space) vs $(obs_space).")
    end
    types = DataType[typeof(c.prior) for c in candidates]
    length(unique(types)) == length(types) ||
        error("family-BMA distinct-type: candidate priors must have distinct prevision types " *
              "(classify dispatches by type); got $(types). A genuinely same-type family pair " *
              "needs a labelled-component wrapper (deferred).")
    for c in candidates
        probe = Kernel(obs_space, obs_space, _family_stub_generate, _family_stub_logdensity;
                       likelihood_family = c.family)
        maybe_conjugate(c.prior, probe) !== nothing ||
            error("family-BMA conjugate-recognised: ($(typeof(c.prior)), $(typeof(c.family))) is " *
                  "not a registered conjugate pair (maybe_conjugate returned nothing).")
    end
    FamilyBMA(candidates, obs_space, λ_family, _family_kernel(candidates, obs_space))
end

"""
    build_family_prior(model) -> MixturePrevision

The Family-BMA prior: a mixture over the candidate families, each component its honest within-family
conjugate prior, weighted by `complexity_logprior(L; λ=λ_family)` on the family index (uniform at the
default `λ_family=0`, where the marginal likelihood does all the Occam work). Heterogeneous
components (distinct families) ⇒ a typed `Prevision[]` literal — the sanctioned form of
`untyped-mixture-construction` (genuine heterogeneity, not a lazy `Any[]`).
"""
function build_family_prior(model::FamilyBMA)
    comps = Prevision[c.prior for c in model.candidates]
    lw = Float64[complexity_logprior(c.L; λ = model.λ_family) for c in model.candidates]
    MixturePrevision(comps, lw)
end

"""
    family_observe(model, mixture, obs) -> MixturePrevision

Bayesian update of the family posterior on one observation — a single `condition` through the
canalised path. Each component resolves its own family (`DispatchByComponent`) and updates via its
conjugate; the family weights are reweighted by each family's marginal likelihood (the Occam work).
"""
family_observe(model::FamilyBMA, mixture::MixturePrevision, obs) =
    condition(mixture, model.kernel, obs)

"""
    family_posterior(mixture) -> Vector{Float64}

The readout: the **full** posterior over families. `average-not-collapse` — callers MARGINALISE over
this (`argmax_a expect(mixture, u_a)`); they never `argmax_m` it to pick a family. There is,
deliberately, no `select_family`.
"""
family_posterior(mixture::MixturePrevision) = weights(mixture)
