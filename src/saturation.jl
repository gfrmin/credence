# saturation.jl — the belief-aware saturation signal: the residual-plateau regime belief.
# Exploration-budget Move 2. Included inside module Ontology by ontology.jl, AFTER family_bma.jl.
#
# The belief-side half of the saturation signal (the prior-side half is `compression_exhausted` in
# perturbation.jl). A 2-regime BMA over the per-step DECREMENT Δ = ℓ_prev − ℓ_now of the predictive
# log-loss (Δ > 0 ⇒ the loss fell ⇒ the belief is still improving):
#
#   :plateaued — Δ ~ N(0, σ²), σ² INFERRED  (ZeroMeanGammaPrevision). Improvement is in the noise.
#   :improving — Δ ~ N(μ, σ²), μ>0 favoured, σ² INFERRED  (NormalGammaPrevision, positive prior mean).
#
# SCALE-FREE Bayesian signal detection (the load-bearing ratified refinement, design §8): the noise
# scale is inferred in BOTH regimes, so a slow-but-CONSISTENT improver stays :improving — its drift is
# detectable above its own inferred noise, even at Δ ≈ 0.08 — never falsely plateaued by a fixed σ. The
# carried posterior weight on :plateaued is the signal (average-not-collapse). It is a SOFT prior for
# Move 3's exploration EU, never a hard gate (Q3). Reuses the Family-BMA machinery — no new mechanism;
# the BMA reweighting in condition(::MixturePrevision) does the signal detection (evidence comparison).

# Component order = candidate order in `_build_regime_model` (build_family_prior + condition preserve it).
const _REGIME_PLATEAUED = 1
const _REGIME_IMPROVING = 2

function _build_regime_model()
    obs = Euclidean(1)
    # SCALE-FREE priors (the load-bearing ratified refinement). The precision (1/σ²) prior is
    # near-Jeffreys diffuse — Gamma(0.01, 0.01) ≈ the scale-invariant reference prior p(σ²) ∝ 1/σ² — in
    # BOTH regimes, so the noise floor is the DATA's (inferred), never imposed: a slow-but-consistent
    # improver (Δ≈0.08) is detected above its own tiny noise, not read as plateaued. A Gamma(1,1) would
    # pin the scale to ~1 and false-plateau slow tasks (the exact bug the design §8 stall gate refuses).
    # The improving regime is the clean nested Bayesian t-test: μ free (μ₀=0, κ=0.1 weak ⇒ the μ prior
    # width σ²/κ is σ-relative, so detection is scale-free) vs plateaued's μ≡0 — and the BMA evidence
    # comparison in condition(::MixturePrevision) IS the signal detection. (μ₀=0 is symmetric: a
    # sustained NEGATIVE drift — the loss diverging — also reads as "not plateaued". Acceptable for a
    # SOFT signal, and arguably right: a worsening belief is not a saturated alphabet, so Move 3's
    # lookahead should be free to consider it rather than be deferred.)
    plateaued = FamilyCandidate(ZeroMeanGammaLikelihood(),
                                ZeroMeanGammaPrevision(0.01, 0.01), obs, 0.0)
    improving = FamilyCandidate(NormalGammaLikelihood(),
                                NormalGammaPrevision(0.1, 0.0, 0.01, 0.01), obs, 0.0)
    build_family_model([plateaued, improving])
end

# Built once at load (the model is stateless; the kernel's DispatchByComponent routes per regime).
const _REGIME_MODEL = _build_regime_model()

"""
    initial_learning_regime() → MixturePrevision

The uniform-prior 2-regime belief `{:plateaued, :improving}`. This is also the reset target on every
grammar change (`reset_learning_regime!`) — pre-change residuals are stale under a new alphabet.
"""
initial_learning_regime() = build_family_prior(_REGIME_MODEL)

"""
    update_learning_regime(regime, ℓ_prev, ℓ_now) → MixturePrevision

Condition the regime belief on the step's decrement `Δ = ℓ_prev − ℓ_now`, through Tier-1 `condition`
(the BMA reweighting is the signal detection). At cold start (`ℓ_prev === nothing`, fewer than two
residuals) there is no decrement, so the regime is returned unchanged.
"""
function update_learning_regime(regime::MixturePrevision, ℓ_prev, ℓ_now)
    ℓ_prev === nothing && return regime
    condition(regime, _REGIME_MODEL.kernel, ℓ_prev - ℓ_now)
end

"""
    plateau_probability(regime) → Float64

The carried posterior weight on `:plateaued` — the belief-side saturation signal. A SOFT, overridable
prior for Move 3's exploration EU, NEVER a hard gate (Q3): "still improving" is not a proof of zero VOI,
so it may never block a positive-EU explore. Read via the public `weights` accessor.
"""
plateau_probability(regime::MixturePrevision)::Float64 = weights(regime)[_REGIME_PLATEAUED]
