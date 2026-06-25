# stdlib.jl — Derived functions and convenience accessors
#
# These are compositions of the axiom-constrained functions with
# ordinary computation. They are convenience, not capability.
# Their interfaces are negotiable and will evolve.

# ── WeightsDomainError: informative error for continuous Previsions ──

struct WeightsDomainError <: Exception
    message::String
end

Base.showerror(io::IO, e::WeightsDomainError) = print(io, "WeightsDomainError: ", e.message)

const _WEIGHTS_DOMAIN_MSG =
    "weights is defined only for finite-support Previsions; " *
    "for continuous Previsions, use probability(p, e::Event) with a declared Event " *
    "to obtain a measure of an event's mass, or expect(p, f) for an integrated functional."

# ── Stdlib one-liners over expect ──

mean(p::Prevision) = expect(p, Identity())

function variance(p::Prevision)
    μ = mean(p)
    expect(p, CenteredSquare(μ))
end

variance(p::BetaPrevision) = p.alpha * p.beta / ((p.alpha + p.beta)^2 * (p.alpha + p.beta + 1))
variance(p::GaussianPrevision) = p.sigma^2
# Per-coordinate marginal variances diag(Σ) — exact (a Gaussian's marginals are
# read off the covariance diagonal). Cross-covariance is in `p.Sigma`.
variance(p::MvGaussianPrevision) = [p.Sigma[i, i] for i in 1:length(p.mu)]

# The exact i-th marginal of a multivariate Gaussian is the scalar Gaussian
# N(μᵢ, Σᵢᵢ) — what a consumer persisting per-feature {mu, sigma} reads back.
# Marginalisation drops cross-covariance by construction; that projection is the
# consumer's explicit choice (the joint Σ stays available on `p`).
marginal(p::MvGaussianPrevision, i::Int) =
    GaussianPrevision(p.mu[i], sqrt(p.Sigma[i, i]))

# The i-th coordinate marginal of a product-grid posterior: sum the (normalised) product weights
# over every OTHER axis, returning a scalar `QuadraturePrevision` on axis i so the usual scalar
# readouts (`mean`, `variance`, `expect`) apply. The engine integrates the other coordinates over
# ITS OWN grid — the consumer ships only the axis index, never a grid `shape` (this is what retires
# the grid-coupled `marginalise(state, shape, axis)` verb). Cross-coordinate structure stays on `p`.
function _axis_index(flat0::Int, dims::Vector{Int}, i::Int)
    stride = 1
    for j in (i + 1):length(dims)
        stride *= dims[j]
    end
    (flat0 ÷ stride) % dims[i]   # row-major: last axis fastest (matches `_mv_points`)
end

function marginal(p::MvQuadraturePrevision, i::Int)
    1 <= i <= length(p.axes) || error("marginal: axis $i out of range for $(length(p.axes)) dims")
    lw = p.log_weights
    mx = maximum(lw)
    w = exp.(lw .- mx)
    w ./= sum(w)
    dims = length.(p.axes)
    marg = zeros(length(p.axes[i]))
    for k in eachindex(w)
        marg[_axis_index(k - 1, dims, i) + 1] += w[k]
    end
    QuadraturePrevision(copy(p.axes[i]), log.(max.(marg, 1e-300)))
end

probability(p::Prevision, e::Event) = expect(p, Indicator(e))

function weights(p::CategoricalPrevision)
    lw = p.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./ sum(w)
end

# A labelled categorical's probabilities are its inner categorical's — the label is opaque.
weights(p::LabelledCategoricalPrevision) = weights(p.categorical)

function weights(p::MixturePrevision)
    lw = p.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./ sum(w)
end

weights(p::ParticlePrevision) = begin
    lw = p.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./ sum(w)
end

weights(p::QuadraturePrevision) = begin
    lw = p.log_weights
    max_lw = maximum(lw)
    w = exp.(lw .- max_lw)
    w ./ sum(w)
end

weights(p::BetaPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))
weights(p::TaggedBetaPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))
weights(p::GaussianPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))
weights(p::MvGaussianPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))
weights(p::GammaPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))
weights(p::DirichletPrevision) = p.alpha ./ sum(p.alpha)
weights(p::NormalGammaPrevision) = throw(WeightsDomainError(_WEIGHTS_DOMAIN_MSG))

function marginal(p::MixturePrevision, indices::Vector{Int})
    w = weights(p)
    marginal_logw = Float64[]
    for i in indices
        push!(marginal_logw, p.log_weights[i])
    end
    MixturePrevision([p.components[i] for i in indices], marginal_logw)
end

# NOTE: the grid-coupled `marginalise(state, shape, axis)` verb was retired when the coupled
# utility fold moved engine-side (Phase B). The consumer no longer constructs a product grid or
# ships a `shape`; it declares a continuous `truncated_mv_gaussian`, and `marginal(p, i)` reads a
# coordinate marginal off the engine's OWN product grid (see `marginal(::MvQuadraturePrevision, i)`).

"""
    with_components(p::MixturePrevision, components) -> MixturePrevision

Re-key a mixture: keep `p`'s weights verbatim, replace its per-component beliefs.
The dual of `marginal` (which selects components+weights by index) — this preserves
the weights and swaps the components, one-for-one. The structure posterior is
carried in log-space unchanged (no normalisation assumption is touched). Used to
build a per-context decision view from a structure-BMA belief.
"""
function with_components(p::MixturePrevision, components)
    length(components) == length(p.components) ||
        error("with_components: $(length(components)) components for a $(length(p.components))-way mixture")
    MixturePrevision(collect(components), copy(p.log_weights))
end

# ── Typed decision stdlib: argmax / EVPI over a functional-per-action preference ──
#
# These are the typed-Functional encoding of the ONE decision mechanism
# (argmax_a expect(belief, φ_a)); the `.bdsl` `optimise`/`value`/`voi`/`net-voi`
# (src/stdlib.bdsl) are the lambda-pref surface of the same ops. This Julia form
# exists because a typed preference keeps EU closed-form (Identity / Projection /
# LinearCombination dispatch) instead of forcing the opaque-closure quadrature a
# `(lambda (h) (pref h a))` would — the Invariant-2 reason. The skin's
# `functional_per_action` handler (apps/skin/server.jl `handle_optimise`/
# `handle_value`) mirrors this loop over a JSON spec; this is the in-process
# canonical home for in-Julia callers (the credence-pi feature brain). `actions`
# fixes a deterministic iteration order and strict `>` makes the first action win
# a tie — no host-dependent tie-break.
#
# `fpa` maps each action to a `Functional`; `eu(belief, fpa[a]) = expect(...)`.
# Asserted exact against a hand oracle in test/test_typed_decision.jl.

eu(belief, φ) = Float64(expect(belief, φ))

function optimise(belief, actions, fpa::AbstractDict)
    best_a = nothing
    best_eu = -Inf
    for a in actions
        e = eu(belief, fpa[a])
        if e > best_eu
            best_eu = e
            best_a = a
        end
    end
    best_a === nothing && error("optimise: empty action set")
    best_a
end

function value(belief, actions, fpa::AbstractDict)
    best_eu = -Inf
    for a in actions
        best_eu = max(best_eu, eu(belief, fpa[a]))
    end
    best_eu === -Inf && error("value: empty action set")
    best_eu
end

# Marginal predictive P(obs), marginalised over the belief through the kernel.
predictive_prob(belief, k::Kernel, obs) = exp(_predictive_ll(belief, k, obs))

# voi: EVPI of one observation from `k` for the decision over `actions`/`fpa`.
#   voi = Σ_o P(o)·value(condition(belief,k,o)) − value(belief)
# with P(o) the marginal predictive, renormalised over `possible_obs`. Conditioning
# routes through the axiom-constrained `condition` (Invariant 1); no belief is
# modified outside it.
function voi(belief, k::Kernel, actions, fpa::AbstractDict, possible_obs)
    base = value(belief, actions, fpa)
    preds = [predictive_prob(belief, k, o) for o in possible_obs]
    total = sum(preds)
    total <= 0.0 && return 0.0
    posterior_val = 0.0
    for (i, o) in enumerate(possible_obs)
        posterior_val += (preds[i] / total) * value(condition(belief, k, o), actions, fpa)
    end
    posterior_val - base
end

# net-voi: VOI minus the cost of observing — the `ask`-gate EU.
net_voi(belief, k::Kernel, actions, fpa::AbstractDict, possible_obs, cost) =
    voi(belief, k, actions, fpa, possible_obs) - cost

# ── decide_with_voi: the proceed/block/ask EU decision template ──
#
# One typed decision primitive over the canonical three actions (proceed/block/ask),
# parameterised by utility DATA. It is the engine-side home of the EU coefficient
# assembly that used to live in the credence-pi app brain (`decide`/`decide_multi`);
# lifting it here is what lets a non-embedding consumer drive the decision over the
# wire (a wire client ships the scalars, the engine does all the arithmetic —
# Invariant 1 by construction). The math:
#
#   tf = 1 + m   (expected calls a block prevents: this one + the look-ahead tail m)
#   EU(proceed) = 0                                                  (the baseline)
#   EU(block)   = c·tf − c·[tf+λ]·θ_a + w_time·E[time]·tf  [+ H·θ_u] (waste tail [+ harm])
#   EU(ask)     = net_voi(θ_a) − q                       (EVPI of the user resolving θ_a)
#
# expressed as `LinearCombination` Functionals maximised by the ONE canonical
# `optimise`. With no harm coordinate the block payoff is `Identity` over the approve
# belief; with one, it is `Projection`s over the `ProductMeasure` joint (approve, unsafe).
# `harm_belief === nothing` AND `expected_repeats == 0` AND `w_time == 0` reduce it
# bit-for-bit to the single-outcome myopic decision (asserted in test_decide_with_voi.jl).
# `decision_kernel` is the consumer's Bernoulli decision kernel (e.g. the structure-BMA
# `structure_decision_kernel()`); the template carries no domain specifics.

const _ID = Identity()
_lin(coeff::Float64, off::Float64) = LinearCombination(Tuple{Float64, TestFunction}[(coeff, _ID)], off)
_const(off::Float64) = LinearCombination(Tuple{Float64, TestFunction}[], off)

function decide_with_voi(approve_belief, decision_kernel::Kernel;
                         cost::Float64, aversion::Float64, interrupt_cost::Float64,
                         expected_repeats::Float64 = 0.0,
                         w_time::Float64 = 0.0, exp_time::Float64 = 0.0,
                         harm_belief = nothing, harm_cost::Float64 = 0.0)
    tf = 1.0 + expected_repeats                       # calls a block prevents: this one + the tail m
    tcost = w_time * exp_time * tf                    # wall-clock a block saves (w_time $/sec × E[time]·tf)
    # EU(ask) = net_voi − q against the SAME tail+time-aware block payoff, so resolving a
    # likely-long loop inherits the (1+m)× value. The VOI is always over the approve belief
    # alone — a harmful action is one-shot, so "should I let this through?" is the EVPI question.
    ask_block = _lin(-cost * (tf + aversion), cost * tf + tcost)
    eu_ask = net_voi(approve_belief, decision_kernel, [:proceed, :block],
                     Dict(:proceed => _const(0.0), :block => ask_block), [0, 1], interrupt_cost)
    if harm_belief === nothing
        # Single-outcome: block payoff is the Identity-based ask_block over the approve belief.
        fpa = Dict(:proceed => _const(0.0), :block => ask_block, :ask => _const(eu_ask))
        return optimise(approve_belief, [:proceed, :block, :ask], fpa)
    end
    # Multi-outcome: one EU integrating waste AND harm in one currency, over the joint
    # (approve, unsafe). The waste cutoff slides with the harm belief AND the detected tail.
    joint = ProductMeasure(Measure[wrap_in_measure(approve_belief), wrap_in_measure(harm_belief)])
    block_fn = LinearCombination(Tuple{Float64, TestFunction}[
        (-cost * (tf + aversion), Projection(1)), (harm_cost, Projection(2))], cost * tf + tcost)
    fpa = Dict(:proceed => _const(0.0), :block => block_fn, :ask => _const(eu_ask))
    optimise(joint, [:proceed, :block, :ask], fpa)
end
