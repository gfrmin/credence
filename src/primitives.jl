"""
    primitives.jl — The three primitives. There are no others.

AXIOMS (asserted, not derived):
  A1. Preferences are complete and transitive        (Savage P1)
  A2. Beliefs are consistent with the product rule   (Cox)
  A3. No sure loss                                   (Dutch book coherence)

THEOREMS (uniquely forced by the axioms):
  → Beliefs must be probabilities                    (Cox)
  → Learning must be Bayesian conditioning           (Bayes/de Finetti)
  → Action must maximise expected utility            (Savage)

PRIMITIVES (the unique computational realisation):
  belief  — weighted hypotheses
  update  — Bayesian conditioning (reweight by likelihood)
  decide  — argmax expected utility

Everything else is a derived combinator.
"""
module Primitives

export Belief, update, decide
export weights, hypotheses, expected_utility, weighted_sum

# =================================================================
# PRIMITIVE 1: belief
# =================================================================
#
# A belief is a finite weighted set of hypotheses.
# Each hypothesis is a world-program: it carries its own
# observation model and value function.
#
# Weights are stored in log-space to avoid underflow.
# They are normalised (sum to 1 in probability space)
# as required by Cox's theorem.

struct Belief{H}
    hyps::Vector{H}
    logw::Vector{Float64}

    function Belief{H}(hyps::Vector{H}, logw::Vector{Float64}) where H
        length(hyps) == length(logw) || error("hyps and logw must have same length")
        length(hyps) > 0 || error("belief must have at least one hypothesis")
        # Normalise
        max_lw = maximum(logw)
        log_total = max_lw + log(sum(exp.(logw .- max_lw)))
        new{H}(hyps, logw .- log_total)
    end
end

# Uniform prior: the maximum entropy starting point
function Belief(hyps::Vector{H}) where H
    n = length(hyps)
    Belief{H}(hyps, fill(0.0, n))  # all equal → normalises to 1/n each
end

hypotheses(b::Belief) = b.hyps

function weights(b::Belief)
    max_lw = maximum(b.logw)
    w = exp.(b.logw .- max_lw)
    w ./ sum(w)
end

Base.length(b::Belief) = length(b.hyps)


# =================================================================
# PRIMITIVE 2: update
# =================================================================
#
# Bayesian conditioning. The ONLY learning mechanism.
#
# Each hypothesis scores the observation via its own
# observation model: ℓ(h, o) → log-probability.
# Weights are shifted by the log-likelihood.
#
# This is Bayes' rule:
#   P(h|o) ∝ P(o|h) · P(h)
# in log-space:
#   logw'[i] = logw[i] + ℓ(hyps[i], o)

function update(belief::Belief{H}, observation, likelihood) where H
    new_logw = copy(belief.logw)
    for i in eachindex(belief.hyps)
        ll = likelihood(belief.hyps[i], observation)
        !isnan(ll) || error("likelihood returned NaN for hypothesis $i")
        new_logw[i] += ll
    end
    # Check for complete extinction (all -Inf)
    if all(lw -> lw == -Inf, new_logw)
        error("observation is impossible under all hypotheses — model misspecification")
    end
    Belief{H}(belief.hyps, new_logw)
end

# Fold over a sequence of observations (same likelihood)
function update(belief::Belief, observations::AbstractVector, likelihood)
    b = belief
    for obs in observations
        b = update(b, obs, likelihood)
    end
    b
end


# =================================================================
# PRIMITIVE 3: decide
# =================================================================
#
# Expected utility maximisation. The ONLY decision mechanism.
#
# For each action a:
#   EU(a) = Σ_i w[i] · u(hyps[i], a)
#
# Returns the action with highest EU.
# This is Savage's theorem: the unique coherent decision rule.

function expected_utility(belief::Belief, action, utility)
    w = weights(belief)
    sum(w[i] * utility(belief.hyps[i], action) for i in eachindex(w))
end

function decide(belief::Belief, actions, utility)
    length(actions) > 0 || error("must have at least one action")

    best_action = first(actions)
    best_eu = expected_utility(belief, best_action, utility)

    for a in Iterators.drop(actions, 1)
        eu = expected_utility(belief, a, utility)
        if eu > best_eu
            best_eu = eu
            best_action = a
        end
    end

    (action=best_action, eu=best_eu)
end

# Full EU table (useful for inspection)
function decide_verbose(belief::Belief, actions, utility)
    eus = Dict(a => expected_utility(belief, a, utility) for a in actions)
    best = argmax(eus)
    (action=best, eu=eus[best], all_eu=eus)
end

# =================================================================
# weighted_sum: generic expectation under a belief
# =================================================================
#
# Σ_i w_i · fn(h_i)
# This is what expected_utility does with the action curried in.
# Exposed as a supporting computation form, not a new primitive.

function weighted_sum(belief::Belief, fn)
    w = weights(belief)
    sum(w[i] * fn(belief.hyps[i]) for i in eachindex(w))
end

end # module Primitives
