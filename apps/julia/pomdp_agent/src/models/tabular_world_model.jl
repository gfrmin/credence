# Role: brain-side application
"""
    TabularWorldModel

A Bayesian world model for discrete state-action spaces using:
- Dirichlet-Categorical for transition probabilities
- Normal-Gamma conjugate prior for reward distributions (via credence's NormalGammaMeasure)

Supports Thompson Sampling via sampling from posteriors.
"""

using Distributions

"""
    NormalGammaPosterior

Type alias for credence's NormalGammaMeasure. All conjugate updates, sampling,
and mean computation go through credence's condition(), draw(), and mean().
"""
const NormalGammaPosterior = NormalGammaMeasure

# Normal-Gamma kernel used for conditioning reward posteriors.
# Built once, shared across all update sites.
const _NORMAL_GAMMA_KERNEL = Kernel(
    ProductSpace(Space[Euclidean(1), PositiveReals()]),
    Euclidean(1),
    h -> error("generate not used"),
    (h, o) -> -0.5 * log(2π * h[2]) - (o - h[1])^2 / (2.0 * h[2]);
    params = Dict{Symbol,Any}(:normal_gamma => true),
    likelihood_family = PushOnly())

"""
    TabularWorldModel

Maintains Bayesian beliefs over transition and reward dynamics.

Optionally maintains feature-level reward posteriors alongside the tabular model.
When a `feature_extractor` is provided, reward predictions combine tabular and
feature posteriors via precision-weighted averaging — features with more observations
(higher κ) contribute more. This enables generalisation: seeing reward for "eat" in
Kitchen informs "eat" in Pantry via shared (:action_type, :interact) features.
"""
mutable struct TabularWorldModel <: WorldModel
    # Transition model: (state, action) → Dirichlet over next states
    # Stored as counts: transition_counts[(s,a)][s'] = count
    transition_counts::Dict{Tuple{Any,Any}, Dict{Any, Float64}}

    # Reward model: (state, action) → NormalGammaMeasure posterior
    reward_posterior::Dict{Tuple{Any,Any}, NormalGammaMeasure}

    # Feature-level reward posteriors: (feature_key, action) → NormalGammaMeasure
    # Enables generalisation across states sharing features (location, action type, etc.)
    feature_reward_posterior::Dict{Tuple{Any,Any}, NormalGammaMeasure}

    # Feature extractor: abstract_state → Vector of feature keys
    # nothing means pure tabular (existing behaviour)
    feature_extractor::Union{Nothing, Function}

    # Prior hyperparameters
    transition_prior::Float64  # Dirichlet concentration (pseudocount per state)
    reward_prior::NormalGammaMeasure  # Prior for unseen (s,a) pairs

    # Known states (discovered during interaction)
    known_states::Set{Any}

    # Confirmed self-loops: (state, action) pairs that deterministically produce no change
    confirmed_selfloops::Set{Tuple{Any,Any}}
end

"""
    TabularWorldModel(; transition_prior=0.1, reward_prior_mean=0.0, reward_prior_variance=1.0, feature_extractor=nothing)

Create a new tabular world model with specified priors.

The reward prior is a Normal-Gamma with:
- κ₀ = 1.0 (one pseudo-observation worth of mean certainty)
- μ₀ = reward_prior_mean
- α₀ = 1.0 (weakly informative variance prior)
- β₀ = reward_prior_variance (prior scale for variance)
"""
function TabularWorldModel(;
    transition_prior::Float64 = 0.1,
    reward_prior_mean::Float64 = 0.0,
    reward_prior_variance::Float64 = 1.0,
    feature_extractor::Union{Nothing, Function} = nothing
)
    prior = NormalGammaMeasure(1.0, reward_prior_mean, 1.0, reward_prior_variance)
    return TabularWorldModel(
        Dict{Tuple{Any,Any}, Dict{Any, Float64}}(),
        Dict{Tuple{Any,Any}, NormalGammaMeasure}(),
        Dict{Tuple{Any,Any}, NormalGammaMeasure}(),
        feature_extractor,
        transition_prior,
        prior,
        Set{Any}(),
        Set{Tuple{Any,Any}}()
    )
end

"""
    mark_selfloop!(model::TabularWorldModel, s, a)

Mark a (state, action) pair as a confirmed self-loop (deterministic no-op).
"""
function mark_selfloop!(model::TabularWorldModel, s, a)
    push!(model.confirmed_selfloops, (s, a))
end

"""
    is_selfloop(model::TabularWorldModel, s, a) → Bool

Check whether a (state, action) pair is a confirmed self-loop.
"""
function is_selfloop(model::TabularWorldModel, s, a)
    return (s, a) in model.confirmed_selfloops
end

"""
    action_features(action) → Vector

Extract feature keys from an action string for the factored reward model.
Returns a vector of (feature_type, feature_value) tuples.
"""
function action_features(action)
    a = lowercase(string(action))
    if startswith(a, "go ") || a in ["north","south","east","west","up","down","ne","nw","se","sw","n","s","e","w"]
        return [(:action_type, :movement)]
    elseif startswith(a, "take ") || startswith(a, "get ") || startswith(a, "pick ")
        return [(:action_type, :take)]
    elseif startswith(a, "drop ") || startswith(a, "put ")
        return [(:action_type, :drop)]
    elseif startswith(a, "look") || startswith(a, "examine") || startswith(a, "x ") || a == "x" || startswith(a, "read ")
        return [(:action_type, :examine)]
    elseif startswith(a, "open ") || startswith(a, "close ") || startswith(a, "unlock ")
        return [(:action_type, :manipulate)]
    else
        return [(:action_type, :interact)]
    end
end

"""
    update!(model::TabularWorldModel, s, a, r, s′)

Update the model with an observed transition.
Uses credence's condition() for Normal-Gamma conjugate update.
"""
function update!(model::TabularWorldModel, s, a, r, s′)
    key = (s, a)

    # Update transition counts
    if !haskey(model.transition_counts, key)
        model.transition_counts[key] = Dict{Any, Float64}()
    end
    model.transition_counts[key][s′] = get(model.transition_counts[key], s′, model.transition_prior) + 1.0

    # Update reward posterior via credence's condition()
    p = get(model.reward_posterior, key, model.reward_prior)
    model.reward_posterior[key] = condition(p, _NORMAL_GAMMA_KERNEL, r)

    # Feature-level reward updates
    if !isnothing(model.feature_extractor)
        for fkey in model.feature_extractor(s)
            feature_key = (fkey, a)
            fp = get(model.feature_reward_posterior, feature_key, model.reward_prior)
            model.feature_reward_posterior[feature_key] = condition(fp, _NORMAL_GAMMA_KERNEL, r)
        end
        for afkey in action_features(a)
            feature_key = (afkey, :any)
            fp = get(model.feature_reward_posterior, feature_key, model.reward_prior)
            model.feature_reward_posterior[feature_key] = condition(fp, _NORMAL_GAMMA_KERNEL, r)
        end
    end

    # Track known states
    push!(model.known_states, s)
    push!(model.known_states, s′)

    return nothing
end

"""
    transition_dist(model::TabularWorldModel, s, a) → Dict{state, probability}

Return the posterior predictive distribution over next states.
"""
function transition_dist(model::TabularWorldModel, s, a)
    key = (s, a)

    if !haskey(model.transition_counts, key)
        # No observations — return uniform over known states
        n_states = max(1, length(model.known_states))
        return Dict(state => 1.0 / n_states for state in model.known_states)
    end

    counts = model.transition_counts[key]
    total = sum(values(counts))

    return Dict(state => count / total for (state, count) in counts)
end

"""
    collect_posteriors(model::TabularWorldModel, s, a) → Vector{NormalGammaMeasure}

Collect all relevant posteriors for a (state, action) pair: tabular + features.
"""
function collect_posteriors(model::TabularWorldModel, s, a)
    tab = get(model.reward_posterior, (s, a), model.reward_prior)
    if isnothing(model.feature_extractor)
        return [tab]
    end

    posteriors = NormalGammaMeasure[tab]
    for fkey in model.feature_extractor(s)
        fk = (fkey, a)
        if haskey(model.feature_reward_posterior, fk)
            push!(posteriors, model.feature_reward_posterior[fk])
        end
    end
    for afkey in action_features(a)
        fk = (afkey, :any)
        if haskey(model.feature_reward_posterior, fk)
            push!(posteriors, model.feature_reward_posterior[fk])
        end
    end
    return posteriors
end

"""
    combine_posteriors(posteriors::Vector{NormalGammaMeasure}) → NormalGammaMeasure

Precision-weighted combination of Normal-Gamma posteriors.
This is a modeling choice (consumer-side), not a Bayesian primitive.

κ acts as precision (more observations → higher κ → more weight).
    μ_combined = Σ(κᵢ·μᵢ) / Σ(κᵢ)
    α_combined = mean(αᵢ)
    β_combined = mean(βᵢ)
"""
function combine_posteriors(posteriors::Vector{NormalGammaMeasure})
    κ_total = sum(p.κ for p in posteriors)
    μ_combined = sum(p.κ * p.μ for p in posteriors) / κ_total
    α_combined = sum(p.α for p in posteriors) / length(posteriors)
    β_combined = sum(p.β for p in posteriors) / length(posteriors)
    return NormalGammaMeasure(κ_total, μ_combined, α_combined, β_combined)
end

"""
    reward_dist(model::TabularWorldModel, s, a)

Return the posterior mean reward for a (state, action) pair.

Posterior predictive for Normal-Gamma is Student's t: df=2α, loc=μ,
scale=√(β(κ+1)/(ακ)). Reconstruct via credence's expect if needed
for distributional/risk-sensitive planning.
"""
function reward_dist(model::TabularWorldModel, s, a)
    posteriors = collect_posteriors(model, s, a)
    p = length(posteriors) == 1 ? posteriors[1] : combine_posteriors(posteriors)
    return p
end

"""
    sample_dynamics(model::TabularWorldModel) → SampledDynamics

Sample a concrete dynamics model from the posterior for Thompson Sampling.

For each observed (s,a), samples via credence's draw():
- Transition probs: draw(DirichletMeasure) for simplex vector
- Reward: draw(NormalGammaMeasure) returns (μ_s, σ²_s), use μ_s

For unobserved (s,a), rewards are lazily sampled from the prior.
"""
function sample_dynamics(model::TabularWorldModel)
    # Sample transition probabilities via credence's draw(DirichletMeasure)
    sampled_transitions = Dict{Tuple{Any,Any}, Any}()

    for (key, counts) in model.transition_counts
        states = collect(keys(counts))
        alphas = [counts[s] for s in states]
        # Use credence's DirichletMeasure for sampling
        dm = DirichletMeasure(Simplex(length(states)), Finite(states), alphas)
        θ = draw(dm)
        sampled_transitions[key] = (states=states, probs=θ)
    end

    # Sample rewards via credence's draw(NormalGammaMeasure)
    sampled_rewards = Dict{Tuple{Any,Any}, Float64}()

    for (key, _) in model.reward_posterior
        s, a = key
        posteriors = collect_posteriors(model, s, a)
        p = length(posteriors) == 1 ? posteriors[1] : combine_posteriors(posteriors)
        μ_s, _ = draw(p)
        sampled_rewards[key] = μ_s
    end

    return SampledDynamics(
        sampled_transitions,
        sampled_rewards,
        model.known_states,
        model.reward_prior,
        model.confirmed_selfloops
    )
end

"""
    SampledDynamics

A sampled world model for use in planning.

Lazily samples from the prior for unknown (s,a) pairs — different across
Thompson samples (exploration) but consistent within one sample (coherent planning).
"""
mutable struct SampledDynamics
    transitions::Dict{Tuple{Any,Any}, Any}
    rewards::Dict{Tuple{Any,Any}, Float64}
    known_states::Set{Any}
    reward_prior::NormalGammaMeasure
    confirmed_selfloops::Set{Tuple{Any,Any}}
end

"""
    sample_next_state(dynamics::SampledDynamics, s, a) → s′

Sample a next state from the sampled dynamics.

For unknown (s,a), samples once (50% self-loop, 50% random known state) and caches
the result for consistency within this Thompson sample.
"""
function sample_next_state(dynamics::SampledDynamics, s, a)
    key = (s, a)

    if key in dynamics.confirmed_selfloops
        return s
    end

    if !haskey(dynamics.transitions, key)
        # Unknown transition — sample and cache for consistency
        next = if isempty(dynamics.known_states) || rand() < 0.5
            s  # Self-loop
        else
            rand(collect(dynamics.known_states))
        end
        # Cache as a degenerate categorical so future lookups are consistent
        dynamics.transitions[key] = (states=[next], probs=[1.0])
        return next
    end

    trans = dynamics.transitions[key]
    # Weighted sampling from sampled transition probs
    r = rand()
    cumw = 0.0
    for i in eachindex(trans.probs)
        cumw += trans.probs[i]
        r < cumw && return trans.states[i]
    end
    return trans.states[end]
end

"""
    get_reward(dynamics::SampledDynamics, s, a) → Float64

Get the sampled reward for a state-action pair.

For unknown (s,a), lazily samples from the Normal-Gamma prior via credence's draw()
and caches the result.
"""
function get_reward(dynamics::SampledDynamics, s, a)
    key = (s, a)
    if haskey(dynamics.rewards, key)
        return dynamics.rewards[key]
    end
    # Sample from Normal-Gamma prior via credence and cache
    μ_s, _ = draw(dynamics.reward_prior)
    dynamics.rewards[key] = μ_s
    return μ_s
end

"""
    entropy(model::TabularWorldModel) → Float64

Return the entropy of the model posterior (sum over all state-action pairs).
"""
function entropy(model::TabularWorldModel)
    total_entropy = 0.0

    for (key, counts) in model.transition_counts
        total = sum(values(counts))
        for count in values(counts)
            p = count / total
            if p > 0
                total_entropy -= p * log(p)
            end
        end
    end

    return total_entropy
end
