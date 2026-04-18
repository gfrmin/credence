"""
    FactoredWorldModel (Stage 1: MVBN)

Learns action-conditional factored dynamics over MinimalState variables.

For each action a:
- P(location' | location, action) ~ Categorical with Dirichlet prior
- P(obj ∈ inventory' | obj ∈ inventory, action) ~ Bernoulli with Beta prior

All observations are collected per-action, enabling generalization across
different state variables and sharing of statistical strength.

Parameters are organized as:
  cpds[action][variable] = DirichletCategorical for that variable under that action

This is mathematically equivalent to Dirichlet-Categorical CPDs in a Bayesian network,
but organized by action for computational efficiency.
"""

mutable struct FactoredWorldModel <: WorldModel
    # CPDs per action: cpds[action][variable] = DirichletCategorical
    cpds::Dict{String, Dict{String, DirichletCategorical}}

    # Observed transitions: (s_abstract, action) → [s' that were reached]
    transitions::Dict{Tuple{Any,String}, Vector{Any}}

    # Known locations discovered so far
    known_locations::Set{String}

    # Known objects discovered so far
    known_objects::Set{String}

    # Dynamics prior: concentration for Dirichlet
    dynamics_prior_strength::Float64

    # Reward model prior: NormalGammaMeasure for unseen (state, action) pairs
    reward_prior::NormalGammaMeasure

    # Reward model posterior: (state, action) → NormalGammaMeasure
    reward_posterior::Dict{Tuple{Any,String}, NormalGammaMeasure}

    # Confirmed self-loops: (state, action) where outcome is always unchanged
    confirmed_selfloops::Set{Tuple{Any,String}}

    function FactoredWorldModel(prior_strength::Float64=0.1;
                               reward_prior_mean::Float64=0.0,
                               reward_prior_variance::Float64=1.0)
        prior = NormalGammaMeasure(1.0, reward_prior_mean, 1.0, reward_prior_variance)
        new(
            Dict{String, Dict{String, DirichletCategorical}}(),
            Dict{Tuple{Any,String}, Vector{Any}}(),
            Set{String}(),
            Set{String}(),
            prior_strength,
            prior,
            Dict{Tuple{Any,String}, NormalGammaMeasure}(),
            Set{Tuple{Any,String}}()
        )
    end
end

"""
    SampledFactoredDynamics

Represents a concrete sample of dynamics parameters for a single trajectory.
Used by Thompson Sampling.

Fields:
- model::FactoredWorldModel       # Reference to the model
- sampled_cpds::Dict              # sampled_cpds[action][var] = θ (sampled parameters)
"""
mutable struct SampledFactoredDynamics
    model::FactoredWorldModel
    sampled_cpds::Dict{String, Dict{String, Vector{Float64}}}
    sampled_rewards::Dict{Tuple{Any,String}, Float64}
end

"""
    add_location!(model::FactoredWorldModel, location::String)

Register a new location for tracking.
"""
function add_location!(model::FactoredWorldModel, location::String)
    if !in(location, model.known_locations)
        push!(model.known_locations, location)
    end
end

"""
    add_object!(model::FactoredWorldModel, object::String)

Register a new object for tracking.
"""
function add_object!(model::FactoredWorldModel, object::String)
    if !in(object, model.known_objects)
        push!(model.known_objects, object)
    end
end

"""
    ensure_action_cpds!(model::FactoredWorldModel, action::String)

Initialize CPDs for an action if not yet created.
"""
function ensure_action_cpds!(model::FactoredWorldModel, action::String)
    if !haskey(model.cpds, action)
        model.cpds[action] = Dict{String, DirichletCategorical}()
    end

    cpds_a = model.cpds[action]

    # Ensure CPD for location
    if !haskey(cpds_a, "location")
        locations = collect(model.known_locations)
        if isempty(locations)
            locations = ["unknown"]
        end
        cpds_a["location"] = DirichletCategorical(locations, model.dynamics_prior_strength)
    end

    # Ensure CPDs for each object
    for obj in model.known_objects
        if !haskey(cpds_a, "inventory_$obj")
            domain = [true, false]  # in inventory or not
            cpds_a["inventory_$obj"] = DirichletCategorical(domain, model.dynamics_prior_strength)
        end
    end
end

"""
    update!(model::FactoredWorldModel, s, a, r, s′)

Update model with observed transition (s, a, r, s′).
- s, s′ should be MinimalState
"""
function BayesianAgents.update!(model::FactoredWorldModel, s::MinimalState, a::String, r::Float64, s′::MinimalState)
    # Register locations and objects
    add_location!(model, s.location)
    add_location!(model, s′.location)
    for obj in s.inventory
        add_object!(model, obj)
    end
    for obj in s′.inventory
        add_object!(model, obj)
    end

    # Ensure CPDs for this action exist
    ensure_action_cpds!(model, a)

    cpds_a = model.cpds[a]

    # Update location CPD
    update!(cpds_a["location"], s′.location)

    # Update inventory CPDs for each object
    for obj in model.known_objects
        now_in = obj ∈ s′.inventory

        cpd_key = "inventory_$obj"
        if haskey(cpds_a, cpd_key)
            update!(cpds_a[cpd_key], now_in)
        end
    end

    # Record transition
    key = (s, a)
    if !haskey(model.transitions, key)
        model.transitions[key] = []
    end
    push!(model.transitions[key], s′)

    # Update reward model via credence's condition()
    reward_key = (s, a)
    if !haskey(model.reward_posterior, reward_key)
        model.reward_posterior[reward_key] = NormalGammaMeasure(1.0, 0.0, 1.0, 1.0)
    end

    p = model.reward_posterior[reward_key]
    model.reward_posterior[reward_key] = condition(p, _NORMAL_GAMMA_KERNEL, r)
end

"""
    sample_dynamics(model::FactoredWorldModel) → SampledFactoredDynamics

Sample one concrete dynamics model from the posterior for Thompson Sampling.
Returns a sampled model that can be queried for P(s' | s, a).
"""
function BayesianAgents.sample_dynamics(model::FactoredWorldModel)
    # Sample CPD parameters for each action via credence's draw(DirichletMeasure)
    sampled_cpds = Dict{String, Dict{String, Vector{Float64}}}()

    for (action, cpds_a) in model.cpds
        sampled_cpds[action] = Dict{String, Vector{Float64}}()

        for (var, cpd) in cpds_a
            # Thompson sampling: draw θ from Dirichlet posterior (two draws)
            # First draw: sample simplex vector θ from DirichletMeasure
            theta = draw(cpd.measure)
            sampled_cpds[action][var] = theta
        end
    end

    # Sample rewards via credence's draw(NormalGammaMeasure)
    sampled_rewards = Dict{Tuple{Any,String}, Float64}()
    for (key, p) in model.reward_posterior
        μ_s, _ = draw(p)
        sampled_rewards[key] = μ_s
    end

    return SampledFactoredDynamics(model, sampled_cpds, sampled_rewards)
end

"""
    sample_next_state(sampled::SampledFactoredDynamics, s::MinimalState, a::String) → MinimalState

Sample next state from transition distribution under sampled model.
Interface matches TabularWorldModel for MCTS compatibility.

P(s' | s, a, θ_sampled) = P(location' | location, a, θ) × ∏_obj P(obj ∈ inventory' | obj ∈ inventory, a, θ)
"""
function sample_next_state(sampled::SampledFactoredDynamics, s::MinimalState, a::String)::MinimalState
    model = sampled.model
    sampled_cpds = sampled.sampled_cpds

    if !haskey(sampled_cpds, a)
        # Action never taken; return state unchanged as pessimistic estimate
        return s
    end

    cpds_a_sampled = sampled_cpds[a]

    # Sample location from sampled theta
    if haskey(cpds_a_sampled, "location")
        theta_loc = cpds_a_sampled["location"]
        location_domain = model.cpds[a]["location"].domain
        # Weighted sampling from domain using sampled theta
        r = rand()
        cumw = 0.0
        new_location = location_domain[end]
        for i in eachindex(theta_loc)
            cumw += theta_loc[i]
            if r < cumw
                new_location = location_domain[i]
                break
            end
        end
    else
        new_location = s.location
    end

    # Sample inventory
    new_inventory = Set{String}()
    for obj in model.known_objects
        cpd_key = "inventory_$obj"
        if haskey(cpds_a_sampled, cpd_key)
            theta_inv = cpds_a_sampled[cpd_key]
            # theta_inv = [P(true), P(false)]
            in_inventory = rand() < theta_inv[1]  # First element is P(true)
            if in_inventory
                push!(new_inventory, obj)
            end
        else
            # Not tracked, carry forward
            if obj ∈ s.inventory
                push!(new_inventory, obj)
            end
        end
    end

    return MinimalState(new_location, new_inventory)
end

"""
    get_reward(sampled::SampledFactoredDynamics, s::MinimalState, a::String) → Float64

Get expected reward for state-action pair under sampled model.
Interface matches TabularWorldModel for MCTS compatibility.
"""
function get_reward(sampled::SampledFactoredDynamics, s::MinimalState, a::String)::Float64
    key = (s, a)

    # Return Thompson-sampled reward if available
    if haskey(sampled.sampled_rewards, key)
        return sampled.sampled_rewards[key]
    end

    # No data for this state-action: lazily sample from prior via credence and cache
    μ_s, _ = draw(sampled.model.reward_prior)
    sampled.sampled_rewards[key] = μ_s
    return μ_s
end

"""
    transition_dist(model::FactoredWorldModel, s::MinimalState, a::String) → Distribution

Return posterior predictive distribution P(s' | s, a, data).
For factored model, returns samples of likely next states.
"""
function BayesianAgents.transition_dist(model::FactoredWorldModel, s::MinimalState, a::String)
    # Return observed transitions as empirical distribution
    key = (s, a)
    if haskey(model.transitions, key)
        return Distributions.Categorical(ones(length(model.transitions[key])) / length(model.transitions[key]))
    else
        # No data; uniform over known states (or just s itself)
        return Distributions.Categorical([1.0])
    end
end

"""
    reward_dist(model::FactoredWorldModel, s::MinimalState, a::String)

Return the reward posterior for a (state, action) pair.
"""
function BayesianAgents.reward_dist(model::FactoredWorldModel, s::MinimalState, a::String)
    key = (s, a)
    if haskey(model.reward_posterior, key)
        return model.reward_posterior[key]
    else
        # Prior: zero mean, unit variance
        return Distributions.Normal(0.0, 1.0)
    end
end

"""
    entropy(model::FactoredWorldModel) → Float64

Total entropy across all CPDs (measure of uncertainty).
"""
function BayesianAgents.entropy(model::FactoredWorldModel)::Float64
    h = 0.0
    for (action, cpds_a) in model.cpds
        for (var, cpd) in cpds_a
            h += entropy(cpd)
        end
    end
    return h
end

"""
    observable_key(s::MinimalState) → Tuple

Extract observable state (location, sorted inventory) for selfloop keying.
Hidden variables are excluded because selfloops depend only on game mechanics,
not on inferred hidden state. This prevents false negatives when the heuristic
adds varying knowledge_gained across steps.
"""
function observable_key(s::MinimalState)
    return (s.location, sort(collect(s.inventory)))
end

"""
    mark_selfloop!(model::FactoredWorldModel, s::MinimalState, a::String)

Record that action a in state s produces no observable change (null action).
Keyed on observable state only (location + inventory) so hidden variable
differences don't prevent selfloop detection.
"""
function mark_selfloop!(model::FactoredWorldModel, s::MinimalState, a::String)
    push!(model.confirmed_selfloops, (observable_key(s), a))
end

"""
    is_selfloop(model::FactoredWorldModel, s::MinimalState, a::String) → Bool

Check if action is confirmed to be a self-loop from this observable state.
"""
function is_selfloop(model::FactoredWorldModel, s::MinimalState, a::String)::Bool
    return (observable_key(s), a) ∈ model.confirmed_selfloops
end

export FactoredWorldModel, SampledFactoredDynamics, add_location!, add_object!, mark_selfloop!, is_selfloop, observable_key, sample_next_state
