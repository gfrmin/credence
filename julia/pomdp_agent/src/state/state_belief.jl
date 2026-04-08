"""
    StateBelief (Stage 1: MVBN)

Maintains factored belief distribution over state variables:
- P(location | history)     : DirichletCategorical over observed locations
- P(inventory | history)    : Set of DirichletCategorical, one per object

Mathematical basis:
    P(s | history) = P(location | history) × P(inventory | history)
                   = P(location | history) × ∏_obj P(obj ∈ inventory | history)

Supports:
1. Bayesian updates from observations
2. Sampling for Thompson Sampling (sample s ~ P(s | history))
3. Entropy computation (for exploration)
4. Online learning of state space
"""

mutable struct StateBelief
    # Location belief: location → (prior_α, observed_counts)
    location_belief::DirichletCategorical

    # Inventory belief: object_name → DirichletCategorical(domain={in, out})
    # Each object has binary indicator: true = in inventory, false = not in inventory
    inventory_beliefs::Dict{String, DirichletCategorical}

    # Hidden variable beliefs (Stage 2)
    # spell_name → DirichletCategorical(domain={known, unknown})
    spells_beliefs::Dict{String, DirichletCategorical}

    # object_name → Dict{state_value => DirichletCategorical}
    # E.g., spells_beliefs["lantern"] = {"lit" => DC, "dark" => DC}
    object_state_beliefs::Dict{String, Dict{String, DirichletCategorical}}

    # knowledge_fact → DirichletCategorical(domain={known, unknown})
    knowledge_beliefs::Dict{String, DirichletCategorical}

    # History of observed states (for discovering new variables)
    history::Vector{MinimalState}

    # Known objects and facts discovered so far
    known_objects::Set{String}
    known_spells::Set{String}
    known_facts::Set{String}

    function StateBelief(initial_locations::Vector{String})
        location_belief = DirichletCategorical(initial_locations, 0.1)
        inventory_beliefs = Dict{String, DirichletCategorical}()
        spells_beliefs = Dict{String, DirichletCategorical}()
        object_state_beliefs = Dict{String, Dict{String, DirichletCategorical}}()
        knowledge_beliefs = Dict{String, DirichletCategorical}()
        history = MinimalState[]
        known_objects = Set{String}()
        known_spells = Set{String}()
        known_facts = Set{String}()

        new(location_belief, inventory_beliefs, spells_beliefs, object_state_beliefs, knowledge_beliefs, history, known_objects, known_spells, known_facts)
    end
end

"""
    StateBelief() → StateBelief

Create a fresh belief with common IF game locations.
"""
function StateBelief()
    # Start with common locations; will be extended as discovered
    common_locations = ["Kitchen", "Forest", "Room", "House", "Ground", "outside"]
    return StateBelief(common_locations)
end

"""
    add_object!(belief::StateBelief, object_name::String)

Register a new object to track in inventory.
Creates a binary inventory belief for the object.
"""
function add_object!(belief::StateBelief, object_name::String)
    if !haskey(belief.inventory_beliefs, object_name)
        # Binary: object is in inventory or not
        domain = [true, false]
        alpha = 0.5  # Weak prior: equally likely in or out
        belief.inventory_beliefs[object_name] = DirichletCategorical(domain, alpha)
        push!(belief.known_objects, object_name)
    end
end

"""
    add_spell!(belief::StateBelief, spell_name::String)

Register a new spell to track in spells_known.
"""
function add_spell!(belief::StateBelief, spell_name::String)
    if !haskey(belief.spells_beliefs, spell_name)
        domain = [true, false]  # true = known, false = unknown
        alpha = 0.1  # Weak prior
        belief.spells_beliefs[spell_name] = DirichletCategorical(domain, alpha)
        push!(belief.known_spells, spell_name)
    end
end

"""
    add_object_state!(belief::StateBelief, object_name::String, state_value::String)

Register a new object state.
"""
function add_object_state!(belief::StateBelief, object_name::String, state_value::String)
    if !haskey(belief.object_state_beliefs, object_name)
        belief.object_state_beliefs[object_name] = Dict{String, DirichletCategorical}()
    end
    if !haskey(belief.object_state_beliefs[object_name], state_value)
        domain = [true, false]  # true = object is in this state, false = not
        alpha = 0.1
        belief.object_state_beliefs[object_name][state_value] = DirichletCategorical(domain, alpha)
    end
end

"""
    add_fact!(belief::StateBelief, fact::String)

Register a new knowledge fact to track.
"""
function add_fact!(belief::StateBelief, fact::String)
    if !haskey(belief.knowledge_beliefs, fact)
        domain = [true, false]  # true = fact known, false = unknown
        alpha = 0.1
        belief.knowledge_beliefs[fact] = DirichletCategorical(domain, alpha)
        push!(belief.known_facts, fact)
    end
end

"""
    update_from_state!(belief::StateBelief, state::MinimalState)

Update beliefs given an observed state, including hidden variables.
"""
function update_from_state!(belief::StateBelief, state::MinimalState)
    # Record location
    update!(belief.location_belief, state.location)

    # Record inventory for each known object
    for obj in belief.known_objects
        in_inventory = obj ∈ state.inventory
        update!(belief.inventory_beliefs[obj], in_inventory)
    end

    # Register any new objects observed
    for obj in state.inventory
        add_object!(belief, obj)
    end

    # Update hidden variable beliefs
    # Spells known
    for spell in state.spells_known
        add_spell!(belief, spell)
        update!(belief.spells_beliefs[spell], true)  # true = spell is known
    end

    # Object states
    for (obj, state_val) in state.object_states
        add_object_state!(belief, obj, state_val)
        update!(belief.object_state_beliefs[obj][state_val], true)
    end

    # Knowledge facts
    for fact in state.knowledge_gained
        add_fact!(belief, fact)
        update!(belief.knowledge_beliefs[fact], true)  # true = fact is known
    end

    # Store in history for later analysis
    push!(belief.history, state)
end

"""
    sample_state(belief::StateBelief) → MinimalState

Sample a state from the posterior belief distribution.
Uses Thompson Sampling: sample each variable from posterior, combine.
Includes hidden variables (spells, object states, knowledge).
"""
function sample_state(belief::StateBelief)::MinimalState
    # Sample location
    location = rand(belief.location_belief)

    # Sample inventory: for each object, independently sample in/out
    inventory = Set{String}()
    for obj in belief.known_objects
        in_inventory = rand(belief.inventory_beliefs[obj])
        if in_inventory
            push!(inventory, obj)
        end
    end

    # Sample hidden variables
    spells_known = Set{String}()
    for spell in belief.known_spells
        spell_is_known = rand(belief.spells_beliefs[spell])
        if spell_is_known
            push!(spells_known, spell)
        end
    end

    object_states = Dict{String, String}()
    for (obj, state_dict) in belief.object_state_beliefs
        for (state_val, dist) in state_dict
            state_true = rand(dist)
            if state_true
                object_states[obj] = state_val
                break  # Object in one state only
            end
        end
    end

    knowledge_gained = Set{String}()
    for fact in belief.known_facts
        fact_known = rand(belief.knowledge_beliefs[fact])
        if fact_known
            push!(knowledge_gained, fact)
        end
    end

    return MinimalState(location, inventory, spells_known, object_states, knowledge_gained)
end

"""
    predict_state(belief::StateBelief) → MinimalState

Return the mode (MAP estimate) of the posterior belief.
"""
function predict_state(belief::StateBelief)::MinimalState
    location = mode(belief.location_belief)

    inventory = Set{String}()
    for obj in belief.known_objects
        in_inventory = mode(belief.inventory_beliefs[obj])
        if in_inventory
            push!(inventory, obj)
        end
    end

    return MinimalState(location, inventory)
end

"""
    entropy(belief::StateBelief) → Float64

Compute Shannon entropy of posterior belief.
H[s] = H[location] + Σ_obj H[obj ∈ inventory]
"""
function entropy(belief::StateBelief)::Float64
    h = entropy(belief.location_belief)

    for obj in belief.known_objects
        h += entropy(belief.inventory_beliefs[obj])
    end

    return h
end

"""
    posterior_prob(belief::StateBelief, state::MinimalState) → Float64

Compute P(state | history) under factored model.
P(s) = P(location) × ∏_obj P(obj ∈ inventory)
"""
function posterior_prob(belief::StateBelief, state::MinimalState)::Float64
    probs = predict(belief.location_belief)
    location_idx = findfirst(l -> l == state.location, belief.location_belief.domain)
    if isnothing(location_idx)
        return 0.0
    end
    p = probs[location_idx]

    for obj in belief.known_objects
        in_inventory = obj ∈ state.inventory
        obj_probs = predict(belief.inventory_beliefs[obj])
        # domain = [true, false], find idx for whether obj is in inventory
        idx = findfirst(v -> v == in_inventory, belief.inventory_beliefs[obj].domain)
        p *= obj_probs[idx]
    end

    return p
end

"""
    loglikelihood(belief::StateBelief) → Float64

Sum of log marginal likelihoods for all belief variables.
Used for model comparison (e.g., variable discovery, structure learning).
"""
function loglikelihood(belief::StateBelief)::Float64
    ll = loglikelihood(belief.location_belief)
    for obj in belief.known_objects
        ll += loglikelihood(belief.inventory_beliefs[obj])
    end
    return ll
end

"""
    reset!(belief::StateBelief)

Clear all observations, reset to prior.
Keep hidden variable definitions but reset their counts.
"""
function reset!(belief::StateBelief)
    reset!(belief.location_belief)
    for obj in belief.known_objects
        reset!(belief.inventory_beliefs[obj])
    end
    for spell in belief.known_spells
        reset!(belief.spells_beliefs[spell])
    end
    for (obj, state_dict) in belief.object_state_beliefs
        for (state_val, dist) in state_dict
            reset!(dist)
        end
    end
    for fact in belief.known_facts
        reset!(belief.knowledge_beliefs[fact])
    end
    empty!(belief.history)
end

"""
    copy(belief::StateBelief) → StateBelief

Deep copy of belief state, including hidden variables.
"""
function Base.copy(belief::StateBelief)
    new_belief = StateBelief(copy(belief.location_belief.domain))
    new_belief.location_belief = copy(belief.location_belief)
    new_belief.inventory_beliefs = Dict(k => copy(v) for (k, v) in belief.inventory_beliefs)
    new_belief.spells_beliefs = Dict(k => copy(v) for (k, v) in belief.spells_beliefs)
    new_belief.object_state_beliefs = Dict(
        k => Dict(sv => copy(d) for (sv, d) in v)
        for (k, v) in belief.object_state_beliefs
    )
    new_belief.knowledge_beliefs = Dict(k => copy(v) for (k, v) in belief.knowledge_beliefs)
    new_belief.history = copy(belief.history)
    new_belief.known_objects = copy(belief.known_objects)
    new_belief.known_spells = copy(belief.known_spells)
    new_belief.known_facts = copy(belief.known_facts)
    return new_belief
end

export StateBelief, add_object!, add_spell!, add_object_state!, add_fact!, update_from_state!, sample_state, predict_state, entropy, posterior_prob, loglikelihood
