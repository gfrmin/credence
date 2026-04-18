"""
    BisimulationAbstractor

Learns state equivalence classes based on behavioural signatures.

Two states are bisimilar if they have identical reward and transition distributions
for all actions. In practice, we cluster states by their observed (action → outcome)
signatures.

This solves the problem of the agent looping on functionally equivalent states
(e.g., "wearing trousers" vs "not wearing trousers" when neither affects the goal).
"""

"""
    ActionOutcome

Records the outcome of taking an action from a state.
"""
struct ActionOutcome
    reward::Float64
    next_abstract_state::Any
end

"""
    BehaviouralSignature

The behavioural fingerprint of a state: a mapping from actions to outcomes.
"""
const BehaviouralSignature = Dict{Any, Vector{ActionOutcome}}

"""
    BisimulationAbstractor

Learns state equivalence classes based on observed behaviour.
"""
mutable struct BisimulationAbstractor <: StateAbstractor
    # Mapping from concrete observation to abstract state ID
    observation_to_abstract::Dict{Any, Int}

    # Mapping from abstract state ID to set of concrete observations
    abstract_to_observations::Dict{Int, Set{Any}}

    # Behavioural signatures for each concrete observation
    signatures::Dict{Any, BehaviouralSignature}

    # Counter for abstract state IDs
    next_abstract_id::Int

    # Pending transitions (for delayed signature updates)
    pending_transitions::Vector{NamedTuple{(:s, :a, :r, :s_obs), Tuple{Any, Any, Float64, Any}}}

    # Detected contradictions
    contradictions::Vector{NamedTuple{(:abstract_state, :observations, :differing_outcomes), Tuple{Int, Vector{Any}, Any}}}

    # Threshold for considering signatures different
    similarity_threshold::Float64

    # Threshold for considering reward differences significant
    reward_threshold::Float64

    # Preprocessing function applied to observations before abstraction
    preprocess_fn::Function
end

"""
    BisimulationAbstractor(; similarity_threshold=0.95, reward_threshold=0.1, preprocess_fn=identity)

Create a new bisimulation abstractor.
"""
function BisimulationAbstractor(;
    similarity_threshold::Float64 = 0.95,
    reward_threshold::Float64 = 0.1,
    preprocess_fn::Function = identity
)
    return BisimulationAbstractor(
        Dict{Any, Int}(),
        Dict{Int, Set{Any}}(),
        Dict{Any, BehaviouralSignature}(),
        1,
        [],
        [],
        similarity_threshold,
        reward_threshold,
        preprocess_fn
    )
end

"""
    abstract_state(abstractor::BisimulationAbstractor, observation) → Int

Map an observation to its abstract state ID.
Creates a new abstract state if this observation hasn't been seen before.
"""
function abstract_state(abstractor::BisimulationAbstractor, observation)
    # Preprocess the observation to strip irrelevant fields
    obs = abstractor.preprocess_fn(observation)

    if haskey(abstractor.observation_to_abstract, obs)
        return abstractor.observation_to_abstract[obs]
    end

    # New observation — first check if it matches any existing signature
    if haskey(abstractor.signatures, obs)
        sig = abstractor.signatures[obs]

        # Find matching abstract state
        for (abstract_id, obs_set) in abstractor.abstract_to_observations
            for existing_obs in obs_set
                if haskey(abstractor.signatures, existing_obs)
                    if signatures_match(sig, abstractor.signatures[existing_obs],
                                        abstractor.similarity_threshold, abstractor.reward_threshold)
                        # Match found — add to this equivalence class
                        abstractor.observation_to_abstract[obs] = abstract_id
                        push!(obs_set, obs)
                        return abstract_id
                    end
                end
            end
        end
    end

    # No match — create new abstract state
    abstract_id = abstractor.next_abstract_id
    abstractor.next_abstract_id += 1

    abstractor.observation_to_abstract[obs] = abstract_id
    abstractor.abstract_to_observations[abstract_id] = Set([obs])

    return abstract_id
end

"""
    record_transition!(abstractor::BisimulationAbstractor, s, a, r, s′)

Record a transition to update behavioural signatures.
"""
function record_transition!(abstractor::BisimulationAbstractor, s, a, r, s′)
    # Get the concrete observation for the source state
    # (s might be an abstract state ID, so we need to handle both cases)
    s_obs = get_observation_for_abstract(abstractor, s)

    if isnothing(s_obs)
        # s is already a concrete observation
        s_obs = s
    end

    # Record the transition
    push!(abstractor.pending_transitions, (s=s, a=a, r=r, s_obs=s_obs))

    # Update signature
    if !haskey(abstractor.signatures, s_obs)
        abstractor.signatures[s_obs] = BehaviouralSignature()
    end

    sig = abstractor.signatures[s_obs]
    if !haskey(sig, a)
        sig[a] = ActionOutcome[]
    end

    # Get abstract state of s′ — if s′ is already an abstract state ID
    # (exists in abstract_to_observations), use it directly to avoid
    # creating a spurious abstract state for the integer itself
    s′_abstract = if haskey(abstractor.abstract_to_observations, s′)
        s′
    else
        abstract_state(abstractor, s′)
    end
    push!(sig[a], ActionOutcome(r, s′_abstract))

    return nothing
end

"""
    get_observation_for_abstract(abstractor, abstract_id) → observation or nothing

Get a representative concrete observation for an abstract state.
"""
function get_observation_for_abstract(abstractor::BisimulationAbstractor, abstract_id)
    if haskey(abstractor.abstract_to_observations, abstract_id)
        obs_set = abstractor.abstract_to_observations[abstract_id]
        return isempty(obs_set) ? nothing : first(obs_set)
    end
    return nothing
end

"""
    check_contradiction(abstractor::BisimulationAbstractor) → contradiction or nothing

Check for contradictions: same abstract state, different behavioural outcomes.
"""
function check_contradiction(abstractor::BisimulationAbstractor)
    empty!(abstractor.contradictions)
    
    for (abstract_id, obs_set) in abstractor.abstract_to_observations
        if length(obs_set) <= 1
            continue
        end
        
        # Compare signatures within this equivalence class
        obs_list = collect(obs_set)
        
        for i in 1:(length(obs_list)-1)
            for j in (i+1):length(obs_list)
                obs1, obs2 = obs_list[i], obs_list[j]
                
                if haskey(abstractor.signatures, obs1) && haskey(abstractor.signatures, obs2)
                    sig1, sig2 = abstractor.signatures[obs1], abstractor.signatures[obs2]
                    
                    # Check for conflicting outcomes on shared actions
                    conflict = find_conflict(sig1, sig2, abstractor.reward_threshold)

                    if !isnothing(conflict)
                        contradiction = (
                            abstract_state = abstract_id,
                            observations = [obs1, obs2],
                            differing_outcomes = conflict
                        )
                        push!(abstractor.contradictions, contradiction)
                    end
                end
            end
        end
    end
    
    return isempty(abstractor.contradictions) ? nothing : first(abstractor.contradictions)
end

"""
    find_conflict(sig1, sig2, reward_threshold) → conflict or nothing

Find conflicting outcomes between two signatures.
"""
function find_conflict(sig1::BehaviouralSignature, sig2::BehaviouralSignature, reward_threshold::Float64)
    for (action, outcomes1) in sig1
        if haskey(sig2, action)
            outcomes2 = sig2[action]

            # Compare reward distributions
            rewards1 = [o.reward for o in outcomes1]
            rewards2 = [o.reward for o in outcomes2]

            if !isempty(rewards1) && !isempty(rewards2)
                mean1, mean2 = _sig_mean(rewards1), _sig_mean(rewards2)

                # Significant reward difference?
                if abs(mean1 - mean2) > reward_threshold
                    return (action=action, type=:reward, values=(mean1, mean2))
                end
            end

            # Compare transition distributions
            nexts1 = Set(o.next_abstract_state for o in outcomes1)
            nexts2 = Set(o.next_abstract_state for o in outcomes2)

            # Do they lead to different abstract states?
            if !isempty(setdiff(nexts1, nexts2)) || !isempty(setdiff(nexts2, nexts1))
                return (action=action, type=:transition, values=(nexts1, nexts2))
            end
        end
    end

    return nothing
end

"""
    refine!(abstractor::BisimulationAbstractor, contradiction)

Refine the abstraction to resolve a contradiction by splitting an equivalence class.
"""
function refine!(abstractor::BisimulationAbstractor, contradiction)
    abstract_id = contradiction.abstract_state
    obs_to_split = contradiction.observations
    
    if length(obs_to_split) < 2
        return nothing
    end
    
    # Keep the first observation in the original class
    # Move the second to a new class
    obs_to_move = obs_to_split[2]
    
    # Create new abstract state
    new_abstract_id = abstractor.next_abstract_id
    abstractor.next_abstract_id += 1
    
    # Update mappings
    abstractor.observation_to_abstract[obs_to_move] = new_abstract_id
    abstractor.abstract_to_observations[new_abstract_id] = Set([obs_to_move])
    
    # Remove from old class
    delete!(abstractor.abstract_to_observations[abstract_id], obs_to_move)
    
    # Re-check other observations in the old class
    # They might need to move too based on signature similarity
    reclassify_observations!(abstractor, abstract_id, new_abstract_id)
    
    return nothing
end

"""
    reclassify_observations!(abstractor, old_id, new_id)

After a split, reclassify observations that might better match the new class.
"""
function reclassify_observations!(abstractor::BisimulationAbstractor, old_id::Int, new_id::Int)
    old_obs_set = get(abstractor.abstract_to_observations, old_id, Set())
    new_obs_set = get(abstractor.abstract_to_observations, new_id, Set())
    
    if isempty(old_obs_set) || isempty(new_obs_set)
        return
    end
    
    # Get representative signature for new class
    new_representative = first(new_obs_set)
    if !haskey(abstractor.signatures, new_representative)
        return
    end
    new_sig = abstractor.signatures[new_representative]
    
    # Check each observation in old class
    to_move = Any[]
    for obs in old_obs_set
        if haskey(abstractor.signatures, obs)
            obs_sig = abstractor.signatures[obs]
            
            # Does this observation's signature match the new class better?
            if signatures_match(obs_sig, new_sig, abstractor.similarity_threshold, abstractor.reward_threshold)
                push!(to_move, obs)
            end
        end
    end
    
    # Move matching observations
    for obs in to_move
        abstractor.observation_to_abstract[obs] = new_id
        delete!(old_obs_set, obs)
        push!(new_obs_set, obs)
    end
end

"""
    signatures_match(sig1, sig2, threshold, reward_threshold) → Bool

Check if two behavioural signatures are similar enough to be considered equivalent.
"""
function signatures_match(sig1::BehaviouralSignature, sig2::BehaviouralSignature,
                          threshold::Float64, reward_threshold::Float64)
    # Check shared actions
    shared_actions = intersect(keys(sig1), keys(sig2))

    if isempty(shared_actions)
        # No shared actions — can't determine similarity
        return false
    end

    matches = 0
    total = 0

    for action in shared_actions
        outcomes1 = sig1[action]
        outcomes2 = sig2[action]

        if isempty(outcomes1) || isempty(outcomes2)
            continue
        end

        # Compare rewards
        r1 = _sig_mean(o.reward for o in outcomes1)
        r2 = _sig_mean(o.reward for o in outcomes2)

        if abs(r1 - r2) < reward_threshold
            matches += 1
        end
        total += 1

        # Compare transitions
        nexts1 = Set(o.next_abstract_state for o in outcomes1)
        nexts2 = Set(o.next_abstract_state for o in outcomes2)

        if nexts1 == nexts2
            matches += 1
        end
        total += 1
    end

    return total > 0 && (matches / total) >= threshold
end

"""
    _sig_mean(itr)

Compute the mean of an iterable. Named to avoid shadowing Base/Statistics mean.
"""
function _sig_mean(itr)
    vals = collect(itr)
    return isempty(vals) ? 0.0 : sum(vals) / length(vals)
end

"""
    abstraction_summary(abstractor::BisimulationAbstractor) → String

Return a human-readable summary of the current abstraction.
"""
function abstraction_summary(abstractor::BisimulationAbstractor)
    n_concrete = length(abstractor.observation_to_abstract)
    n_abstract = length(abstractor.abstract_to_observations)
    compression = n_concrete > 0 ? n_abstract / n_concrete : 1.0
    
    return """
    Bisimulation Abstraction:
      Concrete observations: $n_concrete
      Abstract states: $n_abstract
      Compression ratio: $(round(compression, digits=3))
      Pending contradictions: $(length(abstractor.contradictions))
    """
end
