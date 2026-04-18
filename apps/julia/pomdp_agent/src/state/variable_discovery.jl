"""
    Variable Discovery (Stage 2)

Automatically discovers new state variables from observations using:
1. LLM text parsing: extract candidate variables
2. BIC model selection: decide if variable improves fit
3. Dynamic state expansion: add variables to state representation

Mathematical basis:
- P(variable exists | data) ∝ P(data | variable exists) × P(variable exists)
- Use BIC as model selection criterion (balances fit vs complexity)
- BIC = -2 log P(data | variable) + k log n (lower is better)

Algorithm:
1. Query LLM for candidate variables from observation text
2. For each candidate:
   a. Compute BIC with variable included
   b. Compute BIC without variable
   c. If improvement > threshold, accept variable
3. Update state belief to track new variable
4. Reset dynamics learning for new variable
"""

"""
    VariableCandidate

Represents a candidate state variable discovered from text.

Fields:
- name::String              # Variable name: "door_state", "lamp_lit", etc.
- domain::Vector           # Possible values: [true, false] or ["locked", "unlocked", "open"]
- evidence::Float64        # LLM confidence that this variable exists (0-1)
- bic_delta::Float64       # BIC improvement if added (positive = better)
"""
struct VariableCandidate
    name::String
    domain::Vector
    evidence::Float64
    bic_delta::Float64
end

"""
    extract_candidate_variables(text::String) → Vector{VariableCandidate}

Parse observation text to extract candidate state variables.

This is a stub - real implementation would use LLM.
Returns common variables likely mentioned in text:
- door_state, key_found, lamp_lit, etc.
"""
function extract_candidate_variables(text::String)::Vector{VariableCandidate}
    candidates = VariableCandidate[]

    # Simple heuristic parsing (would use LLM in practice)
    text_lower = lowercase(text)

    # Door-related
    if contains(text_lower, "door") || contains(text_lower, "locked") || contains(text_lower, "unlock")
        push!(candidates, VariableCandidate(
            "door_state",
            ["locked", "unlocked", "open"],
            0.8,
            0.0  # Will compute BIC
        ))
    end

    # Light-related
    if contains(text_lower, "lamp") || contains(text_lower, "light") || contains(text_lower, "torch")
        push!(candidates, VariableCandidate(
            "lamp_lit",
            [true, false],
            0.7,
            0.0
        ))
    end

    # Key-related
    if contains(text_lower, "key") && !contains(text_lower, "key pad")
        push!(candidates, VariableCandidate(
            "key_found",
            [true, false],
            0.6,
            0.0
        ))
    end

    # Container/box-related
    if contains(text_lower, "box") || contains(text_lower, "chest") || contains(text_lower, "container")
        push!(candidates, VariableCandidate(
            "container_open",
            [true, false],
            0.7,
            0.0
        ))
    end

    # Puzzle/magic-related (for Enchanter)
    if contains(text_lower, "spell") || contains(text_lower, "magic") || contains(text_lower, "enchant")
        push!(candidates, VariableCandidate(
            "spell_cast",
            [true, false],
            0.6,
            0.0
        ))
    end

    return candidates
end

"""
    compute_bic(belief::StateBelief, data::Vector{MinimalState}) → Float64

Compute BIC score for current belief.
BIC = -2 log P(data | model) + k log n

Lower BIC is better (penalizes complexity).
"""
function compute_bic(belief::StateBelief, data::Vector{MinimalState})::Float64
    if isempty(data)
        return 0.0
    end

    # Log likelihood: sum of log P(observed state | belief)
    log_ll = sum(
        log(max(posterior_prob(belief, state), 1e-10))
        for state in data
    )

    # Number of parameters: one per state variable
    n_params = length(belief.known_objects) + 1  # +1 for location

    # BIC: penalize complexity
    n = length(data)
    bic = -2.0 * log_ll + n_params * log(n)

    return bic
end

"""
    compute_bic_delta(belief::StateBelief, var_name::String, domain::Vector, data::Vector{MinimalState}) → Float64

Compute change in BIC if variable is added.
Positive = improvement (lower BIC with variable).
"""
function compute_bic_delta(belief::StateBelief, var_name::String, domain::Vector, data::Vector{MinimalState})::Float64
    # Current BIC
    bic_now = compute_bic(belief, data)

    # BIC with new variable (simulated)
    # We don't actually add it; just estimate the change
    # New parameters: one per value in domain
    n_params_old = length(belief.known_objects) + 1
    n_params_new = n_params_old + length(domain) - 1

    # Improved fit: assume data becomes more predictable with new variable
    # Very rough heuristic: 1% improvement per new parameter
    log_ll_improvement = 0.01 * length(data) * length(domain)

    n = length(data)
    penalty_delta = (n_params_new - n_params_old) * log(n)

    # Delta: negative = improvement
    bic_delta = -2.0 * log_ll_improvement + penalty_delta

    return -bic_delta  # Positive = better
end

"""
    should_accept_variable(candidate::VariableCandidate, bic_delta::Float64, threshold::Float64 = 1.0) → Bool

Decide whether to accept a variable based on BIC improvement and evidence.
"""
function should_accept_variable(candidate::VariableCandidate, bic_delta::Float64, threshold::Float64=1.0)::Bool
    # Both conditions must be met:
    # 1. LLM confidence that variable exists
    # 2. BIC improvement exceeds threshold
    return candidate.evidence > 0.5 && bic_delta > threshold
end

"""
    discover_variables!(belief::StateBelief, observation::NamedTuple; max_new_vars::Int=3) → Vector{String}

Discover and potentially add new variables from observation.

Returns list of newly discovered variable names.
"""
function discover_variables!(belief::StateBelief, observation::NamedTuple; max_new_vars::Int=3)::Vector{String}
    observation_text = hasproperty(observation, :text) ? observation.text : ""

    if isempty(observation_text)
        return String[]
    end

    # Extract candidates from text
    candidates = extract_candidate_variables(observation_text)

    # Filter out already-known variables
    new_candidates = filter(c -> !(c.name in belief.known_objects), candidates)

    if isempty(new_candidates)
        return String[]
    end

    # For each candidate, compute BIC improvement
    bic_delta_candidates = []
    for candidate in new_candidates
        bic_delta = compute_bic_delta(belief, candidate.name, candidate.domain, belief.history)
        candidate_with_bic = VariableCandidate(candidate.name, candidate.domain, candidate.evidence, bic_delta)
        push!(bic_delta_candidates, candidate_with_bic)
    end

    # Sort by BIC improvement (best first)
    sort!(bic_delta_candidates, by=c -> c.bic_delta, rev=true)

    # Accept top candidates that meet threshold
    accepted = String[]
    for (i, candidate) in enumerate(bic_delta_candidates)
        if i > max_new_vars
            break  # Limit new variables per step
        end

        if should_accept_variable(candidate, candidate.bic_delta)
            add_object!(belief, candidate.name)
            push!(accepted, candidate.name)
        end
    end

    return accepted
end

"""
    update_state_belief_with_discovery!(belief::StateBelief, model, observation::NamedTuple)

Integrated function: discover variables and update beliefs.

Note: model type annotation removed to avoid circular dependency.
"""
function update_state_belief_with_discovery!(belief::StateBelief, model, observation::NamedTuple)
    # Extract observed state
    state = extract_minimal_state(observation)
    update_from_state!(belief, state)

    # Discover new variables
    new_vars = discover_variables!(belief, observation)

    # Register new variables in model
    for var in new_vars
        add_object!(model, var)
    end

    return new_vars
end

export VariableCandidate, extract_candidate_variables, compute_bic, compute_bic_delta
export should_accept_variable, discover_variables!, update_state_belief_with_discovery!
