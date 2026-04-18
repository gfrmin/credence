"""
    Bayesian State Inference (Stage 1: MVBN)

Implements P(s | observation, history) ∝ P(observation | s) × P(s | history)

Components:
1. LLM likelihood model: P(text | state) from LLM queries
2. Bayesian update: combine likelihood with prior belief
3. Mean-field approximation: factor updates for tractability
"""

"""
    query_llm_likelihood(llm_sensor::LLMSensor, text::String, variable::String, value::Any) → Float64

Query LLM for likelihood: "How consistent is text with {variable}={value}?"

Returns a score that we convert to likelihood: P(text | variable=value).

In practice:
- LLM returns confidence score (e.g., 0-10 scale)
- We convert: P(text | variable=value) ≈ sigmoid((score - 5) / 2.5)
- This gives ~0.5 when LLM is uncertain, →1 when confident yes, →0 when confident no
"""
function query_llm_likelihood(llm_sensor::LLMSensor, text::String, variable::String, value::Any)::Float64
    # Construct query
    query = """
    Given this text from a text adventure game:
    \"$text\"

    On a scale 0-10, how likely is it that $variable = $value?
    Answer with just a number.
    """

    # Query LLM (simplified; real implementation would use llm_sensor.query)
    # For now, return placeholder
    return 0.5  # TODO: integrate real LLM calls
end

"""
    update_location_belief!(belief::StateBelief, observation_text::String, llm::LLMSensor)

Update P(location | text) using LLM likelihood for each known location.
"""
function update_location_belief!(belief::StateBelief, observation_text::String, llm::LLMSensor)
    # For each known location, query likelihood
    location_likelihoods = Float64[]

    for loc in belief.location_belief.domain
        likelihood = query_llm_likelihood(llm, observation_text, "location", loc)
        push!(location_likelihoods, likelihood)
    end

    # Normalize (treat as unnormalized weights)
    total = sum(location_likelihoods)
    if total > 0
        location_likelihoods ./= total
    else
        location_likelihoods .= 1.0 / length(location_likelihoods)
    end

    # Update beliefs using soft evidence (likelihood weighting)
    # Not a standard Bayesian update, but computationally efficient
    for (i, likelihood) in enumerate(location_likelihoods)
        # Likelihood weighting: adjust belief proportionally
        current_prob = predict(belief.location_belief)[i]
        weight = likelihood / max(current_prob, 0.01)  # Avoid division by zero

        # Add pseudo-observations based on weight
        # (This approximates soft evidence update)
        for _ in 1:Int(round(weight))
            update!(belief.location_belief, belief.location_belief.domain[i])
        end
    end
end

"""
    update_inventory_belief!(belief::StateBelief, observation_text::String, llm::LLMSensor)

Update P(obj ∈ inventory | text) for each known object.
"""
function update_inventory_belief!(belief::StateBelief, observation_text::String, llm::LLMSensor)
    for obj in belief.known_objects
        likelihood_in = query_llm_likelihood(llm, observation_text, "object", obj)

        # Likelihood weighting
        current_prob = predict(belief.inventory_beliefs[obj])[1]  # P(true)
        if likelihood_in > 0.5
            # Object likely in inventory
            weight = likelihood_in / max(current_prob, 0.01)
            for _ in 1:Int(round(weight))
                update!(belief.inventory_beliefs[obj], true)
            end
        else
            # Object likely not in inventory
            weight = (1 - likelihood_in) / max(1 - current_prob, 0.01)
            for _ in 1:Int(round(weight))
                update!(belief.inventory_beliefs[obj], false)
            end
        end
    end
end

"""
    bayesian_update_belief!(belief::StateBelief, observation::NamedTuple, llm::LLMSensor)

Update factored belief given a new observation.

Process:
1. Extract MinimalState from observation (location, inventory)
2. Query LLM for likelihood of observed text given each variable value
3. Update beliefs via likelihood weighting
"""
function bayesian_update_belief!(belief::StateBelief, observation::NamedTuple, llm::LLMSensor)
    # Extract direct observations
    state = extract_minimal_state(observation)
    observation_text = hasproperty(observation, :text) ? observation.text : ""

    # Register any newly seen locations/objects
    add_object!(belief, state.location)
    for obj in state.inventory
        add_object!(belief, obj)
    end

    # Direct update from state observation (most reliable)
    update_from_state!(belief, state)

    # LLM likelihood update (less reliable, but adds information)
    if !isempty(observation_text)
        update_location_belief!(belief, observation_text, llm)
        update_inventory_belief!(belief, observation_text, llm)
    end
end

"""
    predict_from_likelihood(likelihood::Vector{Float64}, prior::Vector{Float64}) → Vector{Float64}

Combine likelihood and prior via Bayes' rule.
"""
function predict_from_likelihood(likelihood::Vector{Float64}, prior::Vector{Float64})::Vector{Float64}
    posterior = likelihood .* prior
    posterior_norm = sum(posterior)
    if posterior_norm > 0
        return posterior ./ posterior_norm
    else
        return prior
    end
end

export update_location_belief!, update_inventory_belief!, bayesian_update_belief!, predict_from_likelihood
