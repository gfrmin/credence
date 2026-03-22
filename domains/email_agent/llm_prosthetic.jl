"""
    llm_prosthetic.jl — LLM as sensor prosthetic (spec §6.4)

The LLM extends the agent's sensory capabilities. When invoked, it
enriches the feature vector by reducing noise on key channels (urgency,
topic classification). The agent decides when to use the prosthetic
via EU, like any other action. The LLM is part of the body, not the brain.

In simulation mode (default for tests), no API call is made — the
enrichment returns ground-truth features, simulating what a well-calibrated
LLM would provide.
"""

const LLM_COST = 0.15

"""
    simulate_llm_enrichment(email, base_features) → Vector{Float64}

Simulate LLM feature enrichment: sharpen urgency and topic channels
by replacing noisy observations with ground-truth values. In production,
this would call the Anthropic API and parse structured output.
"""
function simulate_llm_enrichment(email::Email, base_features::Vector{Float64})::Vector{Float64}
    enriched = copy(base_features)
    # Sharpen urgency (channel 4 → index 5)
    enriched[5] = email.urgency
    # Sharpen topic channels (channels 5-7 → indices 6-8)
    enriched[6] = email.topic == :finance ? 1.0 : 0.0
    enriched[7] = email.topic == :scheduling ? 1.0 : 0.0
    enriched[8] = email.topic == :marketing ? 1.0 : 0.0
    enriched
end

"""
    project_enriched_per_grammar(enriched_features, grammars) → Dict{Int, Vector{Float64}}

Project enriched features through each grammar's sensor config.
Unlike project_email_per_grammar, no additional noise is added —
the LLM has already provided calibrated readings.
"""
function project_enriched_per_grammar(enriched_features::Vector{Float64}, grammars::Vector{Grammar})
    result = Dict{Int, Vector{Float64}}()
    for g in grammars
        readings = Float64[]
        for ch in g.sensor_config.channels
            push!(readings, clamp(enriched_features[ch.source_index + 1], 0.0, 1.0))
        end
        result[g.id] = readings
    end
    result
end
