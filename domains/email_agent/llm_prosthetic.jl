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

using HTTP, JSON3

# ═══════════════════════════════════════
# Ollama client
# ═══════════════════════════════════════

struct LLMConfig
    host::String              # "http://localhost:11434"
    model::String             # "llama3.2" or "phi3"
    max_tokens::Int           # 200
    enabled::Bool             # false for testing without Ollama
    timeout::Float64          # 10.0 seconds
end

function default_llm_config(; enabled::Bool=false)
    LLMConfig("http://localhost:11434", "llama3.1", 200, enabled, 10.0)
end

function call_ollama(config::LLMConfig, prompt::String)::Union{String, Nothing}
    try
        body = JSON3.write(Dict(
            "model" => config.model,
            "prompt" => prompt,
            "stream" => false,
            "options" => Dict("num_predict" => config.max_tokens)
        ))
        resp = HTTP.post(
            "$(config.host)/api/generate",
            ["Content-Type" => "application/json"],
            body;
            readtimeout=round(Int, config.timeout)
        )
        result = JSON3.read(resp.body)
        return String(result.response)
    catch e
        @warn "Ollama call failed" exception=e
        return nothing
    end
end

"""
    llm_enrich_features(config, email, base_features) → Vector{Float64}

Enrich features via Ollama when enabled, otherwise fall back to simulation.
On any failure (network, parse), falls back to simulation transparently.
"""
function llm_enrich_features(config::LLMConfig, email::Email,
                              base_features::Vector{Float64})::Vector{Float64}
    if !config.enabled
        return simulate_llm_enrichment(email, base_features)
    end

    prompt = """Analyse this email. Respond with ONLY a JSON object, no other text.
{"urgency": <0.0-1.0>, "topic": "<finance|scheduling|marketing|personal|technical>", "requires_action": <true|false>}

From: $(email.sender)
Subject: $(email.subject)
Body: [$(email.word_count) words about $(string(email.topic))]"""

    response = call_ollama(config, prompt)

    if response === nothing
        return simulate_llm_enrichment(email, base_features)
    end

    try
        cleaned = replace(response, r"```json\s*" => "", r"```\s*" => "")
        cleaned = strip(cleaned)
        parsed = JSON3.read(cleaned)

        enriched = copy(base_features)

        if haskey(parsed, :urgency)
            enriched[5] = clamp(Float64(parsed.urgency), 0.0, 1.0)
        end
        if haskey(parsed, :topic)
            topic = Symbol(parsed.topic)
            enriched[6] = topic == :finance ? 1.0 : 0.0
            enriched[7] = topic == :scheduling ? 1.0 : 0.0
            enriched[8] = topic == :marketing ? 1.0 : 0.0
        end
        if haskey(parsed, :requires_action)
            enriched[9] = Bool(parsed.requires_action) ? 1.0 : 0.0
        end

        return enriched
    catch e
        @warn "Failed to parse LLM response" exception=e response
        return simulate_llm_enrichment(email, base_features)
    end
end

# ═══════════════════════════════════════
# Costs and simulation
# ═══════════════════════════════════════

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
