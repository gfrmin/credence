# Role: brain-side application
"""
    llm_prosthetic.jl — LLM as sensor prosthetic (spec §6.4)

The LLM extends the agent's sensory capabilities. When invoked, it
can detect features that keyword matching misses (e.g., sarcastic
urgency, implicit action requests). The agent decides when to use
the prosthetic via EU, like any other action. The LLM is part of
the body, not the brain.

In simulation mode (default for tests), no API call is made — the
enrichment returns ground-truth features from the Email struct.
"""

using HTTP, JSON3

# ═══════════════════════════════════════
# Ollama client
# ═══════════════════════════════════════

struct LLMConfig
    host::String              # "http://localhost:11434"
    model::String             # "llama3.1" or "phi3"
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
    llm_enrich_features(config, email, base_features; preview) → Dict{Symbol, Float64}

Enrich features via Ollama when enabled, otherwise fall back to simulation.
The LLM detects features that keyword matching misses — it doesn't
produce semantic labels, just additional binary signals.
"""
function llm_enrich_features(config::LLMConfig, email::Email,
                              base_features::Dict{Symbol, Float64};
                              preview::String="")::Dict{Symbol, Float64}
    if !config.enabled
        return simulate_llm_enrichment(email, base_features)
    end

    body_line = if !isempty(preview)
        "Body: $(first(preview, 300))"
    else
        "Body: [$(email.word_count) words]"
    end

    prompt = """Analyse this email. Respond with ONLY a JSON object, no other text.
{"has_urgent_signal": <true|false>, "has_action_request": <true|false>, "has_money_topic": <true|false>, "has_meeting_topic": <true|false>, "has_question": <true|false>}

From: $(email.sender)
Subject: $(email.subject)
$body_line"""

    response = call_ollama(config, prompt)

    if response === nothing
        return simulate_llm_enrichment(email, base_features)
    end

    try
        cleaned = replace(response, r"```json\s*" => "", r"```\s*" => "")
        cleaned = strip(cleaned)
        parsed = JSON3.read(cleaned)

        enriched = copy(base_features)

        # LLM can detect signals that keyword matching misses
        if haskey(parsed, :has_urgent_signal) && Bool(parsed.has_urgent_signal)
            enriched[:subject_has_urgent_kw] = 1.0
        end
        if haskey(parsed, :has_action_request) && Bool(parsed.has_action_request)
            enriched[:subject_has_action_kw] = 1.0
        end
        if haskey(parsed, :has_money_topic) && Bool(parsed.has_money_topic)
            enriched[:subject_has_money_kw] = 1.0
        end
        if haskey(parsed, :has_meeting_topic) && Bool(parsed.has_meeting_topic)
            enriched[:subject_has_meeting_kw] = 1.0
        end
        if haskey(parsed, :has_question) && Bool(parsed.has_question)
            enriched[:preview_has_question] = 1.0
        end

        return enriched
    catch e
        @warn "Failed to parse LLM response" exception=e response
        return simulate_llm_enrichment(email, base_features)
    end
end

# ═══════════════════════════════════════
# Simulation
# ═══════════════════════════════════════

# LLM cost moved to cost_model.jl — uncertain, time-based NormalGamma belief

"""
    simulate_llm_enrichment(email, base_features) → Dict{Symbol, Float64}

Simulate LLM feature enrichment: the LLM can detect signals that
keyword matching misses. In simulation, this is a no-op since the
synthetic corpus already has ground-truth keyword features.
"""
function simulate_llm_enrichment(email::Email, base_features::Dict{Symbol, Float64})::Dict{Symbol, Float64}
    copy(base_features)
end
