"""
    host.jl — Host driver for the email program-space agent

Orchestrates: grammar pool → program enumeration → kernel compilation →
belief management → action selection → user feedback → repeat.

Stub: validates the Tier 2 API is sufficient for the email domain.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence

include("features.jl")
include("terminals.jl")
include("preferences.jl")
include("metrics.jl")

"""
    run_email_agent(; kwargs...) → (metrics, state, grammar_pool)

Main loop for the email agent. Stub implementation.
"""
function run_email_agent(; kwargs...)
    error("not implemented")
end
