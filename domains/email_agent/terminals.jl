"""
    terminals.jl — Email terminal alphabet and seed grammars

Email-specific seed grammars. These use source indices that map
to email feature channels (sender reputation, urgency, length, etc).
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, SensorChannel, SensorConfig
using Credence: next_grammar_id, reset_grammar_counter!

"""
    generate_email_seed_grammars() → Vector{Grammar}

Generate the email domain seed grammar pool. These grammars use
sensor configs specific to the email observation space.
"""
function generate_email_seed_grammars()::Vector{Grammar}
    error("not implemented")
end
