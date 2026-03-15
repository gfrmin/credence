"""
    persistence.jl — Save/load agent state across sessions.

Uses Julia's Serialization stdlib for reliable roundtrip of
ontology types (CategoricalMeasure, BetaMeasure, etc.),
score totals, and configuration.
"""
module Persistence

using Serialization
using ..Ontology

export save_state, load_state

"""Save agent state to a file."""
function save_state(filepath::String;
                    rel_beliefs, cat_belief, total_score::Float64, total_cost::Float64)
    state = Dict(
        :rel_beliefs => rel_beliefs,
        :cat_belief  => cat_belief,
        :total_score => total_score,
        :total_cost  => total_cost
    )
    open(io -> serialize(io, state), filepath, "w")
end

"""Load agent state from a file. Returns a Dict with keys :rel_beliefs, :cat_belief, :total_score, :total_cost."""
function load_state(filepath::String)
    open(io -> deserialize(io), filepath, "r")
end

end # module Persistence
