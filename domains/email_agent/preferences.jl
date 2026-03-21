"""
    preferences.jl — Email preference types and outcome classification

Maps user reactions to emails (read, archived, replied, deleted, etc.)
to utility values that the agent learns to predict.
"""

"""Classify a user reaction into a utility value in [-1, 1]."""
function classify_outcome(user_reaction::Symbol)::Float64
    error("not implemented")
end
