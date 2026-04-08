"""
    IdentityAbstractor

The simplest state abstractor: no abstraction at all.
Each observation is its own equivalence class.

Use this as a baseline or when state aliasing is not a problem.
"""

"""
    IdentityAbstractor

Maps each observation to itself (no abstraction).
"""
struct IdentityAbstractor <: StateAbstractor end

"""
    abstract_state(::IdentityAbstractor, observation) → observation

Return the observation unchanged.
"""
abstract_state(::IdentityAbstractor, observation) = observation

"""
    record_transition!(::IdentityAbstractor, s, a, r, s′)

No-op for identity abstractor.
"""
record_transition!(::IdentityAbstractor, s, a, r, s′) = nothing

"""
    check_contradiction(::IdentityAbstractor) → nothing

Identity abstractor cannot have contradictions.
"""
check_contradiction(::IdentityAbstractor) = nothing

"""
    refine!(::IdentityAbstractor, contradiction)

No-op for identity abstractor.
"""
refine!(::IdentityAbstractor, contradiction) = nothing
