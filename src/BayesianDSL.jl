"""
    BayesianDSL — Three primitives. Three axioms. Everything composes.
"""
module BayesianDSL

include("parse.jl")
include("primitives.jl")
include("persistence.jl")
include("eval.jl")

using .Parse
using .Primitives
using .Persistence
using .Eval

export run_dsl, load_dsl, parse_sexpr, parse_all
export Belief, update, decide, weights, hypotheses, weighted_sum
export BetaBelief, mean, variance, to_belief
export save_state, load_state

end # module
