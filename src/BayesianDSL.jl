"""
    BayesianDSL — Three primitives. Three axioms. Everything composes.
"""
module BayesianDSL

include("parse.jl")
include("primitives.jl")
include("eval.jl")

using .Parse
using .Primitives
using .Eval

export run_dsl, parse_sexpr, parse_all
export Belief, update, decide, weights, hypotheses, weighted_sum

end # module
