"""
    Credence — Three types. Axiom-constrained functions.
    Everything else is stdlib.
"""
module Credence

include("parse.jl")
include("ontology.jl")
include("eval.jl")
include("persistence.jl")

using .Parse
using .Ontology
using .Eval
using .Persistence

export run_dsl, load_dsl, parse_sexpr, parse_all
export Space, Finite, Interval, ProductSpace, Simplex, support
export Measure, CategoricalMeasure, BetaMeasure, GaussianMeasure, DirichletMeasure
export Kernel, kernel_source, kernel_target
export condition, expect, push_measure, density, log_density_at
export draw, optimise, value
export weights, mean, variance
export save_state, load_state

end
