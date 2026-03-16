"""
    Credence — Three types. Axiom-constrained functions.
    Everything else is stdlib.
"""
module Credence

include("parse.jl")
include("ontology.jl")
include("eval.jl")
include("persistence.jl")
include("host_helpers.jl")

using .Parse
using .Ontology
import .Ontology: truncate  # resolve ambiguity with Base.truncate
using .Eval
using .Persistence

export run_dsl, load_dsl, parse_sexpr, parse_all
export Space, Finite, Interval, ProductSpace, Simplex, Euclidean, PositiveReals, support
export Measure, CategoricalMeasure, BetaMeasure, GaussianMeasure, DirichletMeasure, ProductMeasure, MixtureMeasure
export Kernel, FactorSelector, kernel_source, kernel_target
export condition, expect, push_measure, density, log_density_at
export draw, optimise, value
export weights, mean, variance, prune, truncate
export save_state, load_state
export initial_rel_state, initial_cov_state, marginalize_betas, update_beta_state
export extract_reliability_means

end
