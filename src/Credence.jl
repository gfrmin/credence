"""
    Credence — Three types. Axiom-constrained functions.
    Everything else is stdlib.

    Tier 1: DSL core (this module)
    Tier 2: Program-space inference (ProgramSpace submodule)
    Tier 3: Domain applications (domains/)
"""
module Credence

include("parse.jl")
include("ontology.jl")
include("eval.jl")
include("persistence.jl")
include("host_helpers.jl")
include("program_space/ProgramSpace.jl")

using .Parse
using .Ontology
import .Ontology: truncate  # resolve ambiguity with Base.truncate
using .Eval
using .Persistence
using .ProgramSpace

export run_dsl, load_dsl, parse_sexpr, parse_all
export Space, Finite, Interval, ProductSpace, Simplex, Euclidean, PositiveReals, support
export Measure, CategoricalMeasure, BetaMeasure, TaggedBetaMeasure, GaussianMeasure, GammaMeasure, ExponentialMeasure, DirichletMeasure, NormalGammaMeasure, ProductMeasure, MixtureMeasure
export Kernel, FactorSelector, kernel_source, kernel_target
export LikelihoodFamily, BetaBernoulli, Flat, FiringByTag, DispatchByComponent
export Functional, Identity, Projection, NestedProjection, Tabular, LinearCombination, OpaqueClosure
export factor, replace_factor
export condition, expect, push_measure, density, log_density_at, log_predictive, log_marginal
export draw
export weights, mean, variance, prune, truncate, logsumexp
export save_state, load_state
export initial_rel_state, initial_cov_state, marginalize_betas, update_beta_state
export extract_reliability_means

# Re-export Tier 2 (ProgramSpace)
export ProgramExpr, GTExpr, LTExpr, AndExpr, OrExpr, NotExpr, NonterminalRef
export PersistsExpr, ChangedExpr, SinceExpr
export show_expr
export ProductionRule, Grammar, compute_grammar_complexity
export Program, CompiledKernel, SubprogramFrequencyTable
export THRESHOLDS
export expr_complexity, expanded_complexity
export enumerate_programs
export compile_kernel, compile_expr, evaluate_predicate
export analyse_posterior_subtrees, extract_subtrees
export propose_nonterminal, perturb_grammar
export expr_equal, collect_threshold_nodes, replace_threshold
export AgentState, sync_prune!, sync_truncate!
export aggregate_grammar_weights, top_k_grammar_ids
export add_programs_to_state!
export next_grammar_id, reset_grammar_counter!

end
