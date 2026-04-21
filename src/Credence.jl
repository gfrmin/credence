"""
    Credence — Three types. Axiom-constrained functions.
    Everything else is stdlib.

    The DSL core: Space, Measure, Kernel; the axiom-constrained functions
    (condition, expect, push); the standard library built on them (optimise,
    value, voi, model, problem, …); and their program-space extensions —
    Grammar (a Space constructor for ASTs), CompiledKernel (a Kernel
    performance variant), enumerate_programs (an execution strategy), and
    perturb_grammar (a stdlib learning operation). Program-related files
    are grouped under program_space/ for cohesion, not as a separate tier.

    Applications (Julia domains, Python surfaces, JSON-RPC bridge) live
    under apps/.
"""
module Credence

include("parse.jl")
include("prevision.jl")
include("ontology.jl")
include("eval.jl")
include("persistence.jl")
include("host_helpers.jl")
include("program_space/types.jl")
include("program_space/enumeration.jl")
include("program_space/compilation.jl")
include("program_space/perturbation.jl")
include("program_space/agent_state.jl")

using .Parse
using .Previsions
using .Ontology
import .Ontology: truncate  # resolve ambiguity with Base.truncate
using .Eval
using .Persistence

export run_dsl, load_dsl, parse_sexpr, parse_all
export Space, Finite, Interval, ProductSpace, Simplex, Euclidean, PositiveReals, support
export Measure, CategoricalMeasure, BetaMeasure, TaggedBetaMeasure, GaussianMeasure, GammaMeasure, ExponentialMeasure, DirichletMeasure, NormalGammaMeasure, ProductMeasure, MixtureMeasure
export Kernel, FactorSelector, kernel_source, kernel_target
export LikelihoodFamily, LeafFamily, PushOnly, BetaBernoulli, Flat, FiringByTag, DispatchByComponent, DepthCapExceeded
export NormalNormal, Categorical, NormalGammaLikelihood, Exponential
export Event, TagSet, FeatureEquals, FeatureInterval, Conjunction, Disjunction, Complement
export indicator_kernel, feature_value, BOOLEAN_SPACE
export Functional, Identity, Projection, NestedProjection, Tabular, LinearCombination, OpaqueClosure
export Prevision, TestFunction, Indicator, apply
export MixturePrevision, ExchangeablePrevision, decompose
export ParticlePrevision, QuadraturePrevision, EnumerationPrevision
export factor, replace_factor
export condition, expect, push_measure, density, log_density_at, log_predictive, log_marginal
export draw
export weights, mean, variance, prune, truncate, logsumexp
export save_state, load_state, MigrationError
export initial_rel_state, initial_cov_state, marginalize_betas, update_beta_state
export extract_reliability_means

# Program-space extensions (Grammar/Program types, compilation, enumeration,
# perturbation, AgentState). Defined directly in this module.
export ProgramExpr, GTExpr, LTExpr, AndExpr, OrExpr, NotExpr, NonterminalRef
export PersistsExpr, ChangedExpr, SinceExpr
export show_expr
export ProductionRule, Grammar, compute_grammar_complexity
export Program, CompiledKernel, SubprogramFrequencyTable
export THRESHOLDS
export expr_complexity, expanded_complexity
export enumerate_programs, enumerate_programs_as_prevision
export compile_kernel, compile_expr, evaluate_predicate
export analyse_posterior_subtrees, extract_subtrees
export propose_nonterminal, perturb_grammar
export expr_equal, collect_threshold_nodes, replace_threshold
export AgentState, sync_prune!, sync_truncate!
export aggregate_grammar_weights, top_k_grammar_ids
export add_programs_to_state!
export next_grammar_id, reset_grammar_counter!

end
