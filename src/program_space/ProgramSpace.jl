"""
    ProgramSpace — Program-space inference layer (Tier 2)

Domain-independent machinery for grammar-based program enumeration,
compilation, perturbation, and belief management. Each domain provides
its own terminal alphabet, seed grammars, and sensor configuration.
"""
module ProgramSpace

include("types.jl")
include("enumeration.jl")
include("compilation.jl")
include("perturbation.jl")
include("agent_state.jl")

export ProgramExpr, GTExpr, LTExpr, AndExpr, OrExpr, NotExpr, NonterminalRef
export PersistsExpr, ChangedExpr, SinceExpr
export ActionExpr, IfExpr
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

end # module ProgramSpace
