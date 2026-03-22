"""
    types.jl — AST types, Grammar, Program, CompiledKernel

Core type definitions for the program-space inference layer.
These are domain-independent: any domain that uses program-space
inference over grammars shares these types.
"""

# ═══════════════════════════════════════
# AST types for program expressions
# ═══════════════════════════════════════

abstract type ProgramExpr end

struct GTExpr <: ProgramExpr
    feature::Symbol
    threshold::Float64
end

struct LTExpr <: ProgramExpr
    feature::Symbol
    threshold::Float64
end

struct AndExpr <: ProgramExpr
    left::ProgramExpr
    right::ProgramExpr
end

struct OrExpr <: ProgramExpr
    left::ProgramExpr
    right::ProgramExpr
end

struct NotExpr <: ProgramExpr
    child::ProgramExpr
end

struct NonterminalRef <: ProgramExpr
    name::Symbol
end

# Temporal operators
struct PersistsExpr <: ProgramExpr
    child::ProgramExpr
    n::Int             # number of steps (2, 3, or 5)
end

struct ChangedExpr <: ProgramExpr
    child::ProgramExpr
end

struct SinceExpr <: ProgramExpr
    p::ProgramExpr     # has p been true since q was last true?
    q::ProgramExpr
end

# Action expressions — programs evaluate to action symbols
struct ActionExpr <: ProgramExpr
    action::Symbol
end

struct IfExpr <: ProgramExpr
    predicate::ProgramExpr     # Boolean-valued (gt, lt, and, or, not, changed, persists, nonterminal)
    then_branch::ProgramExpr   # action-valued (ActionExpr or nested IfExpr)
    else_branch::ProgramExpr   # action-valued (ActionExpr or nested IfExpr)
end

# ── Pretty printing ──

function show_expr(e::GTExpr)
    "(gt :$(e.feature) $(e.threshold))"
end
function show_expr(e::LTExpr)
    "(lt :$(e.feature) $(e.threshold))"
end
function show_expr(e::AndExpr)
    "AND($(show_expr(e.left)),$(show_expr(e.right)))"
end
function show_expr(e::OrExpr)
    "OR($(show_expr(e.left)),$(show_expr(e.right)))"
end
function show_expr(e::NotExpr)
    "NOT($(show_expr(e.child)))"
end
function show_expr(e::NonterminalRef)
    string(e.name)
end
function show_expr(e::PersistsExpr)
    "PERSISTS($(show_expr(e.child)),$(e.n))"
end
function show_expr(e::ChangedExpr)
    "CHANGED($(show_expr(e.child)))"
end
function show_expr(e::SinceExpr)
    "SINCE($(show_expr(e.p)),$(show_expr(e.q)))"
end
function show_expr(e::ActionExpr)
    string(e.action)
end
function show_expr(e::IfExpr)
    "IF($(show_expr(e.predicate)),$(show_expr(e.then_branch)),$(show_expr(e.else_branch)))"
end

# ═══════════════════════════════════════
# Grammar types
# ═══════════════════════════════════════

struct ProductionRule
    name::Symbol
    body::ProgramExpr
end

struct Grammar
    feature_set::Set{Symbol}
    rules::Vector{ProductionRule}
    complexity::Float64  # precomputed |G|
    id::Int
end

function compute_grammar_complexity(feature_set::Set{Symbol}, rules::Vector{ProductionRule})
    feature_cost = Float64(length(feature_set))
    rules_cost = sum(1.0 + expr_complexity(r.body) for r in rules; init=0.0)
    feature_cost + rules_cost
end

function Grammar(feature_set::Set{Symbol}, rules::Vector{ProductionRule}, id::Int)
    Grammar(feature_set, rules, compute_grammar_complexity(feature_set, rules), id)
end

# ═══════════════════════════════════════
# Program type
# ═══════════════════════════════════════

struct Program
    expr::ProgramExpr        # full expression tree, evaluates to a Symbol
    complexity::Int          # derivation length
    grammar_id::Int
end

# ═══════════════════════════════════════
# CompiledKernel — NO AST FIELD
# ═══════════════════════════════════════

struct CompiledKernel
    # NO AST FIELD. Expression compiled into closure.
    # This is a type-level constraint: if the AST isn't here, it can't be interpreted.
    evaluate::Function       # (features::Dict{Symbol,Float64}, temporal_state) → Symbol
    complexity::Int
    grammar_id::Int
    program_id::Int
end

# ═══════════════════════════════════════
# SubprogramFrequencyTable — for grammar perturbation
# ═══════════════════════════════════════

struct SubprogramFrequencyTable
    subtrees::Vector{ProgramExpr}
    weighted_frequency::Vector{Float64}
    source_programs::Vector{Vector{Int}}  # which program indices contained each subtree
end

# No public constructor — only created by analyse_posterior_subtrees
