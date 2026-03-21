"""
    types.jl — AST types, Grammar, Program, CompiledKernel

Core type definitions for the program-space inference layer.
These are domain-independent: any domain that uses program-space
inference over grammars shares these types.
"""

# ═══════════════════════════════════════
# Sensor configuration types
# ═══════════════════════════════════════

struct SensorChannel
    source_index::Int       # which dimension of true state
    transform::Symbol       # :identity, :threshold, :delta, :windowed_mean
    noise_σ::Float64        # observation noise
    cost::Float64           # contributes to grammar complexity
end

struct SensorConfig
    channels::Vector{SensorChannel}
end

sensor_config_complexity(sc::SensorConfig) = sum(ch.cost for ch in sc.channels)

n_channels(sc::SensorConfig) = length(sc.channels)

# ═══════════════════════════════════════
# AST types for program expressions
# ═══════════════════════════════════════

abstract type ProgramExpr end

struct GTExpr <: ProgramExpr
    channel::Int       # index into sensor vector (0-based)
    threshold::Float64
end

struct LTExpr <: ProgramExpr
    channel::Int
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

# ── Pretty printing ──

function show_expr(e::GTExpr)
    "GT($(e.channel),$(e.threshold))"
end
function show_expr(e::LTExpr)
    "LT($(e.channel),$(e.threshold))"
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

# ═══════════════════════════════════════
# Grammar types
# ═══════════════════════════════════════

struct ProductionRule
    name::Symbol
    body::ProgramExpr
end

struct Grammar
    sensor_config::SensorConfig
    rules::Vector{ProductionRule}
    complexity::Float64  # precomputed |G|
    id::Int
end

function compute_grammar_complexity(sc::SensorConfig, rules::Vector{ProductionRule})
    sc_cost = sensor_config_complexity(sc)
    rules_cost = sum(1.0 + expr_complexity(r.body) for r in rules; init=0.0)
    sc_cost + rules_cost
end

function Grammar(sc::SensorConfig, rules::Vector{ProductionRule}, id::Int)
    Grammar(sc, rules, compute_grammar_complexity(sc, rules), id)
end

# ═══════════════════════════════════════
# Program type
# ═══════════════════════════════════════

struct Program
    predicate::ProgramExpr   # AST retained for analysis and display
    complexity::Int          # derivation length
    grammar_id::Int
end

# ═══════════════════════════════════════
# CompiledKernel — NO AST FIELD
# ═══════════════════════════════════════

struct CompiledKernel
    # NO AST FIELD. Predicate compiled into closure.
    # This is a type-level constraint: if the AST isn't here, it can't be interpreted.
    evaluate::Function       # (sensor_vector, temporal_state) → Bool
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
