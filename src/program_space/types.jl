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

# The default per-feature threshold grid — the seed every grammar starts from. A threshold (`GTExpr`/
# `LTExpr`) is enumerated at each grid point; `Grammar.thresholds` carries a per-feature grid so the
# exploration budget (Move 3) can REFINE it against the belief's residual without touching other
# grammars. Default grammars copy this for every feature, so enumeration is unchanged. Moved here from
# enumeration.jl in Move 3 because it is now grammar-structural data, not an enumeration-private constant.
const THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]

"""
    Grammar(feature_set, rules, id) — a carrier of the complexity prior over program ASTs.

A Grammar declares the space of programs the agent enumerates. Under
Posture 3's de Finettian framing, the complexity prior
`P(program) = 2^{-|program|}` is not a measure over programs but a
**prevision over program ASTs** — expectations against test functions
declared on the program-space support (subprogram-frequency features,
complexity scoring functions, etc.). Grammar carries the structural
data from which the prevision's support is enumerated.

Fields:
- `feature_set::Set{Symbol}` — predicate features available to programs
  enumerated from this grammar.
- `rules::Vector{ProductionRule}` — nonterminal productions expanding
  into expression subtrees.
- `thresholds::Dict{Symbol, Vector{Float64}}` — per-feature grid the
  predicate atoms (`GTExpr`/`LTExpr`) are enumerated at. Defaults to the
  global `THRESHOLDS` for every feature; the exploration budget (Move 3)
  refines it per-feature against the belief's residual. NOT charged in
  `complexity` — the fineness-Occam is carried by the predictive marginal
  likelihood, not the prior (SPEC §1.3 margin; Move 3 Q1(b)≡Q3(a)).
- `complexity::Float64` — precomputed grammar complexity `|G|`,
  contributing `-complexity * log(2)` to each program's log-prior.
  Threshold-count-invariant by the Q1(b) decision.
- `id::Int` — grammar identifier, used by `add_programs_to_state!` to
  track which grammar produced each mixture component.

See `enumerate_programs_as_measure` for the typed-carrier surface
that returns an `EnumerationMeasure{Program}` over the programs a
Grammar generates. The existing `enumerate_programs` function returns
`Vector{Program}` for backward compatibility; both surfaces draw from
the same enumeration logic.
"""
struct Grammar
    feature_set::Set{Symbol}
    rules::Vector{ProductionRule}
    thresholds::Dict{Symbol, Vector{Float64}}  # per-feature grid (Move 3 refinement target)
    complexity::Float64  # precomputed |G| — threshold-count-invariant (Q1(b)/§1.3 margin)
    id::Int
end

function compute_grammar_complexity(feature_set::Set{Symbol}, rules::Vector{ProductionRule})
    feature_cost = Float64(length(feature_set))
    rules_cost = sum(1.0 + expr_complexity(r.body) for r in rules; init=0.0)
    feature_cost + rules_cost
end

"""
    default_thresholds(feature_set) → Dict{Symbol, Vector{Float64}}

The seed per-feature grid: a copy of the global `THRESHOLDS` for every feature. Every grammar built by
the 3-arg `Grammar(feature_set, rules, id)` constructor uses this, so enumeration is unchanged until a
grammar is explicitly refined (Move 3 `explore_grammar`). A `copy` per feature so a later per-feature
refinement never mutates a shared vector.
"""
default_thresholds(feature_set::Set{Symbol}) =
    Dict{Symbol, Vector{Float64}}(f => copy(THRESHOLDS) for f in feature_set)

# 3-arg convenience (the existing call sites): default per-feature grid, computed complexity.
function Grammar(feature_set::Set{Symbol}, rules::Vector{ProductionRule}, id::Int)
    Grammar(feature_set, rules, default_thresholds(feature_set),
            compute_grammar_complexity(feature_set, rules), id)
end

# 4-arg with an explicit per-feature grid — the refined-grammar constructor (Move 3 `explore_grammar`).
# Complexity is computed identically (threshold-count-invariant): a denser grid does NOT raise |G|.
function Grammar(feature_set::Set{Symbol}, rules::Vector{ProductionRule},
                 thresholds::Dict{Symbol, Vector{Float64}}, id::Int)
    Grammar(feature_set, rules, thresholds, compute_grammar_complexity(feature_set, rules), id)
end

# ═══════════════════════════════════════
# Program type
# ═══════════════════════════════════════

"""
    Program(expr, complexity, grammar_id) — a program-space hypothesis.

A Program is a single hypothesis in the agent's program-space mixture.
Its log-prior under the complexity prior is
`-grammar.complexity * log(2) - program.complexity * log(2)`,
composed with the grammar's structural weight. Programs are the
*support* elements of the Grammar-generated prevision (see Grammar
docstring): expectations against test functions over programs compose
via the mixture weights.

The Program type retains the AST (`expr::ProgramExpr`) for structural
analysis — grammar perturbation, subprogram frequency, complexity
scoring. The evaluation-only sibling is `CompiledKernel` (see
`compilation.jl`), which has NO AST field and is used by the hot-path
kernel evaluation. Per Invariant 3 (single-responsibility representations):
Program is for structural analysis; CompiledKernel is for arithmetic.

Fields:
- `expr::ProgramExpr` — the full expression tree; evaluates to a Symbol
  (action) when invoked with features + temporal state.
- `complexity::Int` — derivation length, contributing to the log-prior.
- `grammar_id::Int` — back-reference to the producing Grammar.
"""
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
    # NT names referenced by ≥1 posterior-support program, or `nothing` when no analysis has run.
    # `:remove_rule` (exploration-budget Move 1) consumes this: a rule absent from a *concrete* set is
    # dead (removable, prior-only). `nothing` ⇒ references unknown ⇒ no rule removable. The sentinel is
    # load-bearing: a concrete empty Set means "analysed, every rule dead" — the OPPOSITE of unknown —
    # because the removal predicate `name ∉ referenced` is vacuously true on the empty set.
    referenced_nonterminals::Union{Nothing, Set{Symbol}}
    # FEATURE names referenced by ≥1 posterior-support program — the symmetric partner consumed by
    # `:remove_feature` (#174). Same support set, same `nothing` sentinel (un-analysed ⇒ no feature
    # removable): a feature in `g.feature_set` absent from a *concrete* set AND from every rule body is
    # dead → reclaimable (prior-only MDL, 1 symbol).
    referenced_features::Union{Nothing, Set{Symbol}}
end

# 3-arg convenience constructor: `nothing` references (un-analysed) for BOTH analysis fields. Hand-built
# tables (tests) get the Scope-A-preserving default — `_removal_payoff`/`_feature_removal_payoff` yield no
# candidates, so `:remove_rule`/`:remove_feature` never fire. `analyse_posterior_subtrees` is the only site
# that populates concrete (possibly empty) Sets.
SubprogramFrequencyTable(subtrees::Vector{ProgramExpr}, weighted_frequency::Vector{Float64},
                         source_programs::Vector{Vector{Int}}) =
    SubprogramFrequencyTable(subtrees, weighted_frequency, source_programs, nothing, nothing)
