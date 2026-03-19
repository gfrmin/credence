"""
    grammar_programs.jl — Grammar, program enumeration, and kernel compilation

This is the core of the program-space agent: the representation of
hypotheses as (grammar, program) pairs, bottom-up enumeration, complexity
scoring, and compilation into closures.

Design invariant: CompiledKernel has NO AST field. Program → CompiledKernel
is a one-way transformation. The type system enforces this.
"""

include("sensor_projection.jl")

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

# ═══════════════════════════════════════
# Complexity scoring
# ═══════════════════════════════════════

function expr_complexity(e::GTExpr) 1 end
function expr_complexity(e::LTExpr) 1 end
function expr_complexity(e::AndExpr) 1 + expr_complexity(e.left) + expr_complexity(e.right) end
function expr_complexity(e::OrExpr) 1 + expr_complexity(e.left) + expr_complexity(e.right) end
function expr_complexity(e::NotExpr) 1 + expr_complexity(e.child) end
function expr_complexity(e::NonterminalRef) 1 end  # nonterminal reference costs 1
function expr_complexity(e::PersistsExpr) 1 + expr_complexity(e.child) end
function expr_complexity(e::ChangedExpr) 1 + expr_complexity(e.child) end
function expr_complexity(e::SinceExpr) 1 + expr_complexity(e.p) + expr_complexity(e.q) end

"""Expanded complexity: cost of expression with nonterminals expanded."""
function expanded_complexity(e::ProgramExpr, rules::Vector{ProductionRule})
    _expanded(e, rules)
end

function _expanded(e::GTExpr, _) 1 end
function _expanded(e::LTExpr, _) 1 end
function _expanded(e::AndExpr, r) 1 + _expanded(e.left, r) + _expanded(e.right, r) end
function _expanded(e::OrExpr, r) 1 + _expanded(e.left, r) + _expanded(e.right, r) end
function _expanded(e::NotExpr, r) 1 + _expanded(e.child, r) end
function _expanded(e::PersistsExpr, r) 1 + _expanded(e.child, r) end
function _expanded(e::ChangedExpr, r) 1 + _expanded(e.child, r) end
function _expanded(e::SinceExpr, r) 1 + _expanded(e.p, r) + _expanded(e.q, r) end
function _expanded(e::NonterminalRef, rules)
    idx = findfirst(r -> r.name == e.name, rules)
    idx === nothing && return 1  # undefined nonterminal
    _expanded(rules[idx].body, rules)
end

# ═══════════════════════════════════════
# Program enumeration (bottom-up)
# ═══════════════════════════════════════

const THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]

"""
    enumerate_programs(grammar, max_depth; include_temporal) → Vector{Program}

Bottom-up enumeration of all PREDICT programs up to max_depth.
Depth-1: single predicates and nonterminal refs.
Depth-2: compositions with AND/OR/NOT.
"""
function enumerate_programs(g::Grammar, max_depth::Int;
                            include_temporal::Bool=false)::Vector{Program}
    n_ch = n_channels(g.sensor_config)

    # Depth 1: atomic predicates + nonterminal references
    atoms = ProgramExpr[]
    for ch_idx in 0:n_ch-1
        for t in THRESHOLDS
            push!(atoms, GTExpr(ch_idx, t))
            push!(atoms, LTExpr(ch_idx, t))
        end
    end
    for r in g.rules
        push!(atoms, NonterminalRef(r.name))
    end

    # Build expressions by depth
    by_depth = Vector{ProgramExpr}[]
    push!(by_depth, atoms)

    for d in 2:max_depth
        prev_all = vcat(by_depth...)
        new_exprs = ProgramExpr[]

        # AND, OR: combine expressions of total depth d
        # Use depth d-1 paired with any smaller expression
        for a in by_depth[d-1]
            for b in atoms  # pair depth d-1 with depth 1
                push!(new_exprs, AndExpr(a, b))
                push!(new_exprs, OrExpr(a, b))
            end
        end

        # NOT of depth d-1
        for a in by_depth[d-1]
            push!(new_exprs, NotExpr(a))
        end

        # Temporal operators (only if enabled)
        if include_temporal && d >= 2
            for a in by_depth[d-1]
                push!(new_exprs, ChangedExpr(a))
                for n in [2, 3, 5]
                    push!(new_exprs, PersistsExpr(a, n))
                end
            end
        end

        push!(by_depth, new_exprs)
    end

    # Collect all expressions and build Programs
    all_exprs = vcat(by_depth...)

    programs = Program[]
    for (i, expr) in enumerate(all_exprs)
        push!(programs, Program(expr, expr_complexity(expr), g.id))
    end
    programs
end

# ═══════════════════════════════════════
# Kernel compilation: AST → closure
# ═══════════════════════════════════════

"""
    compile_kernel(program, grammar) → CompiledKernel

Walks the AST once, produces a closure that captures threshold values,
channel indices, and nonterminal expansions as closed-over constants.
The returned CompiledKernel has no reference to the original AST.
"""
function compile_kernel(program::Program, grammar::Grammar, program_id::Int)::CompiledKernel
    closure = compile_expr(program.predicate, grammar.rules)
    CompiledKernel(closure, program.complexity, grammar.id, program_id)
end

"""Compile an expression into a closure: (sensor_vector, temporal_state) → Bool."""
function compile_expr(e::GTExpr, _rules)
    ch = e.channel + 1  # Julia 1-based
    t = e.threshold
    (sv, _ts) -> sv[ch] > t
end

function compile_expr(e::LTExpr, _rules)
    ch = e.channel + 1
    t = e.threshold
    (sv, _ts) -> sv[ch] < t
end

function compile_expr(e::AndExpr, rules)
    left = compile_expr(e.left, rules)
    right = compile_expr(e.right, rules)
    (sv, ts) -> left(sv, ts) && right(sv, ts)
end

function compile_expr(e::OrExpr, rules)
    left = compile_expr(e.left, rules)
    right = compile_expr(e.right, rules)
    (sv, ts) -> left(sv, ts) || right(sv, ts)
end

function compile_expr(e::NotExpr, rules)
    child = compile_expr(e.child, rules)
    (sv, ts) -> !child(sv, ts)
end

function compile_expr(e::NonterminalRef, rules)
    idx = findfirst(r -> r.name == e.name, rules)
    idx === nothing && error("Undefined nonterminal: $(e.name)")
    compile_expr(rules[idx].body, rules)
end

function compile_expr(e::PersistsExpr, rules)
    child = compile_expr(e.child, rules)
    n = e.n
    (sv, ts) -> begin
        # ts.recent_evaluations is a Vector of recent sensor vectors
        recent = get(ts, :recent, Vector{Float64}[])
        length(recent) < n && return false
        all(i -> child(recent[end - i + 1], ts), 1:n)
    end
end

function compile_expr(e::ChangedExpr, rules)
    child = compile_expr(e.child, rules)
    (sv, ts) -> begin
        recent = get(ts, :recent, Vector{Float64}[])
        isempty(recent) && return false
        current = child(sv, ts)
        prev_sv = last(recent)
        prev = child(prev_sv, ts)
        current != prev
    end
end

function compile_expr(e::SinceExpr, rules)
    p_fn = compile_expr(e.p, rules)
    q_fn = compile_expr(e.q, rules)
    (sv, ts) -> begin
        recent = get(ts, :recent, Vector{Float64}[])
        # Find last time q was true, check p has been true since
        last_q = 0
        for i in length(recent):-1:1
            if q_fn(recent[i], ts)
                last_q = i
                break
            end
        end
        last_q == 0 && return false
        all(i -> p_fn(recent[i], ts), last_q:length(recent)) && p_fn(sv, ts)
    end
end

# ═══════════════════════════════════════
# Predicate evaluation (used only during testing, not at conditioning time)
# ═══════════════════════════════════════

function evaluate_predicate(expr::ProgramExpr, sensor_vector::Vector{Float64},
                            rules::Vector{ProductionRule};
                            temporal_state::Dict{Symbol, Any}=Dict{Symbol, Any}())
    fn = compile_expr(expr, rules)
    fn(sensor_vector, temporal_state)
end

# ═══════════════════════════════════════
# Seed grammars
# ═══════════════════════════════════════

let grammar_counter = Ref(0)
    global function next_grammar_id()
        grammar_counter[] += 1
        grammar_counter[]
    end
    global function reset_grammar_counter!()
        grammar_counter[] = 0
    end
end

function generate_seed_grammars()::Vector{Grammar}
    reset_grammar_counter!()
    grammars = Grammar[]

    # 1. Empty grammar — minimal 2-channel (red + speed), no rules
    push!(grammars, Grammar(minimal_sensor_config(), ProductionRule[], next_grammar_id()))

    # 2. Colour-only grammar — r, g, b channels, no rules
    push!(grammars, Grammar(colour_sensor_config(), ProductionRule[], next_grammar_id()))

    # 3. Motion grammar — speed + wall_dist, no rules
    push!(grammars, Grammar(motion_sensor_config(), ProductionRule[], next_grammar_id()))

    # 4. Full sensor grammar — all 8 channels, no rules
    push!(grammars, Grammar(full_sensor_config(), ProductionRule[], next_grammar_id()))

    # 5. Colour + speed grammar — r, g, b, speed
    push!(grammars, Grammar(colour_speed_sensor_config(), ProductionRule[], next_grammar_id()))

    # 6. Colour grammar with RED nonterminal
    red_body = AndExpr(GTExpr(0, 0.7), AndExpr(LTExpr(1, 0.3), LTExpr(2, 0.3)))
    push!(grammars, Grammar(
        colour_sensor_config(),
        [ProductionRule(:RED, red_body)],
        next_grammar_id()))

    # 7. Colour grammar with BLUE nonterminal
    blue_body = AndExpr(LTExpr(0, 0.3), AndExpr(LTExpr(1, 0.3), GTExpr(2, 0.7)))
    push!(grammars, Grammar(
        colour_sensor_config(),
        [ProductionRule(:BLUE, blue_body)],
        next_grammar_id()))

    # 8. Motion grammar with MOVING nonterminal
    push!(grammars, Grammar(
        motion_sensor_config(),
        [ProductionRule(:MOVING, GTExpr(0, 0.3))],  # channel 0 in motion config = speed
        next_grammar_id()))

    # 9. Motion grammar with NEAR_WALL nonterminal
    push!(grammars, Grammar(
        motion_sensor_config(),
        [ProductionRule(:NEAR_WALL, LTExpr(1, 0.3))],  # channel 1 = wall_dist
        next_grammar_id()))

    # 10. Colour+speed with RED and MOVING nonterminals
    red_cs = AndExpr(GTExpr(0, 0.7), AndExpr(LTExpr(1, 0.3), LTExpr(2, 0.3)))
    moving_cs = GTExpr(3, 0.3)  # channel 3 = speed in colour+speed config
    push!(grammars, Grammar(
        colour_speed_sensor_config(),
        [ProductionRule(:RED, red_cs), ProductionRule(:MOVING, moving_cs)],
        next_grammar_id()))

    # 11. Colour+speed with RED_AND_MOVING compound nonterminal
    ram = AndExpr(
        AndExpr(GTExpr(0, 0.7), AndExpr(LTExpr(1, 0.3), LTExpr(2, 0.3))),
        GTExpr(3, 0.3))
    push!(grammars, Grammar(
        colour_speed_sensor_config(),
        [ProductionRule(:RED_AND_MOVING, ram)],
        next_grammar_id()))

    # 12. Full sensor with colour nonterminals
    push!(grammars, Grammar(
        full_sensor_config(),
        [ProductionRule(:RED, AndExpr(GTExpr(0, 0.7), AndExpr(LTExpr(1, 0.3), LTExpr(2, 0.3)))),
         ProductionRule(:BLUE, AndExpr(LTExpr(0, 0.3), AndExpr(LTExpr(1, 0.3), GTExpr(2, 0.7))))],
        next_grammar_id()))

    grammars
end

# ═══════════════════════════════════════
# Posterior subtree analysis
# ═══════════════════════════════════════

"""
    analyse_posterior_subtrees(programs, weights; ...) → SubprogramFrequencyTable

Walk each program's AST, extract all subtrees of depth ≥ min_complexity.
Weight each subtree occurrence by the program's posterior weight.
Aggregate across programs.
"""
function analyse_posterior_subtrees(
    programs::Vector{Program},
    prog_weights::Vector{Float64};
    min_frequency::Float64=0.1,
    min_complexity::Int=2
)::SubprogramFrequencyTable
    # Collect subtrees with their weighted frequencies
    subtree_map = Dict{String, Tuple{ProgramExpr, Float64, Vector{Int}}}()

    for (i, prog) in enumerate(programs)
        w = prog_weights[i]
        w > 1e-15 || continue  # skip negligible programs
        subtrees = extract_subtrees(prog.predicate, min_complexity)
        for st in subtrees
            key = show_expr(st)
            if haskey(subtree_map, key)
                entry = subtree_map[key]
                subtree_map[key] = (entry[1], entry[2] + w, push!(entry[3], i))
            else
                subtree_map[key] = (st, w, [i])
            end
        end
    end

    # Filter by min_frequency and build table
    subtrees = ProgramExpr[]
    freqs = Float64[]
    sources = Vector{Int}[]

    for (_, (st, freq, src)) in subtree_map
        freq >= min_frequency || continue
        push!(subtrees, st)
        push!(freqs, freq)
        push!(sources, src)
    end

    # Sort by frequency descending
    perm = sortperm(freqs, rev=true)
    SubprogramFrequencyTable(subtrees[perm], freqs[perm], sources[perm])
end

"""Extract all subtrees of an expression with complexity ≥ min_c."""
function extract_subtrees(e::ProgramExpr, min_c::Int)::Vector{ProgramExpr}
    result = ProgramExpr[]
    _extract!(result, e, min_c)
    result
end

function _extract!(result, e::GTExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
end
function _extract!(result, e::LTExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
end
function _extract!(result, e::NonterminalRef, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
end
function _extract!(result, e::AndExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
    _extract!(result, e.left, min_c)
    _extract!(result, e.right, min_c)
end
function _extract!(result, e::OrExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
    _extract!(result, e.left, min_c)
    _extract!(result, e.right, min_c)
end
function _extract!(result, e::NotExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
    _extract!(result, e.child, min_c)
end
function _extract!(result, e::PersistsExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
    _extract!(result, e.child, min_c)
end
function _extract!(result, e::ChangedExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
    _extract!(result, e.child, min_c)
end
function _extract!(result, e::SinceExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
    _extract!(result, e.p, min_c)
    _extract!(result, e.q, min_c)
end

# ═══════════════════════════════════════
# Nonterminal proposal — requires SubprogramFrequencyTable
# ═══════════════════════════════════════

"""
    propose_nonterminal(table) → Union{ProductionRule, Nothing}

Select the subtree with highest weighted frequency.
Propose it as a nonterminal if its compression payoff justifies the
cost of adding a grammar rule.
"""
function propose_nonterminal(table::SubprogramFrequencyTable)::Union{ProductionRule, Nothing}
    isempty(table.subtrees) && return nothing

    # Best candidate: highest weighted frequency
    best_idx = argmax(table.weighted_frequency)
    best_expr = table.subtrees[best_idx]
    best_freq = table.weighted_frequency[best_idx]
    n_sources = length(table.source_programs[best_idx])

    # Compression payoff: each occurrence saves (complexity - 1) derivation steps
    # but adding the rule costs (1 + complexity)
    expr_c = expr_complexity(best_expr)
    savings_per_use = expr_c - 1
    rule_cost = 1 + expr_c
    net_payoff = n_sources * savings_per_use - rule_cost

    net_payoff > 0 || return nothing

    # Generate a name
    name = Symbol("NT_", hash(show_expr(best_expr)) % 10000)
    ProductionRule(name, best_expr)
end

# ═══════════════════════════════════════
# Grammar perturbation
# ═══════════════════════════════════════

"""
    perturb_grammar(g, freq_table) → Grammar

Perturb a grammar by one operator. The freq_table argument is REQUIRED
(enforced by the type system) to ensure posterior analysis before
nonterminal proposal.
"""
function perturb_grammar(g::Grammar, freq_table::SubprogramFrequencyTable)::Grammar
    ops = [:add_channel, :remove_channel, :add_rule, :remove_rule, :modify_threshold]
    op = rand(ops)

    if op == :add_channel
        covered = Set(ch.source_index for ch in g.sensor_config.channels)
        available = setdiff(Set(available_source_indices()), covered)
        isempty(available) && return Grammar(g.sensor_config, g.rules, next_grammar_id())
        new_idx = rand(collect(available))
        new_ch = SensorChannel(new_idx, :identity, 0.05, 1.0)
        new_sc = SensorConfig([g.sensor_config.channels; new_ch])
        return Grammar(new_sc, g.rules, next_grammar_id())

    elseif op == :remove_channel
        length(g.sensor_config.channels) <= 1 && return Grammar(g.sensor_config, g.rules, next_grammar_id())
        idx = rand(1:length(g.sensor_config.channels))
        new_channels = [g.sensor_config.channels[i] for i in eachindex(g.sensor_config.channels) if i != idx]
        return Grammar(SensorConfig(new_channels), g.rules, next_grammar_id())

    elseif op == :add_rule
        proposed = propose_nonterminal(freq_table)
        isnothing(proposed) && return Grammar(g.sensor_config, g.rules, next_grammar_id())
        # Avoid duplicate names
        existing_names = Set(r.name for r in g.rules)
        proposed.name in existing_names && return Grammar(g.sensor_config, g.rules, next_grammar_id())
        return Grammar(g.sensor_config, [g.rules; proposed], next_grammar_id())

    elseif op == :remove_rule
        isempty(g.rules) && return Grammar(g.sensor_config, g.rules, next_grammar_id())
        idx = rand(1:length(g.rules))
        new_rules = [g.rules[i] for i in eachindex(g.rules) if i != idx]
        return Grammar(g.sensor_config, new_rules, next_grammar_id())

    elseif op == :modify_threshold
        # Pick a random threshold in a random rule's body and shift it
        isempty(g.rules) && return Grammar(g.sensor_config, g.rules, next_grammar_id())
        # Just return unchanged for now — threshold modification is a refinement
        return Grammar(g.sensor_config, g.rules, next_grammar_id())
    end

    Grammar(g.sensor_config, g.rules, next_grammar_id())
end

# ═══════════════════════════════════════
# AST structural equality (for subtree matching)
# ═══════════════════════════════════════

function expr_equal(a::GTExpr, b::GTExpr)
    a.channel == b.channel && a.threshold == b.threshold
end
function expr_equal(a::LTExpr, b::LTExpr)
    a.channel == b.channel && a.threshold == b.threshold
end
function expr_equal(a::AndExpr, b::AndExpr)
    expr_equal(a.left, b.left) && expr_equal(a.right, b.right)
end
function expr_equal(a::OrExpr, b::OrExpr)
    expr_equal(a.left, b.left) && expr_equal(a.right, b.right)
end
function expr_equal(a::NotExpr, b::NotExpr)
    expr_equal(a.child, b.child)
end
function expr_equal(a::NonterminalRef, b::NonterminalRef)
    a.name == b.name
end
function expr_equal(::ProgramExpr, ::ProgramExpr) false end
