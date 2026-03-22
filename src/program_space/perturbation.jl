"""
    perturbation.jl — Posterior subtree analysis, nonterminal proposal, grammar perturbation

All perturbation grounded in posterior analysis via SubprogramFrequencyTable.
"""

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
    subtree_map = Dict{String, Tuple{ProgramExpr, Float64, Vector{Int}}}()

    for (i, prog) in enumerate(programs)
        w = prog_weights[i]
        w > 1e-15 || continue
        subtrees = extract_subtrees(prog.expr, min_complexity)
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

    subtrees = ProgramExpr[]
    freqs = Float64[]
    sources = Vector{Int}[]

    for (_, (st, freq, src)) in subtree_map
        freq >= min_frequency || continue
        push!(subtrees, st)
        push!(freqs, freq)
        push!(sources, src)
    end

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
function _extract!(result, e::ActionExpr, _min_c)
    # ActionExpr is not a predicate — never extract as a candidate nonterminal
end
function _extract!(result, e::IfExpr, min_c)
    # IfExpr is not a predicate — don't extract it as a candidate nonterminal.
    # Only extract predicate subtrees (Bool-valued) from the predicate branch.
    _extract!(result, e.predicate, min_c)
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

    best_idx = argmax(table.weighted_frequency)
    best_expr = table.subtrees[best_idx]
    best_freq = table.weighted_frequency[best_idx]
    n_sources = length(table.source_programs[best_idx])

    expr_c = expr_complexity(best_expr)
    savings_per_use = expr_c - 1
    rule_cost = 1 + expr_c
    net_payoff = n_sources * savings_per_use - rule_cost

    net_payoff > 0 || return nothing

    name = Symbol("NT_", hash(show_expr(best_expr)) % 10000)
    ProductionRule(name, best_expr)
end

# ═══════════════════════════════════════
# Grammar perturbation
# ═══════════════════════════════════════

"""
    perturb_grammar(g, freq_table, available_source_indices) → Grammar

Perturb a grammar by one operator. The freq_table argument is REQUIRED
(enforced by the type system) to ensure posterior analysis before
nonterminal proposal. available_source_indices is the set of valid
source channel indices for the domain.
"""
function perturb_grammar(g::Grammar, freq_table::SubprogramFrequencyTable,
                          available_indices::Vector{Int})::Grammar
    ops = [:add_channel, :remove_channel, :add_rule, :remove_rule, :modify_threshold]
    op = rand(ops)

    if op == :add_channel
        covered = Set(ch.source_index for ch in g.sensor_config.channels)
        available = setdiff(Set(available_indices), covered)
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
        existing_names = Set(r.name for r in g.rules)
        proposed.name in existing_names && return Grammar(g.sensor_config, g.rules, next_grammar_id())
        return Grammar(g.sensor_config, [g.rules; proposed], next_grammar_id())

    elseif op == :remove_rule
        isempty(g.rules) && return Grammar(g.sensor_config, g.rules, next_grammar_id())
        idx = rand(1:length(g.rules))
        new_rules = [g.rules[i] for i in eachindex(g.rules) if i != idx]
        return Grammar(g.sensor_config, new_rules, next_grammar_id())

    elseif op == :modify_threshold
        isempty(g.rules) && return Grammar(g.sensor_config, g.rules, next_grammar_id())
        rule_idx = rand(1:length(g.rules))
        rule = g.rules[rule_idx]
        nodes = collect_threshold_nodes(rule.body)
        isempty(nodes) && return Grammar(g.sensor_config, g.rules, next_grammar_id())
        node = rand(nodes)
        other_thresholds = filter(t -> t != node.threshold, THRESHOLDS)
        isempty(other_thresholds) && return Grammar(g.sensor_config, g.rules, next_grammar_id())
        new_t = rand(other_thresholds)
        new_body = replace_threshold(rule.body, node, new_t)
        new_rules = [i == rule_idx ? ProductionRule(rule.name, new_body) : g.rules[i]
                     for i in eachindex(g.rules)]
        return Grammar(g.sensor_config, new_rules, next_grammar_id())
    end

    Grammar(g.sensor_config, g.rules, next_grammar_id())
end

# Backward-compatible 2-argument form using available_source_indices from domain
function perturb_grammar(g::Grammar, freq_table::SubprogramFrequencyTable)::Grammar
    perturb_grammar(g, freq_table, collect(0:7))
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
function expr_equal(a::PersistsExpr, b::PersistsExpr)
    a.n == b.n && expr_equal(a.child, b.child)
end
function expr_equal(a::ChangedExpr, b::ChangedExpr)
    expr_equal(a.child, b.child)
end
function expr_equal(a::SinceExpr, b::SinceExpr)
    expr_equal(a.p, b.p) && expr_equal(a.q, b.q)
end
function expr_equal(a::ActionExpr, b::ActionExpr)
    a.action == b.action
end
function expr_equal(a::IfExpr, b::IfExpr)
    expr_equal(a.predicate, b.predicate) && expr_equal(a.then_branch, b.then_branch) && expr_equal(a.else_branch, b.else_branch)
end
function expr_equal(::ProgramExpr, ::ProgramExpr) false end

# ═══════════════════════════════════════
# Threshold node collection and replacement
# ═══════════════════════════════════════

function collect_threshold_nodes(e::GTExpr) ProgramExpr[e] end
function collect_threshold_nodes(e::LTExpr) ProgramExpr[e] end
function collect_threshold_nodes(e::AndExpr)
    vcat(collect_threshold_nodes(e.left), collect_threshold_nodes(e.right))
end
function collect_threshold_nodes(e::OrExpr)
    vcat(collect_threshold_nodes(e.left), collect_threshold_nodes(e.right))
end
function collect_threshold_nodes(e::NotExpr)
    collect_threshold_nodes(e.child)
end
function collect_threshold_nodes(e::NonterminalRef) ProgramExpr[] end
function collect_threshold_nodes(e::PersistsExpr)
    collect_threshold_nodes(e.child)
end
function collect_threshold_nodes(e::ChangedExpr)
    collect_threshold_nodes(e.child)
end
function collect_threshold_nodes(e::SinceExpr)
    vcat(collect_threshold_nodes(e.p), collect_threshold_nodes(e.q))
end
function collect_threshold_nodes(e::ActionExpr) ProgramExpr[] end
function collect_threshold_nodes(e::IfExpr)
    vcat(collect_threshold_nodes(e.predicate), collect_threshold_nodes(e.then_branch), collect_threshold_nodes(e.else_branch))
end

function replace_threshold(e::GTExpr, old::ProgramExpr, new_t::Float64)
    expr_equal(e, old) ? GTExpr(e.channel, new_t) : e
end
function replace_threshold(e::LTExpr, old::ProgramExpr, new_t::Float64)
    expr_equal(e, old) ? LTExpr(e.channel, new_t) : e
end
function replace_threshold(e::AndExpr, old::ProgramExpr, new_t::Float64)
    AndExpr(replace_threshold(e.left, old, new_t), replace_threshold(e.right, old, new_t))
end
function replace_threshold(e::OrExpr, old::ProgramExpr, new_t::Float64)
    OrExpr(replace_threshold(e.left, old, new_t), replace_threshold(e.right, old, new_t))
end
function replace_threshold(e::NotExpr, old::ProgramExpr, new_t::Float64)
    NotExpr(replace_threshold(e.child, old, new_t))
end
function replace_threshold(e::NonterminalRef, _old::ProgramExpr, _new_t::Float64)
    e
end
function replace_threshold(e::PersistsExpr, old::ProgramExpr, new_t::Float64)
    PersistsExpr(replace_threshold(e.child, old, new_t), e.n)
end
function replace_threshold(e::ChangedExpr, old::ProgramExpr, new_t::Float64)
    ChangedExpr(replace_threshold(e.child, old, new_t))
end
function replace_threshold(e::SinceExpr, old::ProgramExpr, new_t::Float64)
    SinceExpr(replace_threshold(e.p, old, new_t), replace_threshold(e.q, old, new_t))
end
function replace_threshold(e::ActionExpr, _old::ProgramExpr, _new_t::Float64)
    e
end
function replace_threshold(e::IfExpr, old::ProgramExpr, new_t::Float64)
    IfExpr(replace_threshold(e.predicate, old, new_t), replace_threshold(e.then_branch, old, new_t), replace_threshold(e.else_branch, old, new_t))
end
