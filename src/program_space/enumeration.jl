"""
    enumeration.jl — Complexity scoring and bottom-up program enumeration
"""

# ═══════════════════════════════════════
# Complexity scoring
# ═══════════════════════════════════════

const THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]

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

"""
    enumerate_programs(grammar, max_depth; include_temporal) → Vector{Program}

Bottom-up enumeration of all PREDICT programs up to max_depth.
Depth-1: single predicates and nonterminal refs.
Depth-2: compositions with AND/OR/NOT.
"""
function enumerate_programs(g::Grammar, max_depth::Int;
                            include_temporal::Bool=false,
                            min_log_prior::Float64=-20.0,
                            actions::Vector{Symbol}=Symbol[:classify])::Vector{Program}
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
    for expr in all_exprs
        c = expr_complexity(expr)
        -(g.complexity + c) >= min_log_prior || continue
        for action in actions
            push!(programs, Program(expr, action, c, g.id))
        end
    end
    programs
end
