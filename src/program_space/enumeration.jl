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
function expr_complexity(e::ActionExpr) 1 end
function expr_complexity(e::IfExpr) 1 + expr_complexity(e.predicate) + expr_complexity(e.then_branch) + expr_complexity(e.else_branch) end

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
function _expanded(e::ActionExpr, _) 1 end
function _expanded(e::IfExpr, r) 1 + _expanded(e.predicate, r) + _expanded(e.then_branch, r) + _expanded(e.else_branch, r) end

# ═══════════════════════════════════════
# Program enumeration (bottom-up)
# ═══════════════════════════════════════

"""
    enumerate_programs(grammar, max_depth; ...) → Vector{Program}

Bottom-up enumeration of programs as expression trees that evaluate to actions.

Depth semantics (overall program tree depth, not just predicate depth):
- Depth 1: ActionExpr(a) — constant action programs
- Depth 2: IfExpr(depth-1-pred, a1, a2) — single predicates with branching
- Depth 3: IfExpr(depth-2-pred, a1, a2) — compound predicates with branching,
           plus nested IfExpr in branches

Predicate depth = program depth - 1. To get AND/OR/NOT predicates (old depth 2),
use max_depth=3.
"""
function enumerate_programs(g::Grammar, max_depth::Int;
                            include_temporal::Bool=false,
                            min_log_prior::Float64=-20.0,
                            action_space::Vector{Symbol}=Symbol[:classify])::Vector{Program}
    max_complexity = -min_log_prior - g.complexity  # early pruning threshold

    # ── Phase 1: enumerate predicate expressions by depth ──
    # Predicates are Boolean-valued (GT, LT, AND, OR, NOT, temporal, nonterminal)

    atoms = ProgramExpr[]
    for feat in sort(collect(g.feature_set))  # sorted for deterministic enumeration
        for t in THRESHOLDS
            push!(atoms, GTExpr(feat, t))
            push!(atoms, LTExpr(feat, t))
        end
    end
    for r in g.rules
        push!(atoms, NonterminalRef(r.name))
    end

    pred_by_depth = Vector{ProgramExpr}[]
    push!(pred_by_depth, atoms)

    # Build higher-depth predicates up to max_depth-1 (program depth = pred depth + 1)
    max_pred_depth = max_depth - 1
    for d in 2:max_pred_depth
        new_preds = ProgramExpr[]

        # AND, OR: pair depth d-1 with atoms
        for a in pred_by_depth[d-1]
            for b in atoms
                push!(new_preds, AndExpr(a, b))
                push!(new_preds, OrExpr(a, b))
            end
        end

        # NOT of depth d-1
        for a in pred_by_depth[d-1]
            push!(new_preds, NotExpr(a))
        end

        # Temporal operators (only if enabled)
        if include_temporal
            for a in pred_by_depth[d-1]
                push!(new_preds, ChangedExpr(a))
                for n in [2, 3, 5]
                    push!(new_preds, PersistsExpr(a, n))
                end
            end
        end

        push!(pred_by_depth, new_preds)
    end

    # ── Phase 2: build programs from predicates + action_space ──

    programs = Program[]

    # Depth 1: constant action programs (ActionExpr)
    for a in action_space
        c = 1  # expr_complexity(ActionExpr) = 1
        c <= max_complexity || continue
        push!(programs, Program(ActionExpr(a), c, g.id))
    end

    # Depth 2..max_depth: IfExpr with depth-(d-1) predicates and flat action branches
    for pred_depth in 1:max_pred_depth
        pred_depth <= length(pred_by_depth) || continue
        for pred in pred_by_depth[pred_depth]
            pred_c = expr_complexity(pred)
            # Early pruning: IfExpr adds 1 + pred_c + 1 + 1 = 3 + pred_c minimum
            3 + pred_c > max_complexity && continue
            for a1 in action_space, a2 in action_space
                a1 == a2 && continue  # skip tautologies: (if pred a a) ≡ (a)
                expr = IfExpr(pred, ActionExpr(a1), ActionExpr(a2))
                c = 3 + pred_c  # 1 (if) + pred_c + 1 (a1) + 1 (a2)
                push!(programs, Program(expr, c, g.id))
            end
        end
    end

    programs
end
