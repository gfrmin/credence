"""
    compilation.jl — AST → closure compilation

Walks the AST once, produces a closure. The returned CompiledKernel
has no reference to the original AST.
"""

# ═══════════════════════════════════════
# Kernel compilation: AST → closure
# ═══════════════════════════════════════

"""
    compile_kernel(program, grammar) → CompiledKernel

Walks the AST once, produces a closure that captures threshold values,
feature names, and nonterminal expansions as closed-over constants.
The returned CompiledKernel has no reference to the original AST.
"""
function compile_kernel(program::Program, grammar::Grammar, program_id::Int)::CompiledKernel
    closure = compile_expr(program.expr, grammar.rules)
    CompiledKernel(closure, program.complexity, grammar.id, program_id)
end

"""
    compile_num(e::NumExpr) → (features, temporal_state) → Float64

Compile a numeric expression into a closure. `compile_num(FeatureRef)` is the historical
feature read `(f, _ts) -> get(f, feat, 0.0)`, so a lifted bare-feature atom compiles to a
bit-identical closure. `Div` is protected TRUE division (`x/0 = 0.0` — the Koza-closure
artifact, documented; poles are real: rates, inverse distances); `AQ` is the analytic
quotient `x/√(1+y²)` — total and smooth, the enumeration default (design §5 Q2).
"""
function compile_num(e::FeatureRef)
    feat = e.feature
    (features, _ts) -> get(features, feat, 0.0)
end
function compile_num(e::Times)
    l = compile_num(e.left); r = compile_num(e.right)
    (f, ts) -> l(f, ts) * r(f, ts)
end
function compile_num(e::Plus)
    l = compile_num(e.left); r = compile_num(e.right)
    (f, ts) -> l(f, ts) + r(f, ts)
end
function compile_num(e::Minus)
    l = compile_num(e.left); r = compile_num(e.right)
    (f, ts) -> l(f, ts) - r(f, ts)
end
function compile_num(e::Div)
    l = compile_num(e.left); r = compile_num(e.right)
    (f, ts) -> begin
        d = r(f, ts)
        d == 0.0 ? 0.0 : l(f, ts) / d
    end
end
function compile_num(e::AQ)
    l = compile_num(e.left); r = compile_num(e.right)
    (f, ts) -> l(f, ts) / sqrt(1.0 + r(f, ts)^2)
end
function compile_num(e::Neg)
    c = compile_num(e.child)
    (f, ts) -> -c(f, ts)
end

"""Compile an expression into a closure: (features::Dict{Symbol,Float64}, temporal_state) → Bool."""
function compile_expr(e::GTExpr, _rules)
    lhs_fn = compile_num(e.lhs)
    t = e.threshold
    (features, ts) -> lhs_fn(features, ts) > t
end

function compile_expr(e::LTExpr, _rules)
    lhs_fn = compile_num(e.lhs)
    t = e.threshold
    (features, ts) -> lhs_fn(features, ts) < t
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
    (features, ts) -> begin
        recent = get(ts, :recent, Dict{Symbol,Float64}[])
        length(recent) < n && return false
        all(i -> child(recent[end - i + 1], ts), 1:n)
    end
end

function compile_expr(e::ChangedExpr, rules)
    child = compile_expr(e.child, rules)
    (features, ts) -> begin
        recent = get(ts, :recent, Dict{Symbol,Float64}[])
        isempty(recent) && return false
        current = child(features, ts)
        prev_features = last(recent)
        prev = child(prev_features, ts)
        current != prev
    end
end

function compile_expr(e::SinceExpr, rules)
    p_fn = compile_expr(e.p, rules)
    q_fn = compile_expr(e.q, rules)
    (features, ts) -> begin
        recent = get(ts, :recent, Dict{Symbol,Float64}[])
        last_q = 0
        for i in length(recent):-1:1
            if q_fn(recent[i], ts)
                last_q = i
                break
            end
        end
        last_q == 0 && return false
        all(i -> p_fn(recent[i], ts), last_q:length(recent)) && p_fn(features, ts)
    end
end

# ═══════════════════════════════════════
# Action expression compilation: returns Symbol
# ═══════════════════════════════════════

function compile_expr(e::ActionExpr, _rules)
    action = e.action
    (_sv, _ts) -> action
end

function compile_expr(e::IfExpr, rules)
    pred_fn = compile_expr(e.predicate, rules)
    then_fn = compile_expr(e.then_branch, rules)
    else_fn = compile_expr(e.else_branch, rules)
    (sv, ts) -> pred_fn(sv, ts) ? then_fn(sv, ts) : else_fn(sv, ts)
end

# ═══════════════════════════════════════
# Predicate evaluation (used only during testing, not at conditioning time)
# ═══════════════════════════════════════

function evaluate_predicate(expr::ProgramExpr, features::Dict{Symbol, Float64},
                            rules::Vector{ProductionRule};
                            temporal_state::Dict{Symbol, Any}=Dict{Symbol, Any}())
    fn = compile_expr(expr, rules)
    fn(features, temporal_state)
end
