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
