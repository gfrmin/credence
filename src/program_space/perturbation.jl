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
# Compression payoff — the shared description-length arithmetic
# ═══════════════════════════════════════

"""
    _compression_payoff(table) → Union{Tuple{ProductionRule, Int}, Nothing}

The best `:add_rule` candidate and its description-length payoff (symbols saved), or `nothing` if no
posterior subtree compresses (payoff ≤ 0). Defining the most-frequent subtree `s` as a nonterminal
replaces each of its `n_sources` uses (cost `expr_complexity(s)`) with a reference (cost 1) and adds
the rule once (cost `1 + expr_complexity(s)`):

    net_payoff = n_sources · (expr_complexity(s) − 1) − (1 + expr_complexity(s))   [symbols]

This is the two-part-MDL saving (CLAUDE.md §1.3). It is the single home of the payoff arithmetic,
shared by `propose_nonterminal` (the gate) and `net_voc` (the value), so the two never drift.
"""
function _compression_payoff(table::SubprogramFrequencyTable)::Union{Tuple{ProductionRule, Int}, Nothing}
    isempty(table.subtrees) && return nothing
    best_idx = argmax(table.weighted_frequency)
    best_expr = table.subtrees[best_idx]
    n_sources = length(table.source_programs[best_idx])
    expr_c = expr_complexity(best_expr)
    net_payoff = n_sources * (expr_c - 1) - (1 + expr_c)
    net_payoff > 0 || return nothing
    name = Symbol("NT_", hash(show_expr(best_expr)) % 10000)
    (ProductionRule(name, best_expr), net_payoff)
end

"""
    propose_nonterminal(table) → Union{ProductionRule, Nothing}

Select the highest-weighted-frequency subtree and propose it as a nonterminal iff its compression
payoff justifies the rule cost (`net_payoff > 0`). The `:add_rule` candidate generator for
`perturb_grammar`; the payoff arithmetic lives in `_compression_payoff`.
"""
function propose_nonterminal(table::SubprogramFrequencyTable)::Union{ProductionRule, Nothing}
    result = _compression_payoff(table)
    isnothing(result) ? nothing : result[1]
end

# ═══════════════════════════════════════
# net_voc — Value-of-Computation of a grammar perturbation (collapse-towers Phase 5)
# ═══════════════════════════════════════

"""
    net_voc(net_payoff_symbols, compute_cost) → Float64

The Value-of-Computation of a compression perturbation, depth-one, in **log-prior nats** (R1):
`net_value(Δcomplexity_logprior, compute_cost)`. `perturb_grammar` sees only `(g, freq_table,
available_features)` — no belief, no utilities, no re-conditioning — so achievable EU is unaffordable
depth-one (Russell–Wefald); the affordable value-proxy is the change in the program-space complexity
prior (CLAUDE.md §1.3). A compression saving `net_payoff_symbols` raises the log-prior by

    complexity_logprior(−net_payoff_symbols; λ = log(2)) = log(2) · net_payoff_symbols      [nats]

(the program-axis λ is pinned to `log(2)` by §1.3). The structural twin of `net_voi` (`stdlib.jl`):
the same `net_value` form, in the prior's currency rather than utility — the third representation,
after scalar `net_voi` and Functional-offset routing EU. At `compute_cost = 0` the gate `net_voc > 0`
is exactly `propose_nonterminal`'s `net_payoff > 0`. Governs the COMPRESSION class only (`:add_rule`);
generative-change ops (`:modify_threshold`, `:add_feature`, `:remove_feature`) change the likelihood
over un-entertained programs (the escape-mass frontier), invisible here by construction, and are
deferred to an EU-priced exploration mechanism (master plan, named successor).
"""
net_voc(net_payoff_symbols::Real, compute_cost::Real) =
    net_value(complexity_logprior(-net_payoff_symbols; λ = log(2)), compute_cost)

# ═══════════════════════════════════════
# Grammar perturbation — deterministic argmax over net_voc (no rand; Invariant 1 canalised)
# ═══════════════════════════════════════

"""
    perturb_grammar(g, freq_table, available_features; compute_cost = 0.0) → Grammar

Perturb a grammar by the single compression-class meta-action whose `net_voc` is greatest, applied iff
`net_voc > 0`; otherwise a structural no-op (a fresh grammar id over the same feature_set + rules).
The selection is a **deterministic argmax** — the `rand`-based op choice (the Invariant-1 breach this
phase retires) is gone. `freq_table` is REQUIRED (the type system enforces posterior analysis before
nonterminal proposal). `available_features` is retained for signature stability (the deferred
feature-discovery mechanism will consume it); Scope A does not read it.

Scope A (collapse-towers Phase 5): the compression class is `:add_rule` alone — `propose_nonterminal`'s
proposed rule, gated by `net_voc`. `:remove_rule` (dictionary hygiene) awaits a sound nonterminal
reference count; the generative-change ops await an EU-priced exploration budget — both named
successors in `docs/collapse-towers/master-plan.md`.
"""
function perturb_grammar(g::Grammar, freq_table::SubprogramFrequencyTable,
                          available_features::Set{Symbol};
                          compute_cost::Float64 = 0.0)::Grammar
    noop() = Grammar(g.feature_set, g.rules, next_grammar_id())
    candidate = _compression_payoff(freq_table)
    isnothing(candidate) && return noop()                       # no compressing subtree
    rule, net_payoff = candidate
    rule.name in Set(r.name for r in g.rules) && return noop()  # already present
    net_voc(net_payoff, compute_cost) > 0 || return noop()      # VOC gate (forward compute priced in)
    Grammar(g.feature_set, [g.rules; rule], next_grammar_id())
end

# Backward-compatible 2-argument form (default feature set; forwards compute_cost)
function perturb_grammar(g::Grammar, freq_table::SubprogramFrequencyTable;
                          compute_cost::Float64 = 0.0)::Grammar
    perturb_grammar(g, freq_table, g.feature_set; compute_cost = compute_cost)
end

# ═══════════════════════════════════════
# AST structural equality (for subtree matching)
# ═══════════════════════════════════════

function expr_equal(a::GTExpr, b::GTExpr)
    a.feature == b.feature && a.threshold == b.threshold
end
function expr_equal(a::LTExpr, b::LTExpr)
    a.feature == b.feature && a.threshold == b.threshold
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

# Threshold-node collection/replacement (`collect_threshold_nodes`, `replace_threshold`) was the
# machinery of the retired `:modify_threshold` op. `:modify_threshold` is generative-change — it
# changes which programs the grammar generates (a likelihood effect over un-entertained programs),
# invisible to depth-one prior-only `net_voc` by construction — so collapse-towers Phase 5 deferred it
# to an EU-priced exploration budget (master plan, named successor) and deleted the dead machinery
# rather than leave it unreachable.
