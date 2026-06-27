# test_voc_gate.jl — the Value-of-Computation gate on perturb_grammar (collapse-towers Phase 5).
#
# net_voc is the structural twin of net_voi, in LOG-PRIOR currency (R1): perturb_grammar sees only
# (g, freq_table, available_features), so the affordable depth-one value-proxy is the change in the
# program-space complexity prior — net_value(Δcomplexity_logprior, compute_cost). It governs the
# COMPRESSION class only (R2): :add_rule, whose value is exactly propose_nonterminal's net_payoff
# scaled to nats by λ=log(2). The generative-change ops (modify_threshold, add/remove_feature) are
# excluded and deferred — they change the likelihood over un-entertained programs (escape-mass), which
# depth-one prior-only VOC cannot see. The rand-based selection is GONE: which-perturbation is now a
# deterministic argmax of net_voc.
#
# Asserts: (0) net_voc's arithmetic; (1) degenerate reduction — compute_cost=0 reproduces today's
# :add_rule branch bit-for-bit; (2) directional — compute_cost > net_payoff·log(2) suppresses to a
# no-op, the flip relative to the oracle, no magic number; (3) determinism — two runs on identical
# inputs give the same grammar; (4) no `rand(` survives in the perturbation source (the breach is
# closed at the source level, not just behaviourally).
#
# Run from repo root:
#     julia test/test_voc_gate.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, SubprogramFrequencyTable, Program,
                GTExpr, LTExpr, AndExpr, NotExpr, IfExpr, ActionExpr, ProgramExpr,
                perturb_grammar, propose_nonterminal, analyse_posterior_subtrees,
                expr_complexity, show_expr, net_voc

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

# Canonical structural readout of a grammar's rules (id-independent, name+body).
rules_str(g::Grammar) = sort([string(r.name) * "=" * show_expr(r.body) for r in g.rules])

# A freq_table with a known compressing subtree s = And(GT(:red,0.7), LT(:green,0.3)) (expr_complexity
# 3) shared by `n` distinct programs ⇒ net_payoff = n·(3−1) − (1+3) = 2n − 4.
function shared_subtree_table(n::Int)
    s = AndExpr(GTExpr(:red, 0.7), LTExpr(:green, 0.3))
    progs = Program[IfExpr(s, ActionExpr(:a), ActionExpr(:b)) |> e -> Program(e, 6, 1) for _ in 1:n]
    w = fill(1.0 / n, n)
    (analyse_posterior_subtrees(progs, w; min_frequency = 0.0, min_complexity = 2), s)
end

# The independent oracle for net_payoff (symbols) of a freq_table's best subtree.
function oracle_net_payoff(ft::SubprogramFrequencyTable)   # credence-lint: allow — precedent:test-oracle — independent net_payoff for the gate
    best = argmax(ft.weighted_frequency)
    expr_c = expr_complexity(ft.subtrees[best])
    n_sources = length(ft.source_programs[best])
    n_sources * (expr_c - 1) - (1 + expr_c)
end

println("="^64)
println("net_voc — the VOC gate on perturb_grammar (Phase 5)")
println("="^64)

# ── (0) net_voc arithmetic: net_value(Δcomplexity_logprior, compute_cost), Δ = net_payoff·log(2) ──
check("net_voc(4, 0.0) == 4·log(2)", net_voc(4, 0.0) ≈ 4 * log(2), "got $(net_voc(4,0.0))")
check("net_voc(4, 1.0) == 4·log(2) − 1", net_voc(4, 1.0) ≈ 4 * log(2) - 1.0, "got $(net_voc(4,1.0))")
check("net_voc(0, 0.0) == 0", net_voc(0, 0.0) ≈ 0.0, "got $(net_voc(0,0.0))")
check("net_voc folds compute_cost linearly (no clamp)", net_voc(2, 100.0) ≈ 2 * log(2) - 100.0)

# ── (1) degenerate reduction: compute_cost = 0 ≡ today's deterministic :add_rule branch ──
let
    g = Grammar(Set([:red, :green]), ProductionRule[], 1)
    ft, _ = shared_subtree_table(3)                       # net_payoff = 2·3 − 4 = 2 > 0
    proposed = propose_nonterminal(ft)
    @assert proposed !== nothing
    expected = Grammar(g.feature_set, [g.rules; proposed], 999)   # the :add_rule branch's construction
    got = perturb_grammar(g, ft; compute_cost = 0.0)
    check("compute_cost=0 reproduces :add_rule branch (rules)", rules_str(got) == rules_str(expected),
          "got=$(rules_str(got)) expected=$(rules_str(expected))")
    check("compute_cost=0 preserves feature_set", got.feature_set == g.feature_set)
end

# ── (1b) no compressing subtree (net_payoff ≤ 0) ⇒ deterministic no-op ──
let
    g = Grammar(Set([:red, :green]), ProductionRule[], 1)
    empty_ft = SubprogramFrequencyTable(ProgramExpr[], Float64[], Vector{Int}[])
    got = perturb_grammar(g, empty_ft; compute_cost = 0.0)
    check("empty table ⇒ no-op (no rule added)", isempty(got.rules), "rules=$(rules_str(got))")

    ft1, _ = shared_subtree_table(2)                      # net_payoff = 2·2 − 4 = 0, not > 0
    got1 = perturb_grammar(g, ft1; compute_cost = 0.0)
    check("net_payoff = 0 ⇒ no-op (gate is strict >)", isempty(got1.rules), "rules=$(rules_str(got1))")
end

# ── (2) directional: compute_cost flips the gate exactly at net_payoff·log(2) ──
let
    g = Grammar(Set([:red, :green]), ProductionRule[], 1)
    ft, _ = shared_subtree_table(4)                       # net_payoff = 2·4 − 4 = 4 > 0
    payoff = oracle_net_payoff(ft)                        # credence-lint: allow — precedent:test-oracle — the gate flips at payoff·log(2)
    threshold = payoff * log(2)
    below = perturb_grammar(g, ft; compute_cost = threshold - 1e-6)
    above = perturb_grammar(g, ft; compute_cost = threshold + 1e-6)
    check("compute_cost just below payoff·log(2) ⇒ rule added", length(below.rules) == 1,
          "rules=$(rules_str(below))")
    check("compute_cost just above payoff·log(2) ⇒ no-op (forward compute priced out)",
          isempty(above.rules), "rules=$(rules_str(above))")
end

# ── (3) determinism: two runs on identical (g, ft) ⇒ structurally identical grammar (no rand) ──
let
    g = Grammar(Set([:red, :green]), ProductionRule[], 1)
    ft, _ = shared_subtree_table(3)
    a = perturb_grammar(g, ft)
    b = perturb_grammar(g, ft)
    check("two runs ⇒ identical rules", rules_str(a) == rules_str(b), "a=$(rules_str(a)) b=$(rules_str(b))")
    check("two runs ⇒ identical feature_set", a.feature_set == b.feature_set)
end

# ── (4) source-level guard: the breach is closed — no `rand(` in the perturbation path ──
let
    src = read(joinpath(@__DIR__, "..", "src", "program_space", "perturbation.jl"), String)
    check("perturbation.jl contains no rand( call", !occursin("rand(", src),
          "a rand( call survives in perturbation.jl")
end

println("="^64)
println("ALL CHECKS PASSED — net_voc / VOC gate")
println("="^64)
