# test_perturb_consumption.jl — the perturbation consumption path (hardening follow-up to
# collapse-towers Phase 5; adversarial review of PR #160, Finding 1).
#
# A structural no-op `perturb_grammar` must return the grammar with its id UNCHANGED. The downstream
# `add_programs_to_state!` deduplicates by `grammar.id`, so a no-op that mints a FRESH id defeats the
# dedup and re-injects the entire program set as fresh Beta(1,1) duplicates — a silent posterior reset
# (an unsanctioned belief modification, A3) reported as progress. The fix: no-op returns the input
# grammar (same id), so a no-op truly changes nothing.
#
# Run from repo root:
#     julia test/test_perturb_consumption.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, SubprogramFrequencyTable, ProgramExpr, Program,
                AndExpr, GTExpr, LTExpr, ActionExpr, IfExpr,
                perturb_grammar, analyse_posterior_subtrees, enumerate_programs, compile_kernel,
                add_programs_to_state!, AgentState, weights, show_expr
using Credence: TaggedBetaPrevision, BetaPrevision, MixturePrevision, CompiledKernel

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

empty_table() = SubprogramFrequencyTable(ProgramExpr[], Float64[], Vector{Int}[])

# Build a minimal AgentState holding one grammar's enumerated programs (mirrors the host setup).
function state_with_grammar(g::Grammar, depth::Int, action_space::Vector{Symbol})
    programs = enumerate_programs(g, depth; action_space=action_space)
    comps = TaggedBetaPrevision[]
    lw = Float64[]
    meta = Tuple{Int, Int}[]
    ck = CompiledKernel[]
    progs = Program[]
    for (pi, p) in enumerate(programs)
        push!(comps, TaggedBetaPrevision(pi, BetaPrevision(1.0, 1.0)))
        push!(lw, -g.complexity * log(2) - p.complexity * log(2))
        push!(meta, (g.id, pi))
        push!(ck, compile_kernel(p, g, pi))
        push!(progs, p)
    end
    AgentState(MixturePrevision(comps, lw), meta, ck, progs, Dict(g.id => g), depth)
end

println("="^64)
println("perturbation consumption — no-op idempotence (Finding 1)")
println("="^64)

# ── (1) unit: a structural no-op preserves the grammar id (the precise fix) ──
let
    g = Grammar(Set([:red, :green]), ProductionRule[], 7)
    noop = perturb_grammar(g, empty_table())
    check("no-op preserves the grammar id (so dedup-by-id re-adds nothing)", noop.id == g.id,
          "g.id=$(g.id) noop.id=$(noop.id)")
    check("no-op preserves feature_set + rules", noop.feature_set == g.feature_set && isempty(noop.rules))

    # net_payoff = 0 (n_sources=2, expr_c=3) is also a no-op and must preserve the id.
    s = AndExpr(GTExpr(FeatureRef(:red), 0.7), LTExpr(FeatureRef(:green), 0.3))
    progs = Program[Program(IfExpr(s, ActionExpr(:a), ActionExpr(:b)), 6, g.id) for _ in 1:2]
    ft0 = analyse_posterior_subtrees(progs, fill(0.5, 2); min_frequency=0.0, min_complexity=2)
    check("net_payoff≤0 no-op also preserves the id", perturb_grammar(g, ft0).id == g.id)
end

# ── (2) integration: a no-op perturbation re-adds ZERO programs (no duplication, no belief reset) ──
let
    g = Grammar(Set([:red, :green]), ProductionRule[], 1)
    state = state_with_grammar(g, 2, Symbol[:a, :b])
    n_before = length(state.belief.components)
    @assert n_before > 0 "fixture must enumerate some programs"

    noop_g = perturb_grammar(g, empty_table())
    state.grammars[noop_g.id] = noop_g
    n_added = add_programs_to_state!(state, noop_g, 2; action_space=Symbol[:a, :b])

    check("no-op perturbation adds ZERO programs (dedup intact)", n_added == 0, "n_added=$n_added")
    check("no-op leaves the component count unchanged (no fresh-Beta duplicates)",
          length(state.belief.components) == n_before,
          "before=$n_before after=$(length(state.belief.components))")
end

# ── (3) min_frequency threshold semantics (Finding 4): weighted_frequency is a sum of posterior
# weights (≤ 1), so any threshold > 1 is unsatisfiable and silently empties the freq_table. The skin
# server (handle_perturb_grammar) used min_frequency=2 — making every wire perturbation a no-op; the
# hosts (and now the skin) use 0.01. ──
let
    s = AndExpr(GTExpr(FeatureRef(:red), 0.7), LTExpr(FeatureRef(:green), 0.3))
    progs = Program[Program(IfExpr(s, ActionExpr(:a), ActionExpr(:b)), 6, 1) for _ in 1:4]
    w = fill(0.25, 4)                                     # normalised posterior weights, sum = 1
    ft_ok  = analyse_posterior_subtrees(progs, w; min_frequency = 0.01, min_complexity = 2)
    ft_bad = analyse_posterior_subtrees(progs, w; min_frequency = 2.0,  min_complexity = 2)
    check("min_frequency=0.01 finds the shared subtree", !isempty(ft_ok.subtrees),
          "expected a non-empty table")
    check("min_frequency=2 (>1) is unsatisfiable ⇒ empty table (the Finding-4 bug)",
          isempty(ft_bad.subtrees), "a >1 weighted-frequency threshold cannot be met")
end

println("="^64)
println("ALL CHECKS PASSED — perturbation consumption (no-op idempotence)")
println("="^64)
