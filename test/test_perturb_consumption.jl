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
                add_programs_to_state!, ExploreObservation, AgentState, weights, show_expr
using Credence: TaggedBetaPrevision, BetaPrevision, MixturePrevision, CompiledKernel
using Credence: NonterminalRef, collect_nonterminal_refs!, perturbation_voc, propose_nonterminal

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
    n_added = add_programs_to_state!(state, noop_g, 2; observations=ExploreObservation[], action_space=Symbol[:a, :b])

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

# ── (4) cross-grammar dangling nonterminal (the 2026-07-03 gate-run regression) ──
# The frequency table spans the posterior over ALL grammars. The globally best subtree may
# reference a nonterminal defined only in its ORIGIN grammar; installing it into another grammar
# creates a dangling NonterminalRef, and every later enumeration of that lineage crashes in
# compile_expr ("Undefined nonterminal"). The fix: _compression_payoff's argmax is scoped to
# subtrees whose refs resolve in the TARGET grammar — shared by perturb_grammar (transition),
# perturbation_voc (score), and compression_exhausted (signal), so the three stay one function.
let
    origin_rule = ProductionRule(:NT_ORIGIN, GTExpr(FeatureRef(:red), 0.7))
    gA = Grammar(Set([:red, :blue]), [origin_rule], 41)          # defines :NT_ORIGIN
    gB = Grammar(Set([:red, :blue]), ProductionRule[], 42)       # does NOT

    # The dominant subtrees reference :NT_ORIGIN (4 sources × 0.2); plain, fully-resolvable
    # subtrees are next (3 sources × 0.15). All have positive MDL payoff.
    nt_subtree = AndExpr(NonterminalRef(:NT_ORIGIN), GTExpr(FeatureRef(:red), 0.7))
    plain_subtree = AndExpr(GTExpr(FeatureRef(:red), 0.7), LTExpr(FeatureRef(:blue), 0.3))
    progs = Program[]
    for _ in 1:4
        push!(progs, Program(IfExpr(nt_subtree, ActionExpr(:a), ActionExpr(:b)), 6, gA.id))
    end
    for _ in 1:3
        push!(progs, Program(IfExpr(plain_subtree, ActionExpr(:a), ActionExpr(:b)), 6, gB.id))
    end
    w = [fill(0.2, 4); fill(0.15, 3)]
    ft = analyse_posterior_subtrees(progs, w; min_frequency = 0.0, min_complexity = 2)

    proposed = propose_nonterminal(ft)   # the un-targeted legacy surface: global argmax
    check("(4) precondition: the GLOBAL best subtree references the origin-only nonterminal",
          proposed !== nothing && :NT_ORIGIN in collect_nonterminal_refs!(Set{Symbol}(), proposed.body),
          "proposed=$(proposed === nothing ? "nothing" : show_expr(proposed.body))")

    # Target B (lacks :NT_ORIGIN): the invalid subtrees are skipped; the best RESOLVABLE subtree
    # is installed instead — never a dangling reference.
    gB2 = perturb_grammar(gB, ft)
    check("(4) perturbing the foreign grammar still compresses (fallthrough to a valid subtree)",
          gB2.id != gB.id && length(gB2.rules) == 1, "id=$(gB2.id) rules=$(length(gB2.rules))")
    check("(4) the installed rule has NO dangling refs (B has no other rules ⇒ refs must be empty)",
          isempty(collect_nonterminal_refs!(Set{Symbol}(), gB2.rules[1].body)),
          show_expr(gB2.rules[1].body))

    # The crash repro: enumerate + compile the perturbed lineage — must not throw.
    for (pi, p) in enumerate(enumerate_programs(gB2, 3; action_space = Symbol[:a, :b]))
        compile_kernel(p, gB2, pi)
    end
    check("(4) enumerate + compile of the perturbed lineage does not crash", true)

    # Target A (defines :NT_ORIGIN): targeting must not block valid installs — every installed
    # rule's refs resolve within A's own rules.
    gA2 = perturb_grammar(gA, ft)
    rule_names_A = Set(r.name for r in gA2.rules)
    check("(4) the origin grammar still compresses, with every ref resolvable there",
          gA2.id != gA.id &&
          all(issubset(collect_nonterminal_refs!(Set{Symbol}(), r.body), rule_names_A)
              for r in gA2.rules))

    # Score == transition: VOC is positive exactly when the executor changes the grammar,
    # for both targets (the shared _best_compression_candidate core).
    check("(4) perturbation_voc > 0 ⟺ perturb_grammar changes the grammar (both targets)",
          (perturbation_voc(gB, ft) > 0.0) == (gB2.id != gB.id) &&
          (perturbation_voc(gA, ft) > 0.0) == (gA2.id != gA.id))
end

println("="^64)
println("ALL CHECKS PASSED — perturbation consumption (no-op idempotence)")
println("="^64)
