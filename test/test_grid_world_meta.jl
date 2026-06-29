# test_grid_world_meta.jl — the grid-world meta-action EU gates (#174 PR 2).
#
# PR 2 DROPS the compression_exhausted HARD veto from :gw_explore and :gw_add_feature. compression is
# prior-only and never confounds a fit-side VOI (Move 2 Q3), so it is a soft cost-ordering preference —
# already carried by the meta-action cost asymmetry (GW_PERTURB_COST 0.05 < GW_EXPLORE/ADD_FEATURE_COST 0.10)
# in the caller's argmax, the same way :gw_enumerate_more/:gw_deepen compete — NOT a veto. No discount
# constant: the host compares PROXY EUs in a common scale, so the Q5 currency gap (engine-level) never arises
# here; the principled end-state is one net-EU argmax once Move 5 closes Q5. threshold_exhausted STAYS hard in
# :gw_add_feature (it genuinely confounds feature Δℓ). These are the FIRST tests to exercise compute_gw_meta_eu.
#
# Sections:
#   §1  :gw_explore — compression-not-exhausted no longer vetoes (the dropped rung; RED→GREEN).
#   §2  :gw_explore — the compression-EXHAUSTED case is unchanged (stable pin, green pre+post).
#   §3a :gw_add_feature — compression-not-exhausted no longer vetoes (empty buffer ⇒ threshold vacuous).
#   §3b :gw_add_feature — threshold_exhausted STAYS hard (precondition-guarded: explore_grammar refines ⇒ -Inf).
#
# Run: julia test/test_grid_world_meta.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, Program, GTExpr, AndExpr, IfExpr, ActionExpr, NonterminalRef,
                AgentState, MixturePrevision, TaggedBetaPrevision, BetaPrevision, Prevision,
                compile_kernel, update_learning_regime, plateau_probability,
                analyse_posterior_subtrees, compression_exhausted, explore_grammar, ExploreObservation

include(joinpath(@__DIR__, "..", "apps", "julia", "grid_world", "host.jl"))

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

# Minimal AgentState over a single program with a chosen top grammar (template: test_program_space TEST 14c).
function mk_state(g::Grammar, prog::Program; gid::Int = g.id)
    comps = Prevision[TaggedBetaPrevision(1, BetaPrevision(1.0, 1.0))]
    AgentState(MixturePrevision(comps, [0.0]), [(gid, 1)],
               [compile_kernel(prog, g, 1)], Program[prog], Dict(gid => g), 2)
end

# Drive the regime to a PLATEAU (bouncing-flat residuals ⇒ high plateau_probability; cf. test_saturation phaseB).
function plateau!(state)
    prev = nothing
    for ℓ in [0.755, 0.758, 0.752, 0.757, 0.754, 0.756, 0.753, 0.755, 0.754, 0.756]
        state.learning_regime = update_learning_regime(state.learning_regime, prev, ℓ)
        prev = ℓ
    end
    state
end

# A grammar whose top has a DEAD rule ⇒ compression NOT exhausted; the program references only :LIVE.
function compressible_state()
    g = Grammar(Set([:red, :blue]),
                [ProductionRule(:LIVE, GTExpr(:red, 0.7)), ProductionRule(:DEAD, GTExpr(:blue, 0.5))], 1)
    prog = Program(IfExpr(NonterminalRef(:LIVE), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    plateau!(mk_state(g, prog))
end

println("="^64)
println("grid-world meta gates — compression rung dropped (#174 PR 2)")
println("="^64)

# ── §1  :gw_explore — compression-not-exhausted no longer vetoes (the dropped hard rung) ──
let
    state = compressible_state()
    ft = analyse_posterior_subtrees(state.all_programs, Credence.weights(state.belief);
                                    min_frequency = 0.01, min_complexity = 2)
    check("§1 precondition: compression NOT exhausted (dead rule present)",
          compression_exhausted(state.grammars[1], ft) == false)
    pl = plateau_probability(state.learning_regime)
    check("§1 precondition: plateau prior is high (exploration viable)", pl > 0.6, "plateau=$pl")

    eu = compute_gw_meta_eu(state, :gw_explore, 0.0, 10.0, ExploreObservation[]; meta_cost_this_turn = 0.0)
    expected = pl * GW_EXPLORE_BASE - GW_EXPLORE_COST
    # POST-drop: the soft plateau proxy, NOT -Inf. (PRE-drop the hard rung returned -Inf here — the RED.)
    check("§1 :gw_explore NOT vetoed by available compression (hard rung gone)", eu > -Inf, "eu=$eu")
    check("§1 :gw_explore EU == the soft plateau proxy (no compression term)", eu ≈ expected,
          "eu=$eu expected=$expected")
end

# ── §2  :gw_explore — the compression-EXHAUSTED case is unchanged (stable pin, green pre+post) ──
let
    g = Grammar(Set([:red]), ProductionRule[], 2)   # no rules ⇒ nothing to compress ⇒ exhausted
    prog = Program(IfExpr(GTExpr(:red, 0.5), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    state = plateau!(mk_state(g, prog))
    ft = analyse_posterior_subtrees(state.all_programs, Credence.weights(state.belief);
                                    min_frequency = 0.01, min_complexity = 2)
    check("§2 precondition: compression EXHAUSTED (no rules, no compressible subtree)",
          compression_exhausted(state.grammars[2], ft) == true)
    pl = plateau_probability(state.learning_regime)
    eu = compute_gw_meta_eu(state, :gw_explore, 0.0, 10.0, ExploreObservation[]; meta_cost_this_turn = 0.0)
    check("§2 exhausted case unchanged: eu == plateau·BASE − COST",
          eu ≈ pl * GW_EXPLORE_BASE - GW_EXPLORE_COST, "eu=$eu")
end

# ── §3a  :gw_add_feature — compression-not-exhausted no longer vetoes (empty buffer ⇒ threshold vacuous) ──
let
    state = compressible_state()
    pl = plateau_probability(state.learning_regime)
    eu = compute_gw_meta_eu(state, :gw_add_feature, 0.0, 10.0, ExploreObservation[]; meta_cost_this_turn = 0.0)
    expected = pl * GW_ADD_FEATURE_BASE - GW_ADD_FEATURE_COST
    # POST-drop: not vetoed by compression; empty buffer ⇒ explore_grammar no-ops ⇒ threshold vacuously
    # exhausted ⇒ the soft proxy. (PRE-drop the compression rung returned -Inf here — the RED.)
    check("§3a :gw_add_feature NOT vetoed by compression (hard rung gone)", eu ≈ expected,
          "eu=$eu expected=$expected")
end

# ── §3b  :gw_add_feature — threshold_exhausted STAYS hard (precondition-guarded behavioural pin) ──
let
    g = Grammar(Set([:red]), ProductionRule[], 3)   # exhausted (no rules) — isolates the threshold rung
    prog = Program(IfExpr(GTExpr(:red, 0.5), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    state = plateau!(mk_state(g, prog))
    # A buffer separable only at 0.4 (a midpoint off the default grid {0.1,0.3,0.5,0.7,0.9}): 0.35 vs 0.45
    # land on the same side of every existing threshold, so explore_grammar MUST add ~0.4 to fit ⇒ refines.
    buf = ExploreObservation[]
    for _ in 1:6
        push!(buf, ExploreObservation(Dict(:red => 0.35), Dict{Symbol, Any}(), Set([:food]), 1.0))
        push!(buf, ExploreObservation(Dict(:red => 0.45), Dict{Symbol, Any}(), Set([:enemy]), 1.0))
    end
    refined = explore_grammar(g, buf, state.current_max_depth;
                              action_space = Symbol[:food, :enemy], compute_cost = GW_EXPLORE_VOI_FLOOR)
    check("§3b precondition: thresholds NOT exhausted (explore_grammar refines this buffer)",
          refined !== g, "explore_grammar no-opped — the buffer needs to force a refinement")
    eu = compute_gw_meta_eu(state, :gw_add_feature, 0.0, 10.0, buf; meta_cost_this_turn = 0.0)
    check("§3b :gw_add_feature still vetoed when thresholds NOT exhausted (hard rung survives)",
          eu == -Inf, "eu=$eu (threshold_exhausted hard gate should veto)")
end

println("="^64)
println("ALL CHECKS PASSED — grid-world meta gates (compression rung dropped)")
println("="^64)
