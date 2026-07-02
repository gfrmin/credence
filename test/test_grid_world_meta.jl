# test_grid_world_meta.jl — the grid-world meta-action scores under the real single-currency
# argmax (dominance move, Phase 3). Re-baselined from the proxy scheme per dominance-design §0
# ("behaviour shift is intended"): assertions pin the REAL VOI values, not the retired
# plateau·BASE − COST proxies.
#
# The #174 PR 2 gate structure survives in the new currency and is re-asserted here:
#   - compression stays SOFT: perturbation_voc competes in the argmax at its prior-only score,
#     no veto in either direction (§1);
#   - threshold_exhausted stays HARD on :gw_add_feature, now with zero constants — the
#     un-confounded feature baseline exists iff the exploration score is ≤ 0 (§3);
#   - the escape-mass tier (entropy heuristic) is saturation-ordered strictly below any
#     positive exact VOI (§3, §5).
#
# Sections:
#   §1  compressible state, empty buffer — perturb scores its real VOC; explore is exactly 0.
#   §2  exhausted grammar, empty buffer, concentrated belief — everything ≤ 0 ⇒ do_nothing.
#   §3  refinable buffer — explore == plateau·exploration_voi > 0; add_feature hard-gated -Inf;
#       escape tier -Inf (saturation-ordered below the positive exact score).
#   §4  thresholds exhausted, new feature separates — add_feature == plateau·feature_discovery_voi.
#   §5  uncertain multi-component belief, exact tier silent — escape == H − escape_cost;
#       enumerate wins the tie with deepen (breadth before depth, GW_META_ACTIONS order).
#   §6  default_eu_max_policy: act-now floor and deterministic tie order on synthetic dicts.
#
# Run: julia test/test_grid_world_meta.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, Program, FeatureRef, GTExpr, AndExpr, IfExpr, ActionExpr, NonterminalRef,
                AgentState, MixturePrevision, TaggedBetaPrevision, BetaPrevision, Prevision,
                compile_kernel, update_learning_regime, plateau_probability,
                analyse_posterior_subtrees, compression_exhausted, ExploreObservation,
                exploration_voi, feature_discovery_voi, perturbation_voc, entropy, weights

include(joinpath(@__DIR__, "..", "apps", "julia", "grid_world", "host.jl"))

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

# Minimal AgentState over k copies of one program under a chosen top grammar.
function mk_state(g::Grammar, prog::Program; gid::Int = g.id, k::Int = 1)
    comps = Prevision[TaggedBetaPrevision(i, BetaPrevision(1.0, 1.0)) for i in 1:k]
    AgentState(MixturePrevision(comps, zeros(k)), [(gid, i) for i in 1:k],
               [compile_kernel(prog, g, i) for i in 1:k], Program[prog for _ in 1:k],
               Dict(gid => g), 2)
end

# Drive the regime to a PLATEAU (bouncing-flat residuals ⇒ high plateau_probability).
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
                [ProductionRule(:LIVE, GTExpr(FeatureRef(:red), 0.7)), ProductionRule(:DEAD, GTExpr(FeatureRef(:blue), 0.5))], 1)
    prog = Program(IfExpr(NonterminalRef(:LIVE), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    plateau!(mk_state(g, prog))
end

const GW_AS = Symbol[:food, :enemy]

println("="^64)
println("grid-world meta scores — the real single-currency argmax (dominance Phase 3)")
println("="^64)

# ── §1  compressible state, empty buffer: perturb scores its REAL prior-only VOC ──
let
    state = compressible_state()
    ft = analyse_posterior_subtrees(state.all_programs, weights(state.belief);
                                    min_frequency = 0.01, min_complexity = 2)
    check("§1 precondition: compression NOT exhausted (dead rule present)",
          compression_exhausted(state.grammars[1], ft) == false)

    scored = score_gw_meta_actions(state, ExploreObservation[])
    check("§1 :gw_perturb_grammar == its real prior-only net VOC (wiring pin)",
          scored[:gw_perturb_grammar] == perturbation_voc(state.grammars[1], ft),
          "scored=$(scored[:gw_perturb_grammar])")
    check("§1 compression VOC is positive here (a reclaim exists)",
          scored[:gw_perturb_grammar] > 0.0)
    # Empty buffer ⇒ the exact lookahead has nothing to price ⇒ exactly 0, NOT a veto (-Inf):
    # compression availability never gates exploration (#174 PR 2, re-asserted in the new currency).
    check("§1 :gw_explore == 0.0 exactly on an empty buffer (no compression veto)",
          scored[:gw_explore] == 0.0, "scored=$(scored[:gw_explore])")
    check("§1 policy selects the one positive-scored op (perturb)",
          default_eu_max_policy(scored) == :gw_perturb_grammar)
end

# ── §2  exhausted grammar, empty buffer, concentrated single-component belief ⇒ do_nothing ──
let
    g = Grammar(Set([:red]), ProductionRule[], 2)   # no rules ⇒ nothing to compress ⇒ exhausted
    prog = Program(IfExpr(GTExpr(FeatureRef(:red), 0.5), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    state = plateau!(mk_state(g, prog))
    ft = analyse_posterior_subtrees(state.all_programs, weights(state.belief);
                                    min_frequency = 0.01, min_complexity = 2)
    check("§2 precondition: compression EXHAUSTED", compression_exhausted(state.grammars[2], ft))

    scored = score_gw_meta_actions(state, ExploreObservation[])
    check("§2 :gw_perturb_grammar == 0.0 exactly (saturation no-op floor)",
          scored[:gw_perturb_grammar] == 0.0)
    check("§2 :gw_explore == 0.0 exactly (empty buffer)", scored[:gw_explore] == 0.0)
    # Single-component belief ⇒ H == 0 ⇒ the escape tier is priced under water by the declared
    # one-bit compute cost: the belief has concentrated, search stops (§0's interlock).
    check("§2 escape tier == −escape_cost exactly (H == 0, concentrated belief)",
          scored[:gw_enumerate_more] == -GW_ESCAPE_COST_DEFAULT &&
          scored[:gw_deepen] == -GW_ESCAPE_COST_DEFAULT,
          "enum=$(scored[:gw_enumerate_more])")
    check("§2 nothing positive ⇒ act now", default_eu_max_policy(scored) == :gw_do_nothing)
end

# ── §3  refinable buffer: explore scores plateau·(real VOI); add_feature + escape hard-gated ──
let
    g = Grammar(Set([:red]), ProductionRule[], 3)
    prog = Program(IfExpr(GTExpr(FeatureRef(:red), 0.5), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    state = plateau!(mk_state(g, prog))
    # Separable only at ~0.4 (off the default grid {0.1,0.3,0.5,0.7,0.9}): 0.35 vs 0.45 land on
    # the same side of every existing threshold, so refinement MUST add ~0.4 ⇒ positive VOI.
    buf = ExploreObservation[]
    for _ in 1:6
        push!(buf, ExploreObservation(Dict(:red => 0.35), Dict{Symbol, Any}(), Set([:food]), 1.0))
        push!(buf, ExploreObservation(Dict(:red => 0.45), Dict{Symbol, Any}(), Set([:enemy]), 1.0))
    end
    voi = exploration_voi(state.grammars[3], buf, state.current_max_depth; action_space = GW_AS)
    check("§3 precondition: refinement VOI is positive on this buffer", voi > 0.0, "voi=$voi")

    pl = plateau_probability(state.learning_regime)
    scored = score_gw_meta_actions(state, buf)
    check("§3 :gw_explore == plateau · exploration_voi (the regime-marginalised real score)",
          scored[:gw_explore] == pl * voi, "scored=$(scored[:gw_explore]) expected=$(pl * voi)")
    check("§3 :gw_explore is positive (plateau soft gate scales, never vetoes)",
          scored[:gw_explore] > 0.0)
    check("§3 :gw_add_feature == -Inf while thresholds NOT exhausted (hard gate, zero constants)",
          scored[:gw_add_feature] == -Inf)
    check("§3 escape tier == -Inf (saturation-ordered strictly below a positive exact VOI)",
          scored[:gw_enumerate_more] == -Inf && scored[:gw_deepen] == -Inf)
    check("§3 policy selects :gw_explore", default_eu_max_policy(scored) == :gw_explore)
end

# ── §4  thresholds exhausted, a NEW feature separates: add_feature scores its real two-axis VOI ──
let
    g = Grammar(Set([:red]), ProductionRule[], 4)
    prog = Program(IfExpr(GTExpr(FeatureRef(:red), 0.5), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    state = plateau!(mk_state(g, prog))
    # :red is CONSTANT across classes (no refinement can help ⇒ thresholds exhausted, the
    # un-confounded baseline exists); :speed ∈ ALL_GW_FEATURES separates cleanly at the default
    # grid ⇒ feature discovery carries real fit value.
    buf = ExploreObservation[]
    for _ in 1:8
        push!(buf, ExploreObservation(Dict(:red => 0.5, :speed => 0.2), Dict{Symbol, Any}(), Set([:food]), 1.0))
        push!(buf, ExploreObservation(Dict(:red => 0.5, :speed => 0.8), Dict{Symbol, Any}(), Set([:enemy]), 1.0))
    end
    voi_explore = exploration_voi(state.grammars[4], buf, state.current_max_depth; action_space = GW_AS)
    check("§4 precondition: thresholds exhausted (no refinement VOI on a constant feature)",
          voi_explore == 0.0, "voi=$voi_explore")
    fdvoi = feature_discovery_voi(state.grammars[4], buf, ALL_GW_FEATURES,
                                  state.current_max_depth; action_space = GW_AS)
    check("§4 precondition: the new feature clears the two-axis bar (Δℓ − log2 > 0)",
          fdvoi > 0.0, "fdvoi=$fdvoi")

    pl = plateau_probability(state.learning_regime)
    scored = score_gw_meta_actions(state, buf)
    check("§4 :gw_add_feature == plateau · feature_discovery_voi (gate open, real score)",
          scored[:gw_add_feature] == pl * fdvoi,
          "scored=$(scored[:gw_add_feature]) expected=$(pl * fdvoi)")
    check("§4 policy selects :gw_add_feature", default_eu_max_policy(scored) == :gw_add_feature)
end

# ── §5  uncertain multi-component belief, exact tier silent: the escape-mass heuristic fires ──
let
    g = Grammar(Set([:red]), ProductionRule[], 5)
    prog = Program(IfExpr(GTExpr(FeatureRef(:red), 0.5), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    state = plateau!(mk_state(g, prog; k = 4))   # uniform 4-mixture ⇒ H = log 4 > log 2
    scored = score_gw_meta_actions(state, ExploreObservation[])
    h = entropy(state.belief)
    check("§5 escape score == H − escape_cost exactly (the named heuristic, priced)",
          scored[:gw_enumerate_more] == h - GW_ESCAPE_COST_DEFAULT &&
          scored[:gw_deepen] == h - GW_ESCAPE_COST_DEFAULT,
          "enum=$(scored[:gw_enumerate_more]) H=$h")
    check("§5 escape score is positive (belief uncertain enough to pay one bit)",
          scored[:gw_enumerate_more] > 0.0)
    check("§5 tie resolves breadth-before-depth (GW_META_ACTIONS order)",
          default_eu_max_policy(scored) == :gw_enumerate_more)
end

# ── §6  default_eu_max_policy: the act-now floor and deterministic tie order (synthetic dicts) ──
let
    allneg = Dict{Symbol, Float64}(:gw_do_nothing => 0.0, :gw_enumerate_more => -0.1,
                                   :gw_perturb_grammar => -Inf, :gw_deepen => -0.1,
                                   :gw_explore => 0.0, :gw_add_feature => -Inf)
    check("§6 nothing strictly positive ⇒ :gw_do_nothing (0.0 score does not act)",
          default_eu_max_policy(allneg) == :gw_do_nothing)
    tie = Dict{Symbol, Float64}(:gw_do_nothing => 0.0, :gw_enumerate_more => 1.0,
                                :gw_perturb_grammar => -Inf, :gw_deepen => 1.0,
                                :gw_explore => 0.5, :gw_add_feature => -Inf)
    check("§6 exact tie ⇒ first in GW_META_ACTIONS order (enumerate before deepen)",
          default_eu_max_policy(tie) == :gw_enumerate_more)
    check("§6 strict argmax otherwise",
          default_eu_max_policy(Dict{Symbol, Float64}(:gw_do_nothing => 0.0, :gw_explore => 2.0,
                                                      :gw_enumerate_more => 1.0)) == :gw_explore)
end

println("="^64)
println("ALL CHECKS PASSED — grid-world meta scores (real single-currency argmax)")
println("="^64)
