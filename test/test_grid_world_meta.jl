# test_grid_world_meta.jl — the grid-world meta-action scores under belief-derived valuation
# (docs/exploration-budget/belief-derived-valuation-design.md, RATIFIED 2026-07-02).
# Re-baselined from the entropy-heuristic scheme per design §3 ("this move deliberately changes
# selection-seam behaviour"): the escape-tier pins are returns-model pins (fresh-context prior
# optimism, zero-yield collapse, context reset); the growth pins are horizon-completed
# growth_value pins with the one-time prior-Occam placement.
#
# What survives from the #174 PR 2 gate structure, re-asserted here:
#   - compression stays SOFT: perturbation_voc competes in the argmax at its prior-only score,
#     no veto in either direction (§1);
#   - threshold_exhausted stays HARD on :gw_add_feature, zero constants (§4);
#   - the act-now floor and the deterministic tie order (§6).
# What changed (belief-derived valuation):
#   - escape ops are scored by their LEARNED returns beliefs (posterior-mean next yield −
#     declared price), compete FREELY (no saturation-ordering gate), and self-extinguish
#     under zero-yield evidence (§2, §5);
#   - growth scores are horizon-completed: growth_value(fit, n_buf, plateau, H), H = n_buf
#     when no horizon is declared; the feature prior-Occam charge is ONE-TIME, outside both
#     plateau and horizon (§3, §3b, §4).
#
# Sections:
#   §1  compressible state, empty buffer — perturb scores its real VOC; explore exactly 0;
#       escape at fresh-context optimism.
#   §2  exhausted grammar, empty buffer — growth tiers 0; escape == prior optimism exactly
#       (the DESIGNED cold-start probe fires, replacing the old always-on entropy score).
#   §3  refinable buffer — explore == plateau·fit (H = n_buf default); add_feature hard-gated
#       -Inf; horizon completion doubles the fit term exactly (§3b).
#   §4  thresholds exhausted, new feature separates — add_feature == plateau·fit + one-time
#       prior charge (the ratified placement: prior OUTSIDE plateau).
#   §5  returns-model seam integration — fresh tie resolves breadth-before-depth; the
#       self-extinguish loop: zero-yield evidence drives the escape tier below the act-now
#       floor in a bounded number of firings (the zombie churn dies).
#   §6  default_eu_max_policy: act-now floor and deterministic tie order on synthetic dicts.
#
# Run: julia test/test_grid_world_meta.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, Program, FeatureRef, GTExpr, AndExpr, IfExpr, ActionExpr, NonterminalRef,
                AgentState, MixturePrevision, TaggedBetaPrevision, BetaPrevision, Prevision,
                compile_kernel, update_learning_regime, plateau_probability,
                analyse_posterior_subtrees, compression_exhausted, ExploreObservation,
                exploration_voi, feature_discovery_voi, feature_discovery_fit, perturbation_voc,
                growth_value, complexity_logprior, expected_growth_yield, weights

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
# Fresh-context escape score: the returns prior's expected yield (Gamma(2,1) ⇒ β/(α−1) = 1 nat
# of bounded initial optimism) minus the declared one-bit price.
const ESCAPE_FRESH = 1.0 - log(2.0)

println("="^64)
println("grid-world meta scores — belief-derived valuation at the selection seam")
println("="^64)

# ── §1  compressible state, empty buffer: perturb scores its REAL prior-only VOC ──
let
    state = compressible_state()
    ft = analyse_posterior_subtrees(state.all_programs, weights(state.belief);
                                    min_frequency = 0.01, min_complexity = 2)
    check("§1 precondition: compression NOT exhausted (dead rule present)",
          compression_exhausted(state.grammars[1], ft) == false)

    scored = score_gw_meta_actions(state, ExploreObservation[])
    voc = perturbation_voc(state.grammars[1], ft)
    check("§1 :gw_perturb_grammar == its real prior-only net VOC (wiring pin)",
          scored[:gw_perturb_grammar] == voc, "scored=$(scored[:gw_perturb_grammar])")
    check("§1 compression VOC is positive here (a reclaim exists)",
          scored[:gw_perturb_grammar] > 0.0)
    # Empty buffer ⇒ the exact lookahead has nothing to price ⇒ exactly 0, NOT a veto (-Inf):
    # compression availability never gates exploration (#174 PR 2, re-asserted).
    check("§1 :gw_explore == 0.0 exactly on an empty buffer (no compression veto)",
          scored[:gw_explore] == 0.0, "scored=$(scored[:gw_explore])")
    # Escape competes freely at its fresh-context learned value (no eligibility gate).
    check("§1 escape tier == fresh-context optimism exactly (1 − log 2)",
          scored[:gw_enumerate_more] == ESCAPE_FRESH && scored[:gw_deepen] == ESCAPE_FRESH,
          "enum=$(scored[:gw_enumerate_more])")
    # The one argmax decides between the real VOC and the fresh probe — no ordering rule.
    expected = voc > ESCAPE_FRESH ? :gw_perturb_grammar : :gw_enumerate_more
    check("§1 policy is the plain argmax over {VOC, fresh escape} (free competition)",
          default_eu_max_policy(scored) == expected,
          "voc=$voc fresh=$ESCAPE_FRESH chose=$(default_eu_max_policy(scored))")
end

# ── §2  exhausted grammar, empty buffer: growth tiers 0; the DESIGNED cold-start probe ──
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
    # The old scheme scored escape by posterior entropy (fired forever); the new scheme fires
    # the BOUNDED cold-start probe: fresh cells are worth trying exactly until evidence says
    # otherwise (design §2b), so the policy probes rather than acting now.
    check("§2 escape tier == fresh-context optimism exactly (the designed cold-start probe)",
          scored[:gw_enumerate_more] == ESCAPE_FRESH && scored[:gw_deepen] == ESCAPE_FRESH,
          "enum=$(scored[:gw_enumerate_more])")
    check("§2 policy probes (breadth first): :gw_enumerate_more",
          default_eu_max_policy(scored) == :gw_enumerate_more)
end

# ── §3  refinable buffer: explore == plateau·fit; add_feature and the horizon completion ──
let
    g = Grammar(Set([:red]), ProductionRule[], 3)
    prog = Program(IfExpr(GTExpr(FeatureRef(:red), 0.5), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    state = plateau!(mk_state(g, prog))
    # Separable only at ~0.4 (off the default grid {0.1,0.3,0.5,0.7,0.9}): 0.35 vs 0.45 land on
    # the same side of every existing threshold, so refinement MUST add ~0.4 ⇒ positive fit.
    buf = ExploreObservation[]
    for _ in 1:6
        push!(buf, ExploreObservation(Dict(:red => 0.35), Dict{Symbol, Any}(), Set([:food]), 1.0))
        push!(buf, ExploreObservation(Dict(:red => 0.45), Dict{Symbol, Any}(), Set([:enemy]), 1.0))
    end
    voi = exploration_voi(state.grammars[3], buf, state.current_max_depth; action_space = GW_AS)
    check("§3 precondition: refinement fit is positive on this buffer", voi > 0.0, "voi=$voi")

    pl = plateau_probability(state.learning_regime)
    scored = score_gw_meta_actions(state, buf)
    # H defaults to n_buf (no declared horizon) ⇒ the window-total score, exactly (§1's nested pin).
    check("§3 :gw_explore == plateau · fit exactly (H = n_buf default)",
          scored[:gw_explore] == pl * voi, "scored=$(scored[:gw_explore]) expected=$(pl * voi)")
    check("§3 :gw_explore is positive (plateau soft gate scales, never vetoes)",
          scored[:gw_explore] > 0.0)
    check("§3 :gw_add_feature == -Inf while thresholds NOT exhausted (hard gate, zero constants)",
          scored[:gw_add_feature] == -Inf)
    # No eligibility gate: escape competes at its learned value even beside a positive exact tier.
    check("§3 escape tier == fresh-context optimism (free competition, no -Inf ordering gate)",
          scored[:gw_enumerate_more] == ESCAPE_FRESH && scored[:gw_deepen] == ESCAPE_FRESH)
    check("§3 the exact tier outbids the fresh probe on this buffer",
          scored[:gw_explore] > scored[:gw_enumerate_more],
          "explore=$(scored[:gw_explore]) escape=$ESCAPE_FRESH")
    check("§3 policy selects :gw_explore", default_eu_max_policy(scored) == :gw_explore)

    # §3b — the horizon completion: a declared horizon of 2·n_buf doubles the fit term EXACTLY
    # (growth_value computes plateau·fit·(H/n_buf); 2n/n == 2.0 in floats).
    scored_h = score_gw_meta_actions(state, buf; horizon = 2.0 * length(buf))
    check("§3b declared horizon 2·n_buf doubles :gw_explore exactly",
          scored_h[:gw_explore] == 2.0 * (pl * voi),
          "scored=$(scored_h[:gw_explore]) expected=$(2.0 * pl * voi)")
end

# ── §4  thresholds exhausted, a NEW feature separates: the two-axis score, new placement ──
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
    check("§4 precondition: thresholds exhausted (no refinement fit on a constant feature)",
          voi_explore == 0.0, "voi=$voi_explore")
    fit, dc = feature_discovery_fit(state.grammars[4], buf, ALL_GW_FEATURES,
                                    state.current_max_depth; action_space = GW_AS)
    check("§4 precondition: the new feature clears the two-axis bar (fit − log2 > 0)",
          fit + complexity_logprior(dc; λ = log(2)) > 0.0, "fit=$fit dc=$dc")

    pl = plateau_probability(state.learning_regime)
    scored = score_gw_meta_actions(state, buf)
    # The ratified placement (design §2a): the prior-Occam charge is ONE-TIME — outside the
    # plateau multiplication (it is a prior fact, not a measured gain) and outside the horizon.
    expected = growth_value(fit, length(buf), pl, Float64(length(buf));
                            prior_term = complexity_logprior(dc; λ = log(2)))
    check("§4 :gw_add_feature == plateau·fit + prior_term (gate open, one-time charge)",
          scored[:gw_add_feature] == expected,
          "scored=$(scored[:gw_add_feature]) expected=$expected")
    check("§4 placement pin: prior charge NOT plateau-discounted (≠ plateau·(fit + prior))",
          scored[:gw_add_feature] != pl * (fit + complexity_logprior(dc; λ = log(2))))
    check("§4 policy selects :gw_add_feature (it outbids the fresh probe here)",
          scored[:gw_add_feature] > ESCAPE_FRESH &&
          default_eu_max_policy(scored) == :gw_add_feature,
          "add=$(scored[:gw_add_feature])")
end

# ── §5  returns-model seam integration: the fresh tie, and the self-extinguish loop ──
let
    g = Grammar(Set([:red]), ProductionRule[], 5)
    prog = Program(IfExpr(GTExpr(FeatureRef(:red), 0.5), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    state = plateau!(mk_state(g, prog; k = 4))   # uncertain 4-mixture: entropy no longer matters
    scored = score_gw_meta_actions(state, ExploreObservation[])
    check("§5 fresh escape scores are an exact tie at prior optimism (equal priors, Q4)",
          scored[:gw_enumerate_more] == ESCAPE_FRESH && scored[:gw_deepen] == ESCAPE_FRESH)
    check("§5 the cold-start tie resolves breadth-before-depth (GW_META_ACTIONS order)",
          default_eu_max_policy(scored) == :gw_enumerate_more)

    # The self-extinguish loop (design §2b/§4): drive the REAL seam — score, select, execute,
    # observe realised yield — until the policy stops. The zombie churn that fired forever
    # under the entropy score now kills itself in a bounded number of firings: each firing
    # conditions the op's returns cell on its realised (mostly ~0) yield.
    fires = Symbol[]
    for _ in 1:15
        s = score_gw_meta_actions(state, ExploreObservation[])
        chosen = default_eu_max_policy(s)
        chosen == :gw_do_nothing && break
        push!(fires, chosen)
        # The declared depth frontier (task data): one deepen above the start — beyond it,
        # deepen no-ops at zero yield and the returns cells retire it.
        execute_gw_meta_action!(state, chosen, ExploreObservation[]; verbose = false,
                                max_program_depth = 3)
    end
    final = score_gw_meta_actions(state, ExploreObservation[])
    check("§5 the probe sequence terminates (self-extinguish, bounded firings)",
          length(fires) < 15 && default_eu_max_policy(final) == :gw_do_nothing,
          "fires=$fires")
    check("§5 both escape ops end priced under water (posterior mean < declared price)",
          final[:gw_enumerate_more] < 0.0 && final[:gw_deepen] < 0.0,
          "enum=$(final[:gw_enumerate_more]) deepen=$(final[:gw_deepen])")
    check("§5 the learned expectations are Tier-1 reads (score == E[yield] − price)",
          final[:gw_enumerate_more] ==
          expected_growth_yield(state.growth_returns, :gw_enumerate_more) - log(2.0))
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
println("ALL CHECKS PASSED — grid-world meta scores (belief-derived valuation)")
println("="^64)
