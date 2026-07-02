# test_grid_world_meta.jl — the grid-world meta-action scores under belief-derived valuation
# (re-baselined from the dominance-Phase-3 entropy scheme per belief-derived-valuation §2;
# "behaviour shift is intended"): every score is a posterior expectation or declared data.
#
# What survives from the dominance gate structure, re-asserted in the new valuation:
#   - compression stays SOFT: perturbation_voc competes at its prior-only score, no veto (§1);
#   - threshold_exhausted stays HARD on :gw_add_feature, gating on FIT (attribution is a
#     measurement concern, valuation-independent) (§3);
#   - the act-now floor and the GW_META_ACTIONS tie order (§6).
# What changes:
#   - growth scores are horizon-completed: growth_value(fit, n_buf, plateau, H) with the
#     one-time −log2 Occam charge on features (§3, §4);
#   - the escape tier is the LEARNED returns model: escape_score(returns, op, changed) − price,
#     with NO saturation-ordering eligibility gate (ratified Q5) — bounded prior optimism that
#     decays under zero-yield evidence (§5).
#
# Sections:
#   §1  compressible state, empty buffer — perturb -Inf at the seam (provisional; engine voc
#       still prices the reclaim); explore exactly 0.
#   §2  fresh returns prior — escape scores prior-optimism − price at BOTH contexts; a
#       zero-yield-collapsed cell loses to the do-nothing floor (the entropy score never did).
#   §3  refinable buffer — explore == growth_value(fit, n, plateau, H); horizon scaling exact;
#       add_feature hard-gated -Inf; escape ops COMPETE (no -Inf) at their learned scores.
#   §4  thresholds exhausted, new feature separates — add_feature == growth_value(fit, …,
#       prior_term = −log2); defaults reduce to the old plateau·fdvoi.
#   §5  returns-model dynamics through the score seam: prior fires once, three zero yields kill
#       the cell, a real yield sustains it, context cells are independent.
#   §6  default_eu_max_policy: act-now floor and deterministic tie order on synthetic dicts.
#
# Run: julia test/test_grid_world_meta.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, Program, FeatureRef, GTExpr, AndExpr, IfExpr, ActionExpr, NonterminalRef,
                AgentState, MixturePrevision, TaggedBetaPrevision, BetaPrevision, Prevision,
                compile_kernel, update_learning_regime, plateau_probability,
                analyse_posterior_subtrees, compression_exhausted, ExploreObservation,
                exploration_voi, exploration_fit, feature_discovery_voi, feature_discovery_fit,
                perturbation_voc, growth_value, weights,
                GrowthReturns, observe_yield!, escape_score

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
const ESCAPE = Symbol[:gw_enumerate_more, :gw_deepen]
fresh_returns() = GrowthReturns(ESCAPE)
allchanged() = Dict{Symbol, Bool}(:gw_enumerate_more => true, :gw_deepen => true)

println("="^64)
println("grid-world meta scores — belief-derived valuation")
println("="^64)

# ── §1  compressible state, empty buffer: perturb scores its REAL prior-only VOC ──
let
    state = compressible_state()
    ft = analyse_posterior_subtrees(state.all_programs, weights(state.belief);
                                    min_frequency = 0.01, min_complexity = 2)
    check("§1 precondition: compression NOT exhausted (dead rule present)",
          compression_exhausted(state.grammars[1], ft) == false)

    scored = score_gw_meta_actions(state, ExploreObservation[], fresh_returns(), allchanged())
    # PROVISIONAL pin (see host.jl): perturb is out of the argmax pending the removal-consumption
    # redesign; the engine's voc accessor itself still prices the reclaim (asserted directly).
    check("§1 :gw_perturb_grammar == -Inf at the seam (provisional exclusion)",
          scored[:gw_perturb_grammar] == -Inf)
    check("§1 the engine's perturbation_voc still prices the reclaim (> 0)",
          perturbation_voc(state.grammars[1], ft) > 0.0)
    # Empty buffer ⇒ the exact lookahead has nothing to price ⇒ exactly 0, NOT a veto (-Inf):
    # compression availability never gates exploration (#174 PR 2, re-asserted).
    check("§1 :gw_explore == 0.0 exactly on an empty buffer (no compression veto)",
          scored[:gw_explore] == 0.0, "scored=$(scored[:gw_explore])")
end

# ── §2  the returns prior at the score seam: bounded optimism, price-netted; dead cells floor ──
let
    g = Grammar(Set([:red]), ProductionRule[], 2)   # no rules ⇒ nothing to compress ⇒ exhausted
    prog = Program(IfExpr(GTExpr(FeatureRef(:red), 0.5), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    state = plateau!(mk_state(g, prog))
    ft = analyse_posterior_subtrees(state.all_programs, weights(state.belief);
                                    min_frequency = 0.01, min_complexity = 2)
    check("§2 precondition: compression EXHAUSTED", compression_exhausted(state.grammars[2], ft))

    # Fresh prior: expected yield 1 nat, minus the declared one-bit price — positive, fires once.
    scored = score_gw_meta_actions(state, ExploreObservation[], fresh_returns(), allchanged())
    check("§2 fresh escape score == 1.0 − log 2 exactly (prior optimism, price-netted)",
          scored[:gw_enumerate_more] == 1.0 - GW_OP_COMPUTE_COST_DEFAULT,
          "enum=$(scored[:gw_enumerate_more])")
    check("§2 :gw_deepen == -Inf (PROVISIONAL depth-escalation exclusion — see host.jl comment;",
          scored[:gw_deepen] == -Inf)
    check("§2 the fresh enumerate cell wins the argmax",
          default_eu_max_policy(scored) == :gw_enumerate_more)

    # A zero-yield-collapsed cell loses to the act-now floor — the decay the entropy heuristic
    # never had (it fired forever at 4e-5-nat margins).
    gr = fresh_returns()
    for _ in 1:3
        observe_yield!(gr, :gw_enumerate_more, true, 0.0)
    end
    scored2 = score_gw_meta_actions(state, ExploreObservation[], gr, allchanged())
    check("§2 three zero yields put the enumerate cell under the price (< 0)",
          scored2[:gw_enumerate_more] < 0.0,
          "enum=$(scored2[:gw_enumerate_more])")
    check("§2 nothing positive ⇒ act now", default_eu_max_policy(scored2) == :gw_do_nothing)
end

# ── §3  refinable buffer: explore == growth_value(fit, n, plateau, H); escape competes freely ──
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
    fit = exploration_fit(state.grammars[3], buf, state.current_max_depth; action_space = GW_AS)
    check("§3 precondition: refinement fit is positive on this buffer", fit > 0.0, "fit=$fit")

    pl = plateau_probability(state.learning_regime)
    n = length(buf)

    # Window-total horizon (H = n_buf) reproduces the pre-move plateau·voi exactly.
    scored = score_gw_meta_actions(state, buf, fresh_returns(), allchanged();
                                   horizon = Float64(n))
    voi_default = exploration_voi(state.grammars[3], buf, state.current_max_depth;
                                  action_space = GW_AS)
    check("§3 :gw_explore at H == n_buf == plateau · exploration_voi (the pre-move score)",
          scored[:gw_explore] == pl * voi_default,
          "scored=$(scored[:gw_explore]) expected=$(pl * voi_default)")

    # Horizon completion scales the fit term exactly: growth_value through the engine functional.
    scored_h = score_gw_meta_actions(state, buf, fresh_returns(), allchanged(); horizon = 60.0)
    check("§3 :gw_explore horizon-completed == growth_value(fit, n, plateau, 60)",
          scored_h[:gw_explore] == growth_value(fit, n, pl, 60.0),
          "scored=$(scored_h[:gw_explore])")
    check("§3 :gw_add_feature == -Inf while thresholds NOT exhausted (hard gate, on FIT)",
          scored_h[:gw_add_feature] == -Inf)
    # No saturation-ordering gate (ratified Q5): enumerate carries its learned score, not -Inf.
    check("§3 :gw_enumerate_more COMPETES at its learned score (no eligibility -Inf)",
          scored_h[:gw_enumerate_more] == 1.0 - GW_OP_COMPUTE_COST_DEFAULT)
    # With a long horizon the exact tier outbids prior escape optimism here.
    check("§3 policy selects :gw_explore (exact value outbids prior escape optimism)",
          default_eu_max_policy(scored_h) == :gw_explore,
          "explore=$(scored_h[:gw_explore]) enum=$(scored_h[:gw_enumerate_more])")
end

# ── §4  thresholds exhausted, a NEW feature separates: the two-axis horizon-completed score ──
let
    g = Grammar(Set([:red]), ProductionRule[], 4)
    prog = Program(IfExpr(GTExpr(FeatureRef(:red), 0.5), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    state = plateau!(mk_state(g, prog))
    # :red is CONSTANT across classes (no refinement can help ⇒ thresholds exhausted); :speed
    # separates cleanly at the default grid ⇒ feature discovery carries real fit value.
    buf = ExploreObservation[]
    for _ in 1:8
        push!(buf, ExploreObservation(Dict(:red => 0.5, :speed => 0.2), Dict{Symbol, Any}(), Set([:food]), 1.0))
        push!(buf, ExploreObservation(Dict(:red => 0.5, :speed => 0.8), Dict{Symbol, Any}(), Set([:enemy]), 1.0))
    end
    check("§4 precondition: thresholds exhausted (zero refinement fit on a constant feature)",
          exploration_fit(state.grammars[4], buf, state.current_max_depth; action_space = GW_AS) == 0.0)
    ffit = feature_discovery_fit(state.grammars[4], buf, ALL_GW_FEATURES,
                                 state.current_max_depth; action_space = GW_AS)
    check("§4 precondition: the new feature carries positive fit", ffit > 0.0, "ffit=$ffit")

    pl = plateau_probability(state.learning_regime)
    n = length(buf)

    # Window-total horizon reproduces the pre-move plateau·fdvoi... for plateau == 1; in general
    # the NEW semantics differ deliberately: the old score multiplied the WHOLE net (fit − log2)
    # by plateau; the new one plateau-scales the FIT only — the one-time prior charge is a prior,
    # not a gain, so the regime gate does not touch it. Pin the new form exactly.
    scored = score_gw_meta_actions(state, buf, fresh_returns(), allchanged();
                                   horizon = Float64(n))
    check("§4 :gw_add_feature == growth_value(fit, n, plateau, n; prior = −log2) — fit scaled, prior not",
          scored[:gw_add_feature] == growth_value(ffit, n, pl, Float64(n);
                                                  prior_term = GW_FEATURE_PRIOR_TERM),
          "scored=$(scored[:gw_add_feature])")

    scored_h = score_gw_meta_actions(state, buf, fresh_returns(), allchanged(); horizon = 60.0)
    check("§4 horizon completion multiplies the fit term only (one-time Occam charge)",
          scored_h[:gw_add_feature] == growth_value(ffit, n, pl, 60.0;
                                                    prior_term = GW_FEATURE_PRIOR_TERM))
    check("§4 policy selects :gw_add_feature at the long horizon",
          default_eu_max_policy(scored_h) == :gw_add_feature)
end

# ── §5  returns dynamics through the seam: independent contexts, sustain on real yield ──
let
    g = Grammar(Set([:red]), ProductionRule[], 5)
    prog = Program(IfExpr(GTExpr(FeatureRef(:red), 0.5), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    state = plateau!(mk_state(g, prog; k = 4))
    gr = fresh_returns()
    # Kill enumerate's changed-cell; its unchanged-cell stays at prior.
    for _ in 1:3
        observe_yield!(gr, :gw_enumerate_more, true, 0.0)
    end
    scored = score_gw_meta_actions(state, ExploreObservation[], gr, allchanged())
    check("§5 enumerate's changed-cell dead ⇒ act now (deepen provisionally -Inf)",
          scored[:gw_enumerate_more] < 0.0 && scored[:gw_deepen] == -Inf &&
          default_eu_max_policy(scored) == :gw_do_nothing)

    # The unchanged context is a separate cell: still at prior.
    scored_u = score_gw_meta_actions(state, ExploreObservation[], gr,
                                     Dict{Symbol, Bool}(:gw_enumerate_more => false,
                                                        :gw_deepen => false))
    check("§5 (op × changed) cells are independent (unchanged enumerate still at prior)",
          scored_u[:gw_enumerate_more] == 1.0 - GW_OP_COMPUTE_COST_DEFAULT)

    # A real yield sustains a cell: Gamma(2,1) + y=2 → Gamma(3,3) → E = 3/2 (read through
    # escape_score directly — deepen's returns stay TRACKED even while its argmax entry is
    # provisionally -Inf, so the escalate-depth re-entry starts informed).
    observe_yield!(gr, :gw_deepen, true, 2.0)
    check("§5 a 2-nat yield sustains deepen's cell at 3/2 − log 2 exactly (tracked, not scored)",
          escape_score(gr, :gw_deepen, true; compute_cost = GW_OP_COMPUTE_COST_DEFAULT) ==
          1.5 - GW_OP_COMPUTE_COST_DEFAULT)
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
