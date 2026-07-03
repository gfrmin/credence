# test_grid_world_meta.jl — the grid-world meta-action scores under belief-derived valuation
# (re-baselined from the dominance-Phase-3 entropy scheme per belief-derived-valuation §2;
# "behaviour shift is intended"): every score is a posterior expectation or declared data.
#
# What survives from the dominance gate structure, re-asserted in the new valuation:
#   - compression stays SOFT: perturb competes at its exact replacement value, no veto (§1 —
#     re-baselined by the removal-consumption move: the #189 -Inf provisional pin retired);
#   - threshold_exhausted stays HARD on :gw_add_feature, gating on FIT (attribution is a
#     measurement concern, valuation-independent) (§3);
#   - the act-now floor and the GW_META_ACTIONS tie order (§6).
# What changes (re-baselined again by the winners-curse design revs 3–4):
#   - growth scores are the YIELD RULE (§8.2): net_value(prop.yield_nats, op_compute_cost) —
#     the union-over-incumbent window Bayes factor net of the declared price; no plateau, no
#     horizon, no prior term at the seam (§3, §4);
#   - the attribution gate re-expresses as "refinement fires first": add_feature is -Inf iff
#     the threshold proposal's own score clears the floor (§3);
#   - the escape tier is the LEARNED returns model: escape_score(returns, op, changed) − price,
#     with NO saturation-ordering eligibility gate (ratified Q5) — bounded prior optimism that
#     decays under zero-yield evidence (§5).
#
# Sections:
#   §1  compressible state, empty buffer — perturb scores its exact replacement value at the
#       seam (removal-consumption re-entry: score == the engine's replacement_value on the top
#       grammar — one candidate function, T-3.55); explore exactly 0.
#   §2  fresh returns prior — escape scores prior-optimism − price at BOTH contexts; a
#       zero-yield-collapsed cell loses to the do-nothing floor (the entropy score never did).
#   §3  refinable buffer — explore == net_value(yield, price) via the shared proposal;
#       add_feature gated -Inf while refinement clears its price; escape ops COMPETE.
#   §4  thresholds exhausted (no candidates), new feature separates — add_feature ==
#       net_value(yield, price); the Occam charge rides inside the mixture (Q4).
#   §5  returns-model dynamics through the score seam: prior fires once, three zero yields kill
#       the cell, a real yield sustains it, context cells are independent.
#   §6  default_eu_max_policy: act-now floor and deterministic tie order on synthetic dicts.
#
# Run: julia test/test_grid_world_meta.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, Program, FeatureRef, GTExpr, AndExpr, IfExpr, ActionExpr, NonterminalRef,
                AgentState, MixturePrevision, TaggedBetaPrevision, BetaPrevision, Prevision,
                compile_kernel, update_learning_regime,
                analyse_posterior_subtrees, compression_exhausted, ExploreObservation,
                program_space_observation_kernel, condition, log_predictive,
                threshold_growth_proposal, feature_growth_proposal, net_value,
                weights, best_replacement, replacement_value,
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

# Condition the state live and return the honest-residual buffer (the coherence ledger the
# virtual injection replays — synthetic residuals would distort the union's scale).
function live!(state, raw)
    buf = ExploreObservation[]
    for (features, correct) in raw
        k = program_space_observation_kernel(state.compiled_kernels, features,
                                             Dict{Symbol, Any}(), correct)
        res = -log_predictive(state.belief, k, 1.0)
        push!(buf, ExploreObservation(features, Dict{Symbol, Any}(), correct, res))
        state.belief = condition(state.belief, k, 1.0)
    end
    buf
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

# ── §1  compressible state, empty buffer: perturb scores its exact replacement value ──
let
    state = compressible_state()
    check("§1 precondition: the dead rule is a replacement candidate",
          best_replacement(state, 1) !== nothing)

    scored = score_gw_meta_actions(state, ExploreObservation[], fresh_returns(), allchanged())
    # Removal-consumption re-entry (the #189 deviation-3 discharge): perturb competes at the
    # exact realised Δ log-evidence of consuming the top grammar's dead item, net of the declared
    # price — the same candidate function the executor applies (score = transition, T-3.55).
    check("§1 :gw_perturb_grammar == replacement_value at the seam (max over top-k gids)",
          scored[:gw_perturb_grammar] ==
          replacement_value(state, 1; compute_cost = GW_OP_COMPUTE_COST_DEFAULT),
          "scored=$(scored[:gw_perturb_grammar])")
    check("§1 the reclaim is positive (a 2-symbol dead rule at W_G = 1 outbids the one-bit price)",
          scored[:gw_perturb_grammar] > 0.0, "scored=$(scored[:gw_perturb_grammar])")
    # Empty buffer ⇒ no proposal ⇒ the price with nothing to buy: exactly −op_cost, NOT a
    # veto (-Inf): compression availability never gates exploration (#174 PR 2, re-asserted).
    check("§1 :gw_explore == net_value(0, price) exactly on an empty buffer (no compression veto)",
          scored[:gw_explore] == net_value(0.0, GW_OP_COMPUTE_COST_DEFAULT),
          "scored=$(scored[:gw_explore])")
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

# ── §3  refinable buffer: explore == net_value(yield, price); the gate; escape competes ──
let
    g = Grammar(Set([:red]), ProductionRule[], 3)
    prog = Program(IfExpr(GTExpr(FeatureRef(:red), 0.5), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    state = plateau!(mk_state(g, prog))
    # Separable only at ~0.4 (off the default grid {0.1,0.3,0.5,0.7,0.9}): 0.35 vs 0.45 land on
    # the same side of every existing threshold, so refinement MUST add ~0.4 ⇒ real evidence.
    raw = Tuple{Dict{Symbol, Float64}, Set{Symbol}}[]
    for _ in 1:6
        push!(raw, (Dict(:red => 0.35), Set([:food])))
        push!(raw, (Dict(:red => 0.45), Set([:enemy])))
    end
    buf = live!(state, raw)

    prop = threshold_growth_proposal(state, state.grammars[3], buf; action_space = GW_AS)
    check("§3 precondition: the refinement union carries decisive evidence (yield > price)",
          prop !== nothing && net_value(prop.yield_nats, GW_OP_COMPUTE_COST_DEFAULT) > 0.0,
          prop === nothing ? "no proposal" : "yield=$(prop.yield_nats)")

    scored = score_gw_meta_actions(state, buf, fresh_returns(), allchanged())
    # The yield rule at the seam: the score is the realised evidence net of the declared price
    # — the same GrowthProposal the executor adopts (T-3.55; identity-pinned in
    # test_virtual_injection.jl §1). No plateau, no horizon, no prior term.
    check("§3 :gw_explore == net_value(yield, price) exactly",
          scored[:gw_explore] == net_value(prop.yield_nats, GW_OP_COMPUTE_COST_DEFAULT),
          "scored=$(scored[:gw_explore]) expected=$(net_value(prop.yield_nats, GW_OP_COMPUTE_COST_DEFAULT))")
    check("§3 :gw_add_feature == -Inf while refinement clears its price (refinement fires first)",
          scored[:gw_add_feature] == -Inf)
    # No saturation-ordering gate (ratified Q5): enumerate carries its learned score, not -Inf.
    check("§3 :gw_enumerate_more COMPETES at its learned score (no eligibility -Inf)",
          scored[:gw_enumerate_more] == 1.0 - GW_OP_COMPUTE_COST_DEFAULT)
    # Decisive realised evidence outbids prior escape optimism.
    check("§3 policy selects :gw_explore (realised evidence outbids prior escape optimism)",
          default_eu_max_policy(scored) == :gw_explore,
          "explore=$(scored[:gw_explore]) enum=$(scored[:gw_enumerate_more])")
end

# ── §4  thresholds exhausted, a NEW feature separates: the yield rule prices the feature ──
let
    g = Grammar(Set([:red]), ProductionRule[], 4)
    prog = Program(IfExpr(GTExpr(FeatureRef(:red), 0.5), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    state = plateau!(mk_state(g, prog))
    # :red is CONSTANT across classes ⇒ no midpoint candidates ⇒ the threshold class is silent
    # (structurally exhausted); :speed separates cleanly at the default grid.
    raw = Tuple{Dict{Symbol, Float64}, Set{Symbol}}[]
    for _ in 1:8
        push!(raw, (Dict(:red => 0.5, :speed => 0.2), Set([:food])))
        push!(raw, (Dict(:red => 0.5, :speed => 0.8), Set([:enemy])))
    end
    buf = live!(state, raw)
    check("§4 precondition: thresholds exhausted (a constant feature generates no candidates)",
          threshold_growth_proposal(state, state.grammars[4], buf; action_space = GW_AS) === nothing)
    fprop = feature_growth_proposal(state, state.grammars[4], buf, ALL_GW_FEATURES;
                                    action_space = GW_AS)
    check("§4 precondition: the feature union carries decisive evidence",
          fprop !== nothing && fprop.yield_nats > 0.0, "fprop=$(fprop)")

    scored = score_gw_meta_actions(state, buf, fresh_returns(), allchanged())
    # The yield rule: net of the declared price only — the −log2 Occam charge for the feature
    # symbol rides INSIDE the mixture (each newcomer's 2^{−|G|−1−|p|} prior; winners-curse Q4),
    # never at the seam. GW_FEATURE_PRIOR_TERM is retired (pinned in test_virtual_injection §1).
    check("§4 :gw_add_feature == net_value(yield, price) exactly (no seam prior term)",
          scored[:gw_add_feature] == net_value(fprop.yield_nats, GW_OP_COMPUTE_COST_DEFAULT),
          "scored=$(scored[:gw_add_feature])")
    check("§4 policy selects :gw_add_feature",
          default_eu_max_policy(scored) == :gw_add_feature)
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
