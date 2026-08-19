# test_feature_discovery.jl — exploration-budget Move 4 (re-baselined by the winners-curse
# design revs 3–4: the lookahead is a virtual injection; the score is the yield rule §8.2).
# Grow the agent's FEATURE alphabet from a host-furnished candidate set, ALL candidates carried
# in one union (no per-candidate argmax — average-not-collapse applied to the transition).
#
# Sections:
#   §1  candidate generation — the host-furnished set `available_features \ feature_set` (sorted).
#   §2  _add_feature — surgery: feature added, default grid for it, OTHER features' grids preserved,
#       fresh id, complexity +1 (a feature IS a description-length unit — Q2, unlike a threshold).
#   §3  discovery via the virtual injection: the predictive feature's programs earn the union's
#       mass and the yield clears the price; no-op when nothing helps; determinism; empty paths.
#   §4  Q2 under the union mechanism: the prior-Occam charge lives INSIDE the mixture (each
#       newcomer arrives at 2^{−|G|−|p|} with the feature's +1 complexity — winners-curse Q4:
#       no explicit prior term anywhere), so the realised yield is strictly below the fit-axis
#       Δℓ, and a same-window threshold-class union (complexity-invariant candidates) is not so
#       charged. The prior axis is priced, mechanically, with no seam arithmetic.
#
# Run: julia test/test_feature_discovery.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, Program, CompiledKernel, ExploreObservation, THRESHOLDS,
                next_grammar_id, reset_grammar_counter!, enumerate_programs,
                AgentState, MixturePrevision, TaggedBetaPrevision, BetaPrevision, Prevision,
                compile_kernel, complexity_logprior, program_space_observation_kernel,
                condition, log_predictive, net_value, weights, show_expr,
                feature_growth_proposal, adopt!

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("feature-discovery — Move 4")
println("="^64)

# ── §1  candidate generation: the host-furnished set `available_features \ feature_set` (sorted) ──
let
    reset_grammar_counter!()
    g = Grammar(Set([:colour]), ProductionRule[], next_grammar_id())
    available = Set([:colour, :wall_dist, :agent_dist, :speed])

    cands = Credence._feature_candidates(g, available)
    check("§1 candidates are available \\ feature_set",
          Set(cands) == Set([:wall_dist, :agent_dist, :speed]), "got $(cands)")
    check("§1 candidates are sorted (deterministic)",
          cands == sort(cands), "got $(cands)")

    # A grammar already using every available feature ⇒ no candidates.
    g_full = Grammar(Set([:colour, :wall_dist]), ProductionRule[], next_grammar_id())
    check("§1 no candidates when feature_set ⊇ available",
          isempty(Credence._feature_candidates(g_full, Set([:colour, :wall_dist]))),
          "got $(Credence._feature_candidates(g_full, Set([:colour, :wall_dist])))")
end

# ── §2  _add_feature — surgery: a feature is a real description-length unit (Q2), grids preserved ──
let
    reset_grammar_counter!()
    g = Grammar(Set([:colour]), ProductionRule[], next_grammar_id())
    # Refine :colour's grid first, to prove the existing grid SURVIVES the add (Move-3 review lesson).
    g_ref = Credence._refine_grammar(g, :colour, 0.42)        # :colour grid gains 0.42
    refined_grid = g_ref.thresholds[:colour]

    g2 = Credence._add_feature(g_ref, :wall_dist)
    check("§2 the feature is added to feature_set",
          :wall_dist in g2.feature_set && g2.feature_set == Set([:colour, :wall_dist]),
          "got $(g2.feature_set)")
    check("§2 the new feature gets the default grid",
          g2.thresholds[:wall_dist] == THRESHOLDS, "got $(g2.thresholds[:wall_dist])")
    check("§2 the existing (refined) grid SURVIVES the add (not re-defaulted)",
          g2.thresholds[:colour] == refined_grid, "got $(g2.thresholds[:colour])")
    check("§2 complexity rises by exactly 1 (a feature IS a prior-Occam unit — Q2)",
          g2.complexity == g_ref.complexity + 1.0, "got $(g2.complexity) vs $(g_ref.complexity)")
    check("§2 the grammar gets a fresh id", g2.id != g_ref.id, "got $(g2.id)")
end

# ── shared fixture machinery (the test_coherent_injection idiom) ──
function mk_live_state(g::Grammar, AS::Vector{Symbol})
    progs = enumerate_programs(g, 2; action_space = AS)
    comps = TaggedBetaPrevision[TaggedBetaPrevision(i, BetaPrevision(1.0, 1.0))
                                for i in eachindex(progs)]
    lw = Float64[complexity_logprior(g.complexity; λ = log(2)) +
                 complexity_logprior(p.complexity; λ = log(2)) for p in progs]
    AgentState(MixturePrevision(Prevision[comps...], lw),
               [(g.id, i) for i in eachindex(progs)],
               CompiledKernel[compile_kernel(p, g, i) for (i, p) in enumerate(progs)],
               Program[progs...], Dict{Int, Grammar}(g.id => g), 2)
end
function live!(state, raw)
    buf = ExploreObservation[]
    for obs in raw
        k = program_space_observation_kernel(state.compiled_kernels, obs.features,
                                             obs.temporal_state, obs.correct_actions)
        res = -log_predictive(state.belief, k, 1.0)
        push!(buf, ExploreObservation(obs.features, obs.temporal_state,
                                      obs.correct_actions, res))
        state.belief = condition(state.belief, k, 1.0)
    end
    buf
end

# ── §3  discovery via the virtual injection ──
#
# Scenario: the label depends ONLY on :wall_dist (< 0.5 ⇒ :a, ≥ 0.5 ⇒ :b); :colour is uncorrelated noise.
# The colour-only grammar provably cannot separate the classes. The union injection carries the
# :wall_dist candidate's programs; `IF (lt :wall_dist 0.5) :a :b` earns the evidence.
let
    AS = Symbol[:a, :b]
    mk(wall, col, label) = ExploreObservation(
        Dict(:wall_dist => wall, :colour => col), Dict{Symbol, Any}(), Set([label]), 1.0)
    data = ExploreObservation[]
    for _ in 1:5
        push!(data, mk(0.2, 0.1, :a)); push!(data, mk(0.2, 0.9, :a))   # near wall ⇒ :a, colour varies
        push!(data, mk(0.8, 0.1, :b)); push!(data, mk(0.8, 0.9, :b))   # far  wall ⇒ :b, colour varies
    end
    reset_grammar_counter!()
    g = Grammar(Set([:colour]), ProductionRule[], next_grammar_id())
    available = Set([:colour, :wall_dist])

    s = mk_live_state(g, AS)
    buf = live!(s, data)
    prop = feature_growth_proposal(s, g, buf, available; action_space = AS)
    check("§3 the lookahead proposes (the :wall_dist union is non-empty)",
          prop !== nothing && prop.n_added > 0)

    # §3a Discovery: the yield is decisive and the union's mass concentrates on :wall_dist programs.
    check("§3a the yield clears the declared price (v = yield − log2 > 0)",
          net_value(prop.yield_nats, log(2.0)) > 0.0, "yield = $(prop.yield_nats)")
    n_inc = length(s.belief.components)
    w = weights(prop.scratch.belief)
    best_new = argmax(i -> w[i], (n_inc + 1):length(w))
    best_expr = show_expr(prop.scratch.all_programs[best_new].expr)
    check("§3a the top-weighted newcomer is a :wall_dist program (colour could not reach it)",
          occursin("wall_dist", best_expr), "top newcomer: $best_expr")
    check("§3a the union posterior concentrates on the newcomers", prop.p_newcomers > 0.5,
          "p_newcomers = $(prop.p_newcomers)")

    # §3b No evidence ⇒ the yield stays under the price: constant-label data is already fit by
    # the constant program, so the newcomers earn ≈ nothing and the score floor keeps act-now.
    flat = ExploreObservation[mk(0.2, 0.1, :a), mk(0.8, 0.9, :a), mk(0.5, 0.5, :a), mk(0.3, 0.7, :a)]
    s_flat = mk_live_state(g, AS)
    buf_flat = live!(s_flat, flat)
    p_flat = feature_growth_proposal(s_flat, g, buf_flat, available; action_space = AS)
    check("§3b constant label ⇒ the yield stays under the price (act-now floor holds)",
          p_flat === nothing || net_value(p_flat.yield_nats, log(2.0)) <= 0.0,
          p_flat === nothing ? "" : "yield = $(p_flat.yield_nats)")

    # §3c Empty candidate set ⇒ no proposal (every available feature already in the grammar).
    g_full = Credence._add_feature(g, :wall_dist)
    s_full = mk_live_state(g_full, AS)
    buf_full = live!(s_full, data)
    check("§3c empty candidate set ⇒ no proposal",
          feature_growth_proposal(s_full, g_full, buf_full, Set([:colour, :wall_dist]);
                                  action_space = AS) === nothing)

    # §3d Empty buffer ⇒ no proposal.
    check("§3d empty buffer ⇒ no proposal",
          feature_growth_proposal(mk_live_state(g, AS), g, ExploreObservation[], available;
                                  action_space = AS) === nothing)

    # §3e Determinism: identical inputs ⇒ identical proposal.
    sa = mk_live_state(g, AS); bufa = live!(sa, data)
    sb = mk_live_state(g, AS); bufb = live!(sb, data)
    pa = feature_growth_proposal(sa, g, bufa, available; action_space = AS)
    pb = feature_growth_proposal(sb, g, bufb, available; action_space = AS)
    check("§3e determinism: identical inputs ⇒ identical yield and union",
          pa.yield_nats == pb.yield_nats && pa.n_added == pb.n_added,
          "a = $(pa.yield_nats), b = $(pb.yield_nats)")
end

# ── §4  Q2 under the union mechanism: the prior axis is priced INSIDE the mixture ──
#
# Each feature newcomer arrives at 2^{−|G′|−|p|} with |G′| = |G| + 1 — the −log2 Occam charge is
# in its prior BEFORE any evidence flows (winners-curse Q4: no explicit prior term at any seam;
# double-charging is structurally impossible because there is no second site). The observable
# consequence pinned here: on the SAME evidence, the feature union's prior counterfactual mass —
# and hence its realised yield — is strictly depressed relative to what complexity-invariant
# candidates (thresholds, Move 3) would enjoy; and the realised yield is strictly below the
# fit-axis Δℓ the retired mechanism would have credited (the mixture prices what the argmax
# gave away free).
let
    AS = Symbol[:a, :b]
    mk(wall, col, label) = ExploreObservation(
        Dict(:wall_dist => wall, :colour => col), Dict{Symbol, Any}(), Set([label]), 1.0)
    data = ExploreObservation[]
    for _ in 1:5
        push!(data, mk(0.2, 0.1, :a)); push!(data, mk(0.2, 0.9, :a))
        push!(data, mk(0.8, 0.1, :b)); push!(data, mk(0.8, 0.9, :b))
    end
    reset_grammar_counter!()
    g = Grammar(Set([:colour]), ProductionRule[], next_grammar_id())
    available = Set([:colour, :wall_dist])

    s = mk_live_state(g, AS)
    buf = live!(s, data)
    prop = feature_growth_proposal(s, g, buf, available; action_space = AS)

    # The retired fit-axis Δℓ (counter-oracle, hand-replayed through canalised ops).
    # credence-lint: allow — precedent:test-oracle — counter-oracle for the retired fit axis
    function mll_fresh(g2::Grammar)
        progs = enumerate_programs(g2, 2; action_space = AS)
        cks = CompiledKernel[compile_kernel(p, g2, i) for (i, p) in enumerate(progs)]
        comps = TaggedBetaPrevision[TaggedBetaPrevision(i, BetaPrevision(1.0, 1.0))
                                    for i in eachindex(progs)]
        lw = Float64[complexity_logprior(g2.complexity; λ = log(2)) +
                     complexity_logprior(p.complexity; λ = log(2)) for p in progs]
        belief = MixturePrevision(Prevision[comps...], lw)
        mll = 0.0
        for obs in buf
            k = program_space_observation_kernel(cks, obs.features, obs.temporal_state,
                                                 obs.correct_actions)
            mll += -log_predictive(belief, k, 1.0)
            belief = condition(belief, k, 1.0)
        end
        mll
    end
    dl = mll_fresh(g) - mll_fresh(Credence._add_feature(g, :wall_dist))
    check("§4 the fit axis alone is decisive on this data (Δℓ > log2 — the retired credit)",
          dl > log(2), "Δℓ = $dl")
    check("§4 the realised yield is strictly below the fit-axis Δℓ (the mixture prices the prior axis)",
          prop.yield_nats < dl, "yield = $(prop.yield_nats), Δℓ = $dl")
    check("§4 and still clears the declared price (a strong feature is worth its symbol)",
          net_value(prop.yield_nats, log(2.0)) > 0.0, "yield = $(prop.yield_nats)")
end

println("="^64)
println("ALL CHECKS PASSED — feature-discovery")
println("="^64)
