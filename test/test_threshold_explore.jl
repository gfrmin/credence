# test_threshold_explore.jl — exploration-budget Move 3 (re-baselined by the winners-curse
# design revs 3–4: the lookahead is a virtual injection; the score is the yield rule §8.2).
# Sections:
#   §1  capture-before-refactor: a default grammar enumerates IDENTICALLY after the Grammar.thresholds
#       field + enumeration rewire (the 42-program canonical signature, pinned PRE-change on master).
#   §2  candidate generation — midpoints between adjacent observed values (complete finite set).
#   §3  the virtual injection: discovery of an off-grid optimum (Scope A provably cannot reach it),
#       structural completeness (ALL candidates ride in the union — no residual-order early stop to
#       guard), no-op identity, determinism.
#   §4  a refined grid survives compression (perturb_grammar threads g.thresholds).
#
# Run: julia test/test_threshold_explore.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, Program, CompiledKernel, enumerate_programs, show_expr,
                THRESHOLDS, default_thresholds, ExploreObservation,
                next_grammar_id, reset_grammar_counter!,
                perturb_grammar, SubprogramFrequencyTable, ProgramExpr, AndExpr, GTExpr, LTExpr,
                AgentState, MixturePrevision, TaggedBetaPrevision, BetaPrevision, Prevision,
                compile_kernel, complexity_logprior, program_space_observation_kernel,
                condition, log_predictive, net_value, weights,
                threshold_growth_proposal, adopt!

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("threshold-explore — Move 3")
println("="^64)

# ── §1  capture-before-refactor: default-grammar enumeration is bit-identical ──
#
# Canonical signature captured on master (pre-Grammar.thresholds) for
#   Grammar(Set([:red, :speed]), ProductionRule[], 1), enumerate_programs(g, 2; [:food, :enemy])
# 42 programs: 2 constant actions (c=1), then sorted features (:red < :speed) × THRESHOLDS × {GT,LT}
# × {(food,enemy),(enemy,food)} (c=4 each). The default per-feature grid MUST equal the old global grid,
# so this enumeration MUST be unchanged. Asserted `==` (capture-before-refactor; the pin guards the rewire).
let
    g = Grammar(Set([:red, :speed]), ProductionRule[], 1)

    # The default grid is exactly the global THRESHOLDS for every feature.
    check("§1 default_thresholds ≡ global THRESHOLDS per feature",
          g.thresholds[:red] == THRESHOLDS && g.thresholds[:speed] == THRESHOLDS,
          "got red=$(g.thresholds[:red]) speed=$(g.thresholds[:speed])")
    check("§1 complexity is threshold-count-invariant (|G| = #features = 2.0)",
          g.complexity == 2.0, "got $(g.complexity)")

    progs = enumerate_programs(g, 2; action_space = Symbol[:food, :enemy])
    sig = [(show_expr(p.expr), p.complexity) for p in progs]

    expected = Tuple{String, Int}[
        ("food", 1), ("enemy", 1),
        ("IF((gt :red 0.1),food,enemy)", 4), ("IF((gt :red 0.1),enemy,food)", 4),
        ("IF((lt :red 0.1),food,enemy)", 4), ("IF((lt :red 0.1),enemy,food)", 4),
        ("IF((gt :red 0.3),food,enemy)", 4), ("IF((gt :red 0.3),enemy,food)", 4),
        ("IF((lt :red 0.3),food,enemy)", 4), ("IF((lt :red 0.3),enemy,food)", 4),
        ("IF((gt :red 0.5),food,enemy)", 4), ("IF((gt :red 0.5),enemy,food)", 4),
        ("IF((lt :red 0.5),food,enemy)", 4), ("IF((lt :red 0.5),enemy,food)", 4),
        ("IF((gt :red 0.7),food,enemy)", 4), ("IF((gt :red 0.7),enemy,food)", 4),
        ("IF((lt :red 0.7),food,enemy)", 4), ("IF((lt :red 0.7),enemy,food)", 4),
        ("IF((gt :red 0.9),food,enemy)", 4), ("IF((gt :red 0.9),enemy,food)", 4),
        ("IF((lt :red 0.9),food,enemy)", 4), ("IF((lt :red 0.9),enemy,food)", 4),
        ("IF((gt :speed 0.1),food,enemy)", 4), ("IF((gt :speed 0.1),enemy,food)", 4),
        ("IF((lt :speed 0.1),food,enemy)", 4), ("IF((lt :speed 0.1),enemy,food)", 4),
        ("IF((gt :speed 0.3),food,enemy)", 4), ("IF((gt :speed 0.3),enemy,food)", 4),
        ("IF((lt :speed 0.3),food,enemy)", 4), ("IF((lt :speed 0.3),enemy,food)", 4),
        ("IF((gt :speed 0.5),food,enemy)", 4), ("IF((gt :speed 0.5),enemy,food)", 4),
        ("IF((lt :speed 0.5),food,enemy)", 4), ("IF((lt :speed 0.5),enemy,food)", 4),
        ("IF((gt :speed 0.7),food,enemy)", 4), ("IF((gt :speed 0.7),enemy,food)", 4),
        ("IF((lt :speed 0.7),food,enemy)", 4), ("IF((lt :speed 0.7),enemy,food)", 4),
        ("IF((gt :speed 0.9),food,enemy)", 4), ("IF((gt :speed 0.9),enemy,food)", 4),
        ("IF((lt :speed 0.9),food,enemy)", 4), ("IF((lt :speed 0.9),enemy,food)", 4),
    ]

    check("§1 enumeration count unchanged (42)", length(sig) == 42, "got $(length(sig))")
    check("§1 enumeration bit-identical to pre-refactor canonical (==)",
          sig == expected, "enumeration drifted after the Grammar.thresholds rewire")
end

# ── §2  candidate generation: midpoints between adjacent observed values, off-grid only ──
let
    _cand = Credence._threshold_candidates
    reset_grammar_counter!()
    g = Grammar(Set([:x]), ProductionRule[], next_grammar_id())   # default grid [0.1,0.3,0.5,0.7,0.9]
    mk(v) = ExploreObservation(Dict(:x => v), Dict{Symbol,Any}(), Set([:a]), 0.0)

    # Observed :x ∈ {0.2, 0.5, 0.62} → midpoints 0.35, 0.56 (neither on the default grid).
    obs = [mk(0.2), mk(0.5), mk(0.62), mk(0.5)]   # 0.5 repeated → unique handles it
    cands = _cand(g, obs)
    check("§2 candidates are the off-grid midpoints {0.35, 0.56}",
          Set(t for (_, t) in cands) == Set([0.35, 0.56]),
          "got $(cands)")
    check("§2 all candidates are on feature :x", all(f == :x for (f, _) in cands), "got $(cands)")

    # A midpoint landing exactly on an existing grid point is excluded (0.2,0.4 → 0.3 ∈ grid).
    obs2 = [mk(0.2), mk(0.4)]
    check("§2 midpoint coinciding with a grid point is excluded",
          isempty(_cand(g, obs2)), "got $(_cand(g, obs2))")

    # Fewer than two distinct observed values ⇒ no candidate (a threshold needs a gap to split).
    check("§2 <2 distinct values ⇒ no candidates",
          isempty(_cand(g, [mk(0.3), mk(0.3)])), "got $(_cand(g, [mk(0.3), mk(0.3)]))")

    # _refine_grammar inserts the threshold (sorted), fresh id, complexity unchanged (Q1(b)).
    g2 = Credence._refine_grammar(g, :x, 0.56)
    check("§2 refined grid is sorted with the new threshold inserted",
          g2.thresholds[:x] == [0.1, 0.3, 0.5, 0.56, 0.7, 0.9], "got $(g2.thresholds[:x])")
    check("§2 refinement is complexity-invariant (Q1(b))", g2.complexity == g.complexity,
          "got $(g2.complexity) vs $(g.complexity)")
    check("§2 refined grammar has a fresh id", g2.id != g.id, "got $(g2.id)")
end

# ── §3  the virtual injection: discovery, completeness, no-op, determinism ──
#
# Scenario: feature :x, true class boundary at x ≈ 0.62 (OFF the default grid [0.1,…,0.9]). Observed
# values {0.55, 0.60} → :b and {0.65, 0.70} → :a. No on-grid threshold separates them; the midpoint
# candidate 0.625 (between observed 0.60 and 0.65) splits them PERFECTLY. Scope A (compression) can never
# add a threshold, so it provably cannot reach this grammar. Under the winners-curse mechanism the
# lookahead injects ALL midpoint refinements into one scratch union; the evidence concentrates on the
# 0.625 programs and the yield clears the price.
let
    AS = Symbol[:a, :b]
    mk(x, label; r = 1.0) = ExploreObservation(Dict(:x => x), Dict{Symbol, Any}(), Set([label]), r)
    reset_grammar_counter!()
    g = Grammar(Set([:x]), ProductionRule[], next_grammar_id())

    # A live state over g (the test_coherent_injection idiom): fresh complexity-prior belief,
    # conditioned prequentially so each record carries its honest residual ledger.
    function g_state()
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

    data = ExploreObservation[]
    for _ in 1:5
        push!(data, mk(0.55, :b)); push!(data, mk(0.60, :b))
        push!(data, mk(0.65, :a)); push!(data, mk(0.70, :a))
    end
    s = g_state()
    buf = live!(s, data)

    prop = threshold_growth_proposal(s, g, buf; action_space = AS)
    check("§3 the lookahead proposes (candidates exist, union non-empty)",
          prop !== nothing && prop.n_added > 0)

    # §3a Discovery: the union's evidence is decisive and concentrates on the 0.625 split.
    check("§3a the yield clears the declared price (v = yield − log2 > 0)",
          net_value(prop.yield_nats, log(2.0)) > 0.0, "yield = $(prop.yield_nats)")
    n_inc = length(s.belief.components)
    w = weights(prop.scratch.belief)
    best_new = argmax(i -> w[i], (n_inc + 1):length(w))
    best_expr = show_expr(prop.scratch.all_programs[best_new].expr)
    check("§3a the top-weighted newcomer uses the perfect-separation midpoint 0.625",
          occursin("0.625", best_expr), "top newcomer: $best_expr")

    # §3b Structural completeness (supersedes the residual-order early-stop guard): EVERY
    # midpoint candidate's grammar rides in the union — none is screened out, so no
    # positive-evidence candidate can be skipped, by construction.
    cand_ts = Set(t for (_, t) in Credence._threshold_candidates(g, buf))
    scratch_gids = setdiff(keys(prop.scratch.grammars), keys(s.grammars))
    scratch_ts = Set{Float64}()
    for gid in scratch_gids
        for t in prop.scratch.grammars[gid].thresholds[:x]
            t in g.thresholds[:x] || push!(scratch_ts, t)
        end
    end
    check("§3b ALL midpoint candidates ride in the union (inject-all, ratified Q3)",
          cand_ts == scratch_ts, "candidates $(cand_ts) vs union $(scratch_ts)")

    # §3c Adoption = the transition: the scratch becomes the state; the refined grammars are
    # registered and the union's programs are carried (average-not-collapse — no winner picked).
    adopt!(s, prop.scratch)
    check("§3c adoption registers the refined grammars and carries the union",
          length(s.belief.components) == n_inc + prop.n_added &&
          all(haskey(s.grammars, gid) for gid in scratch_gids))

    # §3d No-op identity: empty buffer ⇒ no proposal (the state untouched, nothing to adopt).
    s2 = g_state()
    check("§3d empty buffer ⇒ no proposal",
          threshold_growth_proposal(s2, g, ExploreObservation[]; action_space = AS) === nothing)

    # §3e Determinism: identical inputs ⇒ identical proposal (no rand — the same yield, the
    # same union metadata, componentwise).
    sa = g_state(); bufa = live!(sa, data)
    sb = g_state(); bufb = live!(sb, data)
    pa = threshold_growth_proposal(sa, g, bufa; action_space = AS)
    pb = threshold_growth_proposal(sb, g, bufb; action_space = AS)
    check("§3e determinism: identical inputs ⇒ identical yield and union",
          pa.yield_nats == pb.yield_nats && pa.n_added == pb.n_added &&
          [m[2] for m in pa.scratch.metadata] == [m[2] for m in pb.scratch.metadata],
          "a = $(pa.yield_nats), b = $(pb.yield_nats)")
end

# ── §4  a refined grid survives compression (perturb_grammar threads g.thresholds — review should-fix) ──
let
    reset_grammar_counter!()
    g = Grammar(Set([:x]), ProductionRule[], next_grammar_id())
    g_ref = Credence._refine_grammar(g, :x, 0.42)        # grid [0.1,0.3,0.42,0.5,0.7,0.9]
    refined_grid = g_ref.thresholds[:x]

    # A freq_table whose top subtree compresses: AndExpr (complexity 3) used by 3 programs ⇒
    # net_payoff = 3·(3−1) − (1+3) = 2 > 0 ⇒ :add_rule fires (referenced=nothing ⇒ no removal candidates).
    sub = AndExpr(GTExpr(FeatureRef(:x), 0.3), LTExpr(FeatureRef(:x), 0.7))
    table = SubprogramFrequencyTable(ProgramExpr[sub], [3.0], [[1, 2, 3]])
    g_perturbed = perturb_grammar(g_ref, table, g_ref.feature_set)

    check("§4 perturb compressed the refined grammar (a rule was added)",
          length(g_perturbed.rules) == length(g_ref.rules) + 1, "rules=$(g_perturbed.rules)")
    check("§4 the refined threshold grid SURVIVES compression (not re-defaulted to the global grid)",
          g_perturbed.thresholds[:x] == refined_grid, "got $(g_perturbed.thresholds[:x])")
end

println("="^64)
println("ALL CHECKS PASSED — threshold-explore")
println("="^64)
