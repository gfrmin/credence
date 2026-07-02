# test_threshold_explore.jl — exploration-budget Move 3: compute-budgeted lookahead VOI for threshold
# refinement (the headline move). Sections:
#   §1  capture-before-refactor: a default grammar enumerates IDENTICALLY after the Grammar.thresholds
#       field + enumeration rewire (the 42-program canonical signature, pinned PRE-change on master).
#   §2  candidate generation — midpoints between adjacent observed values (complete finite set).
#   §3  the lookahead VOI + cap-free sequential stop + completeness guard.
#   §4  discovery: explore_grammar finds an off-grid optimum Scope A provably cannot reach.
#
# Run: julia test/test_threshold_explore.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, enumerate_programs, show_expr, THRESHOLDS,
                default_thresholds, explore_grammar, ExploreObservation,
                next_grammar_id, reset_grammar_counter!,
                perturb_grammar, SubprogramFrequencyTable, ProgramExpr, AndExpr, GTExpr, LTExpr

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

# ── §3  the lookahead VOI + explore_grammar: discovery, no-op, cap-free boundary, completeness guard ──
#
# Scenario: feature :x, true class boundary at x ≈ 0.62 (OFF the default grid [0.1,…,0.9]). Observed
# values {0.55, 0.60} → :b and {0.65, 0.70} → :a. No on-grid threshold separates them; the midpoint
# candidate 0.625 (between observed 0.60 and 0.65) splits them PERFECTLY. Scope A (compression) can never
# add a threshold, so it provably cannot reach this grammar.
let
    AS = Symbol[:a, :b]
    mk(x, label; r = 1.0) = ExploreObservation(Dict(:x => x), Dict{Symbol, Any}(), Set([label]), r)
    # 5 copies of each ⇒ the marginal-likelihood gap is decisive.
    data = ExploreObservation[]
    for _ in 1:5
        push!(data, mk(0.55, :b)); push!(data, mk(0.60, :b))
        push!(data, mk(0.65, :a)); push!(data, mk(0.70, :a))
    end
    reset_grammar_counter!()
    g = Grammar(Set([:x]), ProductionRule[], next_grammar_id())

    # The lookahead gives the perfect split strictly lower marginal log-loss than the on-grid baseline.
    baseline = Credence._grammar_marginal_log_loss(g, data, 2, AS)
    g625 = Credence._refine_grammar(g, :x, 0.625)
    mll625 = Credence._grammar_marginal_log_loss(g625, data, 2, AS)
    dl = baseline - mll625
    check("§3 Δℓ > 0: the off-grid 0.625 split beats the on-grid baseline", dl > 0.0,
          "baseline=$baseline mll625=$mll625 Δℓ=$dl")

    # §3a Discovery: explore_grammar refines the grid and adds a split at the true boundary.
    g2 = explore_grammar(g, data, 2; action_space = AS, compute_cost = 0.0)
    check("§3a discovery: grid refined (off-grid optimum found; Scope A cannot)", g2.id != g.id,
          "explore returned the input grammar unchanged")
    check("§3a discovery: the added threshold is the perfect-separation midpoint 0.625",
          any(isapprox(t, 0.625; atol = 1e-9) for t in g2.thresholds[:x]),
          "got grid $(g2.thresholds[:x])")
    check("§3a discovery: exactly one threshold added (one refinement per call)",
          length(g2.thresholds[:x]) == length(g.thresholds[:x]) + 1, "got $(g2.thresholds[:x])")

    # §3b No-op: a compute_cost above every candidate's Δℓ suppresses refinement — the input is returned.
    g_noop = explore_grammar(g, data, 2; action_space = AS, compute_cost = 1.0e6)
    check("§3b no-op when compute_cost exceeds every VOI (same grammar object)", g_noop === g,
          "expected the input grammar; got id $(g_noop.id)")

    # §3c Cap-free graceful boundary: the explore/no-op decision flips CONTINUOUSLY at compute_cost = Δℓ —
    # no cliff, no hard cap. Just below Δℓ ⇒ refine; just above ⇒ no-op (Q3b graceful degradation).
    g_below = explore_grammar(g, data, 2; action_space = AS, compute_cost = dl - 0.1)
    g_above = explore_grammar(g, data, 2; action_space = AS, compute_cost = dl + 0.1)
    check("§3c graceful: compute_cost just below Δℓ ⇒ refine", g_below.id != g.id, "got no-op")
    check("§3c graceful: compute_cost just above Δℓ ⇒ no-op (smooth, no cap)", g_above === g, "got refine")

    # §3d Completeness guard (Q3b one-sidedness): make 0.625 the LOWEST residual-mass candidate (residual
    # concentrated on the 0.55/0.70 extremes, near-zero on 0.60/0.65) — so a residual-order EARLY-STOP
    # would evaluate it LAST and might skip it. Full evaluation still selects it (global VOI argmax),
    # proving no positive-VOI candidate is skipped.
    data_skew = ExploreObservation[]
    for _ in 1:5
        push!(data_skew, mk(0.55, :b; r = 5.0)); push!(data_skew, mk(0.60, :b; r = 0.01))
        push!(data_skew, mk(0.65, :a; r = 0.01)); push!(data_skew, mk(0.70, :a; r = 5.0))
    end
    # 0.625 brackets observed 0.60 & 0.65 (the low-residual pair) ⇒ lowest residual mass of the candidates.
    m575 = Credence._candidate_residual_mass(:x, 0.575, data_skew)
    m625 = Credence._candidate_residual_mass(:x, 0.625, data_skew)
    m675 = Credence._candidate_residual_mass(:x, 0.675, data_skew)
    check("§3d setup: 0.625 has the lowest residual mass (would be evaluated last)",
          m625 < m575 && m625 < m675, "masses 575=$m575 625=$m625 675=$m675")
    g_skew = explore_grammar(g, data_skew, 2; action_space = AS, compute_cost = 0.0)
    check("§3d completeness: full-eval still selects 0.625 (one-sided; lowest-residual ≠ skipped)",
          any(isapprox(t, 0.625; atol = 1e-9) for t in g_skew.thresholds[:x]),
          "got grid $(g_skew.thresholds[:x])")

    # §3e Determinism: two runs on identical inputs produce the same refined grid (no rand, deterministic
    # argmax — the Invariant-1 breach that Phase 5 closed stays closed).
    ga = explore_grammar(g, data, 2; action_space = AS, compute_cost = 0.0)
    gb = explore_grammar(g, data, 2; action_space = AS, compute_cost = 0.0)
    check("§3e determinism: identical inputs ⇒ identical refined grid", ga.thresholds == gb.thresholds,
          "a=$(ga.thresholds) b=$(gb.thresholds)")

    # §3f Empty buffer / no candidates ⇒ no-op (the input grammar, unchanged).
    check("§3f empty buffer ⇒ no-op", explore_grammar(g, ExploreObservation[], 2; action_space = AS) === g,
          "expected no-op on empty buffer")

    # §3g exploration_voi — the scalar the selection layer ranks by; shares _best_threshold_refinement with
    # explore_grammar exactly, so the ranked value == the applied edit's Δℓ (Invariant 3, no drift).
    check("§3g exploration_voi == the winning candidate's Δℓ (cc=0)",
          exploration_voi(g, data, 2; action_space = AS, compute_cost = 0.0) == dl,
          "got $(exploration_voi(g, data, 2; action_space = AS, compute_cost = 0.0)) vs dl=$dl")
    check("§3g exploration_voi == 0.0 when compute_cost suppresses every candidate (no-op floor)",
          exploration_voi(g, data, 2; action_space = AS, compute_cost = 1.0e6) == 0.0,
          "got $(exploration_voi(g, data, 2; action_space = AS, compute_cost = 1.0e6))")
    check("§3g exploration_voi == 0.0 on empty buffer",
          exploration_voi(g, ExploreObservation[], 2; action_space = AS) == 0.0,
          "got $(exploration_voi(g, ExploreObservation[], 2; action_space = AS))")
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
