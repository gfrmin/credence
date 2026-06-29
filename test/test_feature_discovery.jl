# test_feature_discovery.jl — exploration-budget Move 4: feature-discovery lookahead VOI (`:add_feature`).
# The belief-aware sibling of Move 3's threshold refinement, one rung up the fine-before-coarse ladder:
# grow the agent's FEATURE alphabet by EU-max SELECTION over a host-furnished candidate set
# (`available_features \ feature_set`), valued by the same compute-budgeted lookahead.
#
# Sections:
#   §1  candidate generation — the host-furnished set `available_features \ feature_set` (sorted).
#   §2  _add_feature — surgery: feature added, default grid for it, OTHER features' grids preserved,
#       fresh id, complexity +1 (a feature IS a description-length unit — Q2, unlike a threshold).
#   §3  discovery: a grammar missing the predictive feature acquires it; no-op when nothing helps;
#       cap-free graceful boundary; determinism; empty.
#   §4  Q2 — the two-axis pricing is MECHANIZED. The grammar-complexity prior CANCELS inside the
#       normalized marginal log-loss (a per-grammar constant), so the prior-Occam penalty is charged
#       EXPLICITLY at the argmax (`+ complexity_logprior(Δcomplexity; λ=log2)`). The boundary sits at
#       Δℓ − log2, NOT at Δℓ — the signature distinguishing Move 4 (features, dearer) from Move 3
#       (thresholds, complexity-invariant). This is the literal mechanical content of Q1(b)≡Q2's converse.
#
# Run: julia test/test_feature_discovery.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, ExploreObservation, THRESHOLDS,
                next_grammar_id, reset_grammar_counter!, explore_grammar

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

# ── §3  discovery: a grammar missing the predictive feature acquires it via EU-max selection ──
#
# Scenario: the label depends ONLY on :wall_dist (< 0.5 ⇒ :a, ≥ 0.5 ⇒ :b); :colour is uncorrelated noise.
# The colour-only grammar provably cannot separate the classes (no colour program beats chance). Adding
# :wall_dist admits `IF (lt :wall_dist 0.5) :a :b` — a PERFECT separator the colour grammar could not reach.
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

    # The lookahead: adding :wall_dist gives strictly lower marginal log-loss than the colour-only baseline.
    baseline = Credence._grammar_marginal_log_loss(g, data, 2, AS)
    g_wall = Credence._add_feature(g, :wall_dist)
    mll_wall = Credence._grammar_marginal_log_loss(g_wall, data, 2, AS)
    dl = baseline - mll_wall
    check("§3 Δℓ > 0: adding :wall_dist beats the colour-only baseline", dl > 0.0,
          "baseline=$baseline mll_wall=$mll_wall Δℓ=$dl")
    check("§3 Δℓ clears the prior-Occam bar (Δℓ > log2 — a strong feature is worth a symbol)",
          dl > log(2), "Δℓ=$dl log2=$(log(2))")

    # §3a Discovery: explore_features selects :wall_dist (the EU-max selection over host-furnished features).
    g2 = Credence.explore_features(g, data, available, 2; action_space = AS, compute_cost = 0.0)
    check("§3a discovery: the predictive feature is acquired (colour grammar could not reach this)",
          :wall_dist in g2.feature_set, "feature_set=$(g2.feature_set)")
    check("§3a discovery: returns a fresh grammar (not the input)", g2.id != g.id, "got input unchanged")
    check("§3a discovery: exactly the one feature added (selection, not construction)",
          g2.feature_set == Set([:colour, :wall_dist]), "got $(g2.feature_set)")

    # §3b No-op when no candidate feature helps: constant-label data is already fit by the constant
    # program, so no feature improves the marginal likelihood ⇒ Δℓ ≈ 0 < log2 ⇒ rejected (the prior bar).
    flat = ExploreObservation[mk(0.2, 0.1, :a), mk(0.8, 0.9, :a), mk(0.5, 0.5, :a), mk(0.3, 0.7, :a)]
    g_flat = Credence.explore_features(g, flat, available, 2; action_space = AS, compute_cost = 0.0)
    check("§3b no-op when no feature helps (constant label ⇒ Δℓ < log2 ⇒ prior bar rejects)",
          g_flat === g, "expected the input grammar; got id $(g_flat.id)")

    # §3c Empty candidate set ⇒ no-op (every available feature already in the grammar).
    check("§3c empty candidate set ⇒ no-op",
          Credence.explore_features(g_wall, data, Set([:colour, :wall_dist]), 2;
                                    action_space = AS, compute_cost = 0.0) === g_wall,
          "expected no-op when feature_set ⊇ available")

    # §3d Empty buffer ⇒ no-op.
    check("§3d empty buffer ⇒ no-op",
          Credence.explore_features(g, ExploreObservation[], available, 2; action_space = AS) === g,
          "expected no-op on empty buffer")

    # §3e Determinism: identical inputs ⇒ identical result (no rand; deterministic argmax).
    ga = Credence.explore_features(g, data, available, 2; action_space = AS, compute_cost = 0.0)
    gb = Credence.explore_features(g, data, available, 2; action_space = AS, compute_cost = 0.0)
    check("§3e determinism: identical inputs ⇒ identical feature_set",
          ga.feature_set == gb.feature_set, "a=$(ga.feature_set) b=$(gb.feature_set)")
end

# ── §4  Q2 MECHANIZED: the prior-Occam term is charged EXPLICITLY (boundary at Δℓ − log2, not Δℓ) ──
#
# The grammar-complexity prior is a per-grammar CONSTANT added to every component's log-weight, so it
# CANCELS inside the normalized marginal log-loss (`log_predictive`/`condition` normalize). Reusing
# `_grammar_marginal_log_loss` verbatim would therefore price a feature on the FIT axis alone — violating
# Q2. `explore_features` charges the prior penalty `complexity_logprior(Δcomplexity; λ=log2) = −log2`
# explicitly at the argmax. The decisive test: a compute_cost just ABOVE `Δℓ − log2` must no-op. Were the
# log2 term absent (Move-3 thresholds, complexity-invariant), the boundary would sit at Δℓ and that same
# compute_cost would still ADD. The no-op proves the prior axis is priced.
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

    baseline = Credence._grammar_marginal_log_loss(g, data, 2, AS)
    g_wall = Credence._add_feature(g, :wall_dist)
    dl = baseline - Credence._grammar_marginal_log_loss(g_wall, data, 2, AS)

    # Boundary at compute_cost = Δℓ − log2: just below ⇒ add; just above ⇒ no-op.
    below = Credence.explore_features(g, data, available, 2; action_space = AS, compute_cost = dl - log(2) - 0.1)
    above = Credence.explore_features(g, data, available, 2; action_space = AS, compute_cost = dl - log(2) + 0.1)
    check("§4 compute_cost just below Δℓ−log2 ⇒ feature added", :wall_dist in below.feature_set,
          "expected add at cost $(dl - log(2) - 0.1)")
    check("§4 compute_cost just above Δℓ−log2 ⇒ NO-OP (the log2 prior term is charged; boundary ≠ Δℓ)",
          above === g, "expected no-op at cost $(dl - log(2) + 0.1); got id $(above.id)")

    # The contrast that makes §4 decisive: cost just above Δℓ−log2 is still WELL below Δℓ, so a
    # fit-axis-only valuation (no log2) would have ADDED here. The no-op above is the proof.
    check("§4 the no-op cost is strictly below Δℓ (a fit-only rule would have added — it didn't)",
          dl - log(2) + 0.1 < dl, "cost=$(dl - log(2) + 0.1) Δℓ=$dl")
end

println("="^64)
println("ALL CHECKS PASSED — feature-discovery")
println("="^64)
