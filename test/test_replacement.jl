# test_replacement.jl — replacement semantics for applied compression-class removals
# (docs/exploration-budget/removal-consumption-design.md; the #189 deviation-3 discharge).
#
# The constitutional pin for the design's §1 derivation: an applied removal REPLACES its ancestor —
# re-key the group's metadata to the cleaned grammar, shift the group's log-weights by
# Δ = λ·(complexity(g) − complexity(g′)), delete g, register g′ — and this transition is the UNIQUE
# map under which replacement commutes with conditioning (every likelihood term cancels identically;
# only the grammar-complexity prior term differs). The score is the realised Δ log-evidence of firing,
# log1p((e^Δ − 1)·W_G), sharing one candidate function with the transition (T-3.55: score = transition).
# The write is sanctioned by the `coherent-space-edit` precedent, whose named equality test is §1 here.
#
# Sections:
#   §1  commutation — replace-then-condition == counterfactual clean-start-with-g′-then-condition
#       (normalised weights ≤ 1e-12; Beta params, tags, metadata exactly ==).
#   §2  treadmill regression — after firing, the successor has no candidate (the candidate died with
#       its grammar); the consumed gid is unregistered, so the same removal can never re-propose.
#   §3  the OQ-4 soundness case — a live component with normalised weight < 1e-15 that references the
#       feature BLOCKS candidacy (this fails against the SubprogramFrequencyTable's w > 1e-15 support
#       cut and passes against the group-local full-support walk).
#   §4  score exactness — replacement_value == log1p((e^Δ − 1)·W_G) − cost against a hand-built
#       oracle; the realised post-fire group mass equals the score's prediction (score = transition,
#       measured); the W_G → 1 degeneration reproduces the prior-only voc.
#   §5  score/transition unity — best_replacement is pure and deterministic (=== across calls), the
#       scored candidate IS the applied candidate, and the total order (payoff, then name) is pinned.
#   §6  registry hygiene — consumed gid absent, g′ registered with thresholds threaded, group
#       metadata re-keyed, and a later enumerate of g′ dedups to zero (replacement adds nothing).
#
# Run: julia test/test_replacement.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, Program, FeatureRef, GTExpr, LTExpr, AndExpr, IfExpr,
                ActionExpr, NonterminalRef, ProgramExpr,
                AgentState, MixturePrevision, TaggedBetaPrevision, BetaPrevision, Prevision,
                CompiledKernel, compile_kernel, enumerate_programs, complexity_logprior,
                compute_grammar_complexity, expr_complexity, collect_feature_refs!,
                add_programs_to_state!, ExploreObservation, program_space_observation_kernel,
                weights, condition, log_predictive, probability, TagSet, Interval, sync_prune!,
                analyse_posterior_subtrees, net_voc,
                ReplacementCandidate, replacement_candidates, replacement_value, best_replacement,
                replace_grammar_in_state!

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

const AS = Symbol[:food, :enemy]
const DEPTH = 2

# Re-home programs enumerated from a donor grammar under a target grammar id (the post-prune shape:
# a group legitimately holds any SUBSET of its grammar's language, and a group whose surviving
# programs never touch a feature is exactly how that feature goes dead in live state).
rehome(progs::Vector{Program}, gid::Int) = Program[Program(p.expr, p.complexity, gid) for p in progs]

# Build an AgentState over explicit (grammar, programs) groups with complexity-prior log-weights
# (mirrors the hosts' construction; tags positional — the re-tag discipline).
function build_state(pairs::Vector{Tuple{Grammar, Vector{Program}}}; lw_override = nothing)
    comps = TaggedBetaPrevision[]
    lw = Float64[]
    meta = Tuple{Int, Int}[]
    cks = CompiledKernel[]
    progs = Program[]
    i = 0
    for (g, ps) in pairs
        for (pi, p) in enumerate(ps)
            i += 1
            push!(comps, TaggedBetaPrevision(i, BetaPrevision(1.0, 1.0)))
            push!(lw, complexity_logprior(g.complexity; λ = log(2)) +
                      complexity_logprior(p.complexity; λ = log(2)))
            push!(meta, (g.id, pi))
            push!(cks, compile_kernel(p, g, i))
            push!(progs, p)
        end
    end
    lw_override !== nothing && (lw = lw_override)
    AgentState(MixturePrevision(Prevision[comps...], lw), meta, cks, progs,
               Dict{Int, Grammar}(g.id => g for (g, _) in pairs), DEPTH)
end

# Live conditioning exactly as the hosts do it (the test_coherent_injection.jl idiom).
function condition_live!(state::AgentState, raw)
    for (features, correct) in raw
        k = program_space_observation_kernel(state.compiled_kernels, features,
                                             Dict{Symbol, Any}(), correct)
        state.belief = condition(state.belief, k, 1.0)
    end
    state
end

# Canalised group mass (the same Prevision-level read the score uses).
group_mass(state::AgentState, gid::Int) =
    probability(state.belief, TagSet(Interval(0.0, 1.0),
                Set(i for i in eachindex(state.metadata) if state.metadata[i][1] == gid)))

# The §1/§4 fixture: gA carries features {a, b} but its surviving programs test only :a (⇒ :b is
# dead, payoff 1); gB is an independent lineage over :c (the cross-group mass the shift renormalises
# against). Fixed manual ids well above next_grammar_id()'s counter.
gA() = Grammar(Set([:a, :b]), ProductionRule[], 911)
gB() = Grammar(Set([:c]), ProductionRule[], 912)
progsA() = rehome(enumerate_programs(Grammar(Set([:a]), ProductionRule[], 913), DEPTH;
                                     action_space = AS), 911)
progsB() = enumerate_programs(gB(), DEPTH; action_space = AS)

raw_pre() = [
    (Dict(:a => 0.9, :c => 0.2), Set([:food])),
    (Dict(:a => 0.1, :c => 0.8), Set([:enemy])),
    (Dict(:a => 0.8, :c => 0.3), Set([:food])),
]
raw_post() = [
    (Dict(:a => 0.2, :c => 0.9), Set([:enemy])),
    (Dict(:a => 0.7, :c => 0.1), Set([:food])),
    (Dict(:a => 0.3, :c => 0.7), Set([:enemy])),
]

println("="^64)
println("replacement semantics — removal consumption")
println("="^64)

# ── §1  commutation: the coherent-space-edit equality (the precedent's named test) ──
let
    # Path R: condition, REPLACE mid-history, condition more.
    sR = build_state([(gA(), progsA()), (gB(), progsB())])
    condition_live!(sR, raw_pre())
    cand = best_replacement(sR, 911)
    check("§1 precondition: the dead feature :b is the candidate",
          cand !== nothing && cand.kind == :remove_feature && cand.payload === :b,
          "cand=$cand")
    g2 = replace_grammar_in_state!(sR, cand)
    condition_live!(sR, raw_post())

    # Path C: the counterfactual agent that held g′ from the start, conditioning on the SAME history.
    sC = build_state([(Grammar(Set([:a]), ProductionRule[], g2.id), rehome(progsA(), g2.id)),
                      (gB(), progsB())])
    condition_live!(sC, raw_pre())
    condition_live!(sC, raw_post())

    check("§1 g′ complexity strictly below g (the reclaim is real)",
          g2.complexity == gA().complexity - 1, "g2.complexity=$(g2.complexity)")
    check("§1 metadata identical after re-key (group re-homed to g′, program ids kept)",
          sR.metadata == sC.metadata)
    maxdiff = maximum(abs.(sR.belief.log_weights .- sC.belief.log_weights))
    check("§1 log-weights equal ≤ 1e-12 (replacement commutes with conditioning)",
          maxdiff <= 1e-12, "max abs diff = $maxdiff")
    wdiff = maximum(abs.(weights(sR.belief) .- weights(sC.belief)))
    check("§1 normalised weights equal ≤ 1e-12", wdiff <= 1e-12, "max abs diff = $wdiff")
    betas_ok = all(eachindex(sR.belief.components)) do i
        cu = sR.belief.components[i]; ci = sC.belief.components[i]
        cu.tag == ci.tag && cu.beta.alpha == ci.beta.alpha && cu.beta.beta == ci.beta.beta
    end
    check("§1 Beta states and tags == componentwise (evidence untouched by the edit)", betas_ok)
end

# ── §2  treadmill regression: the candidate dies with its grammar ──
let
    s = build_state([(gA(), progsA()), (gB(), progsB())])
    condition_live!(s, raw_pre())
    n_comps = length(s.belief.components)
    cand = best_replacement(s, 911)
    g2 = replace_grammar_in_state!(s, cand)

    check("§2 replacement adds ZERO components (re-description, not exploration — T-3.52)",
          length(s.belief.components) == n_comps,
          "before=$n_comps after=$(length(s.belief.components))")
    check("§2 the successor has no candidate (the dead item was consumed)",
          best_replacement(s, g2.id) === nothing)
    check("§2 replacement_value of the successor is exactly 0.0",
          replacement_value(s, g2.id) == 0.0)
    check("§2 the consumed gid is unregistered (it can never re-propose)",
          !haskey(s.grammars, 911))
    check("§2 the live lineage gB has no candidate either (its feature is referenced)",
          best_replacement(s, 912) === nothing && replacement_value(s, 912) == 0.0)
end

# ── §3  the OQ-4 soundness case: sub-1e-15 live components still block candidacy ──
let
    g = Grammar(Set([:x, :y, :z]), ProductionRule[], 921)
    p_heavy = Program(IfExpr(GTExpr(FeatureRef(:x), 0.5), ActionExpr(:food), ActionExpr(:enemy)), 3, 921)
    p_straggler = Program(IfExpr(GTExpr(FeatureRef(:y), 0.5), ActionExpr(:food), ActionExpr(:enemy)), 3, 921)
    # The straggler's normalised weight is e⁻⁴⁰ ≈ 4.2e-18 < 1e-15: below the frequency table's
    # support cut, yet a live state component (post-prune states legitimately hold such weights —
    # prune keeps RELATIVE mass > e⁻³⁰ while normalisation divides by the full component count).
    s = build_state([(g, Program[p_heavy, p_straggler])]; lw_override = [0.0, -40.0])

    # The defeating precondition: the table-based reference walk does NOT see the straggler,
    # so a table-gated check would wrongly call :y dead.
    ft = analyse_posterior_subtrees(s.all_programs, weights(s.belief);
                                    min_frequency = 0.01, min_complexity = 2)
    check("§3 precondition: the w > 1e-15 table walk misses the straggler's feature :y",
          !(:y in ft.referenced_features), "refs=$(ft.referenced_features)")

    # The sound group-local full-support walk sees it: :y is NOT a candidate; :z (dead for every
    # component) is.
    cands = replacement_candidates(s, 921)
    check("§3 :y is BLOCKED (a live sub-threshold component references it)",
          all(!(c.kind == :remove_feature && c.payload === :y) for c in cands),
          "cands=$cands")
    check("§3 :z (referenced by nothing) IS a candidate",
          any(c.kind == :remove_feature && c.payload === :z for c in cands))
    check("§3 removing :z is what best_replacement picks",
          best_replacement(s, 921).payload === :z)
end

# ── §4  score exactness: the realised Δ log-evidence, and its W_G → 1 degeneration ──
let
    s = build_state([(gA(), progsA()), (gB(), progsB())])
    condition_live!(s, raw_pre())
    cand = best_replacement(s, 911)
    W = group_mass(s, 911)
    check("§4 precondition: both groups hold real mass", 0.0 < W < 1.0, "W=$W")

    Δ = log(2.0) * cand.payoff_symbols       # payoff 1: Δ == the actual complexity difference (§1)
    oracle = log1p((exp(Δ) - 1.0) * W)       # credence-lint: allow — precedent:test-oracle — §4 hand-built score oracle
    check("§4 replacement_value == log1p((e^Δ − 1)·W_G) exactly (cost 0)",
          replacement_value(s, 911) == oracle,
          "value=$(replacement_value(s, 911)) oracle=$oracle")
    check("§4 compute_cost is netted exactly",
          replacement_value(s, 911; compute_cost = 0.25) == oracle - 0.25)
    check("§4 the score is strictly below the prior-only voc (the surrogate overstates by 1 − W_G)",
          replacement_value(s, 911) < net_voc(cand.payoff_symbols, 0.0))

    # Score = transition, measured: firing realises exactly the mass the score priced.
    g2 = replace_grammar_in_state!(s, cand)
    W2 = group_mass(s, g2.id)
    predicted = exp(Δ) * W / (1.0 + (expm1(Δ)) * W)   # credence-lint: allow — precedent:test-oracle — §4 post-fire mass oracle
    check("§4 post-fire group mass == the score's prediction ≤ 1e-12 (T-3.55, measured)",
          abs(W2 - predicted) <= 1e-12, "W2=$W2 predicted=$predicted")

    # W_G → 1: a single-group state degenerates to the prior-only voc, exact up to the mass
    # read's float summation (weights sum to 1 ± 1 ulp; observed 1.0000000000000002) — ≤ 1e-15,
    # not a loose tolerance. The voc identity itself is exact.
    s1 = build_state([(gA(), progsA())])
    check("§4 single-group (W_G = 1) score == net_voc(payoff, 0) == log 2 (≤ 1e-15)",
          abs(replacement_value(s1, 911) - net_voc(1, 0.0)) <= 1e-15 &&
          net_voc(1, 0.0) == log(2.0),
          "value=$(replacement_value(s1, 911))")
end

# ── §5  score/transition unity: one candidate function, deterministic ──
let
    s = build_state([(gA(), progsA()), (gB(), progsB())])
    check("§5 best_replacement is pure and deterministic (=== across calls)",
          best_replacement(s, 911) === best_replacement(s, 911))

    # Total order: a dead RULE (payoff 1 + expr_complexity(body) > 1) outranks a dead feature
    # (payoff 1); equal payoffs tiebreak lexicographically by name.
    dead_rule = ProductionRule(:DEADR, GTExpr(FeatureRef(:a), 0.7))
    g = Grammar(Set([:a, :b, :d]), [dead_rule], 941)
    progs = rehome(enumerate_programs(Grammar(Set([:a]), ProductionRule[], 942), DEPTH;
                                      action_space = AS), 941)
    s2 = build_state([(g, progs)])
    cands = replacement_candidates(s2, 941)
    check("§5 all three dead items are candidates (rule :DEADR, features :b and :d)",
          length(cands) == 3, "cands=$cands")
    best = best_replacement(s2, 941)
    check("§5 the dead rule wins (payoff $(1 + expr_complexity(dead_rule.body)) > 1)",
          best.kind == :remove_rule && best.payload isa ProductionRule &&
          best.payload.name === :DEADR &&
          best.payoff_symbols == 1 + expr_complexity(dead_rule.body))
    # Remove the rule; among the remaining equal-payoff features, :b < :d lexicographically.
    g2 = replace_grammar_in_state!(s2, best)
    best_feat = best_replacement(s2, g2.id)
    check("§5 equal payoffs tiebreak by name (:b before :d)",
          best_feat.kind == :remove_feature && best_feat.payload === :b)
end

# ── §6  registry hygiene: thresholds threaded, dedup proves replacement added nothing ──
let
    # Full enumeration of {a, b} with a REFINED :a grid (the Move-3 survival discipline), then
    # crush and prune every :b-referencing component — the organic way a feature goes dead.
    refined = Dict{Symbol, Vector{Float64}}(:a => [0.1, 0.3, 0.42, 0.5, 0.7, 0.9],
                                            :b => [0.1, 0.3, 0.5, 0.7, 0.9])
    g = Grammar(Set([:a, :b]), ProductionRule[], refined, 951)
    progs = enumerate_programs(g, DEPTH; action_space = AS)
    refs_b = [(:b in collect_feature_refs!(Set{Symbol}(), p.expr)) for p in progs]
    check("§6 precondition: the enumeration holds both :b-free and :b-referencing programs",
          any(refs_b) && !all(refs_b))
    lw = [complexity_logprior(g.complexity; λ = log(2)) +
          complexity_logprior(p.complexity; λ = log(2)) - (refs_b[i] ? 1.0e4 : 0.0)
          for (i, p) in enumerate(progs)]
    s = build_state([(g, progs)]; lw_override = lw)
    sync_prune!(s)
    check("§6 precondition: prune leaves only :b-free components",
          all(!( :b in collect_feature_refs!(Set{Symbol}(), p.expr)) for p in s.all_programs))

    n_before = length(s.belief.components)
    cand = best_replacement(s, 951)
    check("§6 the organically-dead feature is the candidate",
          cand !== nothing && cand.payload === :b)
    g2 = replace_grammar_in_state!(s, cand)

    check("§6 registry: consumed gid deleted, successor registered",
          !haskey(s.grammars, 951) && s.grammars[g2.id] === g2)
    check("§6 g′ drops the feature AND its grid; the refined :a grid survives",
          g2.feature_set == Set([:a]) && !haskey(g2.thresholds, :b) &&
          g2.thresholds[:a] == refined[:a])
    check("§6 every group component re-keyed to g′",
          all(m[1] == g2.id for m in s.metadata))

    # The dedup proof that replacement is pure re-description: enumerating g′ re-adds NOTHING —
    # the group already holds exactly g′'s language at this depth.
    n_added = add_programs_to_state!(s, g2, DEPTH;
                                     observations = ExploreObservation[], action_space = AS)
    check("§6 a later enumerate of g′ dedups to zero", n_added == 0, "n_added=$n_added")
    check("§6 component count unchanged through the whole §6 sequence",
          length(s.belief.components) == n_before)
end

println("="^64)
println("ALL CHECKS PASSED — replacement semantics")
println("="^64)
