# test_coherent_injection.jl — hypothesis addition commutes with conditioning (coherent-injection move).
#
# The constitutional pin for docs/exploration-budget/coherent-injection-design.md §1: the posterior
# over an enlarged hypothesis space given a window of evidence must not depend on WHEN the new
# hypotheses were injected. `MixturePrevision` normalises on construction, so the injection restores
# the cross-group constant via two ledgers (the replay's predictives + the buffer's residuals); the
# equality is exact up to floating-point summation order — pinned at 1e-12 (the deterministic-
# arithmetic tolerance), with Beta states and tags exactly ==.
#
# Sections:
#   §1  commutation: enumerate-union-then-condition == condition-then-inject-coherently
#       (log-weights ≤ 1e-12; Beta states, tags, metadata exactly ==).
#   §2  the enforcement surface: `observations` is REQUIRED (UndefKeywordError when omitted), and
#       an explicitly-empty window is NOT equivalent to the evidence window (ignorance regression).
#   §3  dedup and parallel-array discipline survive the replay path.
#
# Run: julia test/test_coherent_injection.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, Program, CompiledKernel, AgentState,
                add_programs_to_state!, ExploreObservation, program_space_observation_kernel,
                enumerate_programs, compile_kernel, complexity_logprior,
                TaggedBetaPrevision, BetaPrevision, Prevision, MixturePrevision,
                weights, condition, log_predictive, show_expr

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("coherent injection — addition commutes with conditioning")
println("="^64)

const AS = Symbol[:food, :enemy]
const DEPTH = 2

# The starting grammar and the grammar to be injected (a feature-added sibling — the
# :gw_add_feature shape; threshold refinement exercises the identical code path).
g1() = Grammar(Set([:a]), ProductionRule[], 911)
g2() = Grammar(Set([:a, :b]), ProductionRule[], 912)

# The raw evidence: discriminating observations on both features.
raw_window() = [
    (Dict(:a => 0.9, :b => 0.2), Set([:food])),
    (Dict(:a => 0.1, :b => 0.8), Set([:enemy])),
    (Dict(:a => 0.8, :b => 0.3), Set([:food])),
    (Dict(:a => 0.2, :b => 0.9), Set([:enemy])),
    (Dict(:a => 0.7, :b => 0.1), Set([:food])),
    (Dict(:a => 0.3, :b => 0.7), Set([:enemy])),
]

# Fresh state holding exactly g1's programs at DEPTH (the manual construction mirrors
# add_programs_to_state!'s arithmetic; asserted equal to it in test_program_space.jl).
function g1_state()
    g = g1()
    programs = enumerate_programs(g, DEPTH; action_space = AS)
    components = TaggedBetaPrevision[]
    lw = Float64[]
    meta = Tuple{Int, Int}[]
    cks = CompiledKernel[]
    progs = Program[]
    for (pi, p) in enumerate(programs)
        push!(components, TaggedBetaPrevision(pi, BetaPrevision(1.0, 1.0)))
        push!(lw, complexity_logprior(g.complexity; λ = log(2)) +
                  complexity_logprior(p.complexity; λ = log(2)))
        push!(meta, (g.id, pi))
        push!(cks, compile_kernel(p, g, pi))
        push!(progs, p)
    end
    AgentState(MixturePrevision(Prevision[components...], lw), meta, cks, progs,
               Dict{Int, Grammar}(g.id => g), DEPTH)
end

# Live conditioning exactly as the hosts do it: per obs, record the surprise
# (−log_predictive BEFORE conditioning — the coherence ledger), then one mixture condition.
# Returns the populated explore buffer.
function condition_live!(state::AgentState, raw)::Vector{ExploreObservation}
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

# ── §1  commutation ──
let
    raw = raw_window()

    # Path U (union-from-start): inject g2 at t=0 with an honestly-empty window, then condition.
    su = g1_state()
    n_u = add_programs_to_state!(su, g2(), DEPTH;
                                 observations = ExploreObservation[], action_space = AS)
    condition_live!(su, raw)

    # Path I (inject-at-n): condition first, then inject g2 coherently against the window.
    si = g1_state()
    buf = condition_live!(si, raw)
    n_i = add_programs_to_state!(si, g2(), DEPTH; observations = buf, action_space = AS)

    check("§1 both paths inject the same number of programs ($(n_u))", n_u == n_i && n_u > 0,
          "union added $n_u, injection added $n_i")
    check("§1 component counts equal",
          length(su.belief.components) == length(si.belief.components))
    check("§1 metadata identical (same programs, same order)", su.metadata == si.metadata)

    # The constitutional equality: Bayes does not care when you thought of the hypothesis.
    # Exact up to float summation order (different logsumexp groupings) — 1e-12, not 1e-6.
    maxdiff = maximum(abs.(su.belief.log_weights .- si.belief.log_weights))
    check("§1 log-weights equal ≤ 1e-12 (commutation)", maxdiff <= 1e-12,
          "max abs diff = $maxdiff")

    # Beta states and tags componentwise EXACTLY equal (integer-count updates, same order).
    betas_ok = all(eachindex(su.belief.components)) do i
        cu = su.belief.components[i]; ci = si.belief.components[i]
        cu.tag == ci.tag && cu.beta.alpha == ci.beta.alpha && cu.beta.beta == ci.beta.beta
    end
    check("§1 Beta states and tags == componentwise", betas_ok)
    wdiff = maximum(abs.(weights(su.belief) .- weights(si.belief)))
    check("§1 normalised weights equal ≤ 1e-12", wdiff <= 1e-12, "max abs diff = $wdiff")
    check("§1 tags are positional (re-tag discipline)",
          all(si.belief.components[i].tag == i for i in eachindex(si.belief.components)))
end

# ── §2  the enforcement surface ──
let
    raw = raw_window()

    # The window is REQUIRED: omitting it is a signature error, not a silent ignorant injection.
    s = g1_state()
    threw = try
        add_programs_to_state!(s, g2(), DEPTH; action_space = AS)
        false
    catch e
        e isa UndefKeywordError
    end
    check("§2 omitting `observations` throws UndefKeywordError (required, not defaulted)", threw)

    # Ignorance regression: an explicitly-empty window is NOT the evidence window — the evidence
    # actually flows into the injected components' weights.
    s_informed = g1_state(); buf = condition_live!(s_informed, raw)
    s_ignorant = g1_state(); condition_live!(s_ignorant, raw)
    add_programs_to_state!(s_informed, g2(), DEPTH; observations = buf, action_space = AS)
    add_programs_to_state!(s_ignorant, g2(), DEPTH;
                           observations = ExploreObservation[], action_space = AS)
    check("§2 informed injection ≠ ignorant injection on log-weights",
          s_informed.belief.log_weights != s_ignorant.belief.log_weights)
end

# ── §3  dedup + parallel arrays through the replay path ──
let
    s = g1_state()
    buf = condition_live!(s, raw_window())
    n1 = add_programs_to_state!(s, g2(), DEPTH; observations = buf, action_space = AS)
    n_comps = length(s.belief.components)

    # Re-adding the same grammar with the same window dedups to zero and mutates nothing.
    n2 = add_programs_to_state!(s, g2(), DEPTH; observations = buf, action_space = AS)
    check("§3 re-injection dedups to zero", n2 == 0, "added $n2")
    check("§3 parallel arrays in lock-step",
          length(s.belief.components) == n_comps &&
          length(s.metadata) == n_comps &&
          length(s.compiled_kernels) == n_comps &&
          length(s.all_programs) == n_comps)
end

println("="^64)
println("ALL CHECKS PASSED — coherent injection")
println("="^64)
