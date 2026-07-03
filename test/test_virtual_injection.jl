# test_virtual_injection.jl — the lookahead as a virtual injection (winners-curse design,
# revs 3–4; the §8 amendment's yield rule).
#
# The constitutional pin for docs/exploration-budget/winners-curse-design.md §1: a growth op is
# scored by VIRTUALLY PERFORMING its transition — copy the state, coherently inject ALL candidates'
# deduped programs at their two-part complexity priors, condition on the window (the #187 code path,
# verbatim), and read the score off the scratch through existing canalised ops. Firing ADOPTS the
# scratch, so score ≡ transition holds as an identity (T-3.55), not as a shared candidate function.
#
# Sections (design §2):
#   §1  score/transition identity through the host seam — the belief the score priced IS the belief
#       the agent holds after the fire (`===`); Q4 no-double-charge (GW_FEATURE_PRIOR_TERM retired;
#       the score is exactly net_value(yield, op_compute_cost) — no prior term, no plateau, no ×H).
#   §2  the seed-6 regression — a 4-event chance window with K = 5 feature candidates WAITS under
#       the yield rule (§8.2: realised evidence vs the declared price), while the retired
#       per-candidate argmax score (counter-oracle, hand-replayed) fires on the same fixture.
#   §3  the informed fire — a genuinely separating feature over 30 events: the union posterior
#       concentrates on the newcomers, the yield is decisive, the op fires (from n = 8).
#   §4  multiplicity-by-mixture — K chance candidates' union earns no evidence yield (the mixture's
#       own normalisation is the multiplicity correction); dedup keeps the union far smaller than
#       the candidates' summed enumerations; re-injection is idempotent.
#   §5  commutation inheritance — the union injection is one coherent injection, so #187's §1
#       equality holds for the VECTOR method: union-from-start == condition-then-inject-union
#       (log-weights ≤ 1e-12; Beta states and tags exactly ==).
#   §6  hygiene — a spurious adopted union's programs collapse under informative evidence and are
#       pruned; the junk grammar's features go dead and become #193 replacement candidates (the
#       self-healing loop, end-to-end).
#
# Run: julia test/test_virtual_injection.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, Program, CompiledKernel, AgentState,
                add_programs_to_state!, ExploreObservation, program_space_observation_kernel,
                enumerate_programs, compile_kernel, complexity_logprior,
                TaggedBetaPrevision, BetaPrevision, Prevision, MixturePrevision,
                weights, condition, log_predictive, growth_value, net_value,
                GrowthProposal, threshold_growth_proposal, feature_growth_proposal,
                copy_agent_state, adopt!, injection_yield_nats,
                replacement_candidates, sync_prune!,
                GrowthReturns

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("virtual injection — the machinery applied to itself")
println("="^64)

const AS = Symbol[:food, :enemy]
const DEPTH = 2

# ── fixtures ──
# The incumbent grammar sees only :a; the world's signal (when there is one) lives in :b.
# :a is held CONSTANT in every window so the threshold class generates no candidates
# (midpoints need ≥ 2 distinct observed values) — feature discovery is the op under test.
g_inc() = Grammar(Set([:a]), ProductionRule[], 921)

# Fresh state holding exactly the incumbent grammar's programs at DEPTH (the manual
# construction mirrors add_programs_to_state!'s arithmetic — test_coherent_injection.jl idiom).
function inc_state(g::Grammar = g_inc())
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

# The chance window (design §4 step 1): 4 events, labels split 2/2, :b HAPPENS to separate
# them perfectly (with ~5 candidate features and 4 binary events, some candidate chance-fits —
# the winner's-curse seed shape). :c/:d/:e/:f carry no signal.
raw_chance() = [
    (Dict(:a => 0.5, :b => 0.9, :c => 0.4, :d => 0.6, :e => 0.5, :f => 0.3), Set([:food])),
    (Dict(:a => 0.5, :b => 0.1, :c => 0.5, :d => 0.4, :e => 0.6, :f => 0.7), Set([:enemy])),
    (Dict(:a => 0.5, :b => 0.8, :c => 0.6, :d => 0.5, :e => 0.4, :f => 0.6), Set([:food])),
    (Dict(:a => 0.5, :b => 0.2, :c => 0.3, :d => 0.7, :e => 0.5, :f => 0.4), Set([:enemy])),
]

# The informed window (design §4 step 3): 30 events, :b GENUINELY separates (regime 2's
# analogue) — high :b ⇒ food, low :b ⇒ enemy, everything else noise-like but deterministic.
function raw_informed(n::Int = 30)
    raw = Tuple{Dict{Symbol, Float64}, Set{Symbol}}[]
    for i in 1:n
        food = isodd(i)
        b = food ? 0.7 + 0.02 * (i % 5) : 0.1 + 0.02 * (i % 5)
        push!(raw, (Dict(:a => 0.5, :b => b,
                         :c => 0.3 + 0.05 * (i % 7), :d => 0.2 + 0.06 * (i % 5),
                         :e => 0.4 + 0.03 * (i % 6), :f => 0.5 + 0.04 * (i % 4)),
                    Set([food ? :food : :enemy])))
    end
    raw
end

const ALL_FEATURES = Set([:a, :b, :c, :d, :e, :f])

# ── §2  the seed-6 regression (engine level; the design's central falsifiable claim) ──
let
    s = inc_state()
    buf = condition_live!(s, raw_chance())
    g = s.grammars[g_inc().id]

    prop = feature_growth_proposal(s, g, buf, ALL_FEATURES; action_space = AS)
    check("§2 the chance window yields a proposal (candidates exist)", prop !== nothing)

    # The yield rule (design §8.2): fire when the realised evidence — the union-over-incumbent
    # window Bayes factor — clears the declared compute price. No plateau, no horizon.
    v_new = net_value(prop.yield_nats, log(2.0))

    # The retired score's counter-oracle: the per-candidate argmax over fresh-belief marginal
    # log-loss replays, horizon-multiplied. Hand-replayed through the engine's own canalised
    # ops (log_predictive/condition) — the same arithmetic _grammar_marginal_log_loss ran.
    # credence-lint: allow — precedent:test-oracle — counter-oracle for the retired mechanism
    function mll_fresh(g2::Grammar)
        progs = enumerate_programs(g2, DEPTH; action_space = AS)
        isempty(progs) && return Inf
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
    baseline = mll_fresh(g)
    fit_old = maximum(baseline - mll_fresh(Credence._add_feature(g, f))
                      for f in sort(collect(setdiff(ALL_FEATURES, g.feature_set))))
    # The retired coordinates: early-episode plateau ≈ the regime prior (0.5), H = 195.
    v_old = growth_value(fit_old, length(buf), 0.5, 195.0;
                         prior_term = complexity_logprior(1; λ = log(2.0)))

    check("§2 the retired score FIRES on the chance window (the curse; v_old = $(round(v_old, digits = 2)) nats)",
          v_old > 0.0, "v_old = $v_old")
    check("§2 the virtual-injection score WAITS (v_new = $(round(v_new, digits = 3)) ≤ 0 at H = 195)",
          v_new <= 0.0, "v_new = $v_new (yield = $(prop.yield_nats))")
    check("§2 the brake is an order of magnitude (v_new < v_old / 10)",
          v_new < v_old / 10.0, "v_new = $v_new, v_old = $v_old")
    check("§2 the union's evidence yield is near-noise (< 1 nat; computed exactly, not learned)",
          prop.yield_nats < 1.0, "yield = $(prop.yield_nats)")
end

# ── §3  the informed fire (the asymptotic identity — genuine signal still fires) ──
let
    s = inc_state()
    buf = condition_live!(s, raw_informed())
    g = s.grammars[g_inc().id]

    prop = feature_growth_proposal(s, g, buf, ALL_FEATURES; action_space = AS)
    check("§3 the informed window yields a proposal", prop !== nothing)

    v = net_value(prop.yield_nats, log(2.0))
    check("§3 the informed fire clears the floor (v = $(round(v, digits = 2)) > 0)",
          v > 0.0, "v = $v (yield = $(prop.yield_nats))")

    # Genuine signal needs no long window: the same fixture already clears the price at n = 8
    # (the wait-arm re-accumulates fast — §8.3's regeneration assumption for signals that
    # matter).
    s8 = inc_state()
    buf8 = condition_live!(s8, raw_informed(8))
    prop8 = feature_growth_proposal(s8, s8.grammars[g_inc().id], buf8, ALL_FEATURES;
                                    action_space = AS)
    check("§3 the informed fire lands from n = 8 (yield = $(round(prop8.yield_nats, digits = 2)))",
          net_value(prop8.yield_nats, log(2.0)) > 0.0)
    check("§3 the union posterior concentrates on the newcomers (P_newcomers > 0.5)",
          prop.p_newcomers > 0.5, "p_newcomers = $(prop.p_newcomers)")
    check("§3 the computed yield is decisive (> 1 nat)", prop.yield_nats > 1.0,
          "yield = $(prop.yield_nats)")
end

# ── §4  multiplicity-by-mixture + dedup ──
let
    # Chance window again; the union of ALL candidates vs a single candidate, injected through
    # the vector method on two copies of the same conditioned state.
    s1 = inc_state()
    buf = condition_live!(s1, raw_chance())
    sK = copy_agent_state(s1)
    g = s1.grammars[g_inc().id]

    cands = [Credence._add_feature(g, f) for f in sort(collect(setdiff(ALL_FEATURES, g.feature_set)))]

    for c in [cands[1]]
        s1.grammars[c.id] = c
    end
    n1 = add_programs_to_state!(s1, [cands[1]], DEPTH; observations = buf, action_space = AS)
    y1 = injection_yield_nats(s1, n1)

    for c in cands
        sK.grammars[c.id] = c
    end
    nK = add_programs_to_state!(sK, cands, DEPTH; observations = buf, action_space = AS)
    yK = injection_yield_nats(sK, nK)

    check("§4 both injections add programs (n1 = $n1, nK = $nK)", n1 > 0 && nK > n1)

    # Dedup: the union is far smaller than the candidates' summed enumerations — the shared
    # base language (the incumbent's programs re-derivable under every candidate) is never
    # re-injected.
    total_enum = sum(length(enumerate_programs(c, DEPTH; action_space = AS)) for c in cands)
    check("§4 global dedup keeps the union far below Σ|G_c| ($nK < $(total_enum) / 2)",
          nK < total_enum / 2, "nK = $nK, Σ = $total_enum")

    # Idempotence: re-injecting any single candidate after the union dedups to zero.
    n_again = add_programs_to_state!(sK, [cands[1]], DEPTH; observations = buf, action_space = AS)
    check("§4 re-injection after the union dedups to zero", n_again == 0, "added $n_again")

    # Multiplicity: K chance candidates earn no evidence the mixture's own normalisation
    # doesn't price — the yield stays near-noise regardless of K (no selection lift; the
    # max-over-K disappeared as a mechanism).
    check("§4 no selection lift: yield(K) ≈ yield(1) ≈ 0 (y1 = $(round(y1, digits = 3)), yK = $(round(yK, digits = 3)))",
          yK < 1.0 && yK <= y1 + 0.5, "y1 = $y1, yK = $yK")
end

# ── §5  commutation inheritance (the vector method is ONE coherent injection) ──
let
    raw = raw_informed(6)
    g = g_inc()
    c1 = Credence._add_feature(g, :b)
    c2 = Credence._add_feature(g, :c)

    # Path U (union-from-start): inject both candidates at t = 0 against an honestly-empty
    # window, then condition.
    su = inc_state()
    for c in (c1, c2)
        su.grammars[c.id] = c
    end
    n_u = add_programs_to_state!(su, [c1, c2], DEPTH;
                                 observations = ExploreObservation[], action_space = AS)
    condition_live!(su, raw)

    # Path I (inject-at-n): condition first, then union-inject coherently against the window.
    si = inc_state()
    buf = condition_live!(si, raw)
    for c in (c1, c2)
        si.grammars[c.id] = c
    end
    n_i = add_programs_to_state!(si, [c1, c2], DEPTH; observations = buf, action_space = AS)

    check("§5 both paths inject the same number of programs ($(n_u))", n_u == n_i && n_u > 0,
          "union added $n_u, injection added $n_i")
    check("§5 metadata identical (same programs, same order)", su.metadata == si.metadata)
    maxdiff = maximum(abs.(su.belief.log_weights .- si.belief.log_weights))
    check("§5 log-weights equal ≤ 1e-12 (commutation, inherited from #187)", maxdiff <= 1e-12,
          "max abs diff = $maxdiff")
    betas_ok = all(eachindex(su.belief.components)) do i
        cu = su.belief.components[i]; ci = si.belief.components[i]
        cu.tag == ci.tag && cu.beta.alpha == ci.beta.alpha && cu.beta.beta == ci.beta.beta
    end
    check("§5 Beta states and tags == componentwise", betas_ok)
    check("§5 tags are positional (re-tag discipline)",
          all(si.belief.components[i].tag == i for i in eachindex(si.belief.components)))
end

# ── §6  hygiene — the self-healing loop (spurious union → collapse → #193 candidates) ──
let
    s = inc_state()
    buf = condition_live!(s, raw_chance())
    g = s.grammars[g_inc().id]

    # Force-adopt the chance union (what a score-blind baseline does): the junk rides in at
    # prior mass. :b's chance-fitters hold a little more; :c..:f hold ≈ prior.
    prop = feature_growth_proposal(s, g, buf, ALL_FEATURES; action_space = AS)
    junk_gids = sort(collect(setdiff(keys(prop.scratch.grammars), keys(s.grammars))))
    adopt!(s, prop.scratch)
    check("§6 adoption installs the union (grammars registered, components appended)",
          all(haskey(s.grammars, gid) for gid in junk_gids) && length(junk_gids) >= 2)

    # The world keeps speaking :a-and-:b-silence: 40 informative events where NO feature the
    # junk grammars lean on predicts (labels follow :b's genuine rule — the :c..:f programs
    # mispredict half the time and their mass collapses).
    condition_live!(s, raw_informed(40))
    sync_prune!(s; threshold = -30.0)

    # The :c..:f-only grammars' programs die; their features go dead; #193's replacement
    # machinery sees them (the full-support walk finds no surviving reference).
    dead_feature_offered = false
    for gid in junk_gids
        haskey(s.grammars, gid) || continue
        feats = s.grammars[gid].feature_set
        wants = setdiff(feats, Set([:a, :b]))
        isempty(wants) && continue
        cands = replacement_candidates(s, gid)
        if any(c -> c.kind === :remove_feature && (c.payload in wants), cands)
            dead_feature_offered = true
        end
    end
    check("§6 a spurious union's dead feature becomes a #193 replacement candidate",
          dead_feature_offered)
end

# ── §1  score/transition identity through the host seam (runs last: includes the host) ──
include(joinpath(@__DIR__, "..", "apps", "julia", "grid_world", "host.jl"))
let
    # Q4 structural pin: the explicit feature prior term is RETIRED — the Occam charge lives
    # inside the mixture (each newcomer's own complexity prior), so charging it again at the
    # score seam would double-count.
    check("§1 GW_FEATURE_PRIOR_TERM is retired (Q4 no-double-charge)",
          !(@isdefined GW_FEATURE_PRIOR_TERM))

    # Host-native fixture: the host's feature-discovery candidates come from ALL_GW_FEATURES
    # (grid-world names), so the informative signal must live in one of those (:speed here —
    # the seed-6 story's actual feature); the incumbent sees only :red (held constant so the
    # threshold class stays silent and the gate is open).
    g_gw = Grammar(Set([:red]), ProductionRule[], 931)
    raw_gw = [(Dict(:red => 0.5, :green => 0.3 + 0.05 * (i % 7), :blue => 0.2 + 0.06 * (i % 5),
                    :x_norm => 0.4, :y_norm => 0.6,
                    :speed => isodd(i) ? 0.7 + 0.02 * (i % 5) : 0.1 + 0.02 * (i % 5),
                    :wall_dist => 0.5, :agent_dist => 0.5),
               Set([isodd(i) ? :food : :enemy])) for i in 1:30]
    s = inc_state(g_gw)
    buf = condition_live!(s, raw_gw)
    returns = GrowthReturns(Symbol[:gw_enumerate_more, :gw_deepen])
    changed = Dict{Symbol, Bool}(:gw_enumerate_more => true, :gw_deepen => true)
    cache = Dict{Tuple{Symbol, Int, Int, Int, Int}, Union{Nothing, GrowthProposal}}()

    scored = score_gw_meta_actions(s, buf, returns, changed;
                                   op_compute_cost = log(2.0),
                                   growth_cache = cache, cache_epoch = 0)

    # The cached proposal is the one the score priced.
    key = (:add_feature, 0, g_gw.id, length(buf), DEPTH)
    check("§1 the score populated the proposal cache", haskey(cache, key) && cache[key] !== nothing)
    prop = cache[key]

    # No-double-charge, arithmetically: the score is exactly growth_value(fit, n, plateau, H;
    # compute_cost) — no prior term.
    # credence-lint: allow — precedent:test-oracle — independent recomputation of the score seam
    expected = net_value(prop.yield_nats, log(2.0))
    check("§1 score == net_value(yield, op_compute_cost), no prior term, no plateau, no ×H",
          scored[:gw_add_feature] == expected,
          "scored = $(scored[:gw_add_feature]), expected = $expected")
    check("§1 the informed fixture fires through the seam", scored[:gw_add_feature] > 0.0)

    # Execute with the same cache: the transition ADOPTS the scored scratch — identity, not
    # equality (T-3.55 by construction).
    result = execute_gw_meta_action!(s, :gw_add_feature, buf;
                                     growth_cache = cache, cache_epoch = 0)
    check("§1 the fire reports the union's injection count", result.n_added == prop.n_added)
    check("§1 score ≡ transition as IDENTITY: the adopted belief IS the scored scratch's (===)",
          s.belief === prop.scratch.belief)
    check("§1 parallel arrays adopted by identity too",
          s.metadata === prop.scratch.metadata &&
          s.compiled_kernels === prop.scratch.compiled_kernels &&
          s.all_programs === prop.scratch.all_programs)
end

println("="^64)
println("ALL CHECKS PASSED — virtual injection")
println("="^64)
