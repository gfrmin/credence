# test_growth_returns.jl — belief-derived meta-action valuation (belief-derived-valuation move).
#
# The engine half of docs/exploration-budget/belief-derived-valuation-design.md:
#   §1  the returns model: Gamma(2,1) over the Exponential yield rate, conjugate updates exact
#       (tightest invariant — α/β equality, not "close enough"), escape_score = E[1/λ] − cost
#       via the declared ExponentialMean TestFunction (β/(α−1), the GeometricTail pattern).
#   §2  the yield observable: injection_yield_nats = −log(1 − posterior mass of the injected
#       tags) — exact against a manual oracle; exactly 0.0 on a dedup no-op.
#   §3  ExponentialMean guards and closed form.
#   §4  zero-yield observations condition cleanly (the relaxed r ≥ 0 Exponential guard —
#       density λe^{−λ·0} = λ is well-defined; posterior Gamma(α+1, β+0) IS Bayes).
#   §5  growth_value: the horizon-completed Δ log-evidence functional (stdlib; the grid host's
#       growth seam no longer calls it — winners-curse §8.2 — but the functional's arithmetic
#       stays pinned for consumers with an honest persistent-rate claim); defaults reduce
#       BIT-EXACTLY to net_value(fit + prior, cost) (behaviour-preserving pin).
#   §6  ONE SCORE FORM (T-3.53, winners-curse §8.2): escape ops score net_value(E[next yield],
#       cost) — learned; growth ops score net_value(computed yield, cost) — exact. The ranked
#       value and the applied edit share one core (positive value ⟺ grammar applied).
#
# Run: julia test/test_growth_returns.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: GrowthReturns, observe_yield!, escape_score, injection_yield_nats,
                ExponentialMean, growth_value, net_value,
                GammaPrevision, expect, condition, weights, probability, TagSet, Interval,
                Grammar, ProductionRule, Program, CompiledKernel, AgentState,
                add_programs_to_state!, ExploreObservation, program_space_observation_kernel,
                enumerate_programs, compile_kernel, complexity_logprior,
                TaggedBetaPrevision, BetaPrevision, Prevision, MixturePrevision, log_predictive,
                Program, CompiledKernel, AgentState, MixturePrevision, TaggedBetaPrevision,
                BetaPrevision, Prevision, compile_kernel, complexity_logprior,
                program_space_observation_kernel, condition, log_predictive,
                enumerate_programs, threshold_growth_proposal, feature_growth_proposal

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("belief-derived valuation — learned returns + horizon-completed growth")
println("="^64)

# ── §1  the returns model ──
let
    gr = GrowthReturns([:gw_enumerate_more, :gw_deepen])
    # Prior Gamma(2,1): E[1/λ] = β/(α−1) = 1/1 = 1 nat of bounded initial optimism.
    check("§1 prior expected yield == 1.0 nat exactly",
          escape_score(gr, :gw_enumerate_more, true) == 1.0)
    check("§1 declared compute price subtracts",
          escape_score(gr, :gw_enumerate_more, true; compute_cost = log(2.0)) == 1.0 - log(2.0))

    # One zero-yield observation: Gamma(3,1) → E = 1/2. Evidence decays the optimism —
    # the property the entropy score never had.
    observe_yield!(gr, :gw_enumerate_more, true, 0.0)
    check("§1 one zero yield halves the expectation (Gamma(3,1) → 0.5)",
          escape_score(gr, :gw_enumerate_more, true) == 0.5)

    # Two more zeros: Gamma(5,1) → 1/4; below the default price → loses to the do-nothing floor.
    observe_yield!(gr, :gw_enumerate_more, true, 0.0)
    observe_yield!(gr, :gw_enumerate_more, true, 0.0)
    check("§1 three zero yields drop the score below the log2 price",
          escape_score(gr, :gw_enumerate_more, true; compute_cost = log(2.0)) < 0.0)

    # A real yield sustains it: fresh cell, y = 2.0 nats → Gamma(3,3) → E = 3/2.
    observe_yield!(gr, :gw_deepen, true, 2.0)
    check("§1 a 2-nat yield raises deepen's cell to 3/2 exactly",
          escape_score(gr, :gw_deepen, true) == 1.5)

    # Context cells are independent: deepen's (changed=false) cell is untouched at prior.
    check("§1 context cells independent (unchanged cell still at prior)",
          escape_score(gr, :gw_deepen, false) == 1.0)
end

# ── §2  the yield observable ──
let
    g1 = Grammar(Set([:a]), ProductionRule[], 921)
    g2 = Grammar(Set([:a, :b]), ProductionRule[], 922)
    as = Symbol[:food, :enemy]
    programs = enumerate_programs(g1, 2; action_space = as)
    comps = TaggedBetaPrevision[]; lw = Float64[]; meta = Tuple{Int, Int}[]
    cks = CompiledKernel[]; progs = Program[]
    for (pi, p) in enumerate(programs)
        push!(comps, TaggedBetaPrevision(pi, BetaPrevision(1.0, 1.0)))
        push!(lw, complexity_logprior(g1.complexity; λ = log(2)) +
                  complexity_logprior(p.complexity; λ = log(2)))
        push!(meta, (g1.id, pi)); push!(cks, compile_kernel(p, g1, pi)); push!(progs, p)
    end
    state = AgentState(MixturePrevision(Prevision[comps...], lw), meta, cks, progs,
                       Dict{Int, Grammar}(g1.id => g1), 2)
    buf = ExploreObservation[]
    for (feats, correct) in [(Dict(:a => 0.9, :b => 0.2), Set([:food])),
                             (Dict(:a => 0.1, :b => 0.8), Set([:enemy]))]
        k = program_space_observation_kernel(state.compiled_kernels, feats,
                                             Dict{Symbol, Any}(), correct)
        push!(buf, ExploreObservation(feats, Dict{Symbol, Any}(),
                                      correct, -log_predictive(state.belief, k, 1.0)))
        state.belief = condition(state.belief, k, 1.0)
    end

    state.grammars[g2.id] = g2   # hosts register the grammar before injecting
    n_added = add_programs_to_state!(state, g2, 2; observations = buf, action_space = as)
    y = injection_yield_nats(state, n_added)
    # credence-lint: allow — precedent:test-oracle — manual evidence-relative yield oracle
    n = length(state.belief.components)
    mass = probability(state.belief, TagSet(Interval(0.0, 1.0), Set((n - n_added + 1):n)))
    s0 = sum(exp(complexity_logprior(g2.complexity; λ = log(2)) +
                 complexity_logprior(state.all_programs[i].complexity; λ = log(2)))
             for i in (n - n_added + 1):n)
    m0 = s0 / (1.0 + s0)
    check("§2 yield == max(0, log((1−m₀)/(1−m))), exact against the oracle",
          y == max(0.0, log(1.0 - m0) - log(1.0 - mass)), "y=$y mass=$mass m0=$m0")

    # The prior-counterfactual correction (the §5-Q1 wedge): an EMPTY-window injection takes
    # prior mass by count alone — its EVIDENCE yield must be ~0, not nats of "attention".
    comps0 = TaggedBetaPrevision[TaggedBetaPrevision(pi, BetaPrevision(1.0, 1.0))
                                 for pi in eachindex(programs)]
    lw0 = Float64[complexity_logprior(g1.complexity; λ = log(2)) +
                  complexity_logprior(p.complexity; λ = log(2)) for p in programs]
    state0 = AgentState(MixturePrevision(Prevision[comps0...], lw0),
                        [(g1.id, pi) for pi in eachindex(programs)],
                        [compile_kernel(p, g1, pi) for (pi, p) in enumerate(programs)],
                        Program[programs...], Dict{Int, Grammar}(g1.id => g1), 2)
    g2b = Grammar(Set([:a, :b]), ProductionRule[], 923)
    state0.grammars[g2b.id] = g2b
    n0 = add_programs_to_state!(state0, g2b, 2;
                                observations = ExploreObservation[], action_space = as)
    y0 = injection_yield_nats(state0, n0)
    check("§2 empty-window injection yields ~0 (prior mass is not evidence)", y0 <= 1e-12,
          "y0=$y0")

    # Dedup no-op: same grammar again → n_added == 0 → yield exactly 0.0.
    n2 = add_programs_to_state!(state, g2, 2; observations = buf, action_space = as)
    check("§2 dedup no-op yields exactly 0.0",
          n2 == 0 && injection_yield_nats(state, n2) == 0.0)
end

# ── §3  ExponentialMean ──
let
    check("§3 closed form: expect(Gamma(3,2), ExponentialMean()) == 1.0 exactly",
          expect(GammaPrevision(3.0, 2.0), ExponentialMean()) == 1.0)
    threw = try
        expect(GammaPrevision(1.0, 2.0), ExponentialMean()); false
    catch e
        e isa ErrorException
    end
    check("§3 α ≤ 1 divergence guard errors (the GeometricTail pattern)", threw)
end

# ── §4  zero-yield conditioning (relaxed r ≥ 0 guard) ──
let
    gr = GrowthReturns([:op])
    observe_yield!(gr, :op, false, 0.0)   # must not throw
    check("§4 zero observation conditions to Gamma(3,1) (E == 0.5)",
          escape_score(gr, :op, false) == 0.5)
end

# ── §5  growth_value ──
let
    # Behaviour-preserving default: horizon == n_buf, plateau == 1 reduces BIT-EXACTLY to
    # net_value(fit + prior, cost) — the pre-move valuation.
    for (fit, n, prior, cost) in [(3.7, 30, 0.0, 0.0), (0.42, 7, -log(2.0), 0.1),
                                  (13.9, 30, -log(2.0), 0.0)]
        check("§5 defaults reduce to net_value (fit=$fit)",
              growth_value(fit, n, 1.0, Float64(n); prior_term = prior, compute_cost = cost) ==
              net_value(fit + prior, cost))
    end
    # Horizon completion: plateau · fit · (H/n) + prior − cost.
    check("§5 horizon scaling exact",
          growth_value(2.0, 10, 0.5, 40.0) == 0.5 * 2.0 * 4.0)
    check("§5 one-time prior term is NOT horizon-multiplied",
          growth_value(2.0, 10, 1.0, 40.0; prior_term = -log(2.0)) == 2.0 * 4.0 - log(2.0))
    check("§5 empty window is worthless", growth_value(0.0, 0, 1.0, 100.0) == 0.0)
end

# ── §6  one score form: escape learns the yield, growth computes it (T-3.53, §8.2) ──
let
    as = Symbol[:food, :enemy]
    function mk_state(g)
        progs = enumerate_programs(g, 2; action_space = as)
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
        for (features, correct) in raw
            k = program_space_observation_kernel(state.compiled_kernels, features,
                                                 Dict{Symbol, Any}(), correct)
            res = -log_predictive(state.belief, k, 1.0)
            push!(buf, ExploreObservation(features, Dict{Symbol, Any}(), correct, res))
            state.belief = condition(state.belief, k, 1.0)
        end
        buf
    end

    # A refinement inexpressible at the default grid: the boundary sits at :a ≈ 0.62.
    g = Grammar(Set([:a]), ProductionRule[], 923)
    raw = [(Dict(:a => 0.65), Set([:food])), (Dict(:a => 0.60), Set([:enemy])),
           (Dict(:a => 0.66), Set([:food])), (Dict(:a => 0.58), Set([:enemy])),
           (Dict(:a => 0.70), Set([:food])), (Dict(:a => 0.61), Set([:enemy]))]
    s = mk_state(g)
    buf = live!(s, raw)
    prop = threshold_growth_proposal(s, g, buf; action_space = as)
    check("§6 the growth proposal computes a non-negative yield (the same observable escape learns)",
          prop !== nothing && prop.yield_nats >= 0.0)

    # ONE SCORE FORM: both tiers are net_value(yield-quantity, declared price). The escape
    # side's yield-quantity is the posterior-predictive expectation of the SAME observable the
    # growth side computes exactly by virtually firing.
    gr = GrowthReturns(Symbol[:op])
    observe_yield!(gr, :op, true, prop.yield_nats)
    # credence-lint: allow — precedent:test-oracle — the shared net_value form, recomputed
    check("§6 escape tier: score == net_value(E[next yield], cost) — the learned fidelity",
          escape_score(gr, :op, true; compute_cost = 0.25) ==
          net_value(expect(gr.cells[(:op, true)], ExponentialMean()), 0.25))
    check("§6 growth tier: score == net_value(computed yield, cost) — the exact fidelity",
          net_value(prop.yield_nats, 0.25) == prop.yield_nats - 0.25)

    # Score/edit pairing survives the re-founding: the score and the transition project ONE
    # GrowthProposal (identity-pinned end-to-end in test_virtual_injection.jl §1); here the
    # engine-level floor — an evidence-free union prices at the clamp floor, exactly 0.
    g_flat = Grammar(Set([:z]), ProductionRule[], 924)
    raw_flat = [(Dict(:z => 0.5), Set([:food])) for _ in 1:4]
    s_flat = mk_state(g_flat)
    buf_flat = live!(s_flat, raw_flat)
    p_flat = threshold_growth_proposal(s_flat, g_flat, buf_flat; action_space = as)
    check("§6 no observed spread ⇒ no threshold candidates ⇒ no proposal (structural no-op)",
          p_flat === nothing)
end

println("="^64)
println("ALL CHECKS PASSED — belief-derived valuation")
println("="^64)
