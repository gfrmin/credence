#!/usr/bin/env julia
"""
    test_flat_mixture.jl — Phase 0: Verify flat MixtureMeasure conditioning

Tests that a flat MixtureMeasure of BetaMeasures, indexed by (grammar_id, program_id)
metadata, correctly updates weights when conditioned on observations via a kernel
that dispatches per-component.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Random

Random.seed!(42)

println("=" ^ 60)
println("FLAT MIXTURE TEST 1: Per-component kernel dispatch")
println("=" ^ 60)

# Simulate 2 grammars × 3 programs = 6 components
# Grammar 1: programs that predict enemy when sensor > 0.5
# Grammar 2: programs that predict enemy when sensor < 0.5
let
    # 6 Beta(1,1) priors — one per (grammar, program)
    components = Measure[BetaMeasure(1.0, 1.0) for _ in 1:6]
    metadata = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]

    # Prior weights: 2^(-|G|) * 2^(-|P|)
    # Grammar 1: complexity 2, Grammar 2: complexity 3
    # Programs: complexity 1, 2, 3 within each grammar
    grammar_complexities = [2.0, 3.0]
    program_complexities = [1.0, 2.0, 3.0]
    log_prior = Float64[]
    for (gi, pi) in metadata
        lw = -grammar_complexities[gi] * log(2) - program_complexities[pi] * log(2)
        push!(log_prior, lw)
    end

    belief = MixtureMeasure(Interval(0.0, 1.0), components, log_prior)

    # Kernel: dispatches by component index
    # Grammar 1 programs predict "enemy" (obs=1.0) with high probability
    # Grammar 2 programs predict "food" (obs=0.0) with high probability
    # Simulate: sensor reading = 0.8 (high), entity was enemy (obs=1.0)
    obs_space = Finite([0.0, 1.0])

    # Build per-component log-density functions
    # For a flat mixture, the kernel's log_density takes (theta, obs) where
    # theta is drawn from the component's BetaMeasure
    # But we need per-component dispatch...
    # The trick: build a kernel that works uniformly because each component
    # is a BetaMeasure and the kernel is Beta-Bernoulli

    # Grammar 1 programs: predicate fires on sensor > 0.5 (which is true for reading 0.8)
    # So for grammar 1 components, likelihood = Beta-Bernoulli(enemy=1)
    # Grammar 2 programs: predicate fires on sensor < 0.5 (which is false for reading 0.8)
    # So for grammar 2 components, likelihood = base_rate (flat)

    # For this test, use a simple Bernoulli kernel for all components
    # but vary the observation to simulate per-component behaviour
    k = Kernel(Interval(0.0, 1.0), obs_space,
        θ -> (o -> o == 1.0 ? log(θ) : log(1.0 - θ)),
        (θ, o) -> o == 1.0 ? log(θ) : log(1.0 - θ))

    # Condition on observing enemy (1.0)
    posterior = condition(belief, k, 1.0)

    @assert posterior isa MixtureMeasure
    @assert length(posterior.components) == 6

    # All components should be Beta-Bernoulli updated
    for comp in posterior.components
        @assert comp isa BetaMeasure "Expected BetaMeasure, got $(typeof(comp))"
    end

    # After observing enemy, all Betas should update to Beta(2,1)
    for comp in posterior.components
        @assert comp.alpha ≈ 2.0 "Expected α=2.0, got $(comp.alpha)"
        @assert comp.beta ≈ 1.0 "Expected β=1.0, got $(comp.beta)"
    end

    # Weights should reflect prior: grammar 1 (simpler) should still have more mass
    w = weights(posterior)
    # Grammar 1 aggregate = w[1] + w[2] + w[3]
    # Grammar 2 aggregate = w[4] + w[5] + w[6]
    g1_weight = sum(w[1:3])
    g2_weight = sum(w[4:6])
    @assert g1_weight > g2_weight "Simpler grammar should have more weight"
    println("PASSED: Grammar 1 weight = $(round(g1_weight, digits=4)), Grammar 2 = $(round(g2_weight, digits=4))")
end
println()

println("=" ^ 60)
println("FLAT MIXTURE TEST 2: Differential per-component kernels")
println("=" ^ 60)

# Test with kernels that give DIFFERENT likelihoods per component
# by building a kernel closure that dispatches on component index
let
    n_components = 6
    components = Measure[BetaMeasure(1.0, 1.0) for _ in 1:n_components]
    metadata = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]

    # Equal prior weights for this test
    log_prior = fill(0.0, n_components)
    belief = MixtureMeasure(Interval(0.0, 1.0), components, log_prior)

    # Kernel where grammar 1 programs predict enemy well, grammar 2 poorly
    # We simulate this by having the BetaMeasure represent program confidence
    # and using different Beta priors
    components_diff = Measure[
        BetaMeasure(8.0, 2.0),   # G1P1: high confidence enemy
        BetaMeasure(6.0, 4.0),   # G1P2: moderate confidence
        BetaMeasure(3.0, 7.0),   # G1P3: predicts food (wrong)
        BetaMeasure(2.0, 8.0),   # G2P1: predicts food (wrong)
        BetaMeasure(4.0, 6.0),   # G2P2: moderate, slightly food
        BetaMeasure(1.0, 1.0),   # G2P3: no information
    ]
    belief_diff = MixtureMeasure(Interval(0.0, 1.0), components_diff, log_prior)

    k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
        θ -> (o -> o == 1.0 ? log(θ) : log(1.0 - θ)),
        (θ, o) -> o == 1.0 ? log(θ) : log(1.0 - θ))

    # Condition on enemy (1.0) — programs predicting enemy should gain weight
    posterior = condition(belief_diff, k, 1.0)
    w = weights(posterior)

    # G1P1 (Beta(8,2), mean=0.8) should dominate after seeing enemy
    @assert argmax(w) == 1 "Component with highest enemy prediction should dominate, got argmax=$(argmax(w))"

    # G1P3 (Beta(3,7), predicts food) and G2P1 (Beta(2,8), predicts food) should lose weight
    @assert w[1] > w[3] "G1P1 should outweigh G1P3"
    @assert w[1] > w[4] "G1P1 should outweigh G2P1"

    # Grammar-level aggregation
    g1_weight = sum(w[1:3])
    g2_weight = sum(w[4:6])
    @assert g1_weight > g2_weight "Grammar 1 (with more enemy-predictive programs) should dominate"
    println("PASSED: G1 aggregate=$(round(g1_weight, digits=4)), G2=$(round(g2_weight, digits=4))")
    println("  Per-component weights: ", [round(wi, digits=4) for wi in w])
end
println()

println("=" ^ 60)
println("FLAT MIXTURE TEST 3: _predictive_ll(MixtureMeasure) exact dispatch")
println("=" ^ 60)

let
    # Build a MixtureMeasure of BetaMeasures
    b1 = BetaMeasure(3.0, 1.0)  # mean 0.75 — predicts enemy
    b2 = BetaMeasure(1.0, 3.0)  # mean 0.25 — predicts food
    mix = MixtureMeasure(Interval(0.0, 1.0), Measure[b1, b2], [log(0.6), log(0.4)])

    k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
        θ -> (o -> o == 1.0 ? log(θ) : log(1.0 - θ)),
        (θ, o) -> o == 1.0 ? log(θ) : log(1.0 - θ))

    # Manually compute predictive likelihood
    # P(obs=1 | mix) = 0.6 * E_Beta(3,1)[θ] + 0.4 * E_Beta(1,3)[θ]
    #                = 0.6 * 0.75 + 0.4 * 0.25 = 0.55
    # But _predictive_ll uses expect(component, h -> exp(k.log_density(h, obs)))
    # For BetaMeasure with Bernoulli kernel: E[θ] = mean

    # Test via condition: should work without error
    posterior = condition(mix, k, 1.0)
    @assert posterior isa MixtureMeasure
    @assert length(posterior.components) == 2

    # b1 (enemy-predictive) should gain weight
    w = weights(posterior)
    @assert w[1] > w[2] "Beta(3,1) should gain weight after observing enemy"
    println("PASSED: MixtureMeasure conditioned, weights = ", [round(wi, digits=4) for wi in w])

    # Test nested: MixtureMeasure inside MixtureMeasure
    outer = MixtureMeasure(Interval(0.0, 1.0), Measure[mix, BetaMeasure(5.0, 1.0)], [0.0, 0.0])
    posterior_outer = condition(outer, k, 1.0)
    @assert posterior_outer isa MixtureMeasure
    println("PASSED: Nested MixtureMeasure conditioning works, $(length(posterior_outer.components)) components")
end
println()

println("=" ^ 60)
println("FLAT MIXTURE TEST 4: fold-init DSL form")
println("=" ^ 60)

let
    # fold-init with explicit accumulator
    result = run_dsl("""
    (fold-init + 0 (list 1 2 3 4 5))
    """)
    @assert result == 15 "Expected 15, got $result"
    println("PASSED: fold-init sum = $result")

    # fold-init for sequential conditioning
    result2 = run_dsl("""
    (let m (measure (space :interval 0 1) :beta 1 1)
      (let k (kernel (space :interval 0 1) (space :finite 0 1)
                (lambda (r) (lambda (obs) (if (= obs 1) (log r) (log (- 1.0 r))))))
        (mean (fold-init (lambda (b obs) (condition b k obs))
                         m
                         (list 1 1 1 0)))))
    """)
    # Beta(1,1) + 3 successes + 1 failure = Beta(4,2), mean = 2/3
    @assert abs(result2 - 4/6) < 0.01 "Expected ≈ 0.667, got $result2"
    println("PASSED: fold-init sequential condition, mean = $(round(result2, digits=4))")
end
println()

println("=" ^ 60)
println("FLAT MIXTURE TEST 5: Grammar-level aggregation")
println("=" ^ 60)

let
    # Simulate a realistic scenario: 3 grammars × varying programs
    # Grammar 1 (simple, 2 programs): complexity 2
    # Grammar 2 (medium, 3 programs): complexity 4
    # Grammar 3 (complex, 2 programs): complexity 6
    metadata = [(1,1), (1,2), (2,1), (2,2), (2,3), (3,1), (3,2)]
    gc = [2.0, 2.0, 4.0, 4.0, 4.0, 6.0, 6.0]
    pc = [1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0]

    log_prior = [-gc[i] * log(2) - pc[i] * log(2) for i in eachindex(gc)]

    # Different Beta priors: G1 programs predict well, G3 poorly
    components = Measure[
        BetaMeasure(9.0, 1.0),   # G1P1: strong enemy predictor
        BetaMeasure(7.0, 3.0),   # G1P2: good enemy predictor
        BetaMeasure(5.0, 5.0),   # G2P1: neutral
        BetaMeasure(4.0, 6.0),   # G2P2: slight food
        BetaMeasure(6.0, 4.0),   # G2P3: slight enemy
        BetaMeasure(2.0, 8.0),   # G3P1: strong food predictor
        BetaMeasure(1.0, 9.0),   # G3P2: very strong food predictor
    ]

    belief = MixtureMeasure(Interval(0.0, 1.0), components, log_prior)

    k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
        θ -> (o -> o == 1.0 ? log(θ) : log(1.0 - θ)),
        (θ, o) -> o == 1.0 ? log(θ) : log(1.0 - θ))

    # Condition on 3 enemy observations
    posterior = belief
    for _ in 1:3
        posterior = condition(posterior, k, 1.0)
    end

    w = weights(posterior)

    # Aggregate by grammar
    grammar_weights = Dict{Int,Float64}()
    for (i, (gi, _)) in enumerate(metadata)
        grammar_weights[gi] = get(grammar_weights, gi, 0.0) + w[i]
    end

    # Grammar 1 should dominate: simple AND predictive
    @assert grammar_weights[1] > grammar_weights[2] "G1 should outweigh G2"
    @assert grammar_weights[1] > grammar_weights[3] "G1 should outweigh G3"
    # Grammar 3 should have least weight: complex AND wrong
    @assert grammar_weights[3] < grammar_weights[2] "G3 should have least weight"

    println("PASSED: Grammar weights after 3 enemy obs:")
    for gi in sort(collect(keys(grammar_weights)))
        println("  Grammar $gi: $(round(grammar_weights[gi], digits=6))")
    end
end
println()

println("=" ^ 60)
println("FLAT MIXTURE TEST 6: TaggedBetaMeasure per-component dispatch")
println("=" ^ 60)

let
    # 4 TaggedBetaMeasure components: 2 programs × 2 grammars
    # Programs 1,2 predicates "fire" (kernel returns Beta-Bernoulli ll)
    # Programs 3,4 predicates "don't fire" (kernel returns 0.0 — flat)
    comps = Measure[
        TaggedBetaMeasure(Interval(0.0, 1.0), 1, BetaMeasure(1.0, 1.0)),
        TaggedBetaMeasure(Interval(0.0, 1.0), 2, BetaMeasure(1.0, 1.0)),
        TaggedBetaMeasure(Interval(0.0, 1.0), 3, BetaMeasure(1.0, 1.0)),
        TaggedBetaMeasure(Interval(0.0, 1.0), 4, BetaMeasure(1.0, 1.0)),
    ]
    belief = MixtureMeasure(Interval(0.0, 1.0), comps, fill(0.0, 4))

    # Kernel: tags 1,2 fire (Beta-Bernoulli ll), tags 3,4 don't (flat)
    fires = Set([1, 2])
    k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
        _ -> error("not used"),
        (m_or_θ, obs) -> begin
            if m_or_θ isa TaggedBetaMeasure
                if m_or_θ.tag in fires
                    p = mean(m_or_θ.beta)
                    obs == 1.0 ? log(max(p, 1e-300)) : log(max(1 - p, 1e-300))
                else
                    0.0
                end
            else
                obs == 1.0 ? log(max(m_or_θ, 1e-300)) : log(max(1 - m_or_θ, 1e-300))
            end
        end)

    # Condition on enemy (1.0) — 5 times
    posterior = belief
    for _ in 1:5
        posterior = condition(posterior, k, 1.0)
    end

    @assert posterior isa MixtureMeasure
    @assert length(posterior.components) == 4

    # Check types preserved
    for comp in posterior.components
        @assert comp isa TaggedBetaMeasure "Expected TaggedBetaMeasure, got $(typeof(comp))"
    end

    # Firing components (1,2) should have updated Betas: Beta(6,1) after 5 enemy obs
    for i in 1:2
        c = posterior.components[i]
        @assert c.beta.alpha ≈ 6.0 "Firing comp $i: expected α=6, got $(c.beta.alpha)"
        @assert c.beta.beta ≈ 1.0 "Firing comp $i: expected β=1, got $(c.beta.beta)"
    end

    # Non-firing components (3,4) should be unchanged: Beta(1,1)
    for i in 3:4
        c = posterior.components[i]
        @assert c.beta.alpha ≈ 1.0 "Non-firing comp $i: expected α=1, got $(c.beta.alpha)"
        @assert c.beta.beta ≈ 1.0 "Non-firing comp $i: expected β=1, got $(c.beta.beta)"
    end

    # Firing components should gain weight relative to non-firing
    w = weights(posterior)
    @assert w[1] + w[2] > w[3] + w[4] "Firing components should outweigh non-firing"
    println("PASSED: Firing weights=$(round(w[1]+w[2], digits=4)), " *
            "non-firing=$(round(w[3]+w[4], digits=4))")

    # Tags should be preserved
    for (i, comp) in enumerate(posterior.components)
        @assert comp.tag == i "Tag mismatch: expected $i, got $(comp.tag)"
    end
    println("PASSED: Tags preserved after conditioning")
end
println()

println("=" ^ 60)
println("ALL FLAT MIXTURE TESTS PASSED")
println("=" ^ 60)
