#!/usr/bin/env julia
"""
    test_host.jl — Structural invariant tests for ProductMeasure condition
    and host-level reliability functions.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: BetaPrevision, GaussianPrevision, GammaPrevision, CategoricalPrevision  # Posture 4 Move 4
using Credence.Ontology: wrap_in_measure  # Posture 4 Move 4
using Credence: condition, weights, expect, draw, prune
using Credence: CategoricalMeasure, BetaMeasure, MixtureMeasure, ProductMeasure
using Credence: Finite, Interval, ProductSpace, Kernel, FactorSelector, Measure

println("=" ^ 60)
println("HOST TEST 1: condition(ProductMeasure, kernel_with_factor_selector)")
println("=" ^ 60)

# TEST: condition(ProductMeasure, kernel_with_active_factors, obs)
# returns MixtureMeasure of ProductMeasures with only active factor updated
let
    cat = CategoricalMeasure(Finite([0.0, 1.0, 2.0]), CategoricalPrevision(fill(0.0, 3)))  # 3 categories
    betas = [wrap_in_measure(BetaPrevision(3.0, 2.0)), wrap_in_measure(BetaPrevision(5.0, 5.0)), wrap_in_measure(BetaPrevision(2.0, 8.0))]
    prod = ProductMeasure(Measure[cat, betas...])

    binary = Finite([0.0, 1.0])
    k = Kernel(prod.space, binary,
        _ -> error("not used"),
        (h, o) -> let r = h[Int(h[1]) + 2]; o == 1.0 ? log(r) : log(1.0 - r) end;
        factor_selector = FactorSelector(1, c -> [Int(c) + 2]),
        likelihood_family = BetaBernoulli())

    posterior = condition(prod, k, 1.0)
    @assert posterior isa MixtureMeasure
    @assert length(posterior.components) == 3

    for (ci, comp) in enumerate(posterior.components)
        @assert comp isa ProductMeasure
        # Factor 1: point mass categorical at category ci-1
        cat_factor = comp.factors[1]::CategoricalMeasure
        cat_w = weights(cat_factor)
        @assert cat_w[ci] ≈ 1.0
        @assert all(cat_w[j] ≈ 0.0 for j in eachindex(cat_w) if j != ci)

        for j in 2:4  # Beta factors
            beta = comp.factors[j]::BetaMeasure
            orig = betas[j - 1]
            if j == ci + 1  # active factor for this category
                # Should be conditioned: alpha + 1 for success
                @assert beta.alpha == orig.alpha + 1.0
                @assert beta.beta == orig.beta
            else
                # Inactive: identical to prior
                @assert beta.alpha == orig.alpha
                @assert beta.beta == orig.beta
            end
        end
    end
    println("PASSED: condition(ProductMeasure) preserves structure, only active factor updated")
end
println()

println("=" ^ 60)
println("HOST TEST 2: condition(ProductMeasure) on failure observation")
println("=" ^ 60)

let
    cat = CategoricalMeasure(Finite([0.0, 1.0]), CategoricalPrevision(fill(0.0, 2)))  # 2 categories
    betas = [wrap_in_measure(BetaPrevision(4.0, 3.0)), wrap_in_measure(BetaPrevision(2.0, 6.0))]
    prod = ProductMeasure(Measure[cat, betas...])

    binary = Finite([0.0, 1.0])
    k = Kernel(prod.space, binary,
        _ -> error("not used"),
        (h, o) -> let r = h[Int(h[1]) + 2]; o == 1.0 ? log(r) : log(1.0 - r) end;
        factor_selector = FactorSelector(1, c -> [Int(c) + 2]),
        likelihood_family = BetaBernoulli())

    posterior = condition(prod, k, 0.0)  # failure observation
    @assert posterior isa MixtureMeasure
    @assert length(posterior.components) == 2

    # Category 0: active factor is index 2, beta should increase by 1
    comp0 = posterior.components[1]::ProductMeasure
    @assert comp0.factors[2].alpha == 4.0
    @assert comp0.factors[2].beta == 4.0   # 3.0 + 1.0
    @assert comp0.factors[3].alpha == 2.0  # unchanged
    @assert comp0.factors[3].beta == 6.0

    # Category 1: active factor is index 3, beta should increase by 1
    comp1 = posterior.components[2]::ProductMeasure
    @assert comp1.factors[2].alpha == 4.0  # unchanged
    @assert comp1.factors[2].beta == 3.0
    @assert comp1.factors[3].alpha == 2.0
    @assert comp1.factors[3].beta == 7.0   # 6.0 + 1.0

    println("PASSED: condition(ProductMeasure) on failure correctly updates beta parameter")
end
println()

println("=" ^ 60)
println("HOST TEST 3: condition(ProductMeasure) without factor_selector uses fallback")
println("=" ^ 60)

using Random
let
    Random.seed!(42)
    cat = CategoricalMeasure(Finite([0, 1]), CategoricalPrevision(fill(0.0, 2)))
    theta = wrap_in_measure(BetaPrevision(2.0, 2.0))
    pm = ProductMeasure(Measure[cat, theta])

    obs_space = Finite([0, 1])
    k = Kernel(pm.space, obs_space,
        h -> (o -> begin; t = h[2]; o == 1 ? log(t) : log(1.0 - t) end),
        (h, o) -> begin; t = h[2]; o == 1 ? log(t) : log(1.0 - t) end;
        likelihood_family = BetaBernoulli())

    # No factor_selector → falls back to importance sampling
    posterior = condition(pm, k, 1; n_particles=5000)
    @assert posterior isa CategoricalMeasure
    println("PASSED: ProductMeasure without factor_selector uses importance sampling fallback")
end
println()

println("=" ^ 60)
println("HOST TEST 4: marginalize_betas produces MixtureMeasure of BetaMeasures")
println("=" ^ 60)

let
    function _marginalize_betas(rel_state::MixtureMeasure, cat_w::Vector{Float64})
        components = Measure[]
        log_wts = Float64[]
        for (i, comp) in enumerate(rel_state.components)
            prod = comp::ProductMeasure
            for (c, w_c) in enumerate(cat_w)
                push!(components, prod.factors[c])
                push!(log_wts, rel_state.log_weights[i] + log(max(w_c, 1e-300)))
            end
        end
        prune(MixtureMeasure(Interval(0.0, 1.0), components, log_wts))
    end

    betas = Measure[wrap_in_measure(BetaPrevision(3.0, 2.0)), wrap_in_measure(BetaPrevision(5.0, 5.0)), wrap_in_measure(BetaPrevision(2.0, 8.0))]
    prod = ProductMeasure(betas)
    rel_state = MixtureMeasure(prod.space, Measure[prod], [0.0])
    cat_w = [0.6, 0.3, 0.1]

    eff = _marginalize_betas(rel_state, cat_w)
    @assert eff isa MixtureMeasure
    @assert eff.space == Interval(0.0, 1.0)
    for comp in eff.components
        @assert comp isa BetaMeasure
    end
    w = weights(eff)
    @assert abs(w[1] - 0.6) < 0.01
    @assert abs(w[2] - 0.3) < 0.01
    @assert abs(w[3] - 0.1) < 0.01
    println("PASSED: marginalize_betas produces MixtureMeasure of BetaMeasures")
end
println()

println("=" ^ 60)
println("HOST TEST 5: MixtureMeasure of ProductMeasures conditions correctly (update_beta_state flow)")
println("=" ^ 60)

let
    # Simulates the update_beta_state flow:
    # MixtureMeasure of ProductMeasures, each with [categorical, Beta, Beta]
    cat = CategoricalMeasure(Finite([0.0, 1.0]), CategoricalPrevision(fill(0.0, 2)))
    b1 = wrap_in_measure(BetaPrevision(2.0, 3.0))
    b2 = wrap_in_measure(BetaPrevision(4.0, 1.0))
    prod = ProductMeasure(Measure[cat, b1, b2])
    joint = MixtureMeasure(prod.space, Measure[prod], [0.0])

    binary = Finite([0.0, 1.0])
    k = Kernel(prod.space, binary,
        _ -> error("not used"),
        (h, o) -> let r = h[Int(h[1]) + 2]; o == 1.0 ? log(r) : log(1.0 - r) end;
        factor_selector = FactorSelector(1, c -> [Int(c) + 2]),
        likelihood_family = BetaBernoulli())

    posterior = condition(joint, k, 1.0)
    @assert posterior isa MixtureMeasure
    # 1 component × 2 categories = 2 components
    @assert length(posterior.components) == 2

    # Each component should be a ProductMeasure with point-mass categorical
    for comp in posterior.components
        @assert comp isa ProductMeasure
        cat_f = comp.factors[1]::CategoricalMeasure
        cat_w = weights(cat_f)
        @assert count(w -> w ≈ 1.0, cat_w) == 1
    end

    println("PASSED: MixtureMeasure of ProductMeasures conditions via FactorSelector")
end
println()

println("=" ^ 60)
println("HOST TEST 6: initial_cov_state produces informative Beta priors")
println("=" ^ 60)

let
    function _initial_cov_state(n_categories::Int, prior_coverage::Vector{Float64}; strength::Float64=10.0)
        factors = Measure[wrap_in_measure(BetaPrevision(max(p * strength, 0.01), max((1-p) * strength, 0.01)))
                          for p in prior_coverage]
        prod = ProductMeasure(factors)
        MixtureMeasure(prod.space, Measure[prod], [0.0])
    end

    cov_state = _initial_cov_state(2, [0.8, 0.6])
    @assert cov_state isa MixtureMeasure
    @assert length(cov_state.components) == 1

    prod = cov_state.components[1]::ProductMeasure
    @assert length(prod.factors) == 2

    b1 = prod.factors[1]::BetaMeasure
    @assert abs(b1.alpha - 8.0) < 1e-10  # 0.8 * 10
    @assert abs(b1.beta - 2.0) < 1e-10   # 0.2 * 10

    b2 = prod.factors[2]::BetaMeasure
    @assert abs(b2.alpha - 6.0) < 1e-10  # 0.6 * 10
    @assert abs(b2.beta - 4.0) < 1e-10   # 0.4 * 10

    println("PASSED: initial_cov_state produces informative Beta priors from declared coverage")
end
println()

println("=" ^ 60)
println("HOST TEST 7: update_beta_state on coverage responded/not-responded")
println("=" ^ 60)

let
    using Credence: Credence as C

    function _marginalize_betas(rel_state::MixtureMeasure, cat_w::Vector{Float64})
        components = Measure[]
        log_wts = Float64[]
        for (i, comp) in enumerate(rel_state.components)
            prod = comp::ProductMeasure
            for (c, w_c) in enumerate(cat_w)
                push!(components, prod.factors[c])
                push!(log_wts, rel_state.log_weights[i] + log(max(w_c, 1e-300)))
            end
        end
        prune(MixtureMeasure(Interval(0.0, 1.0), components, log_wts))
    end

    function _update_beta_state(rel_state::MixtureMeasure,
                                cat_belief::CategoricalMeasure, obs)
        n_cat = length(cat_belief.space.values)
        joint_components = Measure[]
        for comp in rel_state.components
            prod = comp::ProductMeasure
            push!(joint_components, ProductMeasure(Measure[cat_belief, prod.factors...]))
        end
        joint_space = joint_components[1].space
        joint = MixtureMeasure(joint_space, joint_components, copy(rel_state.log_weights))

        binary = Finite([0.0, 1.0])
        k = Kernel(joint_space, binary,
            _ -> error("generate not used"),
            (h, o) -> let r = h[Int(h[1]) + 2]; o == 1.0 ? log(r) : log(1.0 - r) end;
            factor_selector = FactorSelector(1, c -> [Int(c) + 2]),
            likelihood_family = BetaBernoulli())
        posterior = condition(joint, k, obs)

        cat_log_post = fill(-Inf, n_cat)
        for (i, comp) in enumerate(posterior.components)
            prod = comp::ProductMeasure
            c_val = prod.factors[1]::CategoricalMeasure
            ci = findfirst(==(c_val.space.values[1]), cat_belief.space.values)
            lw = posterior.log_weights[i]
            if cat_log_post[ci] == -Inf
                cat_log_post[ci] = lw
            else
                mx = max(cat_log_post[ci], lw)
                cat_log_post[ci] = mx + log(exp(cat_log_post[ci] - mx) + exp(lw - mx))
            end
        end
        new_cat_belief = CategoricalMeasure(cat_belief.space, CategoricalPrevision(cat_log_post))

        stripped = Measure[ProductMeasure(Measure[comp.factors[2:end]...])
                           for comp in posterior.components]
        new_state = MixtureMeasure(stripped[1].space, stripped, copy(posterior.log_weights))
        new_state = C.truncate(prune(new_state); max_components=20)
        (new_state, new_cat_belief)
    end

    # Start with coverage Beta(8,2) and Beta(6,4) for 2 categories
    factors = Measure[wrap_in_measure(BetaPrevision(8.0, 2.0)), wrap_in_measure(BetaPrevision(6.0, 4.0))]
    prod = ProductMeasure(factors)
    cov_state = MixtureMeasure(prod.space, Measure[prod], [0.0])
    cat_belief = CategoricalMeasure(Finite([0.0, 1.0]), CategoricalPrevision(fill(0.0, 2)))

    # Responded (obs=1.0): alpha should increase for the active category
    (new_state, _) = _update_beta_state(cov_state, cat_belief, 1.0)
    @assert new_state isa MixtureMeasure
    for comp in new_state.components
        prod_c = comp::ProductMeasure
        for f in prod_c.factors
            @assert f isa BetaMeasure
        end
    end

    # Not responded (obs=0.0): beta should increase for the active category
    (new_state2, _) = _update_beta_state(cov_state, cat_belief, 0.0)
    @assert new_state2 isa MixtureMeasure

    println("PASSED: update_beta_state on coverage responded/not-responded updates correctly")
end
println()

println("=" ^ 60)
println("HOST TEST 8: marginalize_betas on coverage state gives expected mean")
println("=" ^ 60)

let
    using Credence: expect

    function _marginalize_betas(state::MixtureMeasure, cat_w::Vector{Float64})
        components = Measure[]
        log_wts = Float64[]
        for (i, comp) in enumerate(state.components)
            prod = comp::ProductMeasure
            for (c, w_c) in enumerate(cat_w)
                push!(components, prod.factors[c])
                push!(log_wts, state.log_weights[i] + log(max(w_c, 1e-300)))
            end
        end
        prune(MixtureMeasure(Interval(0.0, 1.0), components, log_wts))
    end

    # Coverage state: Beta(8,2) for cat0, Beta(6,4) for cat1
    factors = Measure[wrap_in_measure(BetaPrevision(8.0, 2.0)), wrap_in_measure(BetaPrevision(6.0, 4.0))]
    prod = ProductMeasure(factors)
    cov_state = MixtureMeasure(prod.space, Measure[prod], [0.0])

    # Category weights: 0.6 for cat0, 0.4 for cat1
    cat_w = [0.6, 0.4]

    eff = _marginalize_betas(cov_state, cat_w)
    # Expected coverage = 0.6 * mean(Beta(8,2)) + 0.4 * mean(Beta(6,4))
    #                    = 0.6 * 0.8 + 0.4 * 0.6 = 0.48 + 0.24 = 0.72
    expected_cov = expect(eff, r -> r)
    @assert abs(expected_cov - 0.72) < 0.01 "Expected ~0.72, got $expected_cov"

    println("PASSED: marginalize_betas on coverage state marginalizes correctly (expected=$(round(expected_cov, digits=4)))")
end
println()

println("=" ^ 60)
println("ALL HOST TESTS PASSED")
println("=" ^ 60)
