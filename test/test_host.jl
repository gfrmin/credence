#!/usr/bin/env julia
"""
    test_host.jl — Structural invariant tests for ProductMeasure condition
    and host-level reliability functions.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: condition, weights, expect, draw, prune
using Credence: CategoricalMeasure, BetaMeasure, MixtureMeasure, ProductMeasure
using Credence: Finite, Interval, ProductSpace, Kernel, FactorSelector, Measure

println("=" ^ 60)
println("HOST TEST 1: condition(ProductMeasure, kernel_with_factor_selector)")
println("=" ^ 60)

# TEST: condition(ProductMeasure, kernel_with_active_factors, obs)
# returns MixtureMeasure of ProductMeasures with only active factor updated
let
    cat = CategoricalMeasure(Finite([0.0, 1.0, 2.0]))  # 3 categories
    betas = [BetaMeasure(3.0, 2.0), BetaMeasure(5.0, 5.0), BetaMeasure(2.0, 8.0)]
    prod = ProductMeasure(Measure[cat, betas...])

    binary = Finite([0.0, 1.0])
    k = Kernel(prod.space, binary,
        _ -> error("not used"),
        (h, o) -> let r = h[Int(h[1]) + 2]; o == 1.0 ? log(r) : log(1.0 - r) end,
        FactorSelector(1, c -> [Int(c) + 2]))

    posterior = condition(prod, k, 1.0)
    @assert posterior isa MixtureMeasure
    @assert length(posterior.components) == 3

    for (ci, comp) in enumerate(posterior.components)
        @assert comp isa ProductMeasure
        # Factor 1: point mass categorical at category ci-1
        cat_factor = comp.factors[1]::CategoricalMeasure
        @assert length(cat_factor.space.values) == 1
        @assert cat_factor.space.values[1] == Float64(ci - 1)

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
    cat = CategoricalMeasure(Finite([0.0, 1.0]))  # 2 categories
    betas = [BetaMeasure(4.0, 3.0), BetaMeasure(2.0, 6.0)]
    prod = ProductMeasure(Measure[cat, betas...])

    binary = Finite([0.0, 1.0])
    k = Kernel(prod.space, binary,
        _ -> error("not used"),
        (h, o) -> let r = h[Int(h[1]) + 2]; o == 1.0 ? log(r) : log(1.0 - r) end,
        FactorSelector(1, c -> [Int(c) + 2]))

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
    cat = CategoricalMeasure(Finite([0, 1]))
    theta = BetaMeasure(2.0, 2.0)
    pm = ProductMeasure(Measure[cat, theta])

    obs_space = Finite([0, 1])
    k = Kernel(pm.space, obs_space,
        h -> (o -> begin; t = h[2]; o == 1 ? log(t) : log(1.0 - t) end),
        (h, o) -> begin; t = h[2]; o == 1 ? log(t) : log(1.0 - t) end)

    # No factor_selector → falls back to importance sampling
    posterior = condition(pm, k, 1; n_particles=5000)
    @assert posterior isa CategoricalMeasure
    println("PASSED: ProductMeasure without factor_selector uses importance sampling fallback")
end
println()

println("=" ^ 60)
println("HOST TEST 4: effective_reliability produces MixtureMeasure of BetaMeasures")
println("=" ^ 60)

let
    function _effective_reliability(rel_state::MixtureMeasure, cat_w::Vector{Float64})
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

    betas = Measure[BetaMeasure(3.0, 2.0), BetaMeasure(5.0, 5.0), BetaMeasure(2.0, 8.0)]
    prod = ProductMeasure(betas)
    rel_state = MixtureMeasure(prod.space, Measure[prod], [0.0])
    cat_w = [0.6, 0.3, 0.1]

    eff = _effective_reliability(rel_state, cat_w)
    @assert eff isa MixtureMeasure
    @assert eff.space == Interval(0.0, 1.0)
    for comp in eff.components
        @assert comp isa BetaMeasure
    end
    w = weights(eff)
    @assert abs(w[1] - 0.6) < 0.01
    @assert abs(w[2] - 0.3) < 0.01
    @assert abs(w[3] - 0.1) < 0.01
    println("PASSED: effective_reliability produces MixtureMeasure of BetaMeasures")
end
println()

println("=" ^ 60)
println("HOST TEST 5: MixtureMeasure of ProductMeasures conditions correctly")
println("=" ^ 60)

let
    # Simulates the update_tool_reliability flow:
    # MixtureMeasure of ProductMeasures, each with [categorical, Beta, Beta]
    cat = CategoricalMeasure(Finite([0.0, 1.0]))
    b1 = BetaMeasure(2.0, 3.0)
    b2 = BetaMeasure(4.0, 1.0)
    prod = ProductMeasure(Measure[cat, b1, b2])
    joint = MixtureMeasure(prod.space, Measure[prod], [0.0])

    binary = Finite([0.0, 1.0])
    k = Kernel(prod.space, binary,
        _ -> error("not used"),
        (h, o) -> let r = h[Int(h[1]) + 2]; o == 1.0 ? log(r) : log(1.0 - r) end,
        FactorSelector(1, c -> [Int(c) + 2]))

    posterior = condition(joint, k, 1.0)
    @assert posterior isa MixtureMeasure
    # 1 component × 2 categories = 2 components
    @assert length(posterior.components) == 2

    # Each component should be a ProductMeasure with point-mass categorical
    for comp in posterior.components
        @assert comp isa ProductMeasure
        cat_f = comp.factors[1]::CategoricalMeasure
        @assert length(cat_f.space.values) == 1
    end

    println("PASSED: MixtureMeasure of ProductMeasures conditions via FactorSelector")
end
println()

println("=" ^ 60)
println("ALL HOST TESTS PASSED")
println("=" ^ 60)
