#!/usr/bin/env julia
"""
    test.jl — Tests for the three-types DSL.

Verifies: type constructors, axiom-constrained functions
(condition, expect, push, density), and derived stdlib
(optimise, value, voi, predictive).
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence

println("=" ^ 60)
println("TEST 1: Spaces are first-class typed objects")
println("=" ^ 60)

result = run_dsl("""
(list (space :finite 1 2 3)
      (space :interval 0 1))
""")
@assert result[1] isa Finite
@assert result[2] isa Interval
println("PASSED: Finite and Interval spaces constructed")
println()

println("=" ^ 60)
println("TEST 2: Measures are distributions over spaces")
println("=" ^ 60)

result = run_dsl("""
(let s (space :finite a b c)
  (weights (measure s :uniform)))
""")
@assert length(result) == 3
@assert all(w -> abs(w - 1/3) < 1e-10, result)
println("PASSED: Uniform measure over 3 values: ", result)

result = run_dsl("""
(mean (measure (space :interval 0 1) :beta 8 2))
""")
@assert abs(result - 0.8) < 1e-10
println("PASSED: Beta(8,2) mean = ", result)
println()

println("=" ^ 60)
println("TEST 3: Kernels are typed conditional distributions")
println("=" ^ 60)

# A kernel from hypothesis space to observation space
result = run_dsl("""
(let H (space :finite 0.3 0.7)
  (let O (space :finite 0 1)
    (let k (kernel H O
              (lambda (theta)
                (lambda (obs)
                  (if (= obs 1) (log theta) (log (- 1.0 theta))))))
      (list (density k 0.7 1)
            (density k 0.7 0)
            (density k 0.3 1)))))
""")
@assert abs(result[1] - log(0.7)) < 1e-10
@assert abs(result[2] - log(0.3)) < 1e-10
@assert abs(result[3] - log(0.3)) < 1e-10
println("PASSED: Kernel densities correct")
println()

println("=" ^ 60)
println("TEST 4: expect — integration against a measure")
println("=" ^ 60)

# E[h] under uniform{0.3, 0.7} = 0.5
result = run_dsl("""
(let m (measure (space :finite 0.3 0.7) :uniform)
  (expect m (lambda (h) h)))
""")
@assert abs(result - 0.5) < 1e-10
println("PASSED: E[h] under uniform{0.3, 0.7} = ", result)

# E[r] under Beta(5,3) = 5/8
result = run_dsl("""
(expect (measure (space :interval 0 1) :beta 5 3)
        (lambda (r) r))
""")
@assert abs(result - 0.625) < 0.01
println("PASSED: E[r] under Beta(5,3) = ", result)
println()

println("=" ^ 60)
println("TEST 5: condition — Bayesian inversion (discrete)")
println("=" ^ 60)

# Uniform prior on {0.3, 0.7}, observe heads twice
result = run_dsl("""
(let H (space :finite 0.3 0.7)
  (let m (measure H :uniform)
    (let k (kernel H (space :finite 0 1)
              (lambda (theta)
                (lambda (obs)
                  (if (= obs 1) (log theta) (log (- 1.0 theta))))))
      (weights (condition (condition m k 1) k 1)))))
""")
# P(0.7|HH) ∝ 0.49, P(0.3|HH) ∝ 0.09
@assert result[2] > result[1]
@assert abs(result[2] - 0.49/0.58) < 0.01
println("PASSED: After HH, P(θ=0.7) = ", result[2])
println()

println("=" ^ 60)
println("TEST 6: condition — Beta conjugate update")
println("=" ^ 60)

# Beta(1,1) + success → Beta(2,1), mean = 2/3
result = run_dsl("""
(let m (measure (space :interval 0 1) :beta 1 1)
  (let k (kernel (space :interval 0 1) (space :finite 0 1)
            (lambda (r) (lambda (obs) (if (= obs 1) (log r) (log (- 1.0 r))))))
    (mean (condition m k 1))))
""")
@assert abs(result - 2/3) < 0.01
println("PASSED: Beta(1,1) + success → mean = ", result)

# Beta(1,1) + 3 successes + 1 failure → Beta(4,2), mean = 2/3
result = run_dsl("""
(let m (measure (space :interval 0 1) :beta 1 1)
  (let k (kernel (space :interval 0 1) (space :finite 0 1)
            (lambda (r) (lambda (obs) (if (= obs 1) (log r) (log (- 1.0 r))))))
    (mean (condition (condition (condition (condition m k 1) k 1) k 1) k 0))))
""")
@assert abs(result - 4/6) < 0.01
println("PASSED: Beta(1,1) + 3S + 1F → mean = ", result)
println()

println("=" ^ 60)
println("TEST 7: optimise — EU maximisation (stdlib)")
println("=" ^ 60)

result = run_dsl("""
(let H (space :finite 0.3 0.7)
  (let A (space :finite 1 0)
    (let m (measure H :categorical 0.15 0.85)
      (let pref (lambda (h a)
                  (if (= a 1)
                    (if (> h 0.5) 1.0 -1.0)
                    (if (< h 0.5) 1.0 -1.0)))
        (optimise m A pref)))))
""")
@assert result == 1  # should bet high
println("PASSED: optimise selects action ", result, " (bet high)")

# value returns the EU
result = run_dsl("""
(let H (space :finite 0.3 0.7)
  (let A (space :finite 1 0)
    (let m (measure H :categorical 0.15 0.85)
      (let pref (lambda (h a)
                  (if (= a 1)
                    (if (> h 0.5) 1.0 -1.0)
                    (if (< h 0.5) 1.0 -1.0)))
        (value m A pref)))))
""")
# EU(high) = 0.85*1 + 0.15*(-1) = 0.7
@assert abs(result - 0.7) < 0.01
println("PASSED: value = ", result, " (expected 0.7)")
println()

println("=" ^ 60)
println("TEST 8: predictive — P(observation | beliefs)")
println("=" ^ 60)

result = run_dsl("""
(let H (space :finite 0.3 0.7)
  (let m (measure H :uniform)
    (let k (kernel H (space :finite 0 1)
              (lambda (theta)
                (lambda (obs)
                  (if (= obs 1) (log theta) (log (- 1.0 theta))))))
      (list (predictive m k 1) (predictive m k 0)))))
""")
@assert abs(result[1] - 0.5) < 1e-10
@assert abs(result[2] - 0.5) < 1e-10
println("PASSED: P(H) = ", result[1], ", P(T) = ", result[2])
println()

println("=" ^ 60)
println("TEST 9: VOI — derived from condition + expect")
println("=" ^ 60)

# 80% reliable sensor, binary hypothesis
result = run_dsl("""
(let H (space :finite 1 2)
  (let A H
    (let m (measure H :uniform)
      (let k (kernel H H
                (lambda (h)
                  (lambda (o) (if (= h o) (log 0.8) (log 0.2)))))
        (let pref (lambda (h a) (if (= h a) 1.0 -1.0))
          (voi m k A pref (list 1 2)))))))
""")
@assert abs(result - 0.6) < 0.01
println("PASSED: VOI (80% sensor) = ", result, " (expected 0.6)")

# Flat kernel → VOI = 0 (tested by stdlib self-test too)
result = run_dsl("""
(let H (space :finite 1 2)
  (let m (measure H :uniform)
    (let k (kernel H H (lambda (h) (lambda (o) 0.0)))
      (voi m k H (lambda (h a) (if (= h a) 1.0 0.0))
           (list 1 2)))))
""")
@assert abs(result) < 0.001
println("PASSED: VOI (flat kernel) = ", result, " (expected 0.0)")
println()

println("=" ^ 60)
println("TEST 10: push — measure composition (pushforward)")
println("=" ^ 60)

# Prior: uniform over {0.3, 0.7}
# Kernel: Bernoulli(theta)
# Pushforward: P(obs=1) = 0.5*0.3 + 0.5*0.7 = 0.5
result = run_dsl("""
(let H (space :finite 0.3 0.7)
  (let O (space :finite 0 1)
    (let m (measure H :uniform)
      (let k (kernel H O
                (lambda (theta)
                  (lambda (obs)
                    (if (= obs 1) (log theta) (log (- 1.0 theta))))))
        (weights (push m k))))))
""")
@assert abs(result[1] - 0.5) < 0.01  # P(0)
@assert abs(result[2] - 0.5) < 0.01  # P(1)
println("PASSED: Pushforward weights = ", result)

# Non-uniform prior: {0.3: 0.1, 0.7: 0.9}
# P(obs=1) = 0.1*0.3 + 0.9*0.7 = 0.66
result = run_dsl("""
(let H (space :finite 0.3 0.7)
  (let O (space :finite 0 1)
    (let m (measure H :categorical 0.1 0.9)
      (let k (kernel H O
                (lambda (theta)
                  (lambda (obs)
                    (if (= obs 1) (log theta) (log (- 1.0 theta))))))
        (weights (push m k))))))
""")
@assert abs(result[2] - 0.66) < 0.01  # P(1)
println("PASSED: Non-uniform pushforward P(1) = ", result[2], " (expected 0.66)")
println()

println("=" ^ 60)
println("TEST 11: Full pipeline — learn and decide")
println("=" ^ 60)

# Agent starts uncertain about coin bias, observes H H H T,
# then decides whether to bet on heads
result = run_dsl("""
(let H (space :finite 0.1 0.3 0.5 0.7 0.9)
  (let O (space :finite 0 1)
    (let A (space :finite 1 0)
      (let k (kernel H O
                (lambda (theta)
                  (lambda (obs)
                    (if (= obs 1) (log theta) (log (- 1.0 theta))))))
        (let pref (lambda (theta action)
                    (if (= action 1)
                      (- (* 2.0 theta) 1.0)
                      (- 1.0 (* 2.0 theta))))
          (let m0 (measure H :uniform)
            (let m1 (condition (condition (condition (condition m0 k 1) k 1) k 1) k 0)
              (do
                (print (weights m1))
                (list (optimise m1 A pref)
                      (value m1 A pref))))))))))
""")
println("After HHHT: action=", result[1], " value=", result[2])
@assert result[1] == 1  # bet heads
@assert result[2] > 0   # positive expected value
println()

println("=" ^ 60)
println("TEST 12: condition, expect, push are env values, not special forms")
println("=" ^ 60)

# Prove they can be passed as arguments to higher-order functions
result = run_dsl("""
(let apply-to-measure
  (lambda (f m)
    (f m (lambda (h) h)))
  (apply-to-measure expect (measure (space :finite 3 7) :uniform)))
""")
@assert abs(result - 5.0) < 1e-10
println("PASSED: expect passed as value to higher-order function, result = ", result)
println()

println("=" ^ 60)
println("TEST 13: DirichletMeasure — weights")
println("=" ^ 60)

result = run_dsl("""
(weights (measure (space :simplex 3) :dirichlet (space :finite 0 1 2) 2 3 5))
""")
@assert abs(result[1] - 0.2) < 1e-10
@assert abs(result[2] - 0.3) < 1e-10
@assert abs(result[3] - 0.5) < 1e-10
println("PASSED: Dir(2,3,5) weights = ", result)
println()

println("=" ^ 60)
println("TEST 14: DirichletMeasure — conjugate update via condition(m, k, obs)")
println("=" ^ 60)

# Construct Dirichlet and Categorical kernel in Julia directly
# (kernel source is Simplex, target is Finite)
let
    cats = Finite([0, 1, 2])
    m = DirichletMeasure(Simplex(3), cats, [1.0, 1.0, 1.0])
    k = Kernel(Simplex(3), cats,
        θ -> (o -> begin; idx = findfirst(==(o), cats.values); log(θ[idx]); end),
        (θ, o) -> begin; idx = findfirst(==(o), cats.values); log(θ[idx]); end)

    m2 = condition(m, k, 1)
    w = weights(m2)
    @assert abs(w[1] - 0.25) < 1e-10
    @assert abs(w[2] - 0.50) < 1e-10
    @assert abs(w[3] - 0.25) < 1e-10
    println("PASSED: Dir(1,1,1) + observe 1 → weights ", w, " (expected [0.25, 0.5, 0.25])")

    # Two more observations
    m3 = condition(condition(m2, k, 0), k, 2)
    w3 = weights(m3)
    # Dir(2,2,2) → [1/3, 1/3, 1/3]
    @assert all(abs.(w3 .- 1/3) .< 1e-10)
    println("PASSED: Dir(1,1,1) + observe 1,0,2 → weights ", w3)
end
println()

println("=" ^ 60)
println("TEST 15: DirichletMeasure — draw sums to 1")
println("=" ^ 60)

let
    d = DirichletMeasure(Simplex(3), Finite([0, 1, 2]), [2.0, 3.0, 5.0])
    for _ in 1:100
        s = draw(d)
        @assert abs(sum(s) - 1.0) < 1e-10
        @assert all(x -> x > 0, s)
        @assert length(s) == 3
    end
    # Also test with small alpha (exercises α < 1 path)
    d2 = DirichletMeasure(Simplex(3), Finite([0, 1, 2]), [0.5, 0.5, 0.5])
    for _ in 1:100
        s = draw(d2)
        @assert abs(sum(s) - 1.0) < 1e-10
        @assert all(x -> x >= 0, s)
    end
    println("PASSED: 200 draws all sum to 1.0, all components positive")
end
println()

println("=" ^ 60)
println("TEST 16: DirichletMeasure — expect over the simplex (Monte Carlo)")
println("=" ^ 60)

using Random
let
    Random.seed!(42)
    d = DirichletMeasure(Simplex(3), Finite([0, 1, 2]), [2.0, 3.0, 5.0])
    # E[θ_1] = alpha_1 / sum(alpha) = 0.2
    result = expect(d, θ -> θ[1])
    @assert abs(result - 0.2) < 0.05
    println("PASSED: E[θ₁] under Dir(2,3,5) = ", round(result, digits=4), " (expected ≈ 0.2)")

    # E[θ_1 * θ_2] = α₁α₂ / (S(S+1)) where S = sum(α) = 10
    result2 = expect(d, θ -> θ[1] * θ[2]; n_samples=5000)
    expected = 2.0 * 3.0 / (10.0 * 11.0)  # ≈ 0.0545
    @assert abs(result2 - expected) < 0.02
    println("PASSED: E[θ₁θ₂] under Dir(2,3,5) = ", round(result2, digits=5), " (expected ≈ ", round(expected, digits=5), ")")
end
println()

println("=" ^ 60)
println("TEST 17: DirichletMeasure — posterior predictive via push_measure")
println("=" ^ 60)

let
    cats = Finite([10, 20, 30])
    d = DirichletMeasure(Simplex(3), cats, [2.0, 3.0, 5.0])
    k = Kernel(Simplex(3), cats,
        θ -> (o -> begin; idx = findfirst(==(o), cats.values); log(θ[idx]); end),
        (θ, o) -> begin; idx = findfirst(==(o), cats.values); log(θ[idx]); end)

    pred = push_measure(d, k)
    @assert pred isa CategoricalMeasure
    w = weights(pred)
    @assert abs(w[1] - 0.2) < 1e-10
    @assert abs(w[2] - 0.3) < 1e-10
    @assert abs(w[3] - 0.5) < 1e-10
    println("PASSED: push_measure → CategoricalMeasure with weights ", w)

    # E[category_value] via the pushforward
    result = expect(pred, x -> x)
    @assert abs(result - 23.0) < 1e-10
    println("PASSED: E[X] via push_measure = ", result, " (expected 23.0)")
end
println()

println("=" ^ 60)
println("ALL TESTS PASSED")
println("=" ^ 60)
