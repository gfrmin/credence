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
println("TEST 18: General condition fallback — unknown measure-kernel combo")
println("=" ^ 60)

using Random
let
    Random.seed!(123)
    # BetaMeasure with a non-Bernoulli kernel (Interval → Finite with 3 outcomes)
    m = BetaMeasure(2.0, 5.0)  # mean = 2/7 ≈ 0.286
    obs_space = Finite([:low, :mid, :high])
    k = Kernel(Interval(0.0, 1.0), obs_space,
        θ -> (o -> begin
            if o == :low;  log(1.0 - θ)
            elseif o == :mid; log(0.5)
            else; log(θ)
            end
        end),
        (θ, o) -> begin
            if o == :low;  log(1.0 - θ)
            elseif o == :mid; log(0.5)
            else; log(θ)
            end
        end)

    posterior = condition(m, k, :high; n_particles=5000)
    @assert posterior isa CategoricalMeasure
    w = weights(posterior)
    @assert abs(sum(w) - 1.0) < 1e-10
    # Observing :high favours higher θ, so posterior mean should be > prior mean
    post_mean = expect(posterior, x -> x)
    @assert post_mean > mean(m)
    println("PASSED: General fallback fires. Prior mean=", round(mean(m), digits=3),
            ", posterior mean=", round(post_mean, digits=3))
end
println()

println("=" ^ 60)
println("TEST 19: ProductMeasure — draw and expect")
println("=" ^ 60)

let
    Random.seed!(42)
    beta = BetaMeasure(2.0, 3.0)  # mean = 0.4
    cat = CategoricalMeasure(Finite([:a, :b, :c]))
    pm = ProductMeasure(Measure[beta, cat])

    s = draw(pm)
    @assert length(s) == 2
    @assert s[1] isa Float64
    @assert s[2] in [:a, :b, :c]

    e = expect(pm, x -> x[1]; n_samples=5000)
    @assert abs(e - 0.4) < 0.05
    println("PASSED: ProductMeasure draw returns 2-vector, E[x₁] ≈ ", round(e, digits=3))
end
println()

println("=" ^ 60)
println("TEST 20: ProductMeasure condition via general fallback")
println("=" ^ 60)

let
    Random.seed!(42)
    cat = CategoricalMeasure(Finite([0, 1]))  # uniform over {0, 1}
    theta0 = BetaMeasure(2.0, 2.0)  # mean = 0.5
    theta1 = BetaMeasure(2.0, 2.0)  # mean = 0.5
    pm = ProductMeasure(Measure[cat, theta0, theta1])

    # Kernel: if cat=0, Bernoulli(θ₀); if cat=1, Bernoulli(θ₁)
    obs_space = Finite([0, 1])
    k = Kernel(ProductSpace(Space[Finite([0, 1]), Interval(0.0, 1.0), Interval(0.0, 1.0)]),
               obs_space,
               h -> (o -> begin
                   c = h[1]; t0 = h[2]; t1 = h[3]
                   θ = c == 0 ? t0 : t1
                   o == 1 ? log(θ) : log(1.0 - θ)
               end),
               (h, o) -> begin
                   c = h[1]; t0 = h[2]; t1 = h[3]
                   θ = c == 0 ? t0 : t1
                   o == 1 ? log(θ) : log(1.0 - θ)
               end)

    posterior = condition(pm, k, 1; n_particles=5000)
    @assert posterior isa CategoricalMeasure
    # Posterior is a CategoricalMeasure over sampled product vectors
    w = weights(posterior)
    @assert abs(sum(w) - 1.0) < 1e-10
    println("PASSED: ProductMeasure conditioned via general fallback, ",
            length(w), " particles")
end
println()

println("=" ^ 60)
println("TEST 21: MixtureMeasure condition preserves structure")
println("=" ^ 60)

let
    Random.seed!(42)
    b1 = BetaMeasure(3.0, 1.0)  # skewed high
    b2 = BetaMeasure(1.0, 3.0)  # skewed low
    mix = MixtureMeasure(Interval(0.0, 1.0), Measure[b1, b2], [0.0, 0.0])  # equal weights

    # Bernoulli kernel on [0,1]
    obs_space = Finite([0, 1])
    k = Kernel(Interval(0.0, 1.0), obs_space,
        θ -> (o -> o == 1 ? log(θ) : log(1.0 - θ)),
        (θ, o) -> o == 1 ? log(θ) : log(1.0 - θ))

    posterior = condition(mix, k, 1)
    @assert posterior isa MixtureMeasure
    @assert length(posterior.components) == 2
    # After observing success, component with higher θ (b1: Beta(3,1)) should get more weight
    w = weights(posterior)
    @assert w[1] > w[2]  # Beta(3,1) component should dominate
    # Components should be updated Beta measures
    @assert posterior.components[1] isa BetaMeasure
    @assert posterior.components[2] isa BetaMeasure
    println("PASSED: MixtureMeasure condition → MixtureMeasure, weights = ", round.(w, digits=3))
end
println()

println("=" ^ 60)
println("TEST 22: MixtureMeasure expect")
println("=" ^ 60)

let
    Random.seed!(42)
    b1 = BetaMeasure(4.0, 2.0)  # mean = 2/3
    b2 = BetaMeasure(2.0, 4.0)  # mean = 1/3
    mix = MixtureMeasure(Interval(0.0, 1.0), Measure[b1, b2], [log(0.7), log(0.3)])

    # E[θ] = 0.7 * 2/3 + 0.3 * 1/3 ≈ 0.567
    result = expect(mix, θ -> θ)
    expected = 0.7 * (4.0/6.0) + 0.3 * (2.0/6.0)
    @assert abs(result - expected) < 0.02
    println("PASSED: MixtureMeasure E[θ] = ", round(result, digits=4),
            " (expected ≈ ", round(expected, digits=4), ")")
end
println()

println("=" ^ 60)
println("TEST 23: MixtureMeasure prune")
println("=" ^ 60)

let
    b1 = BetaMeasure(5.0, 2.0)
    b2 = BetaMeasure(1.0, 1.0)
    b3 = BetaMeasure(2.0, 5.0)
    # b1 dominant, b2 and b3 negligible
    mix = MixtureMeasure(Interval(0.0, 1.0), Measure[b1, b2, b3],
                         [0.0, -25.0, -30.0])

    pruned = prune(mix; threshold=-20.0)
    @assert length(pruned.components) == 1
    # Expect unchanged within tolerance (b2, b3 had negligible weight)
    e_before = expect(mix, θ -> θ)
    e_after = expect(pruned, θ -> θ)
    @assert abs(e_before - e_after) < 0.01
    println("PASSED: Pruned 3 → ", length(pruned.components), " component(s), ",
            "E[θ] before=", round(e_before, digits=4), " after=", round(e_after, digits=4))
end
println()

println("=" ^ 60)
println("TEST 24: MixtureMeasure draw")
println("=" ^ 60)

let
    Random.seed!(42)
    b1 = BetaMeasure(3.0, 3.0)
    b2 = BetaMeasure(1.0, 1.0)
    mix = MixtureMeasure(Interval(0.0, 1.0), Measure[b1, b2], [0.0, 0.0])

    for _ in 1:100
        s = draw(mix)
        @assert s isa Float64
        @assert 0.0 <= s <= 1.0
    end
    println("PASSED: 100 draws from MixtureMeasure all valid Float64 in [0,1]")
end
println()

println("=" ^ 60)
println("TEST 25: BetaMeasure + non-Bernoulli kernel → grid fallback")
println("=" ^ 60)

let
    prior = BetaMeasure(2.0, 2.0)
    # 3-outcome kernel: NOT conjugate (not binary)
    obs_space = Finite([:low, :mid, :high])
    k = Kernel(Interval(0.0, 1.0), obs_space,
        θ -> (o -> begin
            if o == :low;  log(1.0 - θ)
            elseif o == :mid; log(0.5)
            else; log(θ)
            end
        end),
        (θ, o) -> begin
            if o == :low;  log(1.0 - θ)
            elseif o == :mid; log(0.5)
            else; log(θ)
            end
        end)
    post = condition(prior, k, :mid)
    @assert post isa CategoricalMeasure "Expected CategoricalMeasure fallback, got $(typeof(post))"
    println("PASSED: BetaMeasure + 3-outcome kernel → CategoricalMeasure (grid fallback)")
end
println()

println("=" ^ 60)
println("TEST 26: BetaMeasure + Bernoulli kernel → conjugate (unchanged)")
println("=" ^ 60)

let
    prior = BetaMeasure(1.0, 1.0)
    obs_space = Finite([0.0, 1.0])
    k = Kernel(Interval(0.0, 1.0), obs_space,
        θ -> (o -> o == 1.0 ? log(θ) : log(1.0 - θ)),
        (θ, o) -> o == 1.0 ? log(θ) : log(1.0 - θ))
    post = condition(prior, k, 1.0)
    @assert post isa BetaMeasure "Expected BetaMeasure, got $(typeof(post))"
    @assert post.alpha ≈ 2.0 "Expected alpha=2.0, got $(post.alpha)"
    @assert post.beta ≈ 1.0 "Expected beta=1.0, got $(post.beta)"
    println("PASSED: BetaMeasure + Bernoulli kernel → conjugate update (α=$(post.alpha), β=$(post.beta))")
end
println()

println("=" ^ 60)
println("TEST 27: GaussianMeasure + Normal kernel → conjugate")
println("=" ^ 60)

let
    prior = GaussianMeasure(Euclidean(1), 0.0, 1.0)
    # Normal-Normal kernel: σ_obs = 1.0
    sigma_obs = 1.0
    k = Kernel(Euclidean(1), Euclidean(1),
        h -> (o -> -0.5 * ((o - h) / sigma_obs)^2),
        (h, o) -> -0.5 * ((o - h) / sigma_obs)^2,
        nothing, Dict(:sigma_obs => sigma_obs))
    post = condition(prior, k, 2.0)
    @assert post isa GaussianMeasure "Expected GaussianMeasure, got $(typeof(post))"
    @assert abs(post.mu - 1.0) < 1e-10 "Expected μ_post=1.0, got $(post.mu)"
    expected_sigma = 1.0 / sqrt(2.0)
    @assert abs(post.sigma - expected_sigma) < 1e-10 "Expected σ_post=$(expected_sigma), got $(post.sigma)"
    println("PASSED: N(0,1) + observe 2.0 through σ_obs=1 → N($(post.mu), $(round(post.sigma, digits=4)))")
end
println()

println("=" ^ 60)
println("TEST 28: Gaussian precision-weighted mean (strong prior dominates)")
println("=" ^ 60)

let
    prior = GaussianMeasure(Euclidean(1), 10.0, 0.5)  # tight prior at 10
    sigma_obs = 2.0
    k = Kernel(Euclidean(1), Euclidean(1),
        h -> (o -> -0.5 * ((o - h) / sigma_obs)^2),
        (h, o) -> -0.5 * ((o - h) / sigma_obs)^2,
        nothing, Dict(:sigma_obs => sigma_obs))
    post = condition(prior, k, 0.0)
    @assert post isa GaussianMeasure "Expected GaussianMeasure, got $(typeof(post))"
    # τ_prior = 4.0, τ_obs = 0.25, μ_post = (4*10 + 0.25*0) / 4.25 ≈ 9.412
    @assert post.mu > 9.0 "Strong prior should dominate: μ_post=$(post.mu)"
    @assert post.sigma < 0.5 "Posterior should be tighter than prior: σ_post=$(post.sigma)"
    println("PASSED: Strong prior N(10,0.5) + weak obs x=0 → μ_post=$(round(post.mu, digits=4)) (prior dominates)")
end
println()

println("=" ^ 60)
println("TEST 29: Gaussian variance always shrinks")
println("=" ^ 60)

let
    configs = [(0.0, 1.0, 1.0, 0.0), (5.0, 2.0, 0.5, 3.0), (-1.0, 0.1, 10.0, 100.0)]
    for (mu0, sig0, sig_obs, x) in configs
        prior = GaussianMeasure(Euclidean(1), mu0, sig0)
        k = Kernel(Euclidean(1), Euclidean(1),
            h -> (o -> -0.5 * ((o - h) / sig_obs)^2),
            (h, o) -> -0.5 * ((o - h) / sig_obs)^2,
            nothing, Dict(:sigma_obs => sig_obs))
        post = condition(prior, k, x)
        @assert post isa GaussianMeasure "Expected GaussianMeasure"
        @assert post.sigma < sig0 "Variance must shrink: σ_post=$(post.sigma) ≥ σ_prior=$(sig0)"
    end
    println("PASSED: Posterior σ < prior σ for all 3 configs")
end
println()

println("=" ^ 60)
println("TEST 30: GaussianMeasure + non-Gaussian kernel → grid fallback")
println("=" ^ 60)

let
    prior = GaussianMeasure(Euclidean(1), 0.0, 1.0)
    # Kernel to Finite target: NOT Normal-Normal
    obs_space = Finite([:a, :b])
    k = Kernel(Euclidean(1), obs_space,
        h -> (o -> o == :a ? log(0.5 + 0.3 * tanh(h)) : log(0.5 - 0.3 * tanh(h))),
        (h, o) -> o == :a ? log(0.5 + 0.3 * tanh(h)) : log(0.5 - 0.3 * tanh(h)))
    post = condition(prior, k, :a)
    @assert post isa CategoricalMeasure "Expected CategoricalMeasure fallback, got $(typeof(post))"
    println("PASSED: GaussianMeasure + Finite-target kernel → CategoricalMeasure (grid fallback)")
end
println()

println("=" ^ 60)
println("TEST 31: Gaussian kernel without params → grid fallback")
println("=" ^ 60)

let
    # Euclidean → Euclidean but no params → must fall through to grid
    prior = GaussianMeasure(Euclidean(1), 0.0, 1.0)
    k = Kernel(Euclidean(1), Euclidean(1),
        h -> (o -> -0.5 * ((o - h) / 1.0)^2),
        (h, o) -> -0.5 * ((o - h) / 1.0)^2)
    post = condition(prior, k, 1.0)
    @assert post isa CategoricalMeasure "Expected CategoricalMeasure (grid fallback), got $(typeof(post))"
    println("PASSED: Euclidean→Euclidean kernel without :sigma_obs param → grid fallback")
end
println()

println("=" ^ 60)
println("ALL TESTS PASSED")
println("=" ^ 60)
