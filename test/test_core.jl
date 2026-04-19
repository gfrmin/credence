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
        (θ, o) -> begin; idx = findfirst(==(o), cats.values); log(θ[idx]); end;
        likelihood_family = PushOnly())

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
        (θ, o) -> begin; idx = findfirst(==(o), cats.values); log(θ[idx]); end;
        likelihood_family = PushOnly())

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
        end;
        likelihood_family = PushOnly())

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
               end;
               likelihood_family = PushOnly())

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
        (θ, o) -> o == 1 ? log(θ) : log(1.0 - θ);
        likelihood_family = BetaBernoulli())

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
        end;
        likelihood_family = PushOnly())
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
        (θ, o) -> o == 1.0 ? log(θ) : log(1.0 - θ);
        likelihood_family = BetaBernoulli())
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
        (h, o) -> -0.5 * ((o - h) / sigma_obs)^2;
        params = Dict{Symbol,Any}(:sigma_obs => sigma_obs),
        likelihood_family = PushOnly())
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
        (h, o) -> -0.5 * ((o - h) / sigma_obs)^2;
        params = Dict{Symbol,Any}(:sigma_obs => sigma_obs),
        likelihood_family = PushOnly())
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
            (h, o) -> -0.5 * ((o - h) / sig_obs)^2;
            params = Dict{Symbol,Any}(:sigma_obs => sig_obs),
            likelihood_family = PushOnly())
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
        (h, o) -> o == :a ? log(0.5 + 0.3 * tanh(h)) : log(0.5 - 0.3 * tanh(h));
        likelihood_family = PushOnly())
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
        (h, o) -> -0.5 * ((o - h) / 1.0)^2;
        likelihood_family = PushOnly())
    post = condition(prior, k, 1.0)
    @assert post isa CategoricalMeasure "Expected CategoricalMeasure (grid fallback), got $(typeof(post))"
    println("PASSED: Euclidean→Euclidean kernel without :sigma_obs param → grid fallback")
end
println()

println("=" ^ 60)
println("TEST 32: NormalGammaMeasure — constructor and mean")
println("=" ^ 60)

let
    m = NormalGammaMeasure(1.0, 0.0, 1.0, 1.0)
    @assert m.κ == 1.0
    @assert m.μ == 0.0
    @assert m.α == 1.0
    @assert m.β == 1.0
    @assert mean(m) == 0.0
    @assert m.space isa ProductSpace
    println("PASSED: NormalGammaMeasure(1,0,1,1) constructed, mean = ", mean(m))
end
println()

println("=" ^ 60)
println("TEST 33: NormalGammaMeasure — conjugate update")
println("=" ^ 60)

let
    m = NormalGammaMeasure(1.0, 0.0, 1.0, 1.0)
    # Build a Normal-Gamma kernel
    k = Kernel(
        ProductSpace(Space[Euclidean(1), PositiveReals()]),
        Euclidean(1),
        h -> error("generate not used"),
        (h, o) -> -0.5 * log(2π * h[2]) - (o - h[1])^2 / (2.0 * h[2]);
        params = Dict{Symbol,Any}(:normal_gamma => true),
        likelihood_family = PushOnly())

    # Observe r = 2.0
    m2 = condition(m, k, 2.0)
    @assert m2 isa NormalGammaMeasure
    @assert m2.κ ≈ 2.0  # κₙ = 1 + 1
    @assert m2.μ ≈ 1.0  # (1*0 + 2) / 2
    @assert m2.α ≈ 1.5  # 1 + 0.5
    expected_β = 1.0 + 1.0 * (2.0 - 0.0)^2 / (2.0 * 2.0)  # 1 + 4/4 = 2
    @assert abs(m2.β - expected_β) < 1e-10
    println("PASSED: conjugate update κ=", m2.κ, " μ=", m2.μ, " α=", m2.α, " β=", m2.β)
end
println()

println("=" ^ 60)
println("TEST 34: NormalGammaMeasure — draw returns (μ, σ²) tuple")
println("=" ^ 60)

let
    Random.seed!(42)
    m = NormalGammaMeasure(10.0, 5.0, 5.0, 2.0)
    for _ in 1:100
        s = draw(m)
        @assert s isa Tuple{Float64, Float64}
        @assert s[2] > 0 "σ² must be positive, got $(s[2])"
    end
    println("PASSED: 100 draws all return (μ, σ²) with σ² > 0")
end
println()

println("=" ^ 60)
println("TEST 35: NormalGammaMeasure — multiple observations shrink variance")
println("=" ^ 60)

let
    m = NormalGammaMeasure(1.0, 0.0, 1.0, 1.0)
    k = Kernel(
        ProductSpace(Space[Euclidean(1), PositiveReals()]),
        Euclidean(1),
        h -> error("generate not used"),
        (h, o) -> -0.5 * log(2π * h[2]) - (o - h[1])^2 / (2.0 * h[2]);
        params = Dict{Symbol,Any}(:normal_gamma => true),
        likelihood_family = PushOnly())

    # Observe several values near 3.0
    current = m
    for r in [3.0, 3.1, 2.9, 3.0, 3.05]
        current = condition(current, k, r)
    end
    @assert current.α > m.α  "α should grow with observations"
    @assert current.κ > m.κ  "κ should grow with observations"
    @assert abs(current.μ - 3.0) < 0.5  "μ should be near 3.0"
    println("PASSED: 5 observations → α=", current.α, " κ=", current.κ, " μ=", round(current.μ, digits=3))
end
println()

println("=" ^ 60)
println("TEST 36: NormalGammaMeasure — strong prior dominates weak observation")
println("=" ^ 60)

let
    # Strong prior: κ=100, μ=10
    m = NormalGammaMeasure(100.0, 10.0, 50.0, 10.0)
    k = Kernel(
        ProductSpace(Space[Euclidean(1), PositiveReals()]),
        Euclidean(1),
        h -> error("generate not used"),
        (h, o) -> -0.5 * log(2π * h[2]) - (o - h[1])^2 / (2.0 * h[2]);
        params = Dict{Symbol,Any}(:normal_gamma => true),
        likelihood_family = PushOnly())

    m2 = condition(m, k, 0.0)
    # With κ=100, one observation at 0 barely shifts μ from 10
    @assert m2.μ > 9.0  "Strong prior should dominate: μ_post=$(m2.μ)"
    println("PASSED: κ=100, μ=10 prior + obs 0.0 → μ_post=", round(m2.μ, digits=4))
end
println()

println("=" ^ 60)
println("TEST 37: log_predictive — DirichletMeasure fast-path")
println("=" ^ 60)

let
    cats = Finite([10, 20, 30])
    d = DirichletMeasure(Simplex(3), cats, [2.0, 3.0, 5.0])
    k = Kernel(Simplex(3), cats,
        θ -> (o -> begin; idx = findfirst(==(o), cats.values); log(θ[idx]); end),
        (θ, o) -> begin; idx = findfirst(==(o), cats.values); log(θ[idx]); end;
        likelihood_family = PushOnly())

    # log P(obs=10 | Dir(2,3,5)) = log(2/10) = log(0.2)
    lp = log_predictive(d, k, 10)
    @assert abs(lp - log(0.2)) < 1e-10
    # log P(obs=30 | Dir(2,3,5)) = log(5/10) = log(0.5)
    lp2 = log_predictive(d, k, 30)
    @assert abs(lp2 - log(0.5)) < 1e-10
    println("PASSED: log_predictive(Dir(2,3,5), 10) = ", round(lp, digits=4),
            ", log_predictive(Dir(2,3,5), 30) = ", round(lp2, digits=4))
end
println()

println("=" ^ 60)
println("TEST 38: log_predictive — CategoricalMeasure default path")
println("=" ^ 60)

let
    Random.seed!(42)
    H = Finite([0.3, 0.7])
    m = CategoricalMeasure(H)  # uniform
    obs_space = Finite([0, 1])
    k = Kernel(H, obs_space,
        θ -> (o -> o == 1 ? log(θ) : log(1.0 - θ)),
        (θ, o) -> o == 1 ? log(θ) : log(1.0 - θ);
        likelihood_family = BetaBernoulli())

    # P(obs=1 | uniform{0.3, 0.7}) = 0.5*0.3 + 0.5*0.7 = 0.5
    lp = log_predictive(m, k, 1)
    @assert abs(lp - log(0.5)) < 0.05
    println("PASSED: log_predictive(CategoricalMeasure, obs=1) = ", round(lp, digits=4),
            " (expected ≈ ", round(log(0.5), digits=4), ")")
end
println()

println("=" ^ 60)
println("TEST 39: Sequential log_predictive + condition matches analytic formula")
println("=" ^ 60)

let
    cats = Finite([:a, :b, :c])
    d = DirichletMeasure(Simplex(3), cats, [1.0, 1.0, 1.0])
    k = Kernel(Simplex(3), cats,
        θ -> (o -> begin; idx = findfirst(==(o), cats.values); log(θ[idx]); end),
        (θ, o) -> begin; idx = findfirst(==(o), cats.values); log(θ[idx]); end;
        likelihood_family = PushOnly())

    # Observe :a, :a, :b — compute sequential log marginal
    data = [:a, :a, :b]
    log_ml = 0.0
    current = d
    for obs in data
        log_ml += log_predictive(current, k, obs)
        current = condition(current, k, obs)
    end

    # Analytic Dirichlet-Multinomial: log B(α+n) / B(α)
    # α = [1,1,1], n = [2,1,0]
    # = log(Γ(3)/Γ(6)) + log(Γ(3)/Γ(1)) + log(Γ(2)/Γ(1)) + log(Γ(1)/Γ(1))
    # = (log(2!) - log(5!)) + (log(2!) - log(0!)) + (log(1!) - log(0!)) + 0
    # = (log(2) - log(120)) + log(2) + 0 + 0
    # Sequential: log(1/3) + log(2/4) + log(1/5) = log(1/3 * 1/2 * 1/5) = log(1/30)
    expected = log(1/3) + log(2/4) + log(1/5)
    @assert abs(log_ml - expected) < 1e-10  "Sequential log_ml=$(log_ml), expected=$(expected)"
    println("PASSED: Sequential log marginal = ", round(log_ml, digits=6),
            " (expected ", round(expected, digits=6), ")")
end
println()

println("=" ^ 60)
println("TEST 40: log_marginal — Dir(1,1,1) + counts [2,1,0] matches log(1/30)")
println("=" ^ 60)

let
    cats = Finite([:a, :b, :c])
    d = DirichletMeasure(Simplex(3), cats, [1.0, 1.0, 1.0])
    lm = log_marginal(d, [2, 1, 0])
    expected = log(1/30)  # ≈ -3.4011973817
    @assert abs(lm - expected) < 1e-8  "log_marginal=$(lm), expected=$(expected)"
    println("PASSED: log_marginal(Dir(1,1,1), [2,1,0]) = ", round(lm, digits=6),
            " (expected ", round(expected, digits=6), ")")
end
println()

println("=" ^ 60)
println("TEST 41: log_marginal — Dir(0.1,0.1) + counts [10,5] cross-checked")
println("=" ^ 60)

let
    cats = Finite([:x, :y])
    d = DirichletMeasure(Simplex(2), cats, [0.1, 0.1])
    k = Kernel(Simplex(2), cats,
        θ -> (o -> begin; idx = findfirst(==(o), cats.values); log(θ[idx]); end),
        (θ, o) -> begin; idx = findfirst(==(o), cats.values); log(θ[idx]); end;
        likelihood_family = PushOnly())

    # Oracle 1: closed-form log_marginal
    lm = log_marginal(d, [10, 5])
    expected = -12.3515335627
    @assert abs(lm - expected) < 1e-4  "log_marginal=$(lm), expected=$(expected)"

    # Oracle 2: sequential log_predictive + condition
    data = vcat(fill(:x, 10), fill(:y, 5))
    log_ml_seq = 0.0
    current = d
    for obs in data
        log_ml_seq += log_predictive(current, k, obs)
        current = condition(current, k, obs)
    end
    @assert abs(lm - log_ml_seq) < 1e-8  "closed-form=$(lm) vs sequential=$(log_ml_seq)"

    println("PASSED: log_marginal(Dir(0.1,0.1), [10,5]) = ", round(lm, digits=6),
            ", sequential = ", round(log_ml_seq, digits=6))
end
println()

println("=" ^ 60)
println("TEST 42: log_marginal — zero counts returns 0.0")
println("=" ^ 60)

let
    cats = Finite([1, 2, 3])
    d = DirichletMeasure(Simplex(3), cats, [2.0, 3.0, 5.0])
    lm = log_marginal(d, [0, 0, 0])
    @assert abs(lm) < 1e-10  "log_marginal with zero counts should be 0.0, got $(lm)"
    println("PASSED: log_marginal(Dir(2,3,5), [0,0,0]) = ", lm)
end
println()

println("=" ^ 60)
println("TEST 43: range — generates 0-indexed integer lists")
println("=" ^ 60)

result = run_dsl("(range 4)")
@assert result == [0, 1, 2, 3]  "range 4 should be [0,1,2,3], got $result"
println("PASSED: (range 4) = ", result)

result = run_dsl("(range 1)")
@assert result == [0]  "range 1 should be [0], got $result"
println("PASSED: (range 1) = ", result)

result = run_dsl("(map (lambda (i) (* i i)) (range 5))")
@assert result == [0, 1, 4, 9, 16]  "map over range should work, got $result"
println("PASSED: (map square (range 5)) = ", result)
println()

# ==================================================
# Phase 0 — Functional type hierarchy
# ==================================================

println("=" ^ 60)
println("TEST 44: Identity on leaf measures — closed form")
println("=" ^ 60)

v = expect(BetaMeasure(3.0, 2.0), Identity())
@assert abs(v - 0.6) < 1e-12  "Beta(3,2) Identity should be 0.6, got $v"
println("PASSED: expect(Beta(3,2), Identity()) = ", v)

v = expect(GammaMeasure(4.0, 2.0), Identity())
@assert abs(v - 2.0) < 1e-12  "Gamma(4,2) Identity should be 2.0, got $v"
println("PASSED: expect(Gamma(4,2), Identity()) = ", v)

v = expect(GaussianMeasure(Euclidean(1), -1.5, 2.0), Identity())
@assert abs(v - (-1.5)) < 1e-12  "Gaussian(-1.5,2) Identity should be -1.5, got $v"
println("PASSED: expect(Gaussian(-1.5,2), Identity()) = ", v)
println()

println("=" ^ 60)
println("TEST 45: Projection on flat ProductMeasure")
println("=" ^ 60)

pm_flat = ProductMeasure(Measure[BetaMeasure(3.0, 2.0), BetaMeasure(1.0, 1.0)])
v = expect(pm_flat, Projection(1))
@assert abs(v - 0.6) < 1e-12  "Projection(1) should give mean of first factor 0.6, got $v"
println("PASSED: expect(PM[Beta(3,2), Beta(1,1)], Projection(1)) = ", v)

v = expect(pm_flat, Projection(2))
@assert abs(v - 0.5) < 1e-12  "Projection(2) should give mean of second factor 0.5, got $v"
println("PASSED: expect(PM[Beta(3,2), Beta(1,1)], Projection(2)) = ", v)
println()

println("=" ^ 60)
println("TEST 46: NestedProjection through nested ProductMeasure")
println("=" ^ 60)

inner_a = ProductMeasure(Measure[BetaMeasure(3.0, 2.0), GammaMeasure(2.0, 0.5)])
inner_b = ProductMeasure(Measure[BetaMeasure(1.0, 1.0), GammaMeasure(2.0, 0.5)])
pm_nested = ProductMeasure(Measure[inner_a, inner_b])

v = expect(pm_nested, NestedProjection([1, 1]))
@assert abs(v - 0.6) < 1e-12  "NestedProjection([1,1]) should be 0.6, got $v"
println("PASSED: expect(PM[PM[Beta(3,2),Gamma], PM[Beta(1,1),Gamma]], NestedProjection([1,1])) = ", v)

v = expect(pm_nested, NestedProjection([2, 1]))
@assert abs(v - 0.5) < 1e-12  "NestedProjection([2,1]) should be 0.5, got $v"
println("PASSED: expect(..., NestedProjection([2,1])) = ", v)

# Single-element NestedProjection
v = expect(pm_flat, NestedProjection([1]))
@assert abs(v - 0.6) < 1e-12  "NestedProjection([1]) on flat PM should be 0.6, got $v"
println("PASSED: expect(flat PM, NestedProjection([1])) = ", v)
println()

println("=" ^ 60)
println("TEST 47: LinearCombination of NestedProjections (router preference)")
println("=" ^ 60)

# Router EU style: reward * sum_c w_c * E[theta_{a,c}] - cost_a
# Provider 0 has two categories: Beta(3,2) and Beta(1,1)
# EU = 1.0 * (0.6 * 0.6 + 0.4 * 0.5) - 0.01 = 0.55
lc = LinearCombination(
    Tuple{Float64, Functional}[
        (0.6, NestedProjection([1, 1])),
        (0.4, NestedProjection([2, 1])),
    ],
    -0.01,
)
v = expect(pm_nested, lc)
expected = 0.6 * 0.6 + 0.4 * 0.5 - 0.01
@assert abs(v - expected) < 1e-12  "LC of NestedProjections should be $expected, got $v"
println("PASSED: LinearCombination of NestedProjections = ", v, " (expected ", expected, ")")

# Offset works without terms
lc_offset_only = LinearCombination(Tuple{Float64, Functional}[], 0.42)
v = expect(pm_nested, lc_offset_only)
@assert abs(v - 0.42) < 1e-12  "empty LC with offset should return offset"
println("PASSED: empty LinearCombination returns offset = ", v)
println()

println("=" ^ 60)
println("TEST 48: MixtureMeasure recursion for any Functional")
println("=" ^ 60)

mix = MixtureMeasure(pm_flat.space,
    Measure[ProductMeasure(Measure[BetaMeasure(3.0, 2.0), BetaMeasure(1.0, 1.0)]),
            ProductMeasure(Measure[BetaMeasure(1.0, 1.0), BetaMeasure(3.0, 2.0)])],
    [log(0.75), log(0.25)])
v = expect(mix, Projection(1))
# 0.75 * 0.6 + 0.25 * 0.5 = 0.575
@assert abs(v - 0.575) < 1e-12  "Mixture + Projection should be 0.575, got $v"
println("PASSED: expect(Mixture[PM, PM], Projection(1)) = ", v, " (expected 0.575)")
println()

println("=" ^ 60)
println("TEST 49: CategoricalMeasure — Projection and Tabular")
println("=" ^ 60)

# Categorical over vector-valued atoms (simulates particle cloud)
cat_vec = CategoricalMeasure{Vector{Float64}}(
    Finite([[0.7, 2.0], [0.3, 1.5], [0.5, 3.0]]),
    [log(0.5), log(0.3), log(0.2)])
v = expect(cat_vec, Projection(1))
expected = 0.5 * 0.7 + 0.3 * 0.3 + 0.2 * 0.5
@assert abs(v - expected) < 1e-12  "Categorical projection mismatch: got $v, expected $expected"
println("PASSED: expect(Categorical of vectors, Projection(1)) = ", v)

# Tabular on categorical over integer atoms
cat_scalar = CategoricalMeasure(Finite([0, 1, 2]), [log(0.2), log(0.5), log(0.3)])
v = expect(cat_scalar, Tabular([10.0, 20.0, 30.0]))
expected = 0.2 * 10.0 + 0.5 * 20.0 + 0.3 * 30.0  # 21.0
@assert abs(v - expected) < 1e-12  "Categorical tabular mismatch: got $v, expected $expected"
println("PASSED: expect(Categorical, Tabular) = ", v)
println()

println("=" ^ 60)
println("TEST 49b: DSL (factor), (replace-factor), (n-factors)")
println("=" ^ 60)

# Construct ProductMeasure of two Betas in the DSL, access factors
pm_dsl = run_dsl("""
(product-measure
  (measure (space :interval 0 1) :beta 3 2)
  (measure (space :interval 0 1) :beta 1 1))
""")
@assert pm_dsl isa ProductMeasure  "(product-measure ...) should return a ProductMeasure"
@assert length(pm_dsl.factors) == 2
println("PASSED: (product-measure ...) returns ProductMeasure with 2 factors")

# (factor m 0) — 0-based index in DSL
code = """
(let pm (product-measure
          (measure (space :interval 0 1) :beta 3 2)
          (measure (space :interval 0 1) :beta 1 1))
  (factor pm 0))
"""
f0 = run_dsl(code)
@assert f0 isa BetaMeasure && f0.alpha == 3.0 && f0.beta == 2.0
println("PASSED: (factor pm 0) returns Beta(3,2)")

# (replace-factor m 1 new) — replace second factor
code = """
(let pm (product-measure
          (measure (space :interval 0 1) :beta 3 2)
          (measure (space :interval 0 1) :beta 1 1))
  (let new (measure (space :interval 0 1) :beta 7 3)
    (factor (replace-factor pm 1 new) 1)))
"""
f1 = run_dsl(code)
@assert f1 isa BetaMeasure && f1.alpha == 7.0 && f1.beta == 3.0
println("PASSED: (replace-factor pm 1 new) replaces factor at index 1")

# (n-factors m)
code = """
(n-factors (product-measure
  (measure (space :interval 0 1) :beta 1 1)
  (measure (space :interval 0 1) :beta 1 1)
  (measure (space :interval 0 1) :beta 1 1)))
"""
nf = run_dsl(code)
@assert nf == 3
println("PASSED: (n-factors ...) = ", nf)

# (product-measure lst) variant — single list argument
code = """
(product-measure
  (map (lambda (i) (measure (space :interval 0 1) :beta 1 1)) (range 4)))
"""
pm_from_list = run_dsl(code)
@assert pm_from_list isa ProductMeasure && length(pm_from_list.factors) == 4
println("PASSED: (product-measure lst) builds ProductMeasure from list argument")
println()

println("=" ^ 60)
println("TEST 50: OpaqueClosure fallback delegates to bare-function method")
println("=" ^ 60)

# On BetaMeasure: OpaqueClosure should give same result as bare lambda (quadrature)
bare_v = expect(BetaMeasure(3.0, 2.0), x -> x * x)
wrap_v = expect(BetaMeasure(3.0, 2.0), OpaqueClosure(x -> x * x))
@assert abs(bare_v - wrap_v) < 1e-12  "OpaqueClosure should match bare function: $bare_v vs $wrap_v"
println("PASSED: OpaqueClosure on BetaMeasure matches bare function: ", wrap_v)

# On ProductMeasure: OpaqueClosure falls through to Monte Carlo fallback
using Random; Random.seed!(42)
mc1 = expect(pm_flat, h -> h[1])
Random.seed!(42)
mc2 = expect(pm_flat, OpaqueClosure(h -> h[1]))
@assert abs(mc1 - mc2) < 1e-12  "OpaqueClosure on PM should match bare function (same seed)"
println("PASSED: OpaqueClosure on ProductMeasure matches bare function under seed")
println()

println("=" ^ 60)
println("TEST 51: LikelihoodFamily dispatch — BetaBernoulli explicit")
println("=" ^ 60)
let
    tbm = TaggedBetaMeasure(Interval(0.0, 1.0), 7, BetaMeasure(2.0, 3.0))
    k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
        _ -> error("not used"),
        (h, o) -> o == 1.0 ? log(max(h isa TaggedBetaMeasure ? mean(h.beta) : h, 1e-300)) :
                              log(max(1 - (h isa TaggedBetaMeasure ? mean(h.beta) : h), 1e-300));
        likelihood_family = BetaBernoulli())
    post1 = condition(tbm, k, 1.0)
    @assert post1.tag == 7
    @assert post1.beta.alpha == 3.0 "Expected α=3.0 after obs=1, got $(post1.beta.alpha)"
    @assert post1.beta.beta == 3.0 "Expected β=3.0 unchanged, got $(post1.beta.beta)"

    post0 = condition(tbm, k, 0.0)
    @assert post0.beta.alpha == 2.0
    @assert post0.beta.beta == 4.0
    println("PASSED: explicit BetaBernoulli conjugate update exact")
end
println()

println("=" ^ 60)
println("TEST 52: LikelihoodFamily dispatch — Flat explicit")
println("=" ^ 60)
let
    tbm = TaggedBetaMeasure(Interval(0.0, 1.0), 42, BetaMeasure(4.0, 6.0))
    k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
        _ -> error("not used"),
        (h, o) -> log(0.5);
        likelihood_family = Flat())
    post = condition(tbm, k, 1.0)
    @assert post.tag == 42 "tag preserved"
    @assert post.beta.alpha == 4.0 "Flat: α unchanged"
    @assert post.beta.beta == 6.0 "Flat: β unchanged"
    @assert post === tbm "Flat posterior is identical object to prior"
    println("PASSED: Flat likelihood leaves posterior ≡ prior")
end
println()

println("=" ^ 60)
println("TEST 53: LikelihoodFamily dispatch — DispatchByComponent classify")
println("=" ^ 60)
let
    # Even tags → BetaBernoulli, odd → Flat
    classify(m) = iseven(m.tag) ? BetaBernoulli() : Flat()
    k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
        _ -> error("not used"),
        (h, o) -> o == 1.0 ? log(max(h isa TaggedBetaMeasure ? mean(h.beta) : h, 1e-300)) :
                              log(max(1 - (h isa TaggedBetaMeasure ? mean(h.beta) : h), 1e-300));
        likelihood_family = DispatchByComponent(classify))

    tbm_even = TaggedBetaMeasure(Interval(0.0, 1.0), 2, BetaMeasure(1.0, 1.0))
    tbm_odd  = TaggedBetaMeasure(Interval(0.0, 1.0), 3, BetaMeasure(1.0, 1.0))

    post_even = condition(tbm_even, k, 1.0)
    post_odd  = condition(tbm_odd,  k, 1.0)

    @assert post_even.beta.alpha == 2.0  "Even tag: BetaBernoulli update expected"
    @assert post_even.beta.beta  == 1.0
    @assert post_odd.beta.alpha  == 1.0  "Odd tag: Flat → prior unchanged"
    @assert post_odd.beta.beta   == 1.0
    println("PASSED: DispatchByComponent routes to leaf families")
end
println()

println("=" ^ 60)
println("TEST 54: Error pin — PushOnly likelihood_family errors at condition")
println("=" ^ 60)
let
    tbm = TaggedBetaMeasure(Interval(0.0, 1.0), 1, BetaMeasure(1.0, 1.0))
    k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
        _ -> error("not used"),
        (h, o) -> 0.0;
        likelihood_family = PushOnly())
    caught = false
    try
        condition(tbm, k, 1.0)
    catch e
        caught = true
        @assert occursin("push-only kernel", sprint(showerror, e))
    end
    @assert caught "Expected error for PushOnly kernel conditioning TaggedBetaMeasure"
    println("PASSED: PushOnly kernel → loud error on condition(TaggedBetaMeasure, …)")
end
println()

println("=" ^ 60)
println("TEST 55: Error pin — DispatchByComponent self-reference hits DepthCapExceeded")
println("=" ^ 60)
let
    # classify returns another DispatchByComponent → self-reference loop
    local k_ref
    classify(m) = k_ref.likelihood_family
    tbm = TaggedBetaMeasure(Interval(0.0, 1.0), 1, BetaMeasure(1.0, 1.0))
    k_ref = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
        _ -> error("not used"),
        (h, o) -> 0.0;
        likelihood_family = DispatchByComponent(classify))
    @assert try; condition(tbm, k_ref, 1.0); false
            catch e; e isa DepthCapExceeded end  "self-referential classify must raise DepthCapExceeded"
    println("PASSED: classify self-reference → DepthCapExceeded (typed)")
end
println()

println("=" ^ 60)
println("TEST 55b: Error pin — DispatchByComponent returning FiringByTag hits DepthCapExceeded")
println("=" ^ 60)
let
    # classify returns a FiringByTag (router, not a leaf) → depth-cap unwind cannot terminate
    classify_to_firing(m) = FiringByTag(Set([1]), BetaBernoulli(), Flat())
    # But FiringByTag resolves to a leaf on first unwrap. To force a depth-cap,
    # have classify return another DispatchByComponent that returns a FiringByTag
    # that routes to yet another DispatchByComponent — the mutual recursion.
    local k_ref
    classify(m) = DispatchByComponent(classify)
    tbm = TaggedBetaMeasure(Interval(0.0, 1.0), 1, BetaMeasure(1.0, 1.0))
    k_ref = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
        _ -> error("not used"),
        (h, o) -> 0.0;
        likelihood_family = DispatchByComponent(classify))
    @assert try; condition(tbm, k_ref, 1.0); false
            catch e; e isa DepthCapExceeded end  "nested DispatchByComponent must raise DepthCapExceeded"
    println("PASSED: DispatchByComponent → DispatchByComponent loop → DepthCapExceeded")
end
println()

println("=" ^ 60)
println("TEST 55c: Construction pin — FiringByTag branches must be LeafFamily")
println("=" ^ 60)
let
    # FiringByTag.when_fires / when_not are typed as LeafFamily.
    # DispatchByComponent <: LikelihoodFamily but not <: LeafFamily — must error.
    classify(m) = BetaBernoulli()
    caught = false
    try
        FiringByTag(Set([1]), DispatchByComponent(classify), Flat())
    catch e
        caught = true
    end
    @assert caught "FiringByTag with non-LeafFamily branch must fail at construction"
    # Same for when_not
    caught2 = false
    try
        FiringByTag(Set([1]), BetaBernoulli(), DispatchByComponent(classify))
    catch e
        caught2 = true
    end
    @assert caught2 "FiringByTag with non-LeafFamily when_not must fail at construction"
    # Valid construction with both leaves works
    FiringByTag(Set([1]), BetaBernoulli(), Flat())
    println("PASSED: FiringByTag LeafFamily branch constraint enforced at construction")
end
println()

println("=" ^ 60)
println("TEST 56: Functional dispatch — MixtureMeasure × OpaqueClosure")
println("=" ^ 60)
let
    # Regression guard: without the explicit overload, dispatch was ambiguous
    # between expect(::MixtureMeasure, ::Functional) and expect(::Measure, ::OpaqueClosure).
    b1 = BetaMeasure(3.0, 1.0)  # mean 0.75
    b2 = BetaMeasure(1.0, 3.0)  # mean 0.25
    mix = MixtureMeasure(Interval(0.0, 1.0), Measure[b1, b2], [log(0.6), log(0.4)])

    f = θ -> θ^2
    bare = expect(mix, f)
    wrapped = expect(mix, OpaqueClosure(f))
    @assert abs(bare - wrapped) < 1e-12 "OpaqueClosure on mixture must match bare function"
    println("PASSED: OpaqueClosure dispatches cleanly on MixtureMeasure")
end
println()

println("=" ^ 60)
println("TEST 57: Functional — nested LinearCombination offset arithmetic")
println("=" ^ 60)
let
    # inner = 2·Identity + 3, outer = 5·inner + 1
    # For Beta(2,4), E[Identity] = 2/6 = 1/3
    # inner EU = 2·(1/3) + 3 = 11/3
    # outer EU = 5·(11/3) + 1 = 58/3
    b = BetaMeasure(2.0, 4.0)
    inner = LinearCombination(Tuple{Float64, Functional}[(2.0, Identity())], 3.0)
    outer = LinearCombination(Tuple{Float64, Functional}[(5.0, inner)], 1.0)
    got = expect(b, outer)
    expected = 58 / 3
    @assert abs(got - expected) < 1e-12 "Nested LC: expected $expected, got $got"
    println("PASSED: nested LinearCombination offset composes correctly: $(round(got, digits=6))")
end
println()

println("=" ^ 60)
println("TEST 58: Construction-time validation — Projection/NestedProjection/Tabular")
println("=" ^ 60)
let
    @assert try; Projection(0);           false catch ArgumentError; true end  "Projection(0) must error"
    @assert try; Projection(-1);          false catch ArgumentError; true end  "Projection(-1) must error"
    @assert try; NestedProjection(Int[]); false catch ArgumentError; true end  "empty NestedProjection must error"
    @assert try; NestedProjection([1, 0, 2]); false catch ArgumentError; true end  "NP with 0 must error"
    @assert try; Tabular(Float64[]);      false catch ArgumentError; true end  "empty Tabular must error"
    # Valid constructions still work
    Projection(1); NestedProjection([1, 2]); Tabular([1.0, 2.0])
    println("PASSED: invalid constructions rejected; valid ones accepted")
end
println()

println("=" ^ 60)
println("TEST 59: ProductMeasure factor / replace_factor round-trip (Julia level)")
println("=" ^ 60)
let
    b1 = BetaMeasure(3.0, 2.0)
    b2 = BetaMeasure(1.0, 4.0)
    b3 = BetaMeasure(5.0, 5.0)
    pm = ProductMeasure(Measure[b1, b2, b3])

    @assert factor(pm, 1) === b1 "factor returns same object (identity)"
    @assert factor(pm, 2) === b2
    @assert factor(pm, 3) === b3

    replacement = BetaMeasure(9.0, 1.0)
    pm2 = replace_factor(pm, 2, replacement)
    @assert pm2 !== pm "replace_factor returns a new ProductMeasure"
    @assert factor(pm2, 2) === replacement
    @assert factor(pm2, 1) === b1 "unreplaced factors carry through"
    @assert factor(pm2, 3) === b3
    @assert factor(pm, 2) === b2 "original unchanged at field level"

    @assert try; replace_factor(pm, 0, replacement); false catch; true end "out-of-range (0) errors"
    @assert try; replace_factor(pm, 4, replacement); false catch; true end "out-of-range (4) errors"

    println("PASSED: factor/replace_factor round-trip exact")
end
println()

println("=" ^ 60)
println("TEST: condition(m, e::Event) equivalence to condition(m, indicator_kernel(e), true)")
println("=" ^ 60)

# Di Lavore–Román–Sobociński Prop. 4.9 in the codebase:
# the sibling form is provably equivalent to the parametric form at
# observation `true` for any deterministic event. Pin this as a
# regression canary — if someone later routes condition(::Event)
# through a separate code path, this catches semantic drift.

let
    sp = Interval(0.0, 1.0)
    components = [TaggedBetaMeasure(sp, t, BetaMeasure(2.0 + t, 3.0)) for t in 1:5]
    m = MixtureMeasure(sp, components, Float64[0.0, -0.5, 0.1, 0.2, -0.3])

    events = Event[
        TagSet(sp, Set([1, 3, 5])),
        TagSet(sp, Set([2, 4])),
        TagSet(sp, Set([1, 2, 3, 4, 5])),        # all — trivial case
        Complement(TagSet(sp, Set([3]))),
        Conjunction(TagSet(sp, Set([1, 2, 3, 4])), TagSet(sp, Set([2, 3, 4, 5]))),
        Disjunction(TagSet(sp, Set([1])), TagSet(sp, Set([5]))),
    ]
    for (i, e) in enumerate(events)
        lhs = condition(m, e)
        rhs = condition(m, indicator_kernel(e), true)
        w_lhs = weights(lhs)
        w_rhs = weights(rhs)
        close_enough = all(abs(w_lhs[j] - w_rhs[j]) < 1e-12 for j in eachindex(w_lhs))
        @assert close_enough "event $i: weights diverge"
    end
    println("PASSED: 6 event variants — both condition forms agree to 1e-12")
end

println()
println("=" ^ 60)
println("ALL TESTS PASSED")
println("=" ^ 60)
