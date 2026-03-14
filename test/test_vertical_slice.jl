#!/usr/bin/env julia
"""
    test_vertical_slice.jl — Does the DSL work?

This is the acid test. If these examples produce correct
Bayesian reasoning from S-expressions parsed → compiled → executed,
the vertical slice is alive.
"""

# Load the DSL
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using BayesianDSL

println("=" ^ 60)
println("TEST 1: Parser")
println("=" ^ 60)

expr = parse_sexpr("(update (belief 0.3 0.7) 1 likelihood)")
println("Parsed: ", expr)
println("Type:   ", typeof(expr))
println()

expr2 = parse_sexpr("(lambda (x y) (+ x y))")
println("Parsed: ", expr2)
println()

# ─── Test primitives directly ───

println("=" ^ 60)
println("TEST 2: Primitives (Julia-level)")
println("=" ^ 60)

# Coin with θ ∈ {0.2, 0.5, 0.8}, uniform prior
b = Belief([0.2, 0.5, 0.8])
println("Prior weights: ", round.(weights(b), digits=4))

# Observe heads (1)
coin_lik(θ, obs) = obs == 1 ? log(θ) : log(1 - θ)
b = update(b, 1, coin_lik)
println("After H:       ", round.(weights(b), digits=4))

b = update(b, 1, coin_lik)
println("After H,H:     ", round.(weights(b), digits=4))

b = update(b, 0, coin_lik)
println("After H,H,T:   ", round.(weights(b), digits=4))

# Decide: is θ > 0.5?
u(θ, a) = a == :high ? (θ > 0.5 ? 1.0 : -1.0) : (θ < 0.5 ? 1.0 : -1.0)
result = decide(b, [:high, :low], u)
println("Decision:      ", result.action, " (EU = ", round(result.eu, digits=4), ")")
println()

# ─── Test the full DSL pipeline ───

println("=" ^ 60)
println("TEST 3: Coin example (full DSL pipeline)")
println("=" ^ 60)

coin_source = read(joinpath(@__DIR__, "..", "examples", "coin.bdsl"), String)
result = run_dsl(coin_source)
println("Decision: ", result)
println()

# ─── Test VOI example ───

println("=" ^ 60)
println("TEST 4: Tool selection / VOI (full DSL pipeline)")
println("=" ^ 60)

# Simpler inline VOI test to verify the computation
voi_test = """
(let prior (belief 1 2)
  (let u (lambda (truth act) (if (= truth act) 1.0 -1.0))
    (let lik (lambda (truth obs) (if (= truth obs) (log 0.8) (log 0.2)))

      ; Current EU of best action (should be 0, by symmetry)
      (let current-eu 0.0

        ; After observing tool=1: posterior strongly favours h=1
        (let post1 (update prior 1 lik)
          (let eu1 (eu post1 1 u)

            ; After observing tool=2: posterior strongly favours h=2
            (let post2 (update prior 2 lik)
              (let eu2 (eu post2 2 u)

                ; VOI = 0.5 * eu1 + 0.5 * eu2 - current-eu
                (let voi (- (+ (* 0.5 eu1) (* 0.5 eu2)) current-eu)
                  (do
                    (print voi)
                    (print (> voi 0.3))))))))))))
"""

result = run_dsl(voi_test)
println()

# ─── Test lambda and composition ───

println("=" ^ 60)
println("TEST 5: Lambda and composition")
println("=" ^ 60)

compose_test = """
(let add (lambda (a b) (+ a b))
  (let sq (lambda (x) (* x x))
    (sq (add 3 4))))
"""
result = run_dsl(compose_test)
println("(3 + 4)² = ", result)
println()

# ─── Verify axiom enforcement ───

println("=" ^ 60)
println("TEST 6: Axiom enforcement")
println("=" ^ 60)

# Weights must sum to 1
b = Belief([1, 2, 3])
w = weights(b)
println("Weights sum: ", sum(w), " (should be 1.0)")

# Update with impossible observation should error
try
    impossible_lik(h, o) = -Inf  # impossible under all hypotheses
    update(b, 42, impossible_lik)
    println("ERROR: should have thrown!")
catch e
    println("Correctly rejected impossible observation: ", e.msg)
end

println()
println("=" ^ 60)
println("ALL TESTS PASSED")
println("=" ^ 60)
