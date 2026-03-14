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

# ─── Test weighted_sum directly ───

println("=" ^ 60)
println("TEST 2b: weighted_sum (Julia-level)")
println("=" ^ 60)

b_ws = Belief([1, 2, 3])
ws_result = weighted_sum(b_ws, x -> x * 2.0)
println("weighted_sum([1,2,3], x->2x) = ", ws_result, " (expected: 4.0)")
@assert abs(ws_result - 4.0) < 1e-10 "weighted_sum failed"
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
                (let voi-manual (- (+ (* 0.5 eu1) (* 0.5 eu2)) current-eu)
                  (do
                    (print voi-manual)
                    (print (> voi-manual 0.3))))))))))))
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

# ─── Test new supporting forms ───

println("=" ^ 60)
println("TEST 7: map, fold, first, max, variadic +/*")
println("=" ^ 60)

map_test = """(map (lambda (x) (* x x)) (list 1 2 3 4))"""
result = run_dsl(map_test)
println("map square [1,2,3,4] = ", result, " (expected: [1,4,9,16])")
@assert result == [1, 4, 9, 16] "map failed"

fold_test = """(fold + (list 1 2 3 4 5))"""
result = run_dsl(fold_test)
println("fold + [1,2,3,4,5] = ", result, " (expected: 15)")
@assert result == 15 "fold failed"

first_test = """(first (list 42 99))"""
result = run_dsl(first_test)
println("first [42,99] = ", result, " (expected: 42)")
@assert result == 42 "first failed"

max_test = """(max 3 7 2 9 1)"""
result = run_dsl(max_test)
println("max 3 7 2 9 1 = ", result, " (expected: 9)")
@assert result == 9 "max failed"

variadic_plus = """(+ 1 2 3 4)"""
result = run_dsl(variadic_plus)
println("(+ 1 2 3 4) = ", result, " (expected: 10)")
@assert result == 10 "variadic + failed"

variadic_times = """(* 2 3 4)"""
result = run_dsl(variadic_times)
println("(* 2 3 4) = ", result, " (expected: 24)")
@assert result == 24 "variadic * failed"

println()

# ─── Test define ───

println("=" ^ 60)
println("TEST 8: define (top-level only)")
println("=" ^ 60)

define_test = """
(define x 42)
(define double (lambda (n) (* n 2)))
(double x)
"""
result = run_dsl(define_test)
println("define x=42, double(x) = ", result, " (expected: 84)")
@assert result == 84 "define failed"

# define inside lambda should error
try
    run_dsl("""(let f (lambda () (define bad 1)) (f))""")
    println("ERROR: define inside lambda should have thrown!")
catch e
    println("Correctly rejected define inside lambda: ", e.msg)
end

println()

# ─── Test weighted-sum in DSL ───

println("=" ^ 60)
println("TEST 9: weighted-sum in DSL")
println("=" ^ 60)

ws_dsl_test = """
(let b (belief 1 2 3)
  (weighted-sum b (lambda (h) (* h 2.0))))
"""
result = run_dsl(ws_dsl_test)
println("weighted-sum belief(1,2,3) h->2h = ", result, " (expected: 4.0)")
@assert abs(result - 4.0) < 1e-10 "weighted-sum DSL failed"

println()

# ─── Test stdlib VOI matches manual computation ───

println("=" ^ 60)
println("TEST 10: stdlib VOI matches manual VOI")
println("=" ^ 60)

# Manual VOI computation (same as TEST 4)
manual_voi = """
(let prior (belief 1 2)
  (let u (lambda (truth act) (if (= truth act) 1.0 -1.0))
    (let lik (lambda (truth obs) (if (= truth obs) (log 0.8) (log 0.2)))
      (let post1 (update prior 1 lik)
        (let post2 (update prior 2 lik)
          (- (+ (* 0.5 (eu post1 1 u)) (* 0.5 (eu post2 2 u))) 0.0))))))
"""
manual_result = run_dsl(manual_voi)

# Stdlib VOI computation
stdlib_voi = """
(let prior (belief 1 2)
  (let u (lambda (truth act) (if (= truth act) 1.0 -1.0))
    (let lik (lambda (truth obs) (if (= truth obs) (log 0.8) (log 0.2)))
      (voi prior (list 1 2) (list 1 2) u lik))))
"""
stdlib_result = run_dsl(stdlib_voi)

println("Manual VOI:  ", manual_result)
println("Stdlib VOI:  ", stdlib_result)
println("Difference:  ", abs(manual_result - stdlib_result))
@assert abs(manual_result - stdlib_result) < 1e-10 "stdlib VOI does not match manual computation"

println()

# ─── Test credence engine ───

println("=" ^ 60)
println("TEST 11: Credence engine (full example)")
println("=" ^ 60)

credence_source = read(joinpath(@__DIR__, "..", "examples", "credence_engine.bdsl"), String)
result = run_dsl(credence_source)
println("Credence engine completed successfully.")

println()
println("=" ^ 60)
println("ALL TESTS PASSED")
println("=" ^ 60)
