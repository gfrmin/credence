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

# Partial -Inf elimination: one hypothesis ruled out, others survive
partial_b = Belief([:a, :b, :c])
partial_lik(h, o) = h == :a ? -Inf : 0.0
partial_post = update(partial_b, 1, partial_lik)
pw = weights(partial_post)
println("Partial -Inf elimination weights: ", pw)
@assert pw[1] ≈ 0.0 atol=1e-15 "eliminated hypothesis should have weight ≈ 0"
@assert pw[2] ≈ 0.5 atol=1e-10 "surviving hypotheses should share posterior equally"
@assert pw[3] ≈ 0.5 atol=1e-10 "surviving hypotheses should share posterior equally"
println("Partial -Inf elimination: PASSED")

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

# ─── Test new supporting forms: sample, second, nth ───

println("=" ^ 60)
println("TEST 12: sample, second, nth")
println("=" ^ 60)

# sample: deterministic case (single hypothesis)
result = run_dsl("(sample (belief 42))")
@assert result == 42 "sample from single-hypothesis belief should return that hypothesis"
println("sample (belief 42) = ", result, " (expected: 42)")

# sample: statistical test (50 draws from skewed belief)
counts = run_dsl("""
(let b (update (belief 1 2) 1 (lambda (h o) (if (= h o) (log 0.9) (log 0.1))))
  (let draws (map (lambda (_) (sample b)) (list 1 2 3 4 5 6 7 8 9 10
                                                11 12 13 14 15 16 17 18 19 20
                                                21 22 23 24 25 26 27 28 29 30
                                                31 32 33 34 35 36 37 38 39 40
                                                41 42 43 44 45 46 47 48 49 50))
    (fold + (map (lambda (d) (if (= d 1) 1.0 0.0)) draws))))
""")
@assert counts > 30 "sample distribution seems wrong: expected ~45/50 to be 1, got $counts"
println("sample statistical test: $counts/50 draws were h=1 (expected ~45)")

# second
@assert run_dsl("(second (list 10 20 30))") == 20 "second failed"
println("second (list 10 20 30) = 20")

# nth (0-based)
@assert run_dsl("(nth (list 10 20 30 40) 0)") == 10 "nth 0 failed"
@assert run_dsl("(nth (list 10 20 30 40) 3)") == 40 "nth 3 failed"
println("nth: 0→10, 3→40")

println()

# ─── Test stdlib combinators: best-action, bernoulli-lik, answer-lik, update-reliability ───

println("=" ^ 60)
println("TEST 13: Stdlib combinators (best-action, likelihoods, update-reliability)")
println("=" ^ 60)

# best-action agrees with decide
ba_test = run_dsl("""
(let b (update (belief 1 2 3) 2 (lambda (h o) (if (= h o) (log 0.8) (log 0.1))))
  (best-action b (list 1 2 3) (lambda (h a) (if (= h a) 1.0 -1.0))))
""")
@assert ba_test == 2 "best-action should pick hypothesis 2 after strong evidence"
println("best-action after update on h=2: ", ba_test, " (expected: 2)")

# bernoulli-lik
@assert abs(run_dsl("(bernoulli-lik 0.8 1)") - log(0.8)) < 1e-10 "bernoulli-lik correct failed"
@assert abs(run_dsl("(bernoulli-lik 0.8 0)") - log(0.2)) < 1e-10 "bernoulli-lik incorrect failed"
println("bernoulli-lik: correct")

# answer-lik
@assert abs(run_dsl("(answer-lik 2 2 0.8 4)") - log(0.8)) < 1e-10 "answer-lik correct failed"
@assert abs(run_dsl("(answer-lik 2 1 0.8 4)") - log(0.2/3)) < 1e-10 "answer-lik incorrect failed"
println("answer-lik: correct")

# update-reliability: after observing correct, high-r hypotheses gain weight
rel_test = run_dsl("""
(let prior (belief 0.2 0.5 0.8)
  (let posterior (update-reliability prior 1)
    (list (weights prior) (weights posterior))))
""")
prior_w = rel_test[1]
post_w = rel_test[2]
@assert post_w[3] > prior_w[3] "r=0.8 should gain weight after correct observation"
@assert post_w[1] < prior_w[1] "r=0.2 should lose weight after correct observation"
println("update-reliability: r=0.8 gained weight, r=0.2 lost weight after correct obs")

println()

# ─── Load agent env for tests 14-15 ───
agent_env = load_dsl(read(joinpath(@__DIR__, "..", "examples", "credence_agent.bdsl"), String))

# ─── Test joint belief update ───

println("=" ^ 60)
println("TEST 14: Joint (answer, reliability) belief update")
println("=" ^ 60)

joint_test = run_dsl("""
(let joint (belief (list 0 0.3) (list 0 0.7)
                   (list 1 0.3) (list 1 0.7)
                   (list 2 0.3) (list 2 0.7)
                   (list 3 0.3) (list 3 0.7))
  (let updated (update-on-response joint 2 4)
    (weights updated)))
""", env=copy(agent_env))
# Hypotheses: (0,0.3),(0,0.7),(1,0.3),(1,0.7),(2,0.3),(2,0.7),(3,0.3),(3,0.7)
@assert joint_test[6] > joint_test[5] "higher reliability should have more weight for correct answer"
@assert joint_test[5] > joint_test[1] "correct answer should dominate over wrong answer"
println("Joint update: (answer=2, r=0.7) has highest weight after response=2")

println()

# ─── Test agent-step integration ───

println("=" ^ 60)
println("TEST 15: agent-step integration")
println("=" ^ 60)

# Uniform prior over 4 answers, one tool → VOI should exceed cost → agent queries
agent_test = run_dsl("""
(let joint (belief (list 0 0.5) (list 0 0.8)
                   (list 1 0.5) (list 1 0.8)
                   (list 2 0.5) (list 2 0.8)
                   (list 3 0.5) (list 3 0.8))
  (let tool-infos (list (list joint 2.0 0.9 0))
    (let util (lambda (h a) (if (= a -1) 0.0 (if (= (first h) a) 10.0 -5.0)))
      (agent-step tool-infos (list 0 1 2 3) (list -1 0 1 2 3) 4 util))))
""", env=copy(agent_env))
@assert agent_test[1] == 2 "agent should query (action type 2) when VOI > cost"
@assert agent_test[2] == 0 "agent should query tool 0 (only tool available)"
println("agent-step with uniform prior: queries tool 0 (VOI > cost)")

# After strong evidence: agent should submit
submit_test = run_dsl("""
(let joint (belief (list 0 0.5) (list 0 0.8)
                   (list 1 0.5) (list 1 0.8)
                   (list 2 0.5) (list 2 0.8)
                   (list 3 0.5) (list 3 0.8))
  (let updated (update-on-response joint 2 4)
    (let tool-infos (list (list updated 2.0 0.9 0))
      (let util (lambda (h a) (if (= a -1) 0.0 (if (= (first h) a) 10.0 -5.0)))
        (agent-step tool-infos (list 0 1 2 3) (list -1 0 1 2 3) 4 util)))))
""", env=copy(agent_env))
@assert submit_test[1] == 0 "agent should submit after strong evidence"
@assert submit_test[2] == 2 "agent should submit answer 2 (observed response)"
println("agent-step after evidence for answer=2: submits 2")

println()

# ─── Test load_dsl ───

println("=" ^ 60)
println("TEST 16: load_dsl returns environment with callable closures")
println("=" ^ 60)

env = load_dsl("(define double (lambda (x) (* x 2)))")
@assert env[:double](21) == 42 "load_dsl closure should be callable"
println("load_dsl: double(21) = 42")

println()
println("=" ^ 60)
println("ALL TESTS PASSED")
println("=" ^ 60)
