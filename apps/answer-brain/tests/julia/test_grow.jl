#!/usr/bin/env julia
# Role: tests
"""
    test_grow.jl — the gather VOI (the ruling's "B half"): grow actuators priced by a
    structure-BMA `g_mechanism`, argmaxed for which-gather, self-gating on the terminal EU.

The conferred factoring (life-agent `docs/ask-as-connection.md` §4, §7): report/abstain stays the
exact terminal threshold (`terminal_decide`, untouched); only the *gather* decision is offloaded. A
grow actuator's value is `grow_value(g, u_correct, eu, cost) = g·(u_correct − eu) − cost`, where
`g = P(recover | sensors)` is the engine's structure-BMA belief and `eu` is the terminal EU — so the
missing-mass gate is carried by `u_correct − eu` (a confident report prices ≈ −cost), not a `p_none`
branch. `terminal_decide` is never touched.

Run from the credence repo root:
    julia --project=. apps/answer-brain/tests/julia/test_grow.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "src"))
using Credence

include(joinpath(@__DIR__, "..", "..", "brain", "answer_brain.jl"))
using .AnswerBrain

const PASSED = String[]
function check(name::AbstractString, cond::Bool; detail::AbstractString = "")
    if cond
        push!(PASSED, name); println("PASSED: ", name)
    else
        println("FAILED: ", name, " — ", detail); error("assertion failed: $name")
    end
end
approx(a, b; atol = 1e-9) = abs(a - b) <= atol

println("="^64)
println("answer-brain grow — the gather VOI (B half)")
println("="^64)

# ── Slice 1: grow_value — the pure pricing, self-gating on the terminal EU ────────────────
# grow_value(g, u_correct, eu, cost) = g·(u_correct − eu) − cost
let u_c = 1.0, cost = 0.1
    # A confident terminal report (eu ≈ u_correct) has ~zero gain ⇒ net = −cost ⇒ no grow.
    check("grow_value self-gates at a confident report",
          approx(AnswerBrain.grow_value(0.9, u_c, u_c, cost), -cost))
    # A withhold (low eu) with high g clears cost ⇒ positive (grow is worth it).
    check("grow_value is positive for a high-g withhold",
          AnswerBrain.grow_value(0.8, u_c, -0.2, cost) > 0.0)
    # Monotone increasing in g (more likely to recover ⇒ more valuable).
    check("grow_value is monotone in g",
          AnswerBrain.grow_value(0.6, u_c, -0.2, cost) >
          AnswerBrain.grow_value(0.3, u_c, -0.2, cost))
    # The gain shrinks as the terminal EU rises toward u_correct (less to gain by growing).
    check("grow_value decreases as terminal EU rises",
          AnswerBrain.grow_value(0.7, u_c, 0.8, cost) <
          AnswerBrain.grow_value(0.7, u_c, -0.2, cost))
    # Cost enters linearly (exact).
    check("grow_value subtracts cost exactly",
          approx(AnswerBrain.grow_value(0.5, 1.0, 0.0, 0.1), 0.5 * 1.0 - 0.1))
end

println("-"^64)
println("grow slice-1 (grow_value): ", length(PASSED), " checks passed")
