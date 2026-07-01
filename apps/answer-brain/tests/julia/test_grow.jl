#!/usr/bin/env julia
# Role: tests
"""
    test_grow.jl — app-side integration checks for the gather VOI (the ruling's "B half").

The pricing (`grow_value`) and the which-gather argmax (`best_grow`) are ENGINE stdlib
(`src/gather_voi.jl`, behaviour pinned by `test/test_gather_voi.jl`); the app imports and
re-exports them. These checks pin only the integration: the app surface IS the engine
function (no shadowing fork), and one behavioural smoke each through the app name.

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
println("answer-brain grow — engine gather-VOI integration")
println("="^64)

# The app surface is the engine function, not a fork.
check("AnswerBrain.grow_value === Credence.grow_value",
      AnswerBrain.grow_value === Credence.grow_value)
check("AnswerBrain.best_grow === Credence.best_grow",
      AnswerBrain.best_grow === Credence.best_grow)

# One behavioural smoke each through the app name (self-gating + which-gather).
let u_c = 1.0, cost = 0.1
    check("grow_value self-gates at a confident report (≈ −cost)",
          approx(AnswerBrain.grow_value(0.9, u_c, u_c, cost), -cost))
end
let u_c = 1.0, eu = -0.2
    best, bv = AnswerBrain.best_grow([("re-extract", 0.7, 0.1), ("retrieve-wider", 0.4, 0.1)], u_c, eu)
    check("best_grow discriminates which-gather through the app surface",
          best == "re-extract" && approx(bv, AnswerBrain.grow_value(0.7, u_c, eu, 0.1)))
end

println("-"^64)
println("grow integration: ", length(PASSED), " checks passed")
