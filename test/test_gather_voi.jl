# test_gather_voi.jl — the gather VOI (grow pricing + which-gather argmax) lifted into
# engine stdlib (src/gather_voi.jl). A grow (recall/discovery) actuator may enlarge a
# consumer's hypothesis set to admit a missing truth; its value is
# `grow_value(g, u_correct, eu, cost) = g·(u_correct − eu) − cost` where
# `g = P(recover | sensors)` is a structure-BMA belief and `eu` is the consumer's terminal
# EU — so the missing-mass gate is carried by `u_correct − eu` (a solved decision prices
# ≈ −cost), never a hand-coded gate. `best_grow` argmaxes over `(probe, g, cost)` actuators
# and fires only if the winner strictly clears 0 (mirrors the `net_voi > 0` gate).
#
# Lifted from apps/answer-brain/brain/answer_brain.jl (the conferred "B half",
# life-agent docs/ask-as-connection.md §4/§7); Connection-generic, so it lives here.
#
# Run from repo root:
#     julia --project=. test/test_gather_voi.jl

push!(LOAD_PATH, "src")
using Credence

function check(name, cond, detail = "")
    if cond
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("assertion failed: $name")
    end
end
approx(a, b; atol = 1e-9) = abs(a - b) <= atol

println("="^64)
println("gather VOI (engine stdlib) — grow_value + best_grow")
println("="^64)

# ── grow_value: the pure pricing, self-gating on the consumer's terminal EU ────────────
let u_c = 1.0, cost = 0.1
    check("grow_value self-gates at a confident terminal (eu ≈ u_correct ⇒ ≈ −cost)",
          approx(grow_value(0.9, u_c, u_c, cost), -cost))
    check("grow_value is positive for a high-g withhold",
          grow_value(0.8, u_c, -0.2, cost) > 0.0)
    check("grow_value is monotone in g",
          grow_value(0.6, u_c, -0.2, cost) > grow_value(0.3, u_c, -0.2, cost))
    check("grow_value decreases as terminal EU rises",
          grow_value(0.7, u_c, 0.8, cost) < grow_value(0.7, u_c, -0.2, cost))
    check("grow_value subtracts cost exactly",
          approx(grow_value(0.5, 1.0, 0.0, 0.1), 0.5 * 1.0 - 0.1))
end

# ── best_grow: the which-gather argmax (fires only if the best clears 0) ────────────────
let u_c = 1.0, eu = -0.2
    acts = [("re-extract", 0.7, 0.1), ("retrieve-wider", 0.4, 0.1)]
    best, bv = best_grow(acts, u_c, eu)
    check("best_grow picks the higher-value actuator", best == "re-extract")
    check("best_grow value matches grow_value",
          approx(bv, grow_value(0.7, u_c, eu, 0.1)))
    acts2 = [("re-extract", 0.55, 0.5), ("retrieve-wider", 0.5, 0.05)]
    best2, _ = best_grow(acts2, u_c, eu)
    check("best_grow accounts for cost (the cheaper actuator wins)", best2 == "retrieve-wider")
    best3, bv3 = best_grow(acts, u_c, u_c)
    check("best_grow returns nothing when nothing clears cost", best3 === nothing && bv3 == 0.0)
end

println("-"^64)
println("gather VOI: all checks passed")
