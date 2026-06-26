# test_net_value.jl — the single net-expected-value shape `net_value(Δvalue, cost) = Δvalue − cost`
# (collapse-towers Phase 3, src/net_value.jl), the scalar reduction of E[value] − cost. `net_voi`
# (action = observe) is the instance; `net_voc` (Phase 5) is the other. Asserts the unit contract and
# that `net_voi` routed through `net_value` is BIT-IDENTICAL to the old `voi(...) - cost`.
#
# Run from repo root:
#     julia test/test_net_value.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: BetaMeasure, Interval, Finite, Kernel, BetaBernoulli, Identity, LinearCombination,
                TestFunction, mean
using Credence.Ontology: net_value, net_voi, voi

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("net_value — the single net-expected-value shape (Phase 3)")
println("="^64)

# ── (1) unit contract: net_value(Δvalue, cost) = Δvalue − cost ──
check("net_value(0.75, 0.25) == 0.5", net_value(0.75, 0.25) == 0.5)   # credence-lint: allow — precedent:test-oracle — FP-exact hand value 0.75−0.25 (dyadic)
for (v, c) in [(1.0, 0.3), (0.0, 0.5), (-0.2, 0.1), (0.65, 0.65)]      # incl. cost > value ⇒ negative net value
    check("net_value($v, $c) == $v − $c", net_value(v, c) == v - c)    # credence-lint: allow — precedent:test-oracle — the subtraction is the spec
end

# ── (2) net_voi routes through net_value, BIT-IDENTICAL, on a voi>0 fixture ──
# Beta(2,2) (mean 0.5) at the decision boundary (EU(block)=1−2θ flips at θ=0.5) ⇒ observing flips the
# decision ⇒ voi > 0. net_voi internally is net_value(voi(...), cost); the oracle recomputes voi(...)−cost.
const0(off) = LinearCombination(Tuple{Float64, TestFunction}[], off)
lc(coeff, off) = LinearCombination(Tuple{Float64, TestFunction}[(coeff, Identity())], off)
belief = BetaMeasure(Interval(0.0, 1.0), 2.0, 2.0)
acts = [:proceed, :block]
fpa = Dict(:proceed => const0(0.0), :block => lc(-2.0, 1.0))   # EU(block)=1−2θ, EU(proceed)=0
k = Kernel(Interval(0.0, 1.0), Finite([0, 1]), theta -> theta,
           (h, o) -> o == 1 ? log(max(h, 1e-300)) : log(max(1 - h, 1e-300));
           likelihood_family = BetaBernoulli())
v = voi(belief, k, acts, fpa, [0, 1])
check("fixture has voi > 0 (information helps at the boundary)", v > 1e-6, "got $v")   # credence-lint: allow — precedent:test-oracle — asserts the fixture exercises a non-trivial voi
let cost = 0.05, expected = v - cost   # credence-lint: allow — precedent:test-oracle — voi(...)−cost is the independent oracle
    got = net_voi(belief, k, acts, fpa, [0, 1], cost)
    check("net_voi == voi − cost (bit-exact, routed through net_value)", got == expected,   # credence-lint: allow — precedent:test-oracle — net_voi routed through net_value equals the oracle
          "got $got vs $expected")
end

# ── (3) net_voc shape forward-check (documents the Phase-5 instance; no Phase-5 code yet) ──
let Δv = 0.42, compute_cost = 0.1
    check("net_voc shape: net_value(Δv, compute_cost) == Δv − compute_cost",
          net_value(Δv, compute_cost) == Δv - compute_cost)          # credence-lint: allow — precedent:test-oracle — the Phase-5 instance reduces to the same subtraction
end

println("="^64)
println("ALL CHECKS PASSED — net_value")
println("="^64)
