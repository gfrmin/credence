# test_typed_decision.jl — the typed decision stdlib (Code-1b): optimise / value /
# voi / net_voi / eu / predictive_prob over a functional-per-action preference.
#
# This is the in-process canonical form of the decision mechanism the skin's
# `functional_per_action` handler dispatches over a JSON spec; it keeps EU
# closed-form (LinearCombination / Identity dispatch) rather than opaque-closure
# quadrature. Asserted exact against a hand oracle.
#
# Run from repo root:
#     julia test/test_typed_decision.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: BetaPrevision, TaggedBetaPrevision, MixturePrevision, Identity,
                LinearCombination, TestFunction, Kernel, Interval, Finite, BetaBernoulli, mean
using Credence.Ontology: optimise, value, voi, net_voi, eu, predictive_prob, condition, expect

function check(name, cond, detail="")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

lc(coeff, off) = LinearCombination(Tuple{Float64,TestFunction}[(coeff, Identity())], off)
const0(off)    = LinearCombination(Tuple{Float64,TestFunction}[], off)

println("="^60)
println("typed decision stdlib — optimise / value / voi")
println("="^60)

# Mixture of two cells, weights .25/.75. E[θ] = .25·.5 + .75·5/7 = .660714…
mix = MixturePrevision([TaggedBetaPrevision(1, BetaPrevision(2.0, 2.0)),
                        TaggedBetaPrevision(2, BetaPrevision(5.0, 2.0))],
                       [log(0.25), log(0.75)])
p = 0.25 * 0.5 + 0.75 * (5/7)

check("expect(mixture, Identity) = Σ wᵢ·meanᵢ (closed-form)",
      isapprox(expect(mix, Identity()), p; atol=1e-12), "got $(expect(mix, Identity()))")
check("expect(mixture, LinearCombination 2θ−1) exact, disambiguated",
      isapprox(expect(mix, lc(2.0, -1.0)), 2p - 1; atol=1e-12), "got $(expect(mix, lc(2.0,-1.0)))")

# functional-per-action: EU(proceed)=0, EU(block)=1−4θ (c=1, λ=3 ⇒ 1−(1+3)θ).
fpa  = Dict(:proceed => const0(0.0), :block => lc(-4.0, 1.0))
acts = [:proceed, :block]
check("eu(block) = 1 − 4·E[θ]", isapprox(eu(mix, fpa[:block]), 1 - 4p; atol=1e-12))
check("value = max EU = 0 (proceed; block is −1.64)", isapprox(value(mix, acts, fpa), 0.0; atol=1e-12))
check("optimise picks proceed", optimise(mix, acts, fpa) === :proceed,
      "got $(optimise(mix, acts, fpa))")

k = Kernel(Interval(0.0, 1.0), Finite([0, 1]), theta -> theta,
           (m, o) -> (a = mean(m.beta); o == 1 ? log(a) : log(1.0 - a));
           likelihood_family = BetaBernoulli())

check("predictive_prob(o=1) = E[θ] exact", isapprox(predictive_prob(mix, k, 1), p; atol=1e-9),
      "got $(predictive_prob(mix, k, 1))")
check("predictive_prob(o=0) = 1−E[θ] exact", isapprox(predictive_prob(mix, k, 0), 1 - p; atol=1e-9))

v = voi(mix, k, acts, fpa, [0, 1])
check("voi ≥ 0 (information never hurts the optimal decision)", v >= -1e-12, "got $v")
check("net_voi = voi − cost", isapprox(net_voi(mix, k, acts, fpa, [0, 1], 0.05), v - 0.05; atol=1e-12))

# When one action dominates decisively, more information has no value.
check("voi = 0 when the decision is already decisive", isapprox(v, 0.0; atol=1e-9), "got $v")

# A near-50/50 belief: an observation CAN flip the optimal action ⇒ voi > 0.
flat = MixturePrevision([TaggedBetaPrevision(1, BetaPrevision(2.0, 2.0))], [0.0])
fpa2 = Dict(:proceed => const0(0.0), :block => lc(-2.0, 1.0))   # EU(block)=1−2θ, θ*=0.5
v2 = voi(flat, k, [:proceed, :block], fpa2, [0, 1])
check("voi > 0 at a 50/50 belief where an obs flips the action", v2 > 1e-6, "got $v2")

println("="^60)
println("ALL CHECKS PASSED — typed decision stdlib")
println("="^60)
