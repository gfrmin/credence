# test_entropy.jl — the entropy accessor (dominance move, Phase 3 prerequisite).
#
# `entropy(p)` is the sanctioned read path for a belief's concentration — the escape-mass
# heuristic score of dominance-design.md §0/§5.1 reads it instead of doing −Σ w log w on
# `weights(p)` output in the host (the compute-on-weights violation, no escape hatch).
#
# Run: julia test/test_entropy.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: MixturePrevision, CategoricalPrevision, TaggedBetaPrevision, BetaPrevision,
                Prevision, entropy, weights, WeightsDomainError

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("entropy accessor")
println("="^64)

# §1 Uniform mixture: H == log(n) (maximum entropy over n components).
let
    comps = Prevision[TaggedBetaPrevision(i, BetaPrevision(1.0, 1.0)) for i in 1:4]
    p = MixturePrevision(comps, zeros(4))
    h = entropy(p)
    # credence-lint: allow — precedent:test-oracle — manual Shannon-entropy oracle for the accessor under test
    check("§1 uniform 4-mixture: H == log 4", isapprox(h, log(4.0); atol = 1e-14), "h=$h")
end

# §2 Degenerate mixture: all mass on one component ⇒ H == 0.0 exactly (0·log 0 = 0 convention).
let
    comps = Prevision[TaggedBetaPrevision(i, BetaPrevision(1.0, 1.0)) for i in 1:3]
    p = MixturePrevision(comps, [0.0, -Inf, -Inf])
    check("§2 degenerate mixture: H == 0.0 exactly", entropy(p) == 0.0, "h=$(entropy(p))")
end

# §3 Asymmetric two-component oracle: H == −(w₁ log w₁ + w₂ log w₂) from the same weights.
let
    comps = Prevision[TaggedBetaPrevision(i, BetaPrevision(1.0, 1.0)) for i in 1:2]
    p = MixturePrevision(comps, [log(0.75), log(0.25)])
    w = weights(p)
    # credence-lint: allow — precedent:test-oracle — manual Shannon-entropy oracle for the accessor under test
    oracle = -(w[1] * log(w[1]) + w[2] * log(w[2]))
    check("§3 asymmetric oracle: H matches manual −Σ w log w", entropy(p) == oracle,
          "h=$(entropy(p)) oracle=$oracle")
end

# §4 Categorical prevision: same accessor, same convention.
let
    p = CategoricalPrevision(log.([0.5, 0.5]))
    # credence-lint: allow — precedent:test-oracle — manual Shannon-entropy oracle for the accessor under test
    check("§4 categorical: H == log 2", isapprox(entropy(p), log(2.0); atol = 1e-14),
          "h=$(entropy(p))")
end

# §5 Continuous prevision: WeightsDomainError propagates (no silent fallback).
let
    threw = try
        entropy(BetaPrevision(2.0, 3.0)); false
    catch e
        e isa WeightsDomainError
    end
    check("§5 continuous prevision throws WeightsDomainError", threw)
end

println("="^64)
println("ALL CHECKS PASSED — entropy accessor")
println("="^64)
