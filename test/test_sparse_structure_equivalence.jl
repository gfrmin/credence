# Role: tests
# test_sparse_structure_equivalence.jl — the exactness contract for the sparse
# structure-BMA backend (src/sparse_structure.jl).
#
# SparseStructurePrevision is an execution-layer optimisation: it must be
# BIT-IDENTICAL to the dense ProductPrevision of TaggedBeta cells. This test runs
# the SAME random observation sequence through `build_prior` (sparse) and
# `build_prior_dense` (reference) and asserts the structure posteriors AND every
# per-context predictive agree to 1e-12. If this fails, the optimisation is not
# exact and must not ship.
#
# Run from repo root:
#     julia --project=. test/test_sparse_structure_equivalence.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: Identity, weights, expect
using Random: MersenneTwister, rand
include(joinpath(@__DIR__, "..", "apps", "credence-pi", "brain", "feature_brain.jl"))
using .FeatureBrain

function check(name, cond, detail="")
    cond ? println("PASSED: $name") :
           (println("FAILED: $name — $detail"); error("assertion failed: $name"))
end

println("="^64)
println("sparse ≡ dense structure-BMA — exactness contract")
println("="^64)

# Enumerate the full context space (cross-product of feature value-sets).
function all_contexts(vals)
    ctxs = Vector{String}[String[]]
    for col in vals
        ctxs = [vcat(c, [v]) for c in ctxs for v in col]
    end
    ctxs
end

# One scenario: build both priors, replay an identical random observation
# sequence, assert structure posteriors + per-context predictives match @1e-12.
function run_scenario(name, names, vals; n_obs=400, seed=0)
    model = build_model(names, vals; p_edge = 0.5)
    sparse = build_prior(model)
    dense  = build_prior_dense(model)
    ctxs = all_contexts(vals)
    rng = MersenneTwister(seed)

    for _ in 1:n_obs
        X = ctxs[rand(rng, 1:length(ctxs))]
        o = rand(rng, 0:1)
        sparse = observe(model, sparse, X, o)
        dense  = observe(model, dense,  X, o)
    end

    # Structure posteriors identical.
    ws, wd = weights(sparse), weights(dense)
    check("$name: structure posteriors agree @1e-12",
          all(isapprox.(ws, wd; atol=1e-12)),
          "max Δ = $(maximum(abs.(ws .- wd)))")

    # Per-context predictive P(approve|X) identical at EVERY context.
    maxΔ = 0.0
    for X in ctxs
        ps = expect(belief_at_context(model, sparse, X), Identity())
        pd = expect(belief_at_context(model, dense,  X), Identity())
        maxΔ = max(maxΔ, abs(ps - pd))
    end
    check("$name: all $(length(ctxs)) per-context predictives agree @1e-12",
          maxΔ <= 1e-12, "max Δ = $maxΔ")
end

# 2-feature (matches the brain's spike oracle shape).
run_scenario("2-feature", ["tool", "rep"],
             [["bash", "read"], ["rep0", "rep3"]]; n_obs=400, seed=1)

# 3-feature, mixed cardinality.
run_scenario("3-feature", ["tool", "wd", "rep"],
             [["bash", "read", "exec"], ["root", "sub"], ["r0", "r1", "r2"]];
             n_obs=600, seed=2)

# 4-feature, deeper — exercises many structures + repeated contexts.
run_scenario("4-feature", ["a", "b", "c", "d"],
             [["a1","a2","a3"], ["b1","b2"], ["c1","c2"], ["d1","d2","d3"]];
             n_obs=800, seed=3)

println()
println("="^64)
println("ALL SPARSE≡DENSE EQUIVALENCE CHECKS PASSED")
println("="^64)
