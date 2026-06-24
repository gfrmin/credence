# test_structure_bma.jl — the structure-BMA builder + observe + readout lifted into
# engine stdlib (decouple Move 3, src/structure_bma.jl). Asserts the lifted functions are
# EXACT through the engine's own public names (not the credence-pi shim): the sparse prior
# and posterior equal the dense reference bit-for-bit, and the soft-evidence path reduces
# exactly to the hard-label path at the certain-signal corners.
#
# Behaviour-preservation vs the pre-lift app brain is additionally pinned end-to-end by
# apps/credence-pi/tests/julia/test_feature_brain.jl (unchanged, passes through the shim).
#
# Run from repo root:
#     julia --project=. test/test_structure_bma.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: weights

function check(name, cond, detail = "")
    if cond
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("assertion failed: $name")
    end
end

println("="^64)
println("structure-BMA (lifted to engine stdlib) — Move 3")
println("="^64)

# 2 features (tool, rep) ⇒ 4 structures (∅, {tool}, {rep}, {tool,rep}).
model = build_structure_model(["tool", "rep"], [["bash", "read"], ["rep0", "rep3"]];
                              alpha0 = 2.0, beta0 = 2.0, p_edge = 0.5)
check("4 structures enumerated", length(model.structures) == 4, "got $(length(model.structures))")

# Prior: the sparse store is an exact backend of the dense product ⇒ bit-identical weights.
ps = build_structure_prior(model)
pd = build_structure_prior_dense(model)
check("prior weights sparse ≡ dense (exact)", weights(ps) == weights(pd),
      "$(weights(ps)) vs $(weights(pd))")

# Posterior after a sequence of hard observations across two contexts: still bit-identical.
seq = [(["bash", "rep3"], 1), (["bash", "rep3"], 1), (["read", "rep0"], 0), (["bash", "rep0"], 1)]
ts, td = ps, pd
for (X, o) in seq
    global ts = structure_observe(model, ts, X, o)
    global td = structure_observe(model, td, X, o)
end
check("posterior weights sparse ≡ dense (exact, 4 obs)", weights(ts) == weights(td),
      "$(weights(ts)) vs $(weights(td))")

# belief_at_context: the per-decision view carries the structure posterior verbatim ⇒ the
# sparse and dense views agree exactly.
let X = ["bash", "rep3"]
    bs = belief_at_context(model, ts, X)
    bd = belief_at_context(model, td, X)
    check("belief_at_context weights sparse ≡ dense (exact)", weights(bs) == weights(bd),
          "$(weights(bs)) vs $(weights(bd))")
end

# Soft-evidence reduces EXACTLY to the hard label at the certain-signal corners:
# (r,w)=(1,0) ≡ obs=1, (r,w)=(0,1) ≡ obs=0 (a degenerate-corner identity).
let X = ["read", "rep3"]
    check("soft (r,w)=(1,0) ≡ hard obs=1 (exact)",
          weights(structure_observe(model, ps, X, 1)) ==
          weights(structure_observe_soft(model, ps, X, 1.0, 0.0)))
    check("soft (r,w)=(0,1) ≡ hard obs=0 (exact)",
          weights(structure_observe(model, ps, X, 0)) ==
          weights(structure_observe_soft(model, ps, X, 0.0, 1.0)))
end

# firing_tags: exactly one cell per structure fires (globally-unique tags).
let X = ["bash", "rep3"]
    tags = structure_firing_tags(model, X)
    check("one firing tag per structure", length(tags) == length(model.structures),
          "got $(length(tags)) for $(length(model.structures)) structures")
end

# reconstruct_structure_prior_from_data (Move 5 prereq): warm-seeding from inline counts ≡
# replaying structure_observe by hand, bit-for-bit (order-independent ⇒ exact). data=nothing
# is the cold prior.
let counts = Dict("contexts" => [
        Dict("ctx" => ["bash", "rep3"], "n1" => 5, "n0" => 2),
        Dict("ctx" => ["read", "rep0"], "n1" => 1, "n0" => 4)])
    warm = reconstruct_structure_prior_from_data(model, counts)
    manual = build_structure_prior(model)
    for (X, n1, n0) in [(["bash", "rep3"], 5, 2), (["read", "rep0"], 1, 4)]
        for _ in 1:n1; manual = structure_observe(model, manual, X, 1); end
        for _ in 1:n0; manual = structure_observe(model, manual, X, 0); end
    end
    check("warm_counts reconstruction ≡ hand structure_observe replay (bit-exact)",
          weights(warm) == weights(manual), "$(weights(warm)) vs $(weights(manual))")
    check("warm_counts=nothing ≡ the cold prior",
          weights(reconstruct_structure_prior_from_data(model, nothing)) == weights(build_structure_prior(model)))
end

println("="^64)
println("ALL CHECKS PASSED — structure-BMA lift exact")
println("="^64)
