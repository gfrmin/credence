# generate_feature_arithmetic_lift.jl — capture-before-refactor generator for the
# feature-arithmetic move (feature-arithmetic-design.md §3).
#
# MUST be run at the PRE-change commit (before any NumExpr code lands) — the golden values
# are the semantic projection of the old GTExpr(feature::Symbol, t) enumeration, and
# test_feature_arithmetic.jl asserts the lifted enumeration reproduces them ==.
# Provenance: the generating SHA is recorded in test/fixtures/README.md.
#
#   julia test/fixtures/generate_feature_arithmetic_lift.jl
#
# Projection captured for the canonical grammar below at depth 3:
#   - program count
#   - per-program show_expr canonical string
#   - per-program complexity
#   - per-program prior log-weight (via enumerate_programs_as_measure → weights)
#   - posterior mixture weights after a fixed 6-observation conditioning sequence

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, FeatureRef, GTExpr, LTExpr, AndExpr, show_expr,
                enumerate_programs, enumerate_programs_as_measure, weights,
                TaggedBetaPrevision, BetaPrevision, Prevision, MixturePrevision,
                compile_kernel, program_space_observation_kernel, condition,
                ExploreObservation

g = Grammar(Set([:a, :b]),
            [ProductionRule(:HOT, AndExpr(GTExpr(FeatureRef(:a), 0.7), LTExpr(FeatureRef(:b), 0.3)))], 901)
const AS = Symbol[:food, :enemy]

progs = enumerate_programs(g, 3; action_space = AS)
m = enumerate_programs_as_measure(g, 3; action_space = AS)
w = weights(m.prevision)
@assert length(progs) == length(w)

# Fixed conditioning sequence: build the host-shaped mixture and condition on 6 labelled
# observations through the Tier-1 kernel (the same shape test_threshold_explore uses).
comps = Prevision[TaggedBetaPrevision(i, BetaPrevision(1.0, 1.0)) for i in eachindex(progs)]
lw = Float64[log(x) for x in w]
belief = MixturePrevision(comps, lw)
cks = [compile_kernel(p, g, i) for (i, p) in enumerate(progs)]
obs_seq = [
    (Dict(:a => 0.8, :b => 0.2), Set([:food])),
    (Dict(:a => 0.2, :b => 0.8), Set([:enemy])),
    (Dict(:a => 0.9, :b => 0.1), Set([:food])),
    (Dict(:a => 0.1, :b => 0.9), Set([:enemy])),
    (Dict(:a => 0.75, :b => 0.25), Set([:food])),
    (Dict(:a => 0.3, :b => 0.6), Set([:enemy])),
]
for (feats, correct) in obs_seq
    k = program_space_observation_kernel(cks, feats, Dict{Symbol, Any}(), correct)
    global belief = condition(belief, k, 1.0)
end
post = weights(belief)

out = joinpath(@__DIR__, "feature_arithmetic_lift_v1.tsv")
open(out, "w") do io
    println(io, "# capture-before-refactor golden (feature-arithmetic-design.md §3)")
    println(io, "# grammar: {:a,:b} + HOT=(and (gt :a 0.7) (lt :b 0.3)) id=901, depth 3, actions [:food,:enemy]")
    println(io, "# columns: idx  show_expr  complexity  prior_weight  posterior_weight")
    println(io, "# n_programs\t$(length(progs))")
    for (i, p) in enumerate(progs)
        println(io, "$i\t$(show_expr(p.expr))\t$(p.complexity)\t$(w[i])\t$(post[i])")
    end
end
println("captured $(length(progs)) programs → $out")
