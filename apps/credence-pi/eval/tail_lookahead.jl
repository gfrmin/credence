# Role: eval
#
# tail_lookahead.jl — quantify the multi-turn look-ahead's effect on REAL contexts.
#
# For every distinct context the shipped brains cover, compare the MYOPIC decision
# (expected_repeats=0) against the TAIL-AWARE decision (expected_repeats = m, the
# continuation posterior's E[ρ/(1−ρ)] = expect(belief, GeometricTail())), under each
# welfare profile. Reports how many decisions the look-ahead changes, the transitions,
# the m distribution, and the regime where it bites — honestly, including the corpus
# limitation (the rep/ident features cap at 3plus, and high-m contexts mostly already
# have low θ, so the two terms agree). The MECHANISM is proven on the uncertain-persistent
# regime by test_feature_brain.jl §8; this measures its EMPIRICAL reach on ClawsBench.
#
# NON-CAUSAL measurement: calls the public `decide`, reimplements no axiom op.
#
# Run from repo root:
#   julia --project=. apps/credence-pi/eval/tail_lookahead.jl

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using Credence
using Credence: Identity, GeometricTail, expect
using JSON3
include(joinpath(@__DIR__, "brain_env.jl"))
using .FeatureBrain: build_model_from_env, belief_at_context, decide, reconstruct_posterior

const PI = abspath(joinpath(@__DIR__, ".."))

function main()
    env = build_env()
    model = build_model_from_env(env)
    warm = reconstruct_posterior(model, joinpath(PI, "brain", "warm_brain.counts.json"))
    tail = reconstruct_posterior(model, joinpath(PI, "brain", "tail_brain.counts.json"))

    # The distinct contexts the shipped brains cover (tail + warm share the corpus).
    tail_json = JSON3.read(read(joinpath(PI, "brain", "tail_brain.counts.json"), String))
    ctxs = unique([String[String(v) for v in e.ctx] for e in tail_json.contexts])

    profiles = [("cost-hawk", 0.25, 0.05), ("flow-guard", 4.0, 1.0)]
    cost = 0.5
    gt = GeometricTail()

    println("="^92)
    println("  credence-pi — tail-aware look-ahead effect on $(length(ctxs)) shipped contexts (cost=\$$cost)")
    println("="^92)

    # m distribution + how many contexts carry a non-trivial tail.
    ms = [expect(belief_at_context(model, tail, c), gt) for c in ctxs]
    nontrivial = count(>(0.1), ms)
    println("expected-repeats m: max=$(round(maximum(ms);digits=3))  median=$(round(sort(ms)[cld(end,2)];digits=4))  " *
            "contexts with m>0.1: $nontrivial/$(length(ctxs))")
    println()

    summary = Dict{String,Any}()
    for (name, λ, q) in profiles
        changed = Tuple{Vector{String},Float64,Float64,Symbol,Symbol}[]
        for c in ctxs
            θ = expect(belief_at_context(model, warm, c), Identity())
            m = expect(belief_at_context(model, tail, c), gt)
            d_my = decide(model, warm, c, cost; aversion=λ, interrupt_cost=q, expected_repeats=0.0)
            d_la = decide(model, warm, c, cost; aversion=λ, interrupt_cost=q, expected_repeats=m)
            d_my === d_la || push!(changed, (c, θ, m, d_my, d_la))
        end
        trans = Dict{String,Int}()
        for (_, _, _, a, b) in changed; trans["$a→$b"] = get(trans, "$(a)→$(b)", 0) + 1; end
        println("── profile $name (λ=$λ, q=$q): $(length(changed)) / $(length(ctxs)) decisions change ──")
        isempty(changed) || println("   transitions: ", trans)
        for (c, θ, m, a, b) in sort(changed; by = x -> -x[3])[1:min(6, length(changed))]
            println("   $a→$b  θ=$(round(θ;digits=3)) m=$(round(m;digits=3))  ", join(c, "/"))
        end
        summary[name] = Dict("changed"=>length(changed), "transitions"=>trans)
    end

    println()
    println("HONEST READ: the look-ahead is a strict, principled refinement (m=0 ⇒ identical).")
    println("Its empirical reach on ClawsBench is bounded — the rep/ident features cap at 3plus")
    println("(so m maxes ~1.7) and the high-m contexts mostly already have low θ (waste-confident),")
    println("where the tail and waste terms agree. It bites in the UNCERTAIN-persistent regime,")
    println("proven by test_feature_brain.jl §8; a finer repetition feature would surface longer")
    println("tails empirically (the structure-BMA auto-sophisticates over a finer partition).")

    out = Dict("n_contexts"=>length(ctxs), "cost"=>cost,
               "m_max"=>maximum(ms), "m_median"=>sort(ms)[cld(length(ms),2)],
               "contexts_m_gt_0p1"=>nontrivial, "profiles"=>summary,
               "note"=>"Look-ahead = expected_repeats m from the tail (continuation) posterior; " *
                       "m=0 reduces to the myopic per-call EU exactly. Empirical reach bounded by " *
                       "rep/ident feature granularity (caps m) + high-m/low-θ correlation; mechanism " *
                       "proven in test §8 (uncertain-persistent regime).")
    open(joinpath(PI, "eval", "results", "tail_lookahead.summary.json"), "w") do io
        JSON3.pretty(io, out)
    end
    println("\nsummary → eval/results/tail_lookahead.summary.json")
end

main()
