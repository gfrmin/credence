# Paper 1 B4 gating experiment (2): REACHABLE BOUND at this horizon.
#
# Explore-then-exploit, oracle category, sweep the forced-exploration budget k:
# for the first k encounters of each category, force-explore (query ALL tools,
# learn θ from real outcomes, submit majority vote); thereafter exploit (greedy
# on learned θ). Maximise over k. The peak is the best any explore-then-exploit
# schedule can reach given ~10-15 questions/category.
#
# Result: peaks at k=1 = 205.0 then COLLAPSES (k=2=187 < k=0=189) — two rounds is
# worse than not exploring. The horizon affords ~one exploration round, which
# cannot recover the 101-pt gap to the 306 ceiling. So horizon-aware VOI tops out
# at ≤+16 over free greedy: horizon-locked. The bound is conservative (query-all,
# majority-vote, fixed-k all slack it down); the SHAPE is family-robust. Offline:
#   julia --project=. scripts/paper1-reachable.jl
const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "apps", "julia", "qa_benchmark", "host.jl"))
using Random

tools = make_spec_tools(); n_tools = length(tools); n_cats = length(CATEGORIES)
catidx = Dict(c => i for (i, c) in enumerate(CATEGORIES))
avg(v) = sum(v) / length(v)

function ete(k)
    scores = Float64[]
    for seed in 0:19
        rng = MersenneTwister(seed); questions = get_questions(; seed)
        rt = generate_response_table(tools, questions, rng)
        rel = [BetaPrevision(1.0, 1.0) for _ in 1:n_tools, _ in 1:n_cats]
        seen = zeros(Int, n_cats); reward = 0.0; cost = 0.0
        for (qi, q) in enumerate(questions)
            ci = catidx[q.category]; seen[ci] += 1
            if seen[ci] <= k
                resps = Int[]
                for t in 1:n_tools
                    r = rt[qi, t]; push!(resps, r); cost += tools[t].cost
                    rel[t, ci] = condition(rel[t, ci], RELIABILITY_KERNEL, r == q.correct_index ? 1.0 : 0.0)
                end
                cnt = Dict{Int,Int}(); for r in resps; cnt[r] = get(cnt, r, 0) + 1; end
                sub = first(sort(collect(cnt); by = last, rev = true))[1]
            else
                tstar = argmax(t -> mean(rel[t, ci]), 1:n_tools)
                r = rt[qi, tstar]; cost += tools[tstar].cost
                rel[tstar, ci] = condition(rel[tstar, ci], RELIABILITY_KERNEL, r == q.correct_index ? 1.0 : 0.0)
                sub = r
            end
            reward += sub == q.correct_index ? REWARD_CORRECT : PENALTY_WRONG
        end
        push!(scores, reward - cost)
    end
    avg(scores)
end

println("REACHABLE BOUND: explore-then-exploit, oracle category, sweep k")
results = [(k, ete(k)) for k in 0:8]
for (k, s) in results
    println("  k=$k (explore first k/cat)  score=$(round(s, digits=1))")
end
bs, bi = findmax(last.(results))
println("  BEST: k=$(results[bi][1])  score=$(round(bs, digits=1))   " *
        "(refs: greedy_oracle 189.4, known-θ ceiling 306.2)")
