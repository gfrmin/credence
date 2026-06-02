# Paper 1 B4 gating experiment (1): KNOWN-RELIABILITY CEILING.
#
# Seed the agent with the TRUE θ table (concentrated Beta — only the mean
# matters, everything in the answer-kernel/VOI is linear in r), disable
# learning, act myopic-EU under oracle categories. This is the asymptote of
# perfect reliability-exploration (perfect exploration ≡ knowing θ), and it
# bounds how much any action policy could gain from better exploration.
#
# Result (20 seeds): 306.2 ± 43.9, acc 85.1% — far above greedy (189.4) and the
# learning VOI agent (163.7). Headroom exists; experiment (2) [reachable bound,
# paper1-reachable.jl] then shows it is horizon-locked. Run offline:
#   julia --project=. scripts/paper1-ceiling.jl
const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "apps", "julia", "qa_benchmark", "host.jl"))
using Random

tools = make_spec_tools()
n_tools = length(tools); n_cats = length(CATEGORIES)
clamp01(x) = clamp(x, 0.005, 0.995)
const K = 100.0
θ = [clamp01(tools[t].reliability_by_category[CATEGORIES[c]]) for t in 1:n_tools, c in 1:n_cats]
init = [BetaPrevision(K * θ[t, c], K * (1 - θ[t, c])) for t in 1:n_tools, c in 1:n_cats]

scores = Float64[]; accs = Float64[]; costs = Float64[]
for seed in 0:19
    rng = MersenneTwister(seed)
    questions = get_questions(; seed)
    rt = generate_response_table(tools, questions, rng)
    r = run_bayesian_seed(tools, questions, rt; use_voi=true, learn=false, init_reliability=init)
    push!(scores, r.total_score); push!(costs, r.total_tool_cost)
    push!(accs, count(rec -> rec.was_correct === true, r.records) / length(questions))
end
mean_(v) = sum(v) / length(v)
std_(v) = (m = mean_(v); sqrt(sum((x - m)^2 for x in v) / (length(v) - 1)))
println("KNOWN-RELIABILITY CEILING (true θ, oracle category, myopic-EU, no learning)")
println("  score = $(round(mean_(scores), digits=1)) ± $(round(std_(scores), digits=1))   " *
        "acc = $(round(100*mean_(accs), digits=1))%   tool_cost/seed = $(round(mean_(costs), digits=1))")
println("  refs: greedy_oracle 189.4 | bayesian_oracle 163.7 | greedy_inferred 149.6 | bayesian_inferred 110.4")
