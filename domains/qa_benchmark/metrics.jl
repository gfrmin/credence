"""
    metrics.jl — Scoring and summary statistics for the QA benchmark.
"""

using Printf
using JSON3

struct QuestionResult
    question_id::String
    category::String
    tools_queried::Vector{Int}
    tool_responses::Dict{Int,Int}    # tool_idx => response_idx
    submitted::Union{Int,Nothing}    # nothing = abstained
    was_correct::Union{Bool,Nothing}
    reward::Float64
    tool_cost::Float64
end

struct SeedResult
    seed::Int
    records::Vector{QuestionResult}
    total_score::Float64       # reward - tool_cost
    total_reward::Float64      # reward only
    total_tool_cost::Float64   # cost only
    wall_time_s::Float64
end

total_score(r::SeedResult) = r.total_score

function accuracy(r::SeedResult)
    submitted = filter(rec -> rec.submitted !== nothing, r.records)
    isempty(submitted) && return 0.0
    count(rec -> rec.was_correct === true, submitted) / length(submitted)
end

function abstention_rate(r::SeedResult)
    isempty(r.records) && return 0.0
    count(rec -> rec.submitted === nothing, r.records) / length(r.records)
end

function tools_per_question(r::SeedResult)
    isempty(r.records) && return 0.0
    sum(length(rec.tools_queried) for rec in r.records) / length(r.records)
end

function cost_per_question(r::SeedResult)
    isempty(r.records) && return 0.0
    r.total_tool_cost / length(r.records)
end

wall_time_per_question(r::SeedResult) = isempty(r.records) ? 0.0 : r.wall_time_s / length(r.records)

function summary_table(all_results::Dict{String,Vector{SeedResult}})
    header = @sprintf("%-22s %3s %12s %10s %10s %10s %10s %10s",
                      "Agent", "N", "Score", "Accuracy", "Abstain%", "Tools/Q", "Cost/Q", "Time/Q")
    lines = [header, "-"^length(header)]
    for (name, runs) in sort(collect(all_results); by=first)
        n = length(runs)
        scores = [total_score(r) for r in runs]
        accs = [accuracy(r) for r in runs]
        absts = [abstention_rate(r) for r in runs]
        tpq = [tools_per_question(r) for r in runs]
        cpq = [cost_per_question(r) for r in runs]
        wtq = [wall_time_per_question(r) for r in runs]
        push!(lines, @sprintf("%-22s %3d %7.1f±%3.1f %7.3f±%.3f %7.3f±%.3f %7.2f±%.2f %7.2f±%.2f %7.4f±%.4f",
            name, n,
            _mean(scores), _std(scores),
            _mean(accs), _std(accs),
            _mean(absts), _std(absts),
            _mean(tpq), _std(tpq),
            _mean(cpq), _std(cpq),
            _mean(wtq), _std(wtq)))
    end
    join(lines, "\n")
end

_mean(v) = sum(v) / length(v)
_std(v) = let m = _mean(v); sqrt(sum((x - m)^2 for x in v) / length(v)) end

# ─── JSON serialization ───

function _serialize_record(rec::QuestionResult)
    Dict{String,Any}(
        "question_id"    => rec.question_id,
        "category"       => rec.category,
        "tools_queried"  => rec.tools_queried,
        "tool_responses" => Dict(string(k) => v for (k, v) in rec.tool_responses),
        "action_type"    => rec.submitted === nothing ? "abstain" : "submit",
        "was_correct"    => rec.was_correct,
        "reward"         => rec.reward,
        "tool_cost"      => rec.tool_cost,
    )
end

function _serialize_seed(r::SeedResult)
    Dict{String,Any}(
        "seed"            => r.seed,
        "total_score"     => r.total_score,
        "total_reward"    => r.total_reward,
        "total_tool_cost" => r.total_tool_cost,
        "wall_time_s"     => r.wall_time_s,
        "records"         => [_serialize_record(rec) for rec in r.records],
    )
end

function save_results(path::String, all_results::Dict{String,Vector{SeedResult}})
    mkpath(dirname(path))
    data = Dict{String,Any}(name => [_serialize_seed(r) for r in runs]
                            for (name, runs) in all_results)
    open(path, "w") do io
        JSON3.pretty(io, data)
    end
    println(stderr, "Results saved to $path")
end
