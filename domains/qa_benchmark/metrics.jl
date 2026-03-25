"""
    metrics.jl — Scoring and summary statistics for the QA benchmark.
"""

using Printf

struct QuestionResult
    question_id::String
    category::String
    tools_queried::Vector{Int}
    submitted::Union{Int,Nothing}   # nothing = abstained
    was_correct::Union{Bool,Nothing}
    reward::Float64
    tool_cost::Float64
end

struct SeedResult
    seed::Int
    records::Vector{QuestionResult}
    total_score::Float64
    total_reward::Float64
    total_tool_cost::Float64
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

wall_time_per_question(r::SeedResult) = isempty(r.records) ? 0.0 : r.wall_time_s / length(r.records)

function summary_table(all_results::Dict{String,Vector{SeedResult}})
    header = @sprintf("%-22s %3s %12s %10s %10s %10s %10s",
                      "Agent", "N", "Score", "Accuracy", "Abstain%", "Tools/Q", "Time/Q")
    lines = [header, "-"^length(header)]
    for (name, runs) in sort(collect(all_results); by=first)
        n = length(runs)
        scores = [total_score(r) for r in runs]
        accs = [accuracy(r) for r in runs]
        absts = [abstention_rate(r) for r in runs]
        tpq = [tools_per_question(r) for r in runs]
        wtq = [wall_time_per_question(r) for r in runs]
        push!(lines, @sprintf("%-22s %3d %7.1f±%3.1f %7.3f±%.3f %7.3f±%.3f %7.2f±%.2f %7.4f±%.4f",
            name, n,
            mean(scores), std(scores),
            mean(accs), std(accs),
            mean(absts), std(absts),
            mean(tpq), std(tpq),
            mean(wtq), std(wtq)))
    end
    join(lines, "\n")
end

# Helpers
mean(v) = sum(v) / length(v)
std(v) = let m = mean(v); sqrt(sum((x - m)^2 for x in v) / length(v)) end
