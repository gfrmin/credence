"""
    metrics.jl — Scoring, summary statistics, and SQLite storage for the QA benchmark.
"""

using Printf
using SQLite
using DBInterface
using Dates
using JSON3: JSON3  # only for serializing tool_responses/tools_queried to JSON strings

# ─── Result types ───

struct QuestionResult
    question_id::String
    category::String
    tools_queried::Vector{Int}
    tool_responses::Dict{Int,Int}    # tool_idx => response_idx
    submitted::Union{Int,Nothing}    # nothing = abstained
    was_correct::Union{Bool,Nothing}
    reward::Float64
    tool_cost::Float64
    wall_time_s::Float64             # per-question wall time
    api_input_tokens::Int            # 0 for non-LLM agents
    api_output_tokens::Int           # 0 for non-LLM agents
end

struct SeedResult
    seed::Int
    records::Vector{QuestionResult}
    total_score::Float64       # reward - tool_cost
    total_reward::Float64      # reward only
    total_tool_cost::Float64   # cost only
    wall_time_s::Float64
end

# ─── Summary statistics ───

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

# ─── API pricing (USD per million tokens) ───

const API_PRICING = Dict(
    "claude-haiku-4-5-20251001" => (input=1.0, output=5.0),
    "claude-sonnet-4-6"         => (input=3.0, output=15.0),
    "claude-sonnet-4-20250514"  => (input=3.0, output=15.0),
    "claude-opus-4-6"           => (input=5.0, output=25.0),
)

function api_cost_usd(model::String, in_tok::Int, out_tok::Int)
    pricing = get(API_PRICING, model, nothing)
    pricing === nothing && return 0.0
    in_tok / 1e6 * pricing.input + out_tok / 1e6 * pricing.output
end

# ─── SQLite storage ───

function init_db(path::String)
    mkpath(dirname(path))
    db = SQLite.DB(path)
    DBInterface.execute(db, "PRAGMA foreign_keys = ON")
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            agent TEXT NOT NULL,
            seed INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            total_score REAL NOT NULL,
            total_reward REAL NOT NULL,
            total_tool_cost REAL NOT NULL,
            wall_time_s REAL NOT NULL,
            total_api_input_tokens INTEGER NOT NULL DEFAULT 0,
            total_api_output_tokens INTEGER NOT NULL DEFAULT 0,
            total_api_cost_usd REAL NOT NULL DEFAULT 0.0,
            UNIQUE(agent, seed)
        )
    """)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            question_idx INTEGER NOT NULL,
            question_id TEXT NOT NULL,
            category TEXT NOT NULL,
            tools_queried TEXT NOT NULL,
            tool_responses TEXT NOT NULL,
            submitted INTEGER,
            was_correct INTEGER,
            reward REAL NOT NULL,
            tool_cost REAL NOT NULL,
            wall_time_s REAL NOT NULL,
            api_input_tokens INTEGER NOT NULL DEFAULT 0,
            api_output_tokens INTEGER NOT NULL DEFAULT 0,
            api_cost_usd REAL NOT NULL DEFAULT 0.0
        )
    """)
    db
end

function save_results(db_path::String, agent::String, runs::Vector{SeedResult};
                      model::String="")
    db = init_db(db_path)
    ts = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")

    for r in runs
        total_in = sum(rec.api_input_tokens for rec in r.records)
        total_out = sum(rec.api_output_tokens for rec in r.records)
        total_cost = api_cost_usd(model, total_in, total_out)

        # Delete old run for this agent+seed (CASCADE deletes questions)
        DBInterface.execute(db,
            "DELETE FROM runs WHERE agent = ? AND seed = ?",
            (agent, r.seed))

        DBInterface.execute(db, """
            INSERT INTO runs (agent, seed, timestamp, total_score, total_reward,
                              total_tool_cost, wall_time_s,
                              total_api_input_tokens, total_api_output_tokens, total_api_cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (agent, r.seed, ts, r.total_score, r.total_reward,
              r.total_tool_cost, r.wall_time_s, total_in, total_out, total_cost))

        run_id = SQLite.last_insert_rowid(db)

        for (qi, rec) in enumerate(r.records)
            q_cost = api_cost_usd(model, rec.api_input_tokens, rec.api_output_tokens)
            tq_json = JSON3.write(rec.tools_queried)
            tr_json = JSON3.write(Dict(string(k) => v for (k, v) in rec.tool_responses))

            DBInterface.execute(db, """
                INSERT INTO questions (run_id, question_idx, question_id, category,
                                       tools_queried, tool_responses, submitted, was_correct,
                                       reward, tool_cost, wall_time_s,
                                       api_input_tokens, api_output_tokens, api_cost_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, qi, rec.question_id, rec.category,
                  tq_json, tr_json,
                  rec.submitted, rec.was_correct === nothing ? nothing : Int(rec.was_correct),
                  rec.reward, rec.tool_cost, rec.wall_time_s,
                  rec.api_input_tokens, rec.api_output_tokens, q_cost))
        end
    end

    close(db)
    println(stderr, "Results saved to $db_path (agent=$agent, $(length(runs)) seeds)")
end
