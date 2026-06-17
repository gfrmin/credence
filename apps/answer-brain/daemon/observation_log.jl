# Role: brain
"""
    observation_log.jl — append-only JSONL observation log for answer-brain.

The canonical de Finetti record. Each line wraps one event:

    { "schema": "answer-brain/v1",
      "received_at": "2026-06-17T08:34:11.421Z",
      "event": { ... } }

`append_observation!` writes one grounded observation (the unit the brain conditions on —
which candidate index it reports, its ancestry group, and its already-projected reliability
covariates) and fsync's before returning, so a crash between an effector emission and the
append never loses the observation that prompted it.

`replay_observations` reconstructs the conditioning sequence per question in file order;
feeding it back through `AnswerBrain.candidate_posterior` reproduces the posterior exactly
(deterministic replay — the durability property in test_answer_brain.jl §3).

This module is wire/IO only — it never calls `condition` or `expect`. The reasoner is the
brain; the log is a record. Adapted from `apps/credence-pi/daemon/observation_log.jl`.
"""
module ObservationLog

using Dates: now, UTC, format
using JSON3

export append_event!, append_observation!, read_log, replay_observations,
       LogRecord, SCHEMA_VERSION

const SCHEMA_VERSION = "answer-brain/v1"

"""One parsed log line; `event` held as `Dict{String,Any}` (the live event shape)."""
struct LogRecord
    schema::String
    received_at::String
    event::Dict{String, Any}
end

# ── Append ──────────────────────────────────────────────────────────────

"""
    append_event!(path, event::AbstractDict) -> LogRecord

Wrap `event` in `{schema, received_at, event}`, append one JSON line, fsync. The parent
directory must exist (the daemon owns `~/.answer-brain/` lifecycle).
"""
function append_event!(path::AbstractString, event::AbstractDict)::LogRecord
    received_at = format(now(UTC), "yyyy-mm-ddTHH:MM:SS.sssZ")
    record = Dict{String, Any}("schema" => SCHEMA_VERSION,
                               "received_at" => received_at, "event" => event)
    line = JSON3.write(record)
    open(path, "a") do io
        write(io, line)
        write(io, '\n')
        flush(io)
        ret = ccall(:fsync, Cint, (Cint,), Base.fd(io))
        ret == 0 || error("fsync($path) failed: errno=$(Libc.errno())")
    end
    LogRecord(SCHEMA_VERSION, received_at,
              Dict{String, Any}(string(k) => v for (k, v) in event))
end

"""
    append_observation!(path; question_id, reports, group, authority,
                        subject_factor, time_factor) -> LogRecord

Append one grounded observation for `question_id`. The fields are exactly an
`AnswerBrain.Obs` plus the question it belongs to — the integer/float belief inputs, no PII.
"""
function append_observation!(path::AbstractString; question_id,
                             reports::Integer, group::Integer,
                             authority::Real, subject_factor::Real,
                             time_factor::Real)::LogRecord
    append_event!(path, Dict{String, Any}(
        "event_type" => "observation",
        "question_id" => String(question_id),
        "reports" => Int(reports),
        "group" => Int(group),
        "authority" => Float64(authority),
        "subject_factor" => Float64(subject_factor),
        "time_factor" => Float64(time_factor),
    ))
end

# ── Read ────────────────────────────────────────────────────────────────

"""
    read_log(path) -> Vector{LogRecord}

Every line, parsed in file order. Missing file ⇒ empty. Malformed / wrong-schema / no-event
lines are skipped with `@warn` (forward-compatibility under schema bumps), never raised.
"""
function read_log(path::AbstractString)::Vector{LogRecord}
    records = LogRecord[]
    isfile(path) || return records
    open(path, "r") do io
        for (lineno, line) in enumerate(eachline(io))
            isempty(strip(line)) && continue
            record = _parse_line(line, path, lineno)
            record === nothing && continue
            push!(records, record)
        end
    end
    records
end

function _parse_line(line::AbstractString, path::AbstractString,
                     lineno::Integer)::Union{LogRecord, Nothing}
    parsed = try
        JSON3.read(line)
    catch e
        @warn "observation log: skipping malformed JSON line" path lineno error = e
        return nothing
    end
    schema = get(parsed, :schema, nothing)
    if schema === nothing
        @warn "observation log: skipping line with no schema field" path lineno
        return nothing
    end
    if schema != SCHEMA_VERSION
        @warn "observation log: skipping line with unrecognised schema" path lineno schema
        return nothing
    end
    event_obj = get(parsed, :event, nothing)
    if event_obj === nothing
        @warn "observation log: skipping line with no event field" path lineno
        return nothing
    end
    LogRecord(String(schema), String(get(parsed, :received_at, "")),
              Dict{String, Any}(string(k) => v for (k, v) in event_obj))
end

# ── Replay ──────────────────────────────────────────────────────────────

"""
    replay_observations(path) -> Vector{NamedTuple}

The grounded observations in file order, each a NamedTuple
`(question_id, reports, group, authority, subject_factor, time_factor)`. The caller groups by
`question_id` and rebuilds `AnswerBrain.Obs` to reconstruct that question's posterior — the
deterministic replay path. Raw tuples (not `Obs`) keep this module domain-free (wire/IO only).
"""
function replay_observations(path::AbstractString)
    out = NamedTuple[]
    for record in read_log(path)
        get(record.event, "event_type", "") == "observation" || continue
        e = record.event
        push!(out, (question_id = String(get(e, "question_id", "")),
                    reports = Int(e["reports"]), group = Int(e["group"]),
                    authority = Float64(e["authority"]),
                    subject_factor = Float64(e["subject_factor"]),
                    time_factor = Float64(e["time_factor"])))
    end
    out
end

end # module ObservationLog
