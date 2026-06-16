# Role: brain
"""
    observation_log.jl — append-only JSONL observation log for credence-pi.

The canonical de Finetti record. Each line wraps one sensor event:

    { "schema": "credence-pi/v1",
      "received_at": "2026-05-04T08:34:11.421Z",
      "event": { ... verbatim sensor event ... } }

`append_event!` writes the wrapped record and fsync's the underlying
file descriptor before returning. Correctness over throughput, per
SPEC.md §"Observation log"; a daemon crash between an effector signal
emission and a successful append must not lose the observation that
prompted the signal.

`read_log` and `replay_user_responses` skip malformed lines (JSON parse
error, missing/unrecognised schema) with a warning rather than
crashing. The log accumulates over time and forward-compatibility under
schema bumps depends on tolerating lines the current code can't parse.

The reasoner sits in BDSL: this module never calls `condition` or
`expect`. It is wire/IO only.
"""
module ObservationLog

using Dates: now, UTC, format
using JSON3

export append_event!, read_log, replay_user_responses, replay_contexts, LogRecord
export replay_route_outcomes
export SCHEMA_VERSION

const SCHEMA_VERSION = "credence-pi/v1"

"""
    LogRecord(schema, received_at, event)

One parsed log line. `event` is held as `Dict{String,Any}` so downstream
code (BDSL `lookup`, the daemon dispatcher) sees the same shape it
would see for a freshly arrived sensor event.
"""
struct LogRecord
    schema::String
    received_at::String
    event::Dict{String, Any}
end

# ── Append ─────────────────────────────────────────────────────────────

"""
    append_event!(path::AbstractString, event::AbstractDict) -> LogRecord

Wrap `event` in a `{schema, received_at, event}` envelope, append one
JSON line to `path`, and fsync the file descriptor. Returns the wrapped
record.

`path`'s parent directory must exist; this function does not create it
(the daemon owns lifecycle decisions about `~/.credence-pi/`).
"""
function append_event!(path::AbstractString, event::AbstractDict)::LogRecord
    received_at = format(now(UTC), "yyyy-mm-ddTHH:MM:SS.sssZ")
    record = Dict{String, Any}(
        "schema"      => SCHEMA_VERSION,
        "received_at" => received_at,
        "event"       => event,
    )
    line = JSON3.write(record)
    open(path, "a") do io
        write(io, line)
        write(io, '\n')
        flush(io)
        fd = Base.fd(io)
        ret = ccall(:fsync, Cint, (Cint,), fd)
        ret == 0 || error("fsync($path) failed: errno=$(Libc.errno())")
    end
    LogRecord(SCHEMA_VERSION, received_at,
              Dict{String, Any}(string(k) => v for (k, v) in event))
end

# ── Read ───────────────────────────────────────────────────────────────

"""
    read_log(path::AbstractString) -> Vector{LogRecord}

Read every line of `path` and parse into `LogRecord`s in file order.
Returns `LogRecord[]` if the file is missing or empty.

Lines that fail to JSON-parse, or that lack the schema field, or whose
schema does not match `SCHEMA_VERSION`, are skipped with `@warn` rather
than raising — see module docstring for rationale.
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
        @warn "observation log: skipping malformed JSON line" path lineno error=e
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
    received_at = get(parsed, :received_at, "")
    event_obj = get(parsed, :event, nothing)
    if event_obj === nothing
        @warn "observation log: skipping line with no event field" path lineno
        return nothing
    end
    event_dict = Dict{String, Any}(string(k) => v for (k, v) in event_obj)
    LogRecord(String(schema), String(received_at), event_dict)
end

# ── Replay ─────────────────────────────────────────────────────────────

"""
    replay_user_responses(path::AbstractString) -> Vector{Int}

Read the log and return the sequence of binary observations to fold
through the BDSL's `observe-response`: 1 for `response == "yes"`, 0 for
`response == "no"`. `tool-proposed`, `tool-completed`, and
`user-responded` events with `response ∈ {"timeout", …}` are silently
skipped — Pass 1's BDSL conditions only on yes/no.

This is the canonical replay path the daemon uses at startup to
reconstruct the posterior from the log. The single conversion of the
string "yes"/"no" wire format to the integer obs the BDSL takes lives
here.
"""
function replay_user_responses(path::AbstractString)::Vector{Int}
    obs = Int[]
    for record in read_log(path)
        event_type = get(record.event, "event_type", "")
        event_type == "user-responded" || continue
        response = get(record.event, "response", "")
        if response == "yes"
            push!(obs, 1)
        elseif response == "no"
            push!(obs, 0)
        end
    end
    obs
end

"""
    replay_contexts(path::AbstractString) -> Vector{Tuple{Any, Int}}

Pass-2 replay: reconstruct the sequence of `(features, obs)` pairs to fold through
the feature-conditioned `observe-response`. A feature-conditioned update needs
*which cell* — i.e. the features of the `tool-proposed` that this `user-responded`
answers (joined by `in_response_to == tool-proposed.event_id`).

Single pass: remember each `tool-proposed`'s `features` by `event_id`, and when a
`user-responded` (yes/no) arrives, emit `(features, obs)`. A response whose
originating context is missing from the log (malformed / truncated) is skipped —
the daemon cannot place it in a cell, and a guessed context would corrupt belief.
This mirrors the live context-join in `handle_sensor_event`.
"""
function replay_contexts(path::AbstractString)::Vector{Tuple{Any, Int}}
    contexts = Dict{String, Any}()
    out = Tuple{Any, Int}[]
    for record in read_log(path)
        event_type = get(record.event, "event_type", "")
        if event_type == "tool-proposed"
            event_id = string(get(record.event, "event_id", ""))
            features = get(record.event, "features", nothing)
            features === nothing || (contexts[event_id] = features)
        elseif event_type == "user-responded"
            response = get(record.event, "response", "")
            (response == "yes" || response == "no") || continue
            irt = string(get(record.event, "in_response_to", ""))
            features = get(contexts, irt, nothing)
            features === nothing && continue
            push!(out, (features, response == "yes" ? 1 : 0))
        end
    end
    out
end

"""
    replay_route_outcomes(path) -> Vector{Tuple{String, Any, Bool, Union{Nothing, Bool}}}

Pass-2 ROUTING replay: the sequence of `(model_id, features, success, human)` to fold
through `RoutingBrain.route_outcome!` on top of the warm `tops`, reconstructing the live
routing belief after a restart. Reads the derived `route-outcome` events the daemon appends
when it credits a routed model from a tool-completed. Because `route_outcome!` is
deterministic given the event order, replaying this sequence reproduces the live belief
EXACTLY (no summary approximation) — the same durability the governance posterior gets from
`replay_contexts`. A record missing `model`/`features`/`success` is skipped (cannot be
placed). `human` is `nothing` unless a human approve/reject was recorded for the turn.
"""
function replay_route_outcomes(path::AbstractString)
    out = Tuple{String, Any, Bool, Union{Nothing, Bool}}[]
    for record in read_log(path)
        get(record.event, "event_type", "") == "route-outcome" || continue
        model = get(record.event, "model", nothing)
        features = get(record.event, "features", nothing)
        success = get(record.event, "success", nothing)
        (model === nothing || features === nothing || success === nothing) && continue
        h = get(record.event, "human", nothing)
        human = h === nothing ? nothing : Bool(h)
        push!(out, (String(model), features, Bool(success), human))
    end
    out
end

end # module ObservationLog
