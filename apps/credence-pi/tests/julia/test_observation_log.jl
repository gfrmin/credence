#!/usr/bin/env julia
# Role: tests
"""
    test_observation_log.jl — the credence-pi observation log.

Round-trip, schema rejection, malformed-line tolerance, the Pass-1
`replay_user_responses` subset extraction, and the Pass-2
`replay_contexts` join (rejoin each user-responded to the features of its
originating tool-proposed, by in_response_to). The end-to-end "a fresh
daemon replays the log and recovers the same posterior" property lives in
test_server.jl §6, asserted through the public `weights` accessor.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "src"))
using Credence

include(joinpath(@__DIR__, "..", "..", "daemon", "observation_log.jl"))
using .ObservationLog

const PASSED = String[]
function ok(name)
    push!(PASSED, name)
    println("PASSED: ", name)
end

const TOL = 1e-12

# ── 1. Round trip: append → read ────────────────────────────────────────

let path = tempname() * ".jsonl"
    e1 = Dict("event_type" => "tool-proposed", "event_id" => "evt_1")
    e2 = Dict("event_type" => "user-responded", "in_response_to" => "evt_1",
              "response" => "yes")
    e3 = Dict("event_type" => "tool-completed", "event_id" => "evt_3",
              "in_response_to" => "evt_1")

    r1 = append_event!(path, e1)
    r2 = append_event!(path, e2)
    r3 = append_event!(path, e3)

    @assert r1.schema == "credence-pi/v1"
    @assert r1.event["event_type"] == "tool-proposed"
    @assert r2.event["response"] == "yes"

    records = read_log(path)
    @assert length(records) == 3
    @assert all(r -> r.schema == "credence-pi/v1", records)
    @assert [r.event["event_type"] for r in records] ==
            ["tool-proposed", "user-responded", "tool-completed"]
    ok("append_event! → read_log round-trip preserves order, schema, and event payload")

    rm(path)
end

# ── 2. Schema enforcement ───────────────────────────────────────────────
#
# The log file may contain lines from a future schema, hand-written
# debugging entries, or partial writes from a crashed process. These
# must not crash the daemon at startup.

let path = tempname() * ".jsonl"
    # One good line via the public API.
    append_event!(path, Dict("event_type" => "user-responded",
                             "response" => "yes"))

    # Hand-write three bad lines: missing schema, wrong schema, malformed JSON.
    open(path, "a") do io
        write(io, """{"received_at":"x","event":{"event_type":"user-responded","response":"yes"}}\n""")
        write(io, """{"schema":"credence-pi/v999","received_at":"x","event":{"event_type":"user-responded","response":"no"}}\n""")
        write(io, "this is not json at all\n")
    end

    # And one more good line so we know reading continued past the bad ones.
    append_event!(path, Dict("event_type" => "user-responded",
                             "response" => "no"))

    records = redirect_stderr(devnull) do
        read_log(path)
    end
    @assert length(records) == 2
    @assert records[1].event["response"] == "yes"
    @assert records[2].event["response"] == "no"
    ok("read_log skips lines with missing schema, wrong schema, and JSON parse errors")

    rm(path)
end

# ── 3. replay_user_responses extracts the right subset ─────────────────

let path = tempname() * ".jsonl"
    append_event!(path, Dict("event_type" => "tool-proposed", "event_id" => "p1"))
    append_event!(path, Dict("event_type" => "user-responded",
                             "in_response_to" => "p1", "response" => "yes"))
    append_event!(path, Dict("event_type" => "tool-completed",
                             "in_response_to" => "p1"))
    append_event!(path, Dict("event_type" => "user-responded",
                             "response" => "no"))
    append_event!(path, Dict("event_type" => "user-responded",
                             "response" => "timeout"))
    append_event!(path, Dict("event_type" => "user-responded",
                             "response" => "yes"))

    obs = replay_user_responses(path)
    @assert obs == [1, 0, 1]
    ok("replay_user_responses ignores non-user-responded and timeout, maps yes→1 / no→0")

    rm(path)
end

# ── 4. replay_contexts: rejoin each user-responded to its tool-proposed ─
#
# Pass 2: a feature-conditioned update needs WHICH CELL, so replay must carry the
# features of the originating tool-proposed (joined by in_response_to ==
# event_id). Orphan responses (no matching tool-proposed) are dropped — the
# daemon cannot place them in a cell. (The end-to-end "replay reconstructs the
# same posterior" property is asserted in test_server.jl §6.)

let path = tempname() * ".jsonl"
    fA = Dict("tool-name" => "bash", "recent-repetition-count" => "rep-0")
    fB = Dict("tool-name" => "read", "recent-repetition-count" => "rep-3plus")
    append_event!(path, Dict("event_type" => "tool-proposed", "event_id" => "p1", "features" => fA))
    append_event!(path, Dict("event_type" => "user-responded", "in_response_to" => "p1", "response" => "yes"))
    append_event!(path, Dict("event_type" => "tool-completed", "in_response_to" => "p1"))
    # Orphan: a response whose tool-proposed is absent from the log.
    append_event!(path, Dict("event_type" => "user-responded", "in_response_to" => "ghost", "response" => "no"))
    # Timeout is dropped (not yes/no).
    append_event!(path, Dict("event_type" => "tool-proposed", "event_id" => "p2", "features" => fB))
    append_event!(path, Dict("event_type" => "user-responded", "in_response_to" => "p2", "response" => "timeout"))
    append_event!(path, Dict("event_type" => "tool-proposed", "event_id" => "p3", "features" => fB))
    append_event!(path, Dict("event_type" => "user-responded", "in_response_to" => "p3", "response" => "no"))

    ctxs = replay_contexts(path)
    @assert length(ctxs) == 2
    @assert ctxs[1][2] == 1 && ctxs[1][1]["tool-name"] == "bash"
    @assert ctxs[2][2] == 0 && ctxs[2][1]["tool-name"] == "read"
    ok("replay_contexts joins yes/no responses to their tool-proposed features, in order")
    ok("replay_contexts drops orphan responses and timeouts (no cell to place them)")

    rm(path)
end

# ── 5. Replay on a missing log file is a no-op (empty) ─────────────────

let path = tempname() * ".jsonl"   # never created
    @assert !isfile(path)
    @assert read_log(path) == ObservationLog.LogRecord[]
    @assert replay_user_responses(path) == Int[]
    @assert replay_contexts(path) == Tuple{Any, Int}[]
    ok("replay against a missing log file is a no-op (empty contexts)")
end

# ── 6. fsync: hard-power-cut survival is the property we want, but we
# can't simulate that in-process. Instead, we assert the OS-level effect
# we rely on — every line written via append_event! is durable on disk
# at the moment append_event! returns, observable by re-reading the
# file from a different process-style file handle.

let path = tempname() * ".jsonl"
    for i in 1:5
        append_event!(path, Dict("event_type" => "user-responded",
                                 "response" => i % 2 == 0 ? "yes" : "no"))
        # Open a fresh handle each iteration; eachline on it must see
        # exactly i lines.
        n = 0
        open(path, "r") do io
            for line in eachline(io)
                isempty(strip(line)) || (n += 1)
            end
        end
        @assert n == i
    end
    ok("each append_event! is durably visible to a fresh reader before returning")

    rm(path)
end

println()
println("=" ^ 60)
println("ALL ", length(PASSED), " ASSERTIONS PASSED")
println("=" ^ 60)
