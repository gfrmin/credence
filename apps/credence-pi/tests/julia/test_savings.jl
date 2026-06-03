#!/usr/bin/env julia
# Role: tests
"""
    test_savings.jl — the dollars-saved surface (display-only reporting).

Builds a synthetic observation log and asserts savings_report's tallies:
spend (per-turn cost), governance (asked / approved / denied), and the
explicitly-bounded prevented-spend estimate. Also checks the empty-log
case and that the rendered report carries the ESTIMATE caveat.
"""

include(joinpath(@__DIR__, "..", "..", "savings.jl"))
using .Savings: savings_report, format_report

const PASSED = String[]
ok(name) = (push!(PASSED, name); println("PASSED: ", name))

let path = tempname() * ".jsonl"
    A = Savings.ObservationLog.append_event!
    # Two priced turns: $0.10 (1000 tok) + $0.30 (3000 tok); avg $0.20.
    A(path, Dict("event_type" => "turn-cost", "session_id" => "s1", "usd" => 0.10, "total_tokens" => 1000))
    A(path, Dict("event_type" => "turn-cost", "session_id" => "s1", "usd" => 0.30, "total_tokens" => 3000))
    # Three proposed tool calls, with the "decision" records the daemon
    # emits alongside each effector signal:
    #   e1: ask -> user denies -> followup block   (a prevented call)
    #   e2: ask -> user approves -> followup proceed -> completed
    #   e3: auto-block, no ask                       (a prevented call)
    A(path, Dict("event_type" => "tool-proposed", "event_id" => "e1", "session_id" => "s1",
                 "proposed_call" => Dict("tool_name" => "bash")))
    A(path, Dict("event_type" => "decision", "in_response_to" => "e1", "action" => "ask"))
    A(path, Dict("event_type" => "user-responded", "in_response_to" => "e1", "response" => "no"))
    A(path, Dict("event_type" => "decision", "in_response_to" => "e1", "action" => "block"))

    A(path, Dict("event_type" => "tool-proposed", "event_id" => "e2", "session_id" => "s1",
                 "proposed_call" => Dict("tool_name" => "bash")))
    A(path, Dict("event_type" => "decision", "in_response_to" => "e2", "action" => "ask"))
    A(path, Dict("event_type" => "user-responded", "in_response_to" => "e2", "response" => "yes"))
    A(path, Dict("event_type" => "decision", "in_response_to" => "e2", "action" => "proceed"))
    A(path, Dict("event_type" => "tool-completed", "event_id" => "c1", "in_response_to" => "e2"))

    A(path, Dict("event_type" => "tool-proposed", "event_id" => "e3", "session_id" => "s1",
                 "proposed_call" => Dict("tool_name" => "read")))
    A(path, Dict("event_type" => "decision", "in_response_to" => "e3", "action" => "block"))

    r = savings_report(path)
    @assert r.total_events == 13
    @assert r.sessions == 1
    @assert r.priced_turns == 2
    @assert isapprox(r.total_usd, 0.40; atol = 1e-12)
    @assert r.total_tokens == 4000
    @assert isapprox(r.avg_usd_per_turn, 0.20; atol = 1e-12)
    @assert r.tool_calls_proposed == 3
    @assert r.decided_ask == 2
    @assert r.decided_block == 2      # e1 followup block + e3 auto-block
    @assert r.decided_proceed == 1
    @assert r.prevented_calls == 2
    @assert r.asked == 2
    @assert r.approved == 1
    @assert r.denied == 1
    @assert r.completed == 1
    @assert r.denied_tools == ["bash"]
    # 2 blocked calls * $0.20 avg turn = $0.40 estimated prevented.
    @assert isapprox(r.estimated_prevented_usd, 0.40; atol = 1e-12)
    ok("savings_report tallies spend, brain decisions (incl. silent auto-blocks), and the prevented estimate")

    io = IOBuffer()
    format_report(io, r)
    s = String(take!(io))
    @assert occursin("ESTIMATE", s)
    @assert occursin("credence-pi", s)
    ok("format_report renders the report with the estimate caveat")

    rm(path)
end

let path = tempname() * ".jsonl"  # never created
    r = savings_report(path)
    @assert r.total_events == 0
    @assert r.total_usd == 0.0
    @assert r.denied == 0
    io = IOBuffer()
    format_report(io, r)
    @assert occursin("No observations yet", String(take!(io)))
    ok("savings_report on a missing/empty log yields an all-zero report")
    isfile(path) && rm(path)
end

println()
println("=" ^ 60)
println("ALL ", length(PASSED), " ASSERTIONS PASSED")
println("=" ^ 60)
