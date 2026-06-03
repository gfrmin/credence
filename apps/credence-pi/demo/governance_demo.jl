#!/usr/bin/env julia
# Role: demo (display-only; exercises the real Pass-1 brain end-to-end)
"""
    governance_demo.jl — credence-pi governance, end to end, on a synthetic
    session (no live OpenClaw, no real data required).

Scenario: the agent keeps proposing the same wasteful `bash` call. At cold
start credence-pi is uncertain, so it ASKS (the VOI gate). The user denies
the call each time. The global approval posterior concentrates toward
"refuse", and once it is confident enough the brain AUTO-BLOCKS the call by
expected-utility maximisation — no hard-coded rule. Finally the
dollars-saved surface reports the spend governance flagged.

HONEST FRAMING: the Pass-1 brain holds a single GLOBAL approval posterior.
It learns this user's tool-approval rate; it does NOT yet detect loops
per-context (that is feature-conditioned learning — Pass-2 Phase 2, which
needs accumulated real data). What this demonstrates is the real product
mechanic — govern, learn from the user, auto-block when confident, and
report the caught spend — on the actual brain + wire + log, end to end.

Run:  julia --project=. apps/credence-pi/demo/governance_demo.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "src"))
using Credence
using Credence.Previsions: Identity

include(joinpath(@__DIR__, "..", "daemon", "server.jl"))
using .Server: init_state, handle_sensor_event, snapshot, drain!

include(joinpath(@__DIR__, "..", "savings.jl"))
using .Savings: savings_report, format_report

const BDSL_DIR = joinpath(@__DIR__, "..", "bdsl")

# Post one sensor event through the real daemon dispatcher and return the
# effector the brain emitted (or nothing).
function step!(state, event)::Union{String, Nothing}
    drain!(state.signal_queue)                  # clear any prior follow-up
    handle_sensor_event(state, event)
    sigs = snapshot(state.signal_queue)
    isempty(sigs) ? nothing : string(sigs[1]["effector"])
end

function run_demo(io::IO = stdout)
    path = tempname() * ".jsonl"
    state = init_state(; bdsl_dir = BDSL_DIR, log_path = path)

    println(io, "credence-pi governance demo — the agent repeats a wasteful call;")
    println(io, "the brain asks, the user denies, the brain learns to auto-block.\n")
    println(io, "round | P(approve) | brain decision")
    println(io, "------+------------+----------------")

    first_block = 0
    for k in 1:10
        eid = "evt_$k"
        eff = step!(state, Dict{String, Any}(
            "event_type" => "tool-proposed",
            "event_id" => eid,
            "session_id" => "demo",
            "proposed_call" => Dict("tool_name" => "bash",
                                    "input" => Dict("command" => "npm run build")),
        ))
        p = expect(state.posterior[], Identity())
        decision = eff === nothing ? "(none)" : eff
        println(io, lpad(k, 5), " |   ", rpad(round(p; digits = 3), 7), "  | ", decision)

        if eff == "ask"
            # The user denies the wasteful call; the brain conditions on it.
            step!(state, Dict{String, Any}(
                "event_type" => "user-responded",
                "event_id" => "resp_$k",
                "in_response_to" => eid,
                "response" => "no",
            ))
        elseif eff == "block" && first_block == 0
            first_block = k
        end

        # The turn burned tokens regardless.
        step!(state, Dict{String, Any}(
            "event_type" => "turn-cost",
            "session_id" => "demo",
            "usd" => 0.12,
            "total_tokens" => 1500,
            "model" => "claude-opus-4-8",
        ))
    end

    println(io)
    if first_block > 0
        println(io, ">> By round $first_block the brain had learned enough to AUTO-BLOCK")
        println(io, "   the call by expected-utility maximisation — no hard-coded rule.\n")
    else
        println(io, ">> The brain stayed in ask/learn mode over these rounds.\n")
    end

    format_report(io, savings_report(path))
    rm(path; force = true)
    return first_block
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
