#!/usr/bin/env julia
# Role: tests
"""
    test_server.jl — Step 4 of credence-pi: daemon transport.

Covers:
  - SignalQueue: bounded capacity, oldest-dropped-on-overflow.
  - action_to_signal: per-effector wire shape.
  - handle_sensor_event: log-first ordering (the load-bearing invariant)
    including the BDSL-failure case where the log is written but no
    signal is emitted.
  - End-to-end HTTP: POST /sensor returns ack and emits an SSE signal;
    GET /signals receives the signal as a `data:` line.
  - Daemon restart: a fresh DaemonState reconstructs the posterior by
    replaying the log, asserted through `expect`, not struct fields.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "src"))
using Credence
using Credence.Previsions: Identity
using HTTP
using JSON3
using Sockets: listen, getsockname

include(joinpath(@__DIR__, "..", "..", "daemon", "server.jl"))
using .Server
using .Server: DaemonState, SignalQueue, init_state, handle_sensor_event,
               action_to_signal, emit_signal!, snapshot, drain!,
               start_daemon, stop_daemon

const BDSL_DIR = joinpath(@__DIR__, "..", "..", "bdsl")
const TOL = 1e-12

const PASSED = String[]
function ok(name)
    push!(PASSED, name)
    println("PASSED: ", name)
end

# ── 1. SignalQueue overflow ────────────────────────────────────────────

let q = SignalQueue(100)
    for i in 1:101
        Server.enqueue!(q, Dict("signal_id" => "s$i"))
    end
    snap = snapshot(q)
    @assert length(snap) == 100
    @assert snap[1]["signal_id"] == "s2"
    @assert snap[end]["signal_id"] == "s101"
    ok("SignalQueue capacity 100: pushing 101 drops oldest, retains s2..s101")
end

let q = SignalQueue(3)
    for i in 1:5
        Server.enqueue!(q, Dict("signal_id" => "s$i"))
    end
    snap = snapshot(q)
    @assert [s["signal_id"] for s in snap] == ["s3", "s4", "s5"]
    ok("SignalQueue: small capacity case retains the last `cap` signals in order")
end

let q = SignalQueue(100)
    for i in 1:5
        Server.enqueue!(q, Dict("signal_id" => "s$i"))
    end
    drained = drain!(q)
    @assert length(drained) == 5
    @assert isempty(snapshot(q))
    ok("SignalQueue.drain! empties the buffer atomically")
end

# ── 2. action_to_signal ────────────────────────────────────────────────

let
    tool_proposed = Dict{String, Any}(
        "event_type"    => "tool-proposed",
        "event_id"      => "evt_1",
        "proposed_call" => Dict("tool_name" => "bash",
                                "input"     => Dict("command" => "ls -la /tmp")),
    )
    sig = action_to_signal(:ask, tool_proposed)
    @assert sig["effector"] == "ask"
    @assert haskey(sig["parameters"], "text")
    @assert occursin("bash", sig["parameters"]["text"])
    @assert occursin("ls -la /tmp", sig["parameters"]["text"])
    ok("action_to_signal(:ask) renders text from proposed_call.tool_name and input")

    sig_proceed = action_to_signal(:proceed, tool_proposed)
    @assert sig_proceed["effector"] == "proceed"
    @assert isempty(sig_proceed["parameters"])
    ok("action_to_signal(:proceed) carries an empty parameters dict")

    sig_block = action_to_signal(:block, tool_proposed)
    @assert sig_block["effector"] == "block"
    @assert haskey(sig_block["parameters"], "reason")
    @assert !isempty(sig_block["parameters"]["reason"])
    ok("action_to_signal(:block) carries a reason string")

    raised = false
    try
        action_to_signal(:substitute, tool_proposed)
    catch e
        raised = true
        @assert occursin("unknown action", string(e))
    end
    @assert raised
    ok("action_to_signal raises on actions outside the manifest")
end

# ── 3. handle_sensor_event: log-first ordering ─────────────────────────
#
# The user emphasised this in the step-4 instructions: "append to
# observation log FIRST, THEN call into BDSL, THEN emit signal." If
# the BDSL fails, the log must already contain the event. The test
# injects a BDSL-evaluation failure by replacing decide-action in the
# environment with a closure that throws.

function fresh_state(; tmp_log)
    init_state(; bdsl_dir=BDSL_DIR, log_path=tmp_log)
end

let path = tempname() * ".jsonl"
    state = fresh_state(; tmp_log=path)
    # Sabotage decide-action AFTER state init, so the env is otherwise
    # consistent.
    state.env[Symbol("decide-action")] = posterior -> error("simulated BDSL failure")

    event = Dict{String, Any}(
        "event_type"    => "tool-proposed",
        "event_id"      => "evt_failbdsl",
        "session_id"    => "test",
        "proposed_call" => Dict("tool_name" => "bash",
                                "input"     => Dict("command" => "echo")),
    )

    raised = false
    try
        handle_sensor_event(state, event)
    catch e
        raised = true
        @assert occursin("simulated BDSL failure", string(e))
    end
    @assert raised
    ok("handle_sensor_event propagates BDSL failures to the caller")

    # Log already contains the event despite the BDSL failure.
    log_records = Server.ObservationLog.read_log(path)
    @assert length(log_records) == 1
    @assert log_records[1].event["event_id"] == "evt_failbdsl"
    ok("log-first invariant: failed BDSL call still leaves the event in the log")

    # No signal emitted.
    @assert isempty(snapshot(state.signal_queue))
    ok("log-first invariant: no signal emitted when BDSL raises")

    # Restart: fresh env, replay log. Posterior is unchanged (tool-
    # proposed events don't update belief). Assert via expect, not by
    # reading struct fields.
    restarted = fresh_state(; tmp_log=path)
    pristine = fresh_state(; tmp_log=tempname() * ".jsonl")  # empty-log state
    @assert isapprox(expect(restarted.posterior[], Identity()),
                     expect(pristine.posterior[],  Identity());
                     atol=TOL)
    ok("log-first invariant: restart from log reconstructs the same posterior (tool-proposed leaves belief untouched)")

    rm(path)
end

# ── 4. handle_sensor_event: normal flow ────────────────────────────────

let path = tempname() * ".jsonl"
    state = fresh_state(; tmp_log=path)

    # Cold start: tool-proposed should yield :ask (voi gate at Beta(2,2)).
    proposed = Dict{String, Any}(
        "event_type"    => "tool-proposed",
        "event_id"      => "evt_p1",
        "proposed_call" => Dict("tool_name" => "bash",
                                "input"     => Dict("command" => "ls")),
    )
    ack = handle_sensor_event(state, proposed)
    @assert ack["ack"] == true && ack["event_id"] == "evt_p1"
    sigs = snapshot(state.signal_queue)
    @assert length(sigs) == 1
    @assert sigs[1]["effector"] == "ask"
    @assert sigs[1]["in_response_to"] == "evt_p1"
    ok("tool-proposed at cold start: ack returned, single :ask signal queued")

    drain!(state.signal_queue)

    # user-responded yes → conditions on obs=1, then followup-after-response
    # returns :proceed which is emitted with in_response_to=evt_p1.
    user_resp = Dict{String, Any}(
        "event_type"     => "user-responded",
        "event_id"       => "evt_r1",
        "in_response_to" => "evt_p1",
        "response"       => "yes",
    )
    handle_sensor_event(state, user_resp)
    sigs2 = snapshot(state.signal_queue)
    @assert length(sigs2) == 1
    @assert sigs2[1]["effector"] == "proceed"
    @assert sigs2[1]["in_response_to"] == "evt_p1"
    ok("user-responded yes: posterior conditions and a :proceed follow-up is emitted")

    # Posterior should now be Beta(3,2). Verify via expect on Identity:
    # E[θ] of Beta(3,2) = 3/5 = 0.6.
    @assert isapprox(expect(state.posterior[], Identity()), 3.0 / 5.0; atol=TOL)
    ok("user-responded yes folds through observe-response: E[θ] = 3/5 (FP-bounded)")

    drain!(state.signal_queue)

    # tool-completed: logged but no signal, no belief change.
    completed = Dict{String, Any}(
        "event_type"     => "tool-completed",
        "event_id"       => "evt_c1",
        "in_response_to" => "evt_p1",
        "outcome"        => Dict("success" => true, "duration_ms" => 12),
    )
    handle_sensor_event(state, completed)
    @assert isempty(snapshot(state.signal_queue))
    @assert isapprox(expect(state.posterior[], Identity()), 3.0 / 5.0; atol=TOL)
    ok("tool-completed: log-only, no signal emitted, posterior unchanged")

    # Verify log accumulated all three events.
    records = Server.ObservationLog.read_log(path)
    @assert [r.event["event_type"] for r in records] ==
            ["tool-proposed", "user-responded", "tool-completed"]
    ok("handle_sensor_event flow appends log entries in order")

    rm(path)
end

# ── 4b. handle_sensor_event: turn-cost is log-only ─────────────────────
#
# Move 1 (Pass 2): the OpenClaw-plugin body emits per-turn token/USD
# cost via its llm_output hook. The daemon logs it (for the dollars-
# saved surface and the cost-denominated utility, both Move 2) but does
# NOT condition the posterior and emits NO signal — exactly like
# tool-completed.

let path = tempname() * ".jsonl"
    state = fresh_state(; tmp_log=path)
    before = expect(state.posterior[], Identity())

    turn_cost = Dict{String, Any}(
        "event_type"   => "turn-cost",
        "event_id"     => "evt_tc1",
        "session_id"   => "test",
        "usd"          => 0.0123,
        "total_tokens" => 1500,
        "model"        => "claude-opus-4-8",
    )
    ack = handle_sensor_event(state, turn_cost)
    @assert ack["ack"] == true && ack["event_id"] == "evt_tc1"
    @assert isempty(snapshot(state.signal_queue))
    @assert isapprox(expect(state.posterior[], Identity()), before; atol=TOL)
    ok("turn-cost: log-only, no signal emitted, posterior unchanged")

    records = Server.ObservationLog.read_log(path)
    @assert length(records) == 1
    @assert records[1].event["event_type"] == "turn-cost"
    @assert records[1].event["usd"] == 0.0123
    @assert records[1].event["total_tokens"] == 1500
    ok("turn-cost: logged with cost fields intact for the Move-2 dollars-saved surface")

    rm(path)
end

# ── 5. End-to-end: HTTP /sensor + SSE /signals ─────────────────────────
#
# Spin up a real HTTP server on an ephemeral port, open an SSE
# subscriber, POST a sensor event, and verify the SSE stream produces
# the expected `data:` line carrying the effector signal.

function pick_port()
    # Bind ephemeral, read assigned port, close. Race-y but adequate
    # for a single test run on a developer machine.
    sock = listen(0)
    _, port = getsockname(sock)
    close(sock)
    Int(port)
end

let path = tempname() * ".jsonl"
    port = pick_port()
    server, state = start_daemon(; port=port, log_path=path, bdsl_dir=BDSL_DIR)
    try
        # SSE consumer in a background task. Reads up to the first
        # `data:` line and stops.
        sse_url = "http://127.0.0.1:$port/signals"
        sse_received = Channel{String}(1)
        sse_task = @async begin
            try
                HTTP.open("GET", sse_url; readtimeout=10) do http
                    while !eof(http)
                        line = readline(http)
                        if startswith(line, "data: ")
                            put!(sse_received, String(line[7:end]))
                            break
                        end
                    end
                end
            catch e
                # Likely closed mid-read once we hit `break`; tolerate.
            end
        end

        # Give the SSE handler a moment to register before posting.
        sleep(0.2)

        # POST a tool-proposed event.
        proposed = Dict{String, Any}(
            "event_type"    => "tool-proposed",
            "event_id"      => "evt_http_1",
            "proposed_call" => Dict("tool_name" => "bash",
                                    "input"     => Dict("command" => "ls")),
        )
        resp = HTTP.post("http://127.0.0.1:$port/sensor",
                         ["Content-Type" => "application/json"],
                         JSON3.write(proposed); readtimeout=10)
        @assert resp.status == 200
        ack = JSON3.read(String(resp.body), Dict{String, Any})
        @assert ack["ack"] == true
        @assert ack["event_id"] == "evt_http_1"
        ok("HTTP POST /sensor returns 200 with {ack=true, event_id}")

        # Wait for the SSE consumer to receive its `data:` line.
        sse_data = take!(sse_received)
        sig = JSON3.read(sse_data, Dict{String, Any})
        @assert sig["signal_type"] == "effector"
        @assert sig["effector"] == "ask"
        @assert sig["in_response_to"] == "evt_http_1"
        ok("HTTP GET /signals SSE delivers the effector signal as a `data:` line")
    finally
        stop_daemon(server, state)
        rm(path; force=true)
    end
end

# ── 6. Daemon restart reconstructs posterior from log ─────────────────

let path = tempname() * ".jsonl"
    # Session A: post a sequence of user-responded events through the
    # in-process handle_sensor_event, building up the posterior.
    state_a = fresh_state(; tmp_log=path)
    for (eid, response) in (("u1", "yes"), ("u2", "yes"), ("u3", "no"),
                            ("u4", "yes"))
        handle_sensor_event(state_a, Dict{String, Any}(
            "event_type"     => "user-responded",
            "event_id"       => eid,
            "in_response_to" => "p?",
            "response"       => response,
        ))
    end
    mean_a = expect(state_a.posterior[], Identity())

    # Session B: fresh state, same log path. init_state replays.
    state_b = fresh_state(; tmp_log=path)
    mean_b = expect(state_b.posterior[], Identity())

    @assert isapprox(mean_a, mean_b; atol=TOL)
    ok("daemon restart: replay reconstructs posterior; E[θ] matches in-process within 1e-12")

    # Second moment too (routes through Gauss-Jacobi quadrature).
    sq_a = expect(state_a.posterior[], θ -> θ^2)
    sq_b = expect(state_b.posterior[], θ -> θ^2)
    @assert isapprox(sq_a, sq_b; atol=TOL)
    ok("daemon restart: E[θ²] matches in-process within 1e-12")

    rm(path)
end

println()
println("=" ^ 60)
println("ALL ", length(PASSED), " ASSERTIONS PASSED")
println("=" ^ 60)
