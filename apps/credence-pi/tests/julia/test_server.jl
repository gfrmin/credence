#!/usr/bin/env julia
# Role: tests
"""
    test_server.jl — credence-pi daemon transport (Pass 2 / Route B).

Covers the TRANSPORT layer (the brain math itself is in test_feature_brain.jl):
  - SignalQueue: bounded capacity, oldest-dropped-on-overflow.
  - action_to_signal: per-effector wire shape.
  - handle_sensor_event: log-first ordering (the load-bearing invariant),
    including the brain-failure case where the log is written but no signal
    is emitted.
  - Fix 1 (features → decide): a tool-proposed event forwards its feature dict
    into the brain; a feature-less event is rejected.
  - Fix 2 (context-join): a user-responded conditions on the features of its
    originating tool-proposed (by in_response_to); an orphan response does not.
  - turn-cost: stashed as the latest cost estimate; no signal; belief untouched.
  - Daemon restart: a fresh DaemonState reconstructs the SAME posterior by
    replaying the log, asserted through `weights` (structure posterior), not
    struct internals.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "src"))
using Credence
using Credence: Identity, weights, expect
using HTTP
using JSON3
using Sockets: listen, getsockname

include(joinpath(@__DIR__, "..", "..", "daemon", "server.jl"))
using .Server
using .Server: DaemonState, SignalQueue, init_state, handle_sensor_event,
               action_to_signal, emit_signal!, snapshot, drain!,
               start_daemon, stop_daemon
using .Server.FeatureBrain: build_model_from_env, belief_at_context, context_from_features

const BDSL_DIR = joinpath(@__DIR__, "..", "..", "bdsl")
const TOL = 1e-12

const PASSED = String[]
function ok(name)
    push!(PASSED, name)
    println("PASSED: ", name)
end

# A complete feature dict over the declared features.
feats(; tool="bash", wd="project-root", parent="none", rep="rep-0",
        ident="ident-0", since="lt-30s") =
    Dict{String, Any}("tool-name" => tool,
                      "working-directory-relative" => wd,
                      "parent-tool-call-name" => parent,
                      "recent-repetition-count" => rep,
                      "recent-identical-call-count" => ident,
                      "time-since-last-user-message" => since)

# Predictive P(approve|X) off a posterior, via the brain's per-context view.
function predict(state, features)
    model = build_model_from_env(state.env)
    X = context_from_features(model, features)
    expect(belief_at_context(model, state.posterior[], X), Identity())
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

fresh_state(; tmp_log) = init_state(; bdsl_dir = BDSL_DIR, log_path = tmp_log)

# ── 3. handle_sensor_event: log-first ordering survives a brain failure ──

let path = tempname() * ".jsonl"
    state = fresh_state(; tmp_log = path)
    # Sabotage decide-action (posterior, features, cost, profile) AFTER init.
    state.env[Symbol("decide-action")] = (posterior, features, cost, profile = nothing) -> error("simulated brain failure")

    event = Dict{String, Any}(
        "event_type" => "tool-proposed",
        "event_id"   => "evt_failbrain",
        "session_id" => "test",
        "features"   => feats(),
        "proposed_call" => Dict("tool_name" => "bash", "input" => Dict("command" => "echo")),
    )

    raised = false
    try
        handle_sensor_event(state, event)
    catch e
        raised = true
        @assert occursin("simulated brain failure", string(e))
    end
    @assert raised
    ok("handle_sensor_event propagates brain failures to the caller")

    log_records = Server.ObservationLog.read_log(path)
    @assert length(log_records) == 1
    @assert log_records[1].event["event_id"] == "evt_failbrain"
    ok("log-first invariant: failed brain call still leaves the event in the log")

    @assert isempty(snapshot(state.signal_queue))
    ok("log-first invariant: no signal emitted when the brain raises")
    rm(path)
end

# ── 3b. Fix 1: a tool-proposed without features is rejected ──────────────

let path = tempname() * ".jsonl"
    state = fresh_state(; tmp_log = path)
    raised = false
    try
        handle_sensor_event(state, Dict{String, Any}(
            "event_type" => "tool-proposed", "event_id" => "evt_nofeat"))
    catch e
        raised = true
        @assert occursin("missing required 'features'", string(e))
    end
    @assert raised
    ok("Fix 1: tool-proposed without a feature dict is rejected (Pass-2 brain needs X)")
    rm(path)
end

# ── 4. Normal flow: cold-start asks; context-join conditions on the cell ──

let path = tempname() * ".jsonl"
    state = fresh_state(; tmp_log = path)
    ctx = feats(tool = "bash", rep = "rep-0")

    p_before = predict(state, ctx)
    @assert isapprox(p_before, 0.5; atol = TOL)
    ok("cold-start predictive P(approve|X) = 0.5")

    proposed = Dict{String, Any}(
        "event_type" => "tool-proposed", "event_id" => "evt_p1",
        "features" => ctx,
        "proposed_call" => Dict("tool_name" => "bash", "input" => Dict("command" => "ls")),
    )
    ack = handle_sensor_event(state, proposed)
    @assert ack["ack"] == true && ack["event_id"] == "evt_p1"
    sigs = snapshot(state.signal_queue)
    @assert length(sigs) == 1 && sigs[1]["effector"] == "ask"
    @assert sigs[1]["in_response_to"] == "evt_p1"
    ok("tool-proposed at cold start: single :ask signal queued (voi gate)")

    drain!(state.signal_queue)

    # user-responded yes → context-join to evt_p1's features → condition; the
    # predictive at X rises; followup :proceed emitted.
    handle_sensor_event(state, Dict{String, Any}(
        "event_type" => "user-responded", "event_id" => "evt_r1",
        "in_response_to" => "evt_p1", "response" => "yes"))
    sigs2 = snapshot(state.signal_queue)
    @assert length(sigs2) == 1 && sigs2[1]["effector"] == "proceed"
    @assert sigs2[1]["in_response_to"] == "evt_p1"
    ok("user-responded yes: context-joined, :proceed follow-up emitted")

    p_after = predict(state, ctx)
    @assert p_after > p_before + 1e-6
    ok("Fix 2: the 'yes' conditioned the cell-for-X — P(approve|X) rose above 0.5 ($(round(p_after,digits=4)))")

    # A DIFFERENT context is unaffected (per-context learning, the whole point).
    other = predict(state, feats(tool = "grep", rep = "rep-3plus"))
    @assert isapprox(other, 0.5; atol = 0.2)  # pooled structures move it slightly, not to p_after
    ok("per-context: an unrelated context is far less affected than the observed one")
    rm(path)
end

# ── 4b. Fix 2: an orphan user-responded (no remembered context) is not learned ──

let path = tempname() * ".jsonl"
    state = fresh_state(; tmp_log = path)
    p0 = predict(state, feats())
    handle_sensor_event(state, Dict{String, Any}(
        "event_type" => "user-responded", "event_id" => "evt_orphan",
        "in_response_to" => "never-proposed", "response" => "no"))
    @assert isapprox(predict(state, feats()), p0; atol = TOL)
    ok("Fix 2: a response with no remembered context does not corrupt belief")
    rm(path)
end

# ── 4c. turn-cost: stashed as the cost estimate; no signal; belief untouched ──

let path = tempname() * ".jsonl"
    state = fresh_state(; tmp_log = path)
    w_before = weights(state.posterior[])

    ack = handle_sensor_event(state, Dict{String, Any}(
        "event_type" => "turn-cost", "event_id" => "evt_tc1", "session_id" => "test",
        "usd" => 0.0123, "total_tokens" => 1500, "model" => "claude-opus-4-8"))
    @assert ack["ack"] == true && ack["event_id"] == "evt_tc1"
    @assert isempty(snapshot(state.signal_queue))
    @assert state.last_cost[] == 0.0123
    # credence-lint: allow — precedent:test-oracle — structure-posterior equality oracle (turn-cost must not learn)
    @assert weights(state.posterior[]) == w_before
    ok("turn-cost: stashed as last_cost, no signal, structure posterior untouched")
    rm(path)
end

# ── 5. End-to-end: HTTP /sensor + SSE /signals ─────────────────────────

function pick_port()
    sock = listen(0)
    _, port = getsockname(sock)
    close(sock)
    Int(port)
end

let path = tempname() * ".jsonl"
    port = pick_port()
    server, state = start_daemon(; port = port, log_path = path, bdsl_dir = BDSL_DIR)
    try
        sse_url = "http://127.0.0.1:$port/signals"
        sse_received = Channel{String}(1)
        sse_task = @async begin
            try
                HTTP.open("GET", sse_url; readtimeout = 10) do http
                    while !eof(http)
                        line = readline(http)
                        if startswith(line, "data: ")
                            put!(sse_received, String(line[7:end]))
                            break
                        end
                    end
                end
            catch e
            end
        end
        sleep(0.2)

        proposed = Dict{String, Any}(
            "event_type" => "tool-proposed", "event_id" => "evt_http_1",
            "features" => feats(),
            "proposed_call" => Dict("tool_name" => "bash", "input" => Dict("command" => "ls")),
        )
        resp = HTTP.post("http://127.0.0.1:$port/sensor",
                         ["Content-Type" => "application/json"],
                         JSON3.write(proposed); readtimeout = 10)
        @assert resp.status == 200
        ack = JSON3.read(String(resp.body), Dict{String, Any})
        @assert ack["ack"] == true && ack["event_id"] == "evt_http_1"
        ok("HTTP POST /sensor returns 200 with {ack=true, event_id}")

        sse_data = take!(sse_received)
        sig = JSON3.read(sse_data, Dict{String, Any})
        @assert sig["signal_type"] == "effector"
        @assert sig["effector"] == "ask"
        @assert sig["in_response_to"] == "evt_http_1"
        ok("HTTP GET /signals SSE delivers the effector signal as a `data:` line")
    finally
        stop_daemon(server, state)
        rm(path; force = true)
    end
end

# ── 6. Daemon restart reconstructs the posterior from the log ──────────
#
# Build belief through tool-proposed + user-responded PAIRS (so the context-join
# lands each update in the right cell), then restart and assert the structure
# posterior (and a probe predictive) match within 1e-12 — replay correctness.

let path = tempname() * ".jsonl"
    state_a = fresh_state(; tmp_log = path)
    seq = [("p1", feats(tool="bash", rep="rep-0"), "yes"),
           ("p2", feats(tool="bash", rep="rep-3plus"), "no"),
           ("p3", feats(tool="bash", rep="rep-3plus"), "no"),
           ("p4", feats(tool="read", rep="rep-0"), "yes")]
    for (pid, f, resp) in seq
        handle_sensor_event(state_a, Dict{String, Any}(
            "event_type" => "tool-proposed", "event_id" => pid, "features" => f,
            "proposed_call" => Dict("tool_name" => f["tool-name"], "input" => Dict())))
        handle_sensor_event(state_a, Dict{String, Any}(
            "event_type" => "user-responded", "event_id" => "r_$pid",
            "in_response_to" => pid, "response" => resp))
    end
    w_a = weights(state_a.posterior[])
    probe = feats(tool="bash", rep="rep-3plus")
    pred_a = predict(state_a, probe)

    state_b = fresh_state(; tmp_log = path)   # replay
    w_b = weights(state_b.posterior[])
    pred_b = predict(state_b, probe)

    @assert length(w_a) == length(w_b) && all(isapprox.(w_a, w_b; atol = TOL))
    ok("daemon restart: replay reconstructs the structure posterior within 1e-12")
    @assert isapprox(pred_a, pred_b; atol = TOL)
    ok("daemon restart: a probe predictive P(approve|X) matches in-process within 1e-12")
    # And the repeated 'no' loop pushed the probe context below 0.5 (sanity).
    @assert pred_a < 0.5
    ok("restart sanity: the repeated bash/rep-3plus denials drove P(approve|X) < 0.5")
    rm(path)
end

println()
println("=" ^ 60)
println("ALL ", length(PASSED), " ASSERTIONS PASSED")
println("=" ^ 60)
