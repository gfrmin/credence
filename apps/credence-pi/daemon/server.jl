# Role: brain
"""
    server.jl — credence-pi daemon transport.

Wire surface:
    POST /sensor   - accept one sensor event; return {"ack": true, ...}
    GET  /signals  - long-lived Server-Sent Events stream of effector
                     signals; one SSE message per signal.

Order of operations in `handle_sensor_event` is non-negotiable:

    1. append to observation log
    2. dispatch into BDSL (decide-action, observe-response)
    3. emit effector signal

If the daemon crashes between step 2 and step 3, the log already
contains the event and replay reconstructs the posterior on restart.
Reversing this order is a silent corruption bug — the log would be
missing events that already changed observable state.

The daemon does no probabilistic reasoning. `decide-action`,
`observe-response`, `followup-after-response` are looked up in the
BDSL environment and called as opaque closures. action symbols (`:ask`,
`:proceed`, `:block`) are translated to wire-shaped effector signal
dicts host-side; that is templating, not reasoning.
"""
module Server

using Dates: now, UTC
using HTTP
using JSON3
using Random: randstring

include(joinpath(@__DIR__, "observation_log.jl"))
using .ObservationLog: append_event!, replay_user_responses

# Pull the parent's Credence module — server.jl runs under the test/run
# loader which has already done `using Credence`.
import Main: Credence
using .Credence: Eval, Parse

export start_daemon, stop_daemon, DaemonState
export SignalQueue, enqueue!, snapshot, drain!
export handle_sensor_event, action_to_signal, emit_signal!

const SIGNAL_QUEUE_CAPACITY = 100

# ── SignalQueue ────────────────────────────────────────────────────────
#
# Bounded in-memory buffer for effector signals while no SSE consumer is
# attached. `enqueue!` evicts the oldest signal when capacity is
# exceeded — newest-survives semantics, per SPEC.md §"Wire transport".
# `drain!` returns and clears the buffer in one critical section.
# `snapshot` returns a copy without clearing (for inspection / tests).

mutable struct SignalQueue
    cap::Int
    buf::Vector{Dict{String, Any}}
    cond::Threads.Condition
end

SignalQueue(cap::Int=SIGNAL_QUEUE_CAPACITY) =
    SignalQueue(cap, Dict{String, Any}[], Threads.Condition())

function enqueue!(q::SignalQueue, signal::AbstractDict)
    sig = Dict{String, Any}(string(k) => v for (k, v) in signal)
    Base.lock(q.cond) do
        push!(q.buf, sig)
        while length(q.buf) > q.cap
            popfirst!(q.buf)   # drop oldest
        end
        notify(q.cond; all=true)
    end
    sig
end

function drain!(q::SignalQueue)::Vector{Dict{String, Any}}
    Base.lock(q.cond) do
        items = copy(q.buf)
        empty!(q.buf)
        items
    end
end

function snapshot(q::SignalQueue)::Vector{Dict{String, Any}}
    Base.lock(q.cond) do
        copy(q.buf)
    end
end

"""
    wait_for_signal(q::SignalQueue) -> Vector

Block until at least one signal is available, then drain and return it.
Used by the SSE handler.
"""
function wait_for_signal(q::SignalQueue)::Vector{Dict{String, Any}}
    Base.lock(q.cond) do
        while isempty(q.buf)
            wait(q.cond)
        end
        items = copy(q.buf)
        empty!(q.buf)
        items
    end
end

# ── DaemonState ────────────────────────────────────────────────────────

mutable struct DaemonState
    env::Eval.Env
    posterior::Ref{Any}        # Measure; held as Any for type stability across condition()
    signal_queue::SignalQueue
    log_path::String
    bdsl_dir::String
    closing::Threads.Atomic{Bool}
end

"""
    load_env(bdsl_dir) -> Eval.Env

Build a fresh BDSL environment with stdlib + the five Pass-1 programs
loaded in dependency order.
"""
function load_env(bdsl_dir::AbstractString)::Eval.Env
    env = Eval.default_env()
    env[:__toplevel__] = true
    stdlib_path = joinpath(@__DIR__, "..", "..", "..", "src", "stdlib.bdsl")
    for expr in Parse.parse_all(read(stdlib_path, String))
        Eval.eval_dsl(expr, env)
    end
    for fname in ("capabilities.bdsl", "features.bdsl",
                  "prior.bdsl", "kernel.bdsl", "decide.bdsl")
        for expr in Parse.parse_all(read(joinpath(bdsl_dir, fname), String))
            Eval.eval_dsl(expr, env)
        end
    end
    env
end

"""
    init_state(; bdsl_dir, log_path) -> DaemonState

Build a fresh daemon state. Calls `load_env`, builds an empty signal
queue, and reconstructs the posterior by replaying every
`user-responded` event in `log_path` through `observe-response`.
"""
function init_state(; bdsl_dir::AbstractString, log_path::AbstractString)::DaemonState
    env = load_env(bdsl_dir)
    posterior = env[Symbol("make-prior")]()
    obs_fn = env[Symbol("observe-response")]
    for o in replay_user_responses(log_path)
        posterior = obs_fn(posterior, o)
    end
    DaemonState(env, Ref{Any}(posterior), SignalQueue(), log_path, bdsl_dir,
                Threads.Atomic{Bool}(false))
end

# ── Signal templating ──────────────────────────────────────────────────

"""
    action_to_signal(action::Symbol, sensor_event::AbstractDict) -> Dict

Translate a BDSL action symbol into the effector dict shape the body
dispatches against. Templating only — no probabilistic reasoning.

Keys produced: `effector` (string), `parameters` (Dict).

`:ask` renders text from `sensor_event["proposed_call"]`. `:block`
renders a generic refusal reason; Pass 1 has no per-event reason
synthesis (the BDSL only returns the action symbol). `:proceed` has
no parameters.
"""
function action_to_signal(action::Symbol, sensor_event::AbstractDict)::Dict{String, Any}
    if action === :ask
        return Dict{String, Any}(
            "effector"   => "ask",
            "parameters" => Dict{String, Any}(
                "text" => render_ask_text(sensor_event)),
        )
    elseif action === :proceed
        return Dict{String, Any}(
            "effector"   => "proceed",
            "parameters" => Dict{String, Any}(),
        )
    elseif action === :block
        return Dict{String, Any}(
            "effector"   => "block",
            "parameters" => Dict{String, Any}(
                "reason" => "Refused based on prior approval observations."),
        )
    else
        error("action_to_signal: unknown action $(action)")
    end
end

function render_ask_text(sensor_event::AbstractDict)::String
    proposed = get(sensor_event, "proposed_call", nothing)
    proposed === nothing && return "Allow this tool call?"
    tool_name = string(get(proposed, "tool_name", "(unknown)"))
    input_summary = _summarise_input(get(proposed, "input", nothing))
    return "Allow `$tool_name` to run with input `$input_summary`?"
end

_summarise_input(::Nothing) = "(none)"
function _summarise_input(input)
    s = JSON3.write(input)
    length(s) > 80 ? string(SubString(s, 1, 80), "…") : s
end

# ── emit_signal! ──────────────────────────────────────────────────────

const _SIGNAL_ID_COUNTER = Threads.Atomic{Int}(0)

"""
    emit_signal!(state, in_response_to::AbstractString, action::Symbol,
                 sensor_event::AbstractDict) -> Dict

Compose the signal envelope and enqueue it on the signal queue. Returns
the enqueued signal (with `signal_type`, `signal_id`, `in_response_to`,
`effector`, `parameters` populated).
"""
function emit_signal!(state::DaemonState, in_response_to::AbstractString,
                      action::Symbol, sensor_event::AbstractDict)::Dict{String, Any}
    eff = action_to_signal(action, sensor_event)
    n = Threads.atomic_add!(_SIGNAL_ID_COUNTER, 1)
    signal = Dict{String, Any}(
        "signal_type"     => "effector",
        "signal_id"       => string("sig_", randstring(6), "_", n),
        "in_response_to"  => in_response_to,
        "effector"        => eff["effector"],
        "parameters"      => eff["parameters"],
    )
    enqueue!(state.signal_queue, signal)
    signal
end

# ── handle_sensor_event ───────────────────────────────────────────────
#
# Order of operations is the load-bearing invariant of the daemon. Read
# the module docstring before changing this function.

"""
    handle_sensor_event(state, event::AbstractDict) -> Dict

The central daemon dispatcher. Order:

    1. Append the event to the observation log (fsync'd).
    2. Dispatch into BDSL based on event_type. May update posterior.
    3. Emit zero or more effector signals.

If step 2 raises, step 1 has already happened — the log is consistent
with the world. Step 3 is skipped; the corresponding pi-hook awaiter
times out and the body fails open.

Returns the ack envelope sent back on the /sensor response.
"""
function handle_sensor_event(state::DaemonState,
                             event::AbstractDict)::Dict{String, Any}
    event_dict = Dict{String, Any}(string(k) => v for (k, v) in event)

    # 1. Log first, always. Errors here are fatal — the daemon cannot
    #    proceed safely if the de Finetti record is corrupt.
    append_event!(state.log_path, event_dict)

    # 2. Dispatch.
    event_type = get(event_dict, "event_type", "")
    if event_type == "tool-proposed"
        decide = state.env[Symbol("decide-action")]
        action = decide(state.posterior[])
        action isa Symbol || error("decide-action returned non-Symbol: $(typeof(action))")
        emit_signal!(state, string(get(event_dict, "event_id", "")),
                     action, event_dict)

    elseif event_type == "user-responded"
        response = get(event_dict, "response", "")
        if response == "yes" || response == "no"
            obs = response == "yes" ? 1 : 0
            obs_fn = state.env[Symbol("observe-response")]
            state.posterior[] = obs_fn(state.posterior[], obs)
        end
        followup_fn = state.env[Symbol("followup-after-response")]
        followup = followup_fn(event_dict)
        if followup isa Symbol && followup !== :nothing
            in_response_to = string(get(event_dict, "in_response_to", ""))
            emit_signal!(state, in_response_to, followup, event_dict)
        end

    elseif event_type == "tool-completed"
        # Pass 1: log only. Pass 2 will condition on outcome features.

    else
        @warn "handle_sensor_event: unknown event_type" event_type
    end

    Dict{String, Any}(
        "ack"      => true,
        "event_id" => string(get(event_dict, "event_id", "")),
    )
end

# ── HTTP transport ────────────────────────────────────────────────────

function _handle_sensor_stream(state::DaemonState, stream::HTTP.Stream)
    body_bytes = read(stream)
    body_str = String(body_bytes)

    parsed = try
        JSON3.read(body_str, Dict{String, Any})
    catch e
        _write_json_response(stream, 400,
            Dict("ack" => false, "error" => "malformed JSON: $(sprint(showerror, e))"))
        return
    end

    ack = try
        handle_sensor_event(state, parsed)
    catch e
        @warn "handle_sensor_event raised; log already written" event=parsed error=e
        _write_json_response(stream, 500,
            Dict("ack"      => false,
                 "event_id" => string(get(parsed, "event_id", "")),
                 "error"    => sprint(showerror, e)))
        return
    end

    _write_json_response(stream, 200, ack)
end

function _write_json_response(stream::HTTP.Stream, status::Int,
                              payload::AbstractDict)
    body = JSON3.write(payload)
    HTTP.setstatus(stream, status)
    HTTP.setheader(stream, "Content-Type" => "application/json")
    HTTP.setheader(stream, "Content-Length" => string(sizeof(body)))
    HTTP.startwrite(stream)
    write(stream, body)
end

"""
    _signals_sse_handler(state)

Stream-style HTTP handler for `GET /signals`. Drains the signal queue
to the client as Server-Sent Events. On write failure (client
disconnect) the loop exits and any unsent signals stay buffered for
the next consumer; on overflow the queue's bounded-capacity policy
discards the oldest, per SPEC.md.
"""
function _signals_sse_handler(state::DaemonState)
    return function(stream::HTTP.Stream)
        HTTP.setstatus(stream, 200)
        HTTP.setheader(stream, "Content-Type" => "text/event-stream")
        HTTP.setheader(stream, "Cache-Control" => "no-cache")
        HTTP.setheader(stream, "Connection" => "keep-alive")
        HTTP.startwrite(stream)

        # Poll-based delivery loop with SSE-comment heartbeats. The
        # heartbeat write is what detects client disconnect: a closed
        # TCP connection raises on write, which we treat as exit. (A
        # blocking wait on the SignalQueue's Condition would deadlock
        # when no further signals arrive, and Julia's `wait` has no
        # timeout. 50ms latency is fine for transport.)
        while !state.closing[]
            items = drain!(state.signal_queue)
            if !isempty(items)
                for sig in items
                    _write_sse_data(stream, sig) || return
                end
            else
                try
                    write(stream, ": heartbeat\n\n")
                    flush(stream)
                catch
                    return
                end
                sleep(0.05)
            end
        end
    end
end

function _write_sse_data(stream, sig::AbstractDict)::Bool
    try
        write(stream, "data: ")
        write(stream, JSON3.write(sig))
        write(stream, "\n\n")
        flush(stream)
        true
    catch e
        @warn "SSE write failed; closing stream" error=e
        false
    end
end

# ── Public entry points ───────────────────────────────────────────────

"""
    start_daemon(; port, log_path, bdsl_dir, host="127.0.0.1") -> (server, state)

Initialise daemon state (replaying the log to reconstruct the
posterior) and start the HTTP server in a background task. Returns the
server handle and the state object. Use `stop_daemon(server)` to shut
down.
"""
function start_daemon(; port::Int,
                       log_path::AbstractString,
                       bdsl_dir::AbstractString,
                       host::AbstractString="127.0.0.1")
    state = init_state(; bdsl_dir, log_path)
    sse_handler = _signals_sse_handler(state)

    function dispatch(stream::HTTP.Stream)
        method = stream.message.method
        target = stream.message.target
        if method == "POST" && target == "/sensor"
            _handle_sensor_stream(state, stream)
        elseif method == "GET" && target == "/signals"
            sse_handler(stream)
        else
            HTTP.setstatus(stream, 404)
            HTTP.startwrite(stream)
            write(stream, "Not Found\n")
        end
    end

    # `dispatch` is a streaming handler (takes an HTTP.Stream). The
    # `stream=true` kwarg requires HTTP 1.11.x; newer/older builds removed
    # it, which is why HTTP is pinned to ~1.11 in Project.toml [compat].
    server = HTTP.serve!(dispatch, host, port; stream=true)
    (server, state)
end

"""
    stop_daemon(server)

Close the HTTP listener. Outstanding SSE handlers will exit when the
underlying stream is closed; outstanding /sensor handlers complete
their current request and then return.
"""
# stop_daemon(server, state=nothing)
#
# Close the HTTP listener. If `state` is supplied, also flag it as
# `closing` so any active SSE handlers exit on their next poll cycle
# without waiting for a client-side disconnect to surface as a write
# error. Outstanding /sensor handlers complete their current request.
function stop_daemon(server, state=nothing)
    state === nothing || (state.closing[] = true)
    close(server)
end

end # module Server
