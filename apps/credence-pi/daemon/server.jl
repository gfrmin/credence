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
using Serialization: deserialize

include(joinpath(@__DIR__, "observation_log.jl"))
using .ObservationLog: append_event!, replay_user_responses, replay_contexts, replay_route_outcomes

# Pull the parent's Credence module — server.jl runs under the test/run
# loader which has already done `using Credence`.
import Main: Credence
using .Credence: Eval, Parse

# The Route-B feature-conditioned brain (Pass 2): typed Julia that declares the
# structure-BMA family + Functionals and calls the Tier-1 ops. `wire_brain!`
# injects make-prior / decide-action / observe-response / followup-after-response
# into the BDSL env, so the call-sites below are unchanged in shape.
include(joinpath(@__DIR__, "..", "brain", "feature_brain.jl"))
using .FeatureBrain: wire_brain!, build_model_from_env, reconstruct_posterior

# Model routing (Pass 2 / live): wire_routing! installs env[:route-decide] from the
# declared routing manifest (bdsl/routing.bdsl). routing_brain.jl reaches FeatureBrain
# via `..FeatureBrain` (the sibling submodule), so it resolves here under Server exactly
# as it does at Main in the eval. INERT unless a roster is declared.
include(joinpath(@__DIR__, "..", "brain", "routing_brain.jl"))
using .RoutingBrain: wire_routing!, route_outcome!

# Value reporting (display-only, non-causal): the governance + dollars-saved surface, exposed
# over GET /report so the body can show the user what credence-pi delivered — closing the
# "manual `julia savings.jl`" gap. Savings nests its OWN ObservationLog (distinct from this
# module's include above), so the two coexist without a name clash.
include(joinpath(@__DIR__, "..", "savings.jl"))
using .Savings: savings_report

export start_daemon, stop_daemon, DaemonState
export SignalQueue, enqueue!, snapshot, drain!
export handle_sensor_event, action_to_signal, emit_signal!, emit_route_signal!

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
    # Pass 2: per-turn cost (latest priced turn-cost USD, or nothing) feeds the
    # cost-denominated utility; `pending_contexts` rejoins a user-responded to the
    # features of its originating tool-proposed (by event_id) so the brain
    # conditions on the right cell. Both are transport bookkeeping, not reasoning.
    last_cost::Ref{Any}
    pending_contexts::Dict{String, Any}
    # Model routing (Pass 2 / live learning). `routing` is the live `RoutingState` (per-model
    # posteriors + the shared confound belief) or `nothing` for a governance-only install.
    # `pending_routes` rejoins a tool-completed to its turn's routed model by session_id, so
    # the per-turn correctness signal credits the right model (transport bookkeeping, not
    # reasoning — like `pending_contexts`).
    routing::Any
    pending_routes::Dict{String, Any}
end

"""
    load_env(bdsl_dir) -> (Eval.Env, Union{RoutingState, Nothing})

Build a fresh BDSL environment with stdlib + the five Pass-1 programs
loaded in dependency order. Returns the env and the live `RoutingState`
that `wire_routing!` built (or `nothing` for a governance-only install).
"""
function load_env(bdsl_dir::AbstractString)
    env = Eval.default_env()
    env[:__toplevel__] = true
    stdlib_path = joinpath(@__DIR__, "..", "..", "..", "src", "stdlib.bdsl")
    for expr in Parse.parse_all(read(stdlib_path, String))
        Eval.eval_dsl(expr, env)
    end
    # Pass 2 / Route B: the `.bdsl` files carry DECLARED DATA only — the
    # capability manifest, the feature spaces, and the utility constants. The
    # reasoning (make-prior / decide-action / observe-response) is injected by the
    # typed Julia brain via `wire_brain!`, which reads the declared feature spaces
    # and utility constants out of this same env. (The Pass-1 prior/kernel/decide
    # `.bdsl` brain is retired — see docs/credence-pi-pass-2/move-3-design.md.)
    for fname in ("capabilities.bdsl", "features.bdsl", "utility.bdsl")
        for expr in Parse.parse_all(read(joinpath(bdsl_dir, fname), String))
            Eval.eval_dsl(expr, env)
        end
    end
    # Optional routing manifest (model routing). Absent ⇒ routing stays inert and the
    # daemon is governance-only (fully backward-compatible). Loaded BEFORE wire_routing!
    # reads its declared roster + reward.
    routing_path = joinpath(bdsl_dir, "routing.bdsl")
    if isfile(routing_path)
        for expr in Parse.parse_all(read(routing_path, String))
            Eval.eval_dsl(expr, env)
        end
    end
    wire_brain!(env)
    # Optional routing warm-belief override (mirrors CREDENCE_PI_WARM_BRAIN for governance):
    # CREDENCE_PI_ROUTING_BRAIN="" forces a COLD routing start (no warm seed) — used by the
    # live-escalation A/B to score the gate leakage-free, since the shipped warm belief is
    # trained on the same matrix. Absent ⇒ wire_routing! uses its default counts.json.
    haskey(ENV, "CREDENCE_PI_ROUTING_BRAIN") &&
        (env[Symbol("routing-brain-path")] = ENV["CREDENCE_PI_ROUTING_BRAIN"])
    routing = wire_routing!(env)   # RoutingState iff a roster was declared; else nothing
    (env, routing)
end

"""
    load_starting_posterior(env, warm_brain_path) -> Measure

The starting belief BEFORE the local log is replayed. If `warm_brain_path` names a
readable warm brain (trained on corpus replay — see
apps/credence-pi/eval/train_warm_brain.jl), load it so governance works from install
instead of cold `Beta(2,2)`. PREFERRED format is version-stable per-context COUNTS
(`*.json`), reconstructed by replaying `observe` (order-independent ⇒ identical to
the trained posterior; robust across Julia versions). A legacy `Serialization` blob
(`*.jls`) is still accepted but is Julia-version fragile. The warm brain is a PRIOR:
the local log then conditions it via `observe-response`, so each deployment adapts.
Any load failure (missing file, schema/version drift) falls back LOUDLY to the cold
prior — a stale warm brain must never silently mis-govern.
"""
function load_starting_posterior(env, warm_brain_path)
    cold() = env[Symbol("make-prior")]()
    (warm_brain_path === nothing || isempty(warm_brain_path)) && return cold()
    isfile(warm_brain_path) || (@warn "warm brain not found; cold start" path=warm_brain_path; return cold())
    try
        if endswith(lowercase(String(warm_brain_path)), ".json")
            model = build_model_from_env(env)
            p = reconstruct_posterior(model, warm_brain_path)
            @info "loaded warm brain (counts reconstruction)" path=warm_brain_path
            return p
        end
        p = deserialize(warm_brain_path)   # legacy .jls blob (Julia-version fragile)
        @info "loaded warm brain (deserialized)" path=warm_brain_path type=typeof(p)
        p
    catch e
        @warn "warm brain failed to load; cold start" path=warm_brain_path error=e
        cold()
    end
end

"""
    init_state(; bdsl_dir, log_path, warm_brain_path=nothing) -> DaemonState

Build a fresh daemon state. Calls `load_env`, builds an empty signal
queue, loads the starting posterior (warm brain if configured, else cold prior),
and reconstructs on top of it by replaying every `user-responded` event in
`log_path` through `observe-response`. If routing is configured, it likewise replays
every `route-outcome` event through `route_outcome!` on top of the warm `tops` —
deterministic replay reproduces the live routing belief exactly.
"""
function init_state(; bdsl_dir::AbstractString, log_path::AbstractString,
                    warm_brain_path=nothing)::DaemonState
    env, routing = load_env(bdsl_dir)
    posterior = load_starting_posterior(env, warm_brain_path)
    obs_fn = env[Symbol("observe-response")]
    # Pass 2: replay each user-responded REJOINED to its tool-proposed features
    # (by in_response_to), so the feature-conditioned update lands in the right
    # cell. `replay_contexts` yields (features, obs) pairs in log order.
    for (features, o) in replay_contexts(log_path)
        posterior = obs_fn(posterior, features, o)
    end
    # Pass 2 routing: replay route-outcomes onto the warm tops (exact, deterministic).
    if routing !== nothing
        for (model_id, features, success, human) in replay_route_outcomes(log_path)
            route_outcome!(routing, model_id, features, success; human = human)
        end
    end
    DaemonState(env, Ref{Any}(posterior), SignalQueue(), log_path, bdsl_dir,
                Threads.Atomic{Bool}(false), Ref{Any}(nothing), Dict{String, Any}(),
                routing, Dict{String, Any}())
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
    # Log the decision (event_type "decision") before emitting, so the
    # dollars-saved surface (savings.jl) can account for governance the
    # body never reports back — chiefly silent auto-blocks/auto-proceeds
    # that produce no user-responded event. Decision records are DERIVED
    # (replay reconstructs the posterior and hence the decisions from the
    # logged sensor events), so they do not affect replay correctness:
    # replay_user_responses ignores them. action_to_signal above has
    # already validated `action`, so only manifest actions are logged.
    append_event!(state.log_path, Dict{String, Any}(
        "event_type"     => "decision",
        "in_response_to" => string(in_response_to),
        "action"         => string(action),
    ))
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

"""
    emit_route_signal!(state, in_response_to, choice::AbstractDict) -> Dict

Compose and enqueue a `route` effector signal carrying the chosen model. `choice` is
the dict returned by the brain's `route-decide` closure (`model`, `provider`, `name`).
Logs a derived `route-decision` event first (so the dollars-saved surface can account
for routing; ignored by posterior replay, like `decision`). Templating only — the EU-max
already happened brain-side in `route-decide`.
"""
function emit_route_signal!(state::DaemonState, in_response_to::AbstractString,
                            choice::AbstractDict)::Dict{String, Any}
    model    = string(get(choice, "model", ""))
    provider = string(get(choice, "provider", ""))
    name     = string(get(choice, "name", ""))
    append_event!(state.log_path, Dict{String, Any}(
        "event_type"     => "route-decision",
        "in_response_to" => string(in_response_to),
        "model"          => model,
        "provider"       => provider,
    ))
    n = Threads.atomic_add!(_SIGNAL_ID_COUNTER, 1)
    signal = Dict{String, Any}(
        "signal_type"    => "effector",
        "signal_id"      => string("sig_", randstring(6), "_", n),
        "in_response_to" => string(in_response_to),
        "effector"       => "route",
        "parameters"     => Dict{String, Any}(
            "model" => model, "provider" => provider, "name" => name),
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
    ack_extra = Dict{String, Any}()        # request/response payload (e.g. escalate-request)
    if event_type == "tool-proposed"
        # Fix 1: forward the raw feature dict + the latest per-turn cost into the
        # brain. Transport only — the cell-mapping / EU-max happens brain-side.
        # The features are also stashed so the matching user-responded can rejoin
        # them (Fix 2 / context-join).
        features = get(event_dict, "features", nothing)
        event_id = string(get(event_dict, "event_id", ""))
        features === nothing &&
            error("tool-proposed event $event_id missing required 'features' (Pass 2 brain)")
        state.pending_contexts[event_id] = features
        decide = state.env[Symbol("decide-action")]
        # Per-request profile (the user's utility weights) — body-supplied, no daemon restart.
        action = decide(state.posterior[], features, state.last_cost[],
                        get(event_dict, "profile", nothing))
        action isa Symbol || error("decide-action returned non-Symbol: $(typeof(action))")
        emit_signal!(state, event_id, action, event_dict)

    elseif event_type == "user-responded"
        in_response_to = string(get(event_dict, "in_response_to", ""))
        response = get(event_dict, "response", "")
        if response == "yes" || response == "no"
            # Fix 2: rejoin to the originating tool-proposed's features so the
            # update lands in the right cell. A response with no remembered
            # context (e.g. daemon restarted mid-turn — the in-memory map is
            # empty until replay rebuilds it) is logged but not conditioned on,
            # rather than mis-conditioned against a guessed context.
            features = get(state.pending_contexts, in_response_to, nothing)
            if features !== nothing
                obs = response == "yes" ? 1 : 0
                obs_fn = state.env[Symbol("observe-response")]
                state.posterior[] = obs_fn(state.posterior[], features, obs)
            else
                @warn "user-responded with no remembered context; not conditioning" in_response_to
            end
            delete!(state.pending_contexts, in_response_to)
        end
        followup_fn = state.env[Symbol("followup-after-response")]
        followup = followup_fn(event_dict)
        if followup isa Symbol && followup !== :nothing
            emit_signal!(state, in_response_to, followup, event_dict)
        end

    elseif event_type == "tool-completed"
        # Pass 2: the per-turn correctness signal for ROUTING. Credit the routed model of
        # this session's current turn with the exec outcome (did the proposed call execute
        # cleanly?). The signal is NOISY — `route_outcome!` decodes it against the LEARNED
        # tool-reliability confound, so a flaky tool is absorbed by ρ, not blamed on the
        # model. This NEVER touches the governance posterior (a separate belief). A derived
        # `route-outcome` event is logged so a restart replays this learning exactly.
        if state.routing !== nothing
            session_id = string(get(event_dict, "session_id", ""))
            routed = get(state.pending_routes, session_id, nothing)
            if routed !== nothing
                model_id, features = routed
                outcome = get(event_dict, "outcome", nothing)
                success = outcome isa AbstractDict && get(outcome, "success", true) == true
                route_outcome!(state.routing, model_id, features, success)
                append_event!(state.log_path, Dict{String, Any}(
                    "event_type" => "route-outcome",
                    "event_id"   => string(get(event_dict, "event_id", "")),
                    "model"      => model_id,
                    "features"   => features,
                    "success"    => success,
                ))
            end
        end

    elseif event_type == "turn-cost"
        # Per-turn USD cost (OpenClaw-plugin llm_output hook). Logged for the
        # dollars-saved surface and stashed as the latest cost estimate for the
        # cost-denominated utility (decide-action reads `state.last_cost`). The
        # brain does not CONDITION on it and emits no signal. `usd` may be null
        # (model unpriced) — the brain falls back to a configured per-call cost.
        state.last_cost[] = get(event_dict, "usd", nothing)

    elseif event_type == "route-request"
        # Model routing (before_model_resolve). Read the request features and let the
        # brain's route-decide closure pick the EU-max model; emit a `route` signal the
        # body maps to providerOverride/modelOverride. This NEVER touches the governance
        # posterior — routing is a SEPARATE belief (its own per-model posteriors). If
        # routing is not configured (no roster declared ⇒ no route-decide), emit nothing:
        # the body times out and fails open to OpenClaw's default model.
        route_decide = get(state.env, Symbol("route-decide"), nothing)
        if route_decide === nothing
            @warn "route-request received but routing is not configured; ignoring (body fails open)"
        else
            features = get(event_dict, "features", nothing)
            event_id = string(get(event_dict, "event_id", ""))
            features === nothing &&
                error("route-request event $event_id missing required 'features'")
            # The user's LIVE model roster (roster-aware routing): the body sends the models
            # OpenClaw actually has + their costs. Absent ⇒ route-decide falls back to the
            # declared default roster (back-compat with an older body).
            roster = get(event_dict, "models", nothing)
            # Per-request profile (the user's utility weights: reward/w_time) — body-supplied.
            choice = route_decide(features, roster, get(event_dict, "profile", nothing))
            if choice === nothing
                # Inert: fewer than 2 candidate models ⇒ nothing to route between. Emit no
                # signal; the body keeps OpenClaw's model (fail open). No pending-route stash.
            else
                emit_route_signal!(state, event_id, choice)
                # Remember this turn's routed model so its tool-completed outcomes credit it
                # (per-session; overwritten next turn). Transport bookkeeping, like pending_contexts.
                if state.routing !== nothing
                    session_id = string(get(event_dict, "session_id", ""))
                    isempty(session_id) ||
                        (state.pending_routes[session_id] = (choice["model"], features))
                end
            end
        end

    elseif event_type == "escalate-request"
        # Observe-then-escalate (the dominance proof's WINNING gate, now wired live). Like
        # route-request, but the host drives a try→observe→escalate loop: each call passes the
        # rungs already `tried` (failed) this attempt and the per-call `reward` profile; the
        # brain returns the next cheapest positive-EU rung or STOP. The decision rides back
        # SYNCHRONOUSLY in the ack (`route`/`stop`), so the host reads the POST response — no
        # signal-queue round-trip. NEVER touches the governance posterior (separate belief).
        escalate_decide = get(state.env, Symbol("escalate-decide"), nothing)
        if escalate_decide === nothing
            @warn "escalate-request received but routing is not configured; ignoring (body fails open)"
            ack_extra["stop"] = true
        else
            features = get(event_dict, "features", nothing)
            event_id = string(get(event_dict, "event_id", ""))
            features === nothing &&
                error("escalate-request event $event_id missing required 'features'")
            roster = get(event_dict, "models", nothing)
            tried  = get(event_dict, "tried", Int[])
            reward = get(event_dict, "reward", nothing)
            # Per-request profile (reward/w_time override); reward field kept for back-compat.
            choice = escalate_decide(features, roster, tried, reward,
                                     get(event_dict, "profile", nothing))
            choice === nothing ? (ack_extra["stop"] = true) : (ack_extra["route"] = choice)
        end

    elseif event_type == "counterfactual-decision"
        # Body-side shadow-mode telemetry: in observe-only mode the body would
        # have blocked/asked but proceeded, and records that here. Already
        # appended at step 1; no dispatch, no signal, no posterior touch. The
        # dollars-saved surface (savings.jl) reads it to separate counterfactual
        # "would-block" from enforced "prevented". Pure transport, like turn-cost.

    else
        @warn "handle_sensor_event: unknown event_type" event_type
    end

    merge(Dict{String, Any}(
        "ack"      => true,
        "event_id" => string(get(event_dict, "event_id", "")),
    ), ack_extra)
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
                       host::AbstractString="127.0.0.1",
                       warm_brain_path=nothing)
    state = init_state(; bdsl_dir, log_path, warm_brain_path)
    sse_handler = _signals_sse_handler(state)

    function dispatch(stream::HTTP.Stream)
        method = stream.message.method
        target = stream.message.target
        if method == "POST" && target == "/sensor"
            _handle_sensor_stream(state, stream)
        elseif method == "GET" && target == "/signals"
            sse_handler(stream)
        elseif method == "GET" && target == "/ready"
            # Liveness probe (transport only — no reasoning). Used by the
            # Docker HEALTHCHECK and the deploy smoke test.
            HTTP.setstatus(stream, 200)
            HTTP.setheader(stream, "Content-Type" => "text/plain")
            HTTP.startwrite(stream)
            write(stream, "ok\n")
        elseif method == "GET" && target == "/report"
            # The value surface (display-only, non-causal): governance + dollars-saved tallies
            # from the observation log, as JSON, so the body can show the user what credence-pi
            # delivered this run. No reasoning here — reuses Savings.savings_report verbatim.
            body = try
                JSON3.write(savings_report(state.log_path))
            catch e
                @warn "report generation failed" error = e
                JSON3.write(Dict("error" => "report unavailable"))
            end
            HTTP.setstatus(stream, 200)
            HTTP.setheader(stream, "Content-Type" => "application/json")
            HTTP.setheader(stream, "Content-Length" => string(sizeof(body)))
            HTTP.startwrite(stream)
            write(stream, body)
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
