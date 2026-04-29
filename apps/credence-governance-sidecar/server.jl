using HTTP
using JSON3
using Dates

const DEFAULT_PORT = 3100
const DEFAULT_MAX_REPETITIONS = 3

mutable struct SidecarState
    history::Vector{Dict{String,Any}}
    max_repetitions::Int
    start_time::DateTime
end

function make_state(; max_repetitions=DEFAULT_MAX_REPETITIONS)
    SidecarState(Dict{String,Any}[], max_repetitions, now())
end

function canonical_key(tool_name::AbstractString, params::Dict)
    sorted_params = sort(collect(params), by=first)
    return "$tool_name:$(JSON3.write(sorted_params))"
end

function count_repetitions(state::SidecarState, tool_name::AbstractString, params::Dict)
    key = canonical_key(tool_name, params)
    count = 0
    for entry in state.history
        entry_key = canonical_key(
            get(entry, "toolName", ""),
            get(entry, "params", Dict{String,Any}())
        )
        if entry_key == key
            count += 1
        end
    end
    return count
end

function handle_evaluate(state::SidecarState, body::Dict)
    tool_name = get(body, "toolName", "")
    params = get(body, "params", Dict{String,Any}())
    recent = get(body, "recentHistory", Dict{String,Any}[])

    combined_history = vcat(
        [Dict{String,Any}("toolName" => get(r, "toolName", ""),
                          "params" => get(r, "params", Dict{String,Any}()))
         for r in recent],
        [Dict{String,Any}("toolName" => get(h, "toolName", ""),
                          "params" => get(h, "params", Dict{String,Any}()))
         for h in state.history]
    )

    key = canonical_key(tool_name, params)
    count = 0
    for entry in combined_history
        entry_key = canonical_key(
            get(entry, "toolName", ""),
            get(entry, "params", Dict{String,Any}())
        )
        if entry_key == key
            count += 1
        end
    end

    if tool_name == "Read"
        return Dict("action" => "proceed")
    end

    if count >= state.max_repetitions
        return Dict(
            "action" => "block",
            "reason" => "Loop detected: '$tool_name' with identical arguments has been called $count times (threshold: $(state.max_repetitions)). Halting to prevent runaway loop."
        )
    end

    return Dict("action" => "proceed")
end

function handle_observe(state::SidecarState, body::Dict)
    entry = Dict{String,Any}(
        "toolName" => get(body, "toolName", ""),
        "params" => get(body, "params", Dict{String,Any}()),
        "error" => get(body, "error", nothing),
        "durationMs" => get(body, "durationMs", nothing),
        "timestamp" => get(body, "timestamp", time())
    )
    push!(state.history, entry)
    if length(state.history) > 200
        deleteat!(state.history, 1:length(state.history)-200)
    end
    return Dict("status" => "ok")
end

function handle_health(state::SidecarState)
    return Dict(
        "status" => "ok",
        "uptime_seconds" => round(Int, Dates.value(now() - state.start_time) / 1000),
        "history_length" => length(state.history)
    )
end

function handle_reset(state::SidecarState)
    empty!(state.history)
    return Dict("status" => "ok")
end

function run_server(; port=DEFAULT_PORT, max_repetitions=DEFAULT_MAX_REPETITIONS)
    state = make_state(; max_repetitions)

    router = HTTP.Router()

    HTTP.register!(router, "POST", "/evaluate", function(req)
        body = JSON3.read(String(req.body), Dict{String,Any})
        result = handle_evaluate(state, body)
        return HTTP.Response(200, ["Content-Type" => "application/json"],
                            JSON3.write(result))
    end)

    HTTP.register!(router, "POST", "/observe", function(req)
        body = JSON3.read(String(req.body), Dict{String,Any})
        result = handle_observe(state, body)
        return HTTP.Response(200, ["Content-Type" => "application/json"],
                            JSON3.write(result))
    end)

    HTTP.register!(router, "GET", "/health", function(_)
        result = handle_health(state)
        return HTTP.Response(200, ["Content-Type" => "application/json"],
                            JSON3.write(result))
    end)

    HTTP.register!(router, "POST", "/reset", function(_)
        result = handle_reset(state)
        return HTTP.Response(200, ["Content-Type" => "application/json"],
                            JSON3.write(result))
    end)

    println("Credence governance sidecar starting on port $port")
    println("  max_repetitions = $max_repetitions")
    println("  endpoints: POST /evaluate, POST /observe, GET /health, POST /reset")

    HTTP.serve(router, "127.0.0.1", port)
end

port = parse(Int, get(ENV, "CREDENCE_SIDECAR_PORT", string(DEFAULT_PORT)))
max_reps = parse(Int, get(ENV, "CREDENCE_MAX_REPETITIONS", string(DEFAULT_MAX_REPETITIONS)))
run_server(; port, max_repetitions=max_reps)
