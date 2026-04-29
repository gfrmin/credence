# Role: body
using HTTP
using JSON3
using Dates

# Load Credence substrate from repo root
const REPO_ROOT = dirname(dirname(@__DIR__))
push!(LOAD_PATH, REPO_ROOT)
using Credence

include("brain.jl")
include("persistence.jl")

const DEFAULT_PORT = 3100

function make_state(; max_repetitions::Int=3, prototype_fallback::Bool=true)
    config_path = joinpath(@__DIR__, "config", "budgets.json")
    budget_table = load_budget_table(config_path)
    brain = make_brain_state(budget_table;
                             prototype_fallback_enabled=prototype_fallback,
                             max_repetitions=max_repetitions)

    (; user_id, created_at) = load_sidecar_state!(brain)

    history = Dict{String,Any}[]
    state_lock = ReentrantLock()
    return (; brain, history, start_time=now(), user_id, created_at, state_lock)
end

function handle_evaluate(state, body::Dict{String,Any})
    tool_name = get(body, "toolName", "")
    params = Dict{String,Any}(get(body, "params", Dict{String,Any}()))
    recent = get(body, "recentHistory", Dict{String,Any}[])

    combined_history = vcat(
        [Dict{String,Any}("toolName" => get(r, "toolName", ""),
                          "params" => Dict{String,Any}(get(r, "params", Dict{String,Any}())))
         for r in recent],
        [Dict{String,Any}("toolName" => get(h, "toolName", ""),
                          "params" => Dict{String,Any}(get(h, "params", Dict{String,Any}())))
         for h in state.history]
    )

    fallback = check_prototype_fallback(state.brain, combined_history, tool_name, params)
    if fallback !== nothing
        return fallback
    end

    category = infer_category(tool_name, params)
    result = compute_eu(state.brain, tool_name, category)

    response = Dict{String,Any}(
        "action" => result.action,
        "decision" => result.decision,
        "reason" => result.reason,
        "signals" => result.signals,
        "requireApproval" => nothing,
    )

    return response
end

function handle_observe(state, body::Dict{String,Any})
    tool_name = get(body, "toolName", "")
    params = Dict{String,Any}(get(body, "params", Dict{String,Any}()))
    error_str = get(body, "error", nothing)
    duration_ms = get(body, "durationMs", nothing)

    entry = Dict{String,Any}(
        "toolName" => tool_name,
        "params" => params,
        "error" => error_str,
        "durationMs" => duration_ms,
        "timestamp" => get(body, "timestamp", time())
    )
    push!(state.history, entry)
    if length(state.history) > 200
        deleteat!(state.history, 1:length(state.history)-200)
    end

    category = infer_category(tool_name, params)
    success = classify_outcome(state.brain.budget_table, tool_name, category,
                               error_str isa AbstractString ? error_str : nothing,
                               duration_ms isa Real ? duration_ms : nothing)
    update_posterior!(state.brain, tool_name, category, success)

    return Dict{String,Any}("status" => "ok")
end

function handle_compaction_preview(state, body::Dict{String,Any})
    return Dict{String,Any}("status" => "ok")
end

function handle_health(state)
    posterior_summary = Dict{String,Any}()
    for (key, m) in state.brain.tool_outcomes
        # credence-lint: allow — precedent:display-arithmetic — health endpoint display only
        posterior_summary["$(key.tool_name):$(key.category)"] = Dict{String,Any}(
            "alpha" => m.alpha,  # credence-lint: allow — precedent:expect-through-accessor — display for health endpoint
            "beta" => m.beta,  # credence-lint: allow — precedent:expect-through-accessor — display for health endpoint
            "mean" => mean(m),  # credence-lint: allow — precedent:display-arithmetic — health endpoint display
        )
    end
    return Dict{String,Any}(
        "status" => "ok",
        "uptime_seconds" => round(Int, Dates.value(now() - state.start_time) / 1000),
        "observation_count" => state.brain.observation_count,
        "posterior_summary" => posterior_summary,
    )
end

function run_server(; port::Int=DEFAULT_PORT, max_repetitions::Int=3, prototype_fallback::Bool=true)
    state = make_state(; max_repetitions, prototype_fallback)
    router = HTTP.Router()

    HTTP.register!(router, "POST", "/evaluate", function(req)
        body = JSON3.read(String(req.body), Dict{String,Any})
        result = handle_evaluate(state, body)
        return HTTP.Response(200, ["Content-Type" => "application/json"],
                            JSON3.write(result))
    end)

    HTTP.register!(router, "POST", "/observe", function(req)
        body = JSON3.read(String(req.body), Dict{String,Any})
        result = lock(state.state_lock) do
            r = handle_observe(state, body)
            save_sidecar_state(state.brain, state.user_id, state.created_at)
            r
        end
        return HTTP.Response(200, ["Content-Type" => "application/json"],
                            JSON3.write(result))
    end)

    HTTP.register!(router, "POST", "/compaction-preview", function(req)
        body = JSON3.read(String(req.body), Dict{String,Any})
        result = lock(state.state_lock) do
            r = handle_compaction_preview(state, body)
            save_sidecar_state(state.brain, state.user_id, state.created_at)
            r
        end
        return HTTP.Response(200, ["Content-Type" => "application/json"],
                            JSON3.write(result))
    end)

    HTTP.register!(router, "GET", "/health", function(_)
        result = handle_health(state)
        return HTTP.Response(200, ["Content-Type" => "application/json"],
                            JSON3.write(result))
    end)

    println("Credence governance sidecar starting on port $port")
    println("  user_id = $(state.user_id)")
    println("  state_dir = $(get_state_dir())")
    println("  prototype_fallback = $prototype_fallback")
    println("  max_repetitions = $max_repetitions")
    println("  observation_count = $(state.brain.observation_count)")
    println("  endpoints: POST /evaluate, POST /observe, POST /compaction-preview, GET /health")

    HTTP.serve(router, "127.0.0.1", port)
end

port = parse(Int, get(ENV, "CREDENCE_SIDECAR_PORT", string(DEFAULT_PORT)))
max_reps = parse(Int, get(ENV, "CREDENCE_MAX_REPETITIONS", "3"))
fallback = lowercase(get(ENV, "CREDENCE_PROTOTYPE_FALLBACK", "true")) in ("true", "1", "yes")
run_server(; port, max_repetitions=max_reps, prototype_fallback=fallback)
