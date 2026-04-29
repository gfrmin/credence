# Role: body
using HTTP
using JSON3
using Dates

# Load Credence substrate from repo root
const REPO_ROOT = dirname(dirname(@__DIR__))
push!(LOAD_PATH, REPO_ROOT)
using Credence

include("instruction_patterns.jl")
include("brain.jl")
include("detectors.jl")
include("persistence.jl")

const DEFAULT_PORT = 3100

function make_state()
    config_path = joinpath(@__DIR__, "config", "budgets.json")
    budget_table = load_budget_table(config_path)
    brain = make_brain_state(budget_table)

    read_tools_path = joinpath(@__DIR__, "config", "read_tools.json")
    read_tools = load_read_tools(read_tools_path)

    detector_windows_path = joinpath(@__DIR__, "config", "detector_windows.json")
    eu_window_size = if isfile(detector_windows_path)
        windows = load_detector_windows(detector_windows_path)
        get(windows, "eu_proceed", DEFAULT_EU_WINDOW_SIZE)
    else
        DEFAULT_EU_WINDOW_SIZE
    end

    (; user_id, created_at) = load_sidecar_state!(brain)

    history = Dict{String,Any}[]
    state_lock = ReentrantLock()
    return (; brain, history, start_time=now(), user_id, created_at, state_lock,
              read_tools, eu_window_size)
end

function handle_evaluate(state, body::Dict{String,Any})
    tool_name = get(body, "toolName", "")
    params = Dict{String,Any}(get(body, "params", Dict{String,Any}()))

    category = infer_category(tool_name, params)

    sr = check_stationarity(state.brain, state.read_tools, tool_name, params, category)
    if sr !== nothing && sr.fires
        return Dict{String,Any}(
            "action" => "block",
            "decision" => "halt",
            "reason" => "Posterior stationary on repeated tool call: '$(sr.tool_name)' " *
                        "called $(sr.count) times with identical arguments " *
                        "(outcome_var=$(round(sr.outcome_var, digits=4)) ≤ threshold=$(round(sr.threshold, digits=4)), " *
                        "window=$(sr.window_size)). Halting.",
            "signals" => Dict{String,Any}(),
            "requireApproval" => nothing,
        )
    end

    result = compute_eu(state.brain, tool_name, category)

    nc = check_no_confidence(state.brain, tool_name, category, result.eu_proceed, state.eu_window_size)
    if nc
        return Dict{String,Any}(
            "action" => "block",
            "decision" => "halt",
            "reason" => "Posterior over next-action value is flat: EU(proceed) has high " *
                        "coefficient of variation across recent evaluations. Halting.",
            "signals" => result.signals,
            "requireApproval" => nothing,
        )
    end

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
    record_outcome!(state.brain, tool_name, params, success)

    user_approval = get(body, "userApproval", nothing)
    if user_approval !== nothing
        update_instruction_decay!(state.brain, category, user_approval == true)
    end

    return Dict{String,Any}("status" => "ok")
end

function handle_compaction_preview(state, body::Dict{String,Any})
    messages = get(body, "messages", Any[])
    messages_vec = messages isa Vector ? messages : collect(messages)
    matches = match_instructions(messages_vec)
    registered = String[]
    for (matched_text, action_class) in matches
        for pattern in INSTRUCTION_PATTERNS
            if pattern.action_class == action_class && match(pattern.regex, matched_text) !== nothing
                if register_instruction!(state.brain, pattern.id, action_class)
                    push!(registered, "$(pattern.id):$(action_class)")
                end
                break
            end
        end
    end
    return Dict{String,Any}("status" => "ok", "registered" => registered)
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

function run_server(; port::Int=DEFAULT_PORT)
    state = make_state()
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
    println("  observation_count = $(state.brain.observation_count)")
    println("  eu_window_size = $(state.eu_window_size)")
    println("  read_tools = $(state.read_tools)")
    println("  endpoints: POST /evaluate, POST /observe, POST /compaction-preview, GET /health")

    HTTP.serve(router, "127.0.0.1", port)
end

port = parse(Int, get(ENV, "CREDENCE_SIDECAR_PORT", string(DEFAULT_PORT)))
run_server(; port)
