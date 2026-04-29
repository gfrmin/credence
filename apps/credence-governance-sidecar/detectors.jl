# Role: body
#
# Failure-mode detectors: posterior-signature checks that elevate EU
# for the appropriate intervention action. Loaded by server.jl.

using ..Credence

# ── Configuration ──

function load_read_tools(path::AbstractString)::Set{String}
    data = JSON3.read(read(path, String), Vector{String})
    Set(data)
end

function load_detector_windows(path::AbstractString)::Dict{String,Int}
    data = JSON3.read(read(path, String), Dict{String,Any})
    Dict{String,Int}(string(k) => Int(v) for (k, v) in data)
end

# ── Detector 1: Exec-repetition stationarity (#34574) ──

function stationarity_window(m::BetaMeasure)::Int
    # credence-lint: allow — precedent:expect-through-accessor — K derivation from posterior concentration
    concentration = m.alpha + m.beta
    max(2, ceil(Int, sqrt(concentration)))
end

function outcome_variance(outcomes::Vector{Bool})::Float64
    isempty(outcomes) && return 0.0
    # credence-lint: allow — precedent:display-arithmetic — host-level variance of empirical outcomes
    μ = sum(outcomes) / length(outcomes)
    sum((Float64(x) - μ)^2 for x in outcomes) / length(outcomes)
end

function stationarity_threshold(m::BetaMeasure)::Float64
    # credence-lint: allow — precedent:display-arithmetic — host-level threshold derivation from posterior predictive variance
    p = mean(m)
    p * (1.0 - p) * 0.1  # credence-lint: allow — precedent:display-arithmetic — host-level threshold from posterior mean
end

struct StationarityResult
    fires::Bool
    tool_name::String
    count::Int
    outcome_var::Float64
    threshold::Float64
    window_size::Int
end

function check_stationarity(state::BrainState, read_tools::Set{String},
                            tool_name::AbstractString, params::Dict{String,Any},
                            category::AbstractString)::Union{StationarityResult, Nothing}
    tool_name in read_tools && return nothing

    key = canonical_key(tool_name, params)
    outcomes = get(state.tool_args_outcomes, key, Bool[])

    m = get_posterior(state, tool_name, category)
    K = stationarity_window(m)

    length(outcomes) < K && return nothing

    window = outcomes[max(1, end-K+1):end]
    ov = outcome_variance(window)
    thresh = stationarity_threshold(m)
    fires = ov <= thresh

    StationarityResult(fires, string(tool_name), length(outcomes), ov, thresh, K)
end

function record_outcome!(state::BrainState, tool_name::AbstractString,
                         params::Dict{String,Any}, success::Bool)
    key = canonical_key(tool_name, params)
    outcomes = get!(state.tool_args_outcomes, key, Bool[])
    push!(outcomes, success)
    max_window = 200
    if length(outcomes) > max_window
        deleteat!(outcomes, 1:length(outcomes)-max_window)
    end
end

# ── Detector 3: No-confidence dreaming loops (#65550) ──

const DEFAULT_EU_WINDOW_SIZE = 10
const NO_CONFIDENCE_SPAN = 5

function check_no_confidence(state::BrainState, category::AbstractString,
                             eu_proceed::Float64, window_size::Int)::Bool
    m = get_posterior(state, "", category)

    push!(state.eu_history, eu_proceed)
    if length(state.eu_history) > window_size
        deleteat!(state.eu_history, 1:length(state.eu_history)-window_size)
    end

    length(state.eu_history) < window_size && return false

    μ = sum(state.eu_history) / length(state.eu_history)
    abs(μ) < 1e-12 && return false

    # credence-lint: allow — precedent:display-arithmetic — host-level CV diagnostic
    σ² = sum((x - μ)^2 for x in state.eu_history) / length(state.eu_history)
    σ² < 0.0 && return false
    cv = sqrt(σ²) / abs(μ)

    # credence-lint: allow — precedent:expect-through-accessor — threshold from posterior concentration
    concentration = m.alpha + m.beta
    # credence-lint: allow — precedent:display-arithmetic — host-level threshold derivation
    threshold = 1.0 / sqrt(concentration)

    if cv > threshold
        state.no_confidence_consecutive += 1
    else
        state.no_confidence_consecutive = 0
    end

    state.no_confidence_consecutive >= NO_CONFIDENCE_SPAN
end
