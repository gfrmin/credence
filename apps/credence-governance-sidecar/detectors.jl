# Role: body
#
# Failure-mode detectors: posterior-signature checks that elevate EU
# for the appropriate intervention action. Loaded by server.jl.

using ..Credence
using SpecialFunctions: digamma, logbeta

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

# KL(Beta(α₁,β₁) || Beta(α₂,β₂)). Legitimate posteriors have α,β ≥ 1
# (prior is at least Beta(1,1)); digamma is finite there.
function kl_beta(α₁::Float64, β₁::Float64, α₂::Float64, β₂::Float64)::Float64
    # credence-lint: allow — precedent:display-arithmetic — host-level KL divergence computation
    logbeta(α₂, β₂) - logbeta(α₁, β₁) +
    (α₁ - α₂) * digamma(α₁) +
    (β₁ - β₂) * digamma(β₁) +
    (α₂ - α₁ + β₂ - β₁) * digamma(α₁ + β₁)
end

function fit_beta(outcomes::AbstractVector{Bool})::Tuple{Float64, Float64}
    s = count(outcomes)
    (1.0 + s, 1.0 + length(outcomes) - s)
end

function stationarity_threshold(m::BetaMeasure)::Float64
    # credence-lint: allow — precedent:expect-through-accessor — threshold from posterior concentration
    concentration = m.alpha + m.beta
    # credence-lint: allow — precedent:display-arithmetic — host-level KL threshold derivation
    log(1.0 + 1.0 / concentration)
end

struct StationarityResult
    fires::Bool
    tool_name::String
    count::Int
    kl::Float64
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
    half = div(K, 2)
    half < 1 && return nothing
    first_half = window[1:half]
    second_half = window[half+1:end]
    (α₁, β₁) = fit_beta(first_half)
    (α₂, β₂) = fit_beta(second_half)
    kl = kl_beta(α₁, β₁, α₂, β₂)
    thresh = stationarity_threshold(m)
    fires = kl < thresh

    StationarityResult(fires, string(tool_name), length(outcomes), kl, thresh, K)
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

# Under v0.1's typical posteriors, the variance-based ceiling rarely activates
# and the floor of 3 dominates. The formula's structure derives from posterior
# noise properties; the floor captures the architectural minimum span needed to
# distinguish "sustained" from "transient".
function no_confidence_span(m::BetaMeasure)::Int
    # credence-lint: allow — precedent:expect-through-accessor — span derivation from posterior concentration
    α = m.alpha
    β = m.beta
    concentration = α + β
    # EU = 2p - 1 (linear), so variance(EU) = 4 × variance(p) by delta method
    # credence-lint: allow — precedent:display-arithmetic — host-level span derivation
    eu_var = 4.0 * α * β / (concentration^2 * (concentration + 1.0))
    threshold = 1.0 / sqrt(concentration)
    raw = ceil(Int, 2.0 * eu_var / threshold^2)
    clamp(raw, 3, 20)
end

function check_no_confidence(state::BrainState, tool_name::AbstractString,
                             category::AbstractString,
                             eu_proceed::Float64, window_size::Int)::Bool
    m = get_posterior(state, tool_name, category)

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

    state.no_confidence_consecutive >= no_confidence_span(m)
end
