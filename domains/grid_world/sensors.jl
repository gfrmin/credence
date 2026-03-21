"""
    sensors.jl — Grid-world sensor configuration and projection

Source indices into EntityState:
    0 = red, 1 = green, 2 = blue, 3 = x_norm, 4 = y_norm,
    5 = speed, 6 = wall_dist, 7 = agent_dist
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence
using Credence: SensorChannel, SensorConfig, n_channels

# ── Grid-world source dimensions ──

const N_SOURCE_DIMS = 8
const SOURCE_NAMES = [:red, :green, :blue, :x_norm, :y_norm, :speed, :wall_dist, :agent_dist]

available_source_indices() = collect(0:N_SOURCE_DIMS-1)

source_name(idx::Int) = SOURCE_NAMES[idx + 1]

# ── Projection ──

"""
    project(true_state, config, temporal_state) → Vector{Float64}

Project an entity's true state through a sensor config to produce
the noisy sensor vector the agent perceives.
"""
function project(true_state::Vector{Float64}, config::SensorConfig;
                 entity_id::Int=0,
                 temporal_state::Dict{Int, Vector{Vector{Float64}}}=Dict{Int, Vector{Vector{Float64}}}())
    readings = Float64[]
    for ch in config.channels
        raw = true_state[ch.source_index + 1]
        transformed = apply_transform(raw, ch.transform, ch.source_index, entity_id, temporal_state)
        noisy = transformed + randn() * ch.noise_σ
        push!(readings, clamp(noisy, 0.0, 1.0))
    end
    readings
end

function apply_transform(raw::Float64, transform::Symbol, source_idx::Int,
                          entity_id::Int, temporal_state::Dict{Int, Vector{Vector{Float64}}})
    if transform == :identity
        raw
    elseif transform == :threshold
        raw > 0.5 ? 1.0 : 0.0
    elseif transform == :delta
        history = get(temporal_state, entity_id, Vector{Float64}[])
        isempty(history) && return 0.0
        prev = last(history)[source_idx + 1]
        abs(raw - prev)
    elseif transform == :windowed_mean
        history = get(temporal_state, entity_id, Vector{Float64}[])
        isempty(history) && return raw
        n = min(length(history), 5)
        recent = [history[end - i + 1][source_idx + 1] for i in 1:n]
        sum(recent) / n
    else
        error("Unknown transform: $transform")
    end
end

# ── Predefined sensor configs ──

"""Minimal 2-channel config: red + speed."""
function minimal_sensor_config()
    SensorConfig([
        SensorChannel(0, :identity, 0.05, 1.0),  # red
        SensorChannel(5, :identity, 0.05, 1.0),  # speed
    ])
end

"""Colour-only config: r, g, b."""
function colour_sensor_config()
    SensorConfig([
        SensorChannel(0, :identity, 0.05, 1.0),  # red
        SensorChannel(1, :identity, 0.05, 1.0),  # green
        SensorChannel(2, :identity, 0.05, 1.0),  # blue
    ])
end

"""Motion config: speed + wall_dist."""
function motion_sensor_config()
    SensorConfig([
        SensorChannel(5, :identity, 0.05, 1.0),  # speed
        SensorChannel(6, :identity, 0.05, 1.0),  # wall_dist
    ])
end

"""Full config: all 8 channels."""
function full_sensor_config()
    SensorConfig([SensorChannel(i, :identity, 0.05, 1.0) for i in 0:N_SOURCE_DIMS-1])
end

"""Colour + speed config: r, g, b, speed."""
function colour_speed_sensor_config()
    SensorConfig([
        SensorChannel(0, :identity, 0.05, 1.0),  # red
        SensorChannel(1, :identity, 0.05, 1.0),  # green
        SensorChannel(2, :identity, 0.05, 1.0),  # blue
        SensorChannel(5, :identity, 0.05, 1.0),  # speed
    ])
end
