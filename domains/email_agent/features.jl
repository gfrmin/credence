"""
    features.jl — Email feature extraction

Extracts a numeric feature vector from an email for use by
program-space predicates. Each feature maps to a sensor channel.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence
using Credence: SensorChannel, SensorConfig

"""Email feature configuration — defines which features are extracted."""
struct EmailFeatureConfig
    channel_names::Vector{Symbol}     # human-readable names
    sensor_config::SensorConfig       # corresponding sensor config
end

"""
    extract_features(email, config::EmailFeatureConfig) → Vector{Float64}

Extract numeric features from an email. Each feature is normalised
to [0, 1] for compatibility with program-space predicates.
"""
function extract_features(email, config::EmailFeatureConfig)::Vector{Float64}
    error("not implemented")
end
