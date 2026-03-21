"""
    metrics.jl — Email agent metrics tracking

Stub: preference-learning metrics for the email domain.
"""

"""Track email agent performance metrics."""
mutable struct EmailMetricsTracker
    steps::Vector{Int}
    preference_accuracy::Vector{Float64}
    user_satisfaction::Vector{Float64}

    EmailMetricsTracker() = new(Int[], Float64[], Float64[])
end

"""Record a step's metrics."""
function record_email!(m::EmailMetricsTracker; step::Int, accuracy::Float64, satisfaction::Float64)
    error("not implemented")
end
