# Role: brain-side application
# Reading weights for diagnostic logging is sanctioned access.

using Credence

function log_posterior(m::CategoricalMeasure)
    w = weights(m)
    @info "posterior weights" w
end
