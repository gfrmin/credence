# Role: brain-side application
# Typed container literal passed to MixturePrevision — the disciplined pattern.

using Credence

function build_belief(n::Int)
    components = TaggedBetaPrevision[]
    log_prior_weights = Float64[]
    for i in 1:n
        push!(components, TaggedBetaPrevision(i, BetaPrevision(1.0, 1.0)))
        push!(log_prior_weights, -1.0 * log(2))
    end
    belief = MixturePrevision(components, log_prior_weights)
    belief
end
