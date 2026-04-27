# Role: brain-side application
# Any[] flowing into MixturePrevision — the pre-Move-8b anti-pattern.

using Credence

function build_belief(n::Int)
    components = Any[]
    log_prior_weights = Float64[]
    for i in 1:n
        push!(components, TaggedBetaPrevision(i, BetaPrevision(1.0, 1.0)))
        push!(log_prior_weights, -1.0 * log(2))
    end
    belief = MixturePrevision(components, log_prior_weights)
    belief
end
