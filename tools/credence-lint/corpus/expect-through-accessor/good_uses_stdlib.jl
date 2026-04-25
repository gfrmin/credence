# Role: brain-side application
# Legal: uses stdlib accessors instead of reading structural fields.

using Credence

function compute_stats(p::BetaPrevision)
    m = mean(p)
    v = variance(p)
    (m, v)
end

function check_event(m::MixtureMeasure, tags::Set{Int})
    probability(m, TagSet(m.space, tags))
end

function show_weights(m::MixtureMeasure)
    w = weights(m)
    println("weights: $w")
end
