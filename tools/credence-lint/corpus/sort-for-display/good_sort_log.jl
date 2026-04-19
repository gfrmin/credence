# Role: brain-side application
# Sort-for-display with pragma — result goes to @info, not to branching.

using Credence

function log_top_k(m::CategoricalMeasure, k::Int)
    w = weights(m)
    pairs = collect(enumerate(w))
    # credence-lint: allow — precedent:sort-for-display — result is printed, not branched on
    sort!(pairs; by=p -> p[2], rev=true)
    for (i, wi) in pairs[1:k]
        @info "  [$i] $wi"
    end
end
