# Role: brain-side application
# Bare [] flowing into ProductPrevision — also untyped.

using Credence

function build_product()
    factors = []
    push!(factors, BetaPrevision(1.0, 1.0))
    push!(factors, BetaPrevision(2.0, 3.0))
    ProductPrevision(factors)
end
