# spaces.jl — Space types (frozen layer)
#
# A Space is a set of possibilities. The type and its constructors are
# frozen; the constructor roster is not.
#
# Included inside module Ontology by ontology.jl.

abstract type Space end

struct Finite{T} <: Space
    values::Vector{T}
end

struct Interval <: Space
    lo::Float64
    hi::Float64
end

struct ProductSpace <: Space
    factors::Vector{Space}
end

struct Simplex <: Space
    k::Int  # Δ^(k-1): vectors of length k, non-negative, summing to 1
end

struct Euclidean <: Space
    dim::Int
end

struct PositiveReals <: Space end

support(s::Finite) = s.values

const BOOLEAN_SPACE = Finite([false, true])
