# Role: brain-side application
# convert(Vector{Any}, ...) flowing into EnumerationMeasure — the retired pattern.

using Credence

function build_enumeration(programs, log_weights)
    items = convert(Vector{Any}, programs)
    EnumerationMeasure{Program}(CategoricalPrevision(log_weights), items, Finite(programs))
end
