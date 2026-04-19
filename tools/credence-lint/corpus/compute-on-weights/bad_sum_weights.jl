# Role: brain-side application
# Aggregating weights outside expect() — use expect(m, Indicator) instead.

using Credence

function posterior_mass(m::CategoricalMeasure)
    sum(weights(m))  # violation: arithmetic on weights
end
