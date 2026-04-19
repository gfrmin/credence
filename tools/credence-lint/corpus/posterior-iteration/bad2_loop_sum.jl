# Role: brain-side application
# Manual loop over support — rewrite as expect(m, Indicator).

using Credence

function mass_above(m::CategoricalMeasure, threshold::Float64)
    w = weights(m)
    total = 0.0
    for (h, wi) in zip(support(m), w)
        if h > threshold
            total += wi  # violation: arithmetic on weights inside loop
        end
    end
    total
end
