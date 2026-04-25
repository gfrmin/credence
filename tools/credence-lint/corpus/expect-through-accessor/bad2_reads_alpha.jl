# Role: brain-side application
# Reads structural fields to compute mean — should use mean(p).

using Credence

function manual_mean(p)
    m = p.alpha / (p.alpha + p.beta)
    m
end
