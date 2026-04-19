# Role: brain-side application
# Event-conditioning: condition(m, TagSet(...)) followed by expect(_, Identity()).
# This is the declarative path for conditional expectation over a mixture
# filtered by component tag — the replacement for hand-rolled posterior
# iteration. No arithmetic on DSL returns; no pragma needed.

using Credence

function mean_approval_given_fires(m::MixtureMeasure, fires::Set{Int})
    isempty(fires) && return 0.5
    restricted = condition(m, TagSet(m.space, fires))
    expect(restricted, Identity())
end
