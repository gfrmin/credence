# Role: brain-side application
# Boolean algebra on events composes into the same condition(m, e::Event) form.
# Conjunction / Disjunction / Complement are declared structure — no opaque
# predicate closures at the axiom layer.

using Credence

function mean_approval_given_both(m::MixtureMeasure,
                                   fires_a::Set{Int},
                                   fires_b::Set{Int})
    event = Conjunction(TagSet(m.space, fires_a), TagSet(m.space, fires_b))
    expect(condition(m, event), Identity())
end
