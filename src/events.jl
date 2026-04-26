# events.jl — Event types (frozen layer) + indicator_kernel witnesses
#
# An Event is a declared proposition about the state of a Space.
# Events are bearers of probability in the de Finettian sense:
# P(A) is defined directly, not derived from a measure on subsets.
#
# Every Event constructor witnesses a computable indicator_kernel
# into BOOLEAN_SPACE. That witness is the mechanical bridge between
# the event layer and the kernel layer: condition(m, e::Event)
# expands to condition(m, indicator_kernel(e), true).
#
# Included inside module Ontology by ontology.jl.

abstract type Event end

"""
    TagSet(space, tags)

Event stating that a mixture component's tag lies in a declared finite
set of `Int`s. Peer of `FiringByTag` on the expect / posterior side:
declarative tag-based dispatch, no opaque closures.

Applicable to mixtures whose components carry `Int` tags (e.g.
`TaggedBetaMeasure`). The indicator kernel uses `_tag_of(component)`
to read the tag at dispatch time; add a `_tag_of` method for any new
tag-bearing Measure type.
"""
struct TagSet <: Event
    space::Space
    tags::Set{Int}
end

"""
    FeatureEquals(space, feature, value)

Deterministic equality event on a declared feature of hypotheses in
`space`. The indicator kernel queries `feature_value(h, feature)` —
add a method per hypothesis type when needed.

Valid only for discrete features; continuous equality is a
measure-zero event and must go through a disintegration primitive
(not yet implemented) — this constructor does not guard that case at
construction time; the dispatch is undefined on measure-zero events.
"""
struct FeatureEquals{T} <: Event
    space::Space
    feature::Symbol
    value::T
end

"""
    FeatureInterval(space, feature, lo, hi)

Event stating a declared continuous feature lies in the closed
interval [lo, hi]. `feature_value(h, feature)` must return a real
number for hypotheses `h` drawn from `space`.
"""
struct FeatureInterval <: Event
    space::Space
    feature::Symbol
    lo::Float64
    hi::Float64
end

"""
    Conjunction(left, right)

Event that holds iff both `left` and `right` hold. Operands must
share the same Space (checked at `indicator_kernel` construction).
"""
struct Conjunction <: Event
    left::Event
    right::Event
end

"""
    Disjunction(left, right)

Event that holds iff `left` or `right` holds.
"""
struct Disjunction <: Event
    left::Event
    right::Event
end

"""
    Complement(inner)

Event that holds iff `inner` does not.
"""
struct Complement <: Event
    inner::Event
end

# ── Tag accessor: declared-structure dispatch, not an opaque closure ──
_tag_of(m::TaggedBetaMeasure) = m.tag
_tag_of(p::TaggedBetaPrevision) = p.tag

# ── Feature accessor: method dispatch is the registry ──
"""
    feature_value(h, name::Symbol)

Extract the named feature from hypothesis `h`. Default: NamedTuple
index / struct field access. Override for domain-specific hypothesis
types via method dispatch.
"""
feature_value(h::NamedTuple, name::Symbol) = h[name]
feature_value(h, name::Symbol) = getfield(h, name)

# ── indicator_kernel: the mechanical bridge ──

"""
    indicator_kernel(event) → Kernel

Witness that an Event is expressible as a declared indicator kernel
into BOOLEAN_SPACE. Used internally by `condition(::Measure, ::Event)`.
Also the mechanical proof that Invariant 2 is preserved: events reach
the axiom layer only through declared kernels.
"""
function indicator_kernel(e::TagSet)
    ld = (h, o) -> begin
        holds = _tag_of(h) in e.tags
        (o === true && holds) || (o === false && !holds) ? 0.0 : -Inf
    end
    gen = h -> begin
        holds = _tag_of(h) in e.tags
        CategoricalMeasure(
            BOOLEAN_SPACE,
            [holds ? -Inf : 0.0,   # logw at false
             holds ? 0.0 : -Inf],  # logw at true
        )
    end
    Kernel(e.space, BOOLEAN_SPACE, gen, ld; likelihood_family = Flat())
end

function indicator_kernel(e::FeatureEquals{T}) where T
    ld = (h, o) -> begin
        holds = feature_value(h, e.feature) == e.value
        (o === true && holds) || (o === false && !holds) ? 0.0 : -Inf
    end
    gen = h -> begin
        holds = feature_value(h, e.feature) == e.value
        CategoricalMeasure(
            BOOLEAN_SPACE,
            [holds ? -Inf : 0.0, holds ? 0.0 : -Inf],
        )
    end
    Kernel(e.space, BOOLEAN_SPACE, gen, ld; likelihood_family = Flat())
end

function indicator_kernel(e::FeatureInterval)
    ld = (h, o) -> begin
        v = feature_value(h, e.feature)
        holds = e.lo <= v <= e.hi
        (o === true && holds) || (o === false && !holds) ? 0.0 : -Inf
    end
    gen = h -> begin
        v = feature_value(h, e.feature)
        holds = e.lo <= v <= e.hi
        CategoricalMeasure(
            BOOLEAN_SPACE,
            [holds ? -Inf : 0.0, holds ? 0.0 : -Inf],
        )
    end
    Kernel(e.space, BOOLEAN_SPACE, gen, ld; likelihood_family = Flat())
end

function indicator_kernel(e::Conjunction)
    kl = indicator_kernel(e.left)
    kr = indicator_kernel(e.right)
    kl.source === kr.source ||
        error("Conjunction: operands must share the same Space instance")
    ld = (h, o) -> begin
        holds = kl.log_density(h, true) == 0.0 && kr.log_density(h, true) == 0.0
        (o === true && holds) || (o === false && !holds) ? 0.0 : -Inf
    end
    gen = h -> error("indicator_kernel(Conjunction).generate not used in condition")
    Kernel(kl.source, BOOLEAN_SPACE, gen, ld; likelihood_family = Flat())
end

function indicator_kernel(e::Disjunction)
    kl = indicator_kernel(e.left)
    kr = indicator_kernel(e.right)
    kl.source === kr.source ||
        error("Disjunction: operands must share the same Space instance")
    ld = (h, o) -> begin
        holds = kl.log_density(h, true) == 0.0 || kr.log_density(h, true) == 0.0
        (o === true && holds) || (o === false && !holds) ? 0.0 : -Inf
    end
    gen = h -> error("indicator_kernel(Disjunction).generate not used in condition")
    Kernel(kl.source, BOOLEAN_SPACE, gen, ld; likelihood_family = Flat())
end

function indicator_kernel(e::Complement)
    ki = indicator_kernel(e.inner)
    ld = (h, o) -> begin
        holds = ki.log_density(h, true) != 0.0  # inner did NOT hold
        (o === true && holds) || (o === false && !holds) ? 0.0 : -Inf
    end
    gen = h -> error("indicator_kernel(Complement).generate not used in condition")
    Kernel(ki.source, BOOLEAN_SPACE, gen, ld; likelihood_family = Flat())
end
