#!/usr/bin/env julia
"""
    test_events.jl — Tests for the Event type hierarchy and indicator_kernel.

Verifies: Event constructors declare structure correctly, indicator_kernel
witnesses produce 0 / -Inf log_density in the right places, and Boolean
algebra (Conjunction, Disjunction, Complement) composes indicators as
expected.

The `condition(::Measure, ::Event)` sibling form is tested in
`test_core.jl`'s equivalence section; here we test the kernels
themselves.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: BetaPrevision, GaussianPrevision, GammaPrevision, CategoricalPrevision  # Posture 4 Move 4
using Credence.Ontology: wrap_in_measure  # Posture 4 Move 4

passed = 0
failed = 0
macro check(name, expr)
    quote
        try
            if $(esc(expr))
                global passed += 1
                println("PASSED: ", $(esc(name)))
            else
                global failed += 1
                println("FAILED: ", $(esc(name)))
            end
        catch e
            global failed += 1
            println("FAILED: ", $(esc(name)), " (exception: ", e, ")")
        end
    end
end

println("=" ^ 60)
println("TEST 1: TagSet indicator kernel")
println("=" ^ 60)

let
    e = TagSet(Interval(0.0, 1.0), Set([1, 3]))
    k = indicator_kernel(e)

    @check "source is the declared Space" k.source === Interval(0.0, 1.0) || k.source == Interval(0.0, 1.0)
    @check "target is BOOLEAN_SPACE" k.target === BOOLEAN_SPACE
    @check "likelihood_family is Flat" k.likelihood_family isa Flat

    tbm_1 = TaggedBetaMeasure(Interval(0.0, 1.0), 1, BetaPrevision(1.0, 1.0))
    tbm_2 = TaggedBetaMeasure(Interval(0.0, 1.0), 2, BetaPrevision(1.0, 1.0))
    tbm_3 = TaggedBetaMeasure(Interval(0.0, 1.0), 3, BetaPrevision(1.0, 1.0))

    @check "log_density(tag=1, true) == 0.0  [tag in fires]" k.log_density(tbm_1, true) == 0.0
    @check "log_density(tag=1, false) == -Inf [tag in fires]" k.log_density(tbm_1, false) == -Inf
    @check "log_density(tag=2, true) == -Inf [tag not in fires]" k.log_density(tbm_2, true) == -Inf
    @check "log_density(tag=2, false) == 0.0 [tag not in fires]" k.log_density(tbm_2, false) == 0.0
    @check "log_density(tag=3, true) == 0.0  [tag in fires]" k.log_density(tbm_3, true) == 0.0
end
println()

println("=" ^ 60)
println("TEST 2: Complement inverts the inner event")
println("=" ^ 60)

let
    inner = TagSet(Interval(0.0, 1.0), Set([1, 3]))
    outer = Complement(inner)
    ki = indicator_kernel(inner)
    ko = indicator_kernel(outer)

    tbm_1 = TaggedBetaMeasure(Interval(0.0, 1.0), 1, BetaPrevision(1.0, 1.0))
    tbm_2 = TaggedBetaMeasure(Interval(0.0, 1.0), 2, BetaPrevision(1.0, 1.0))

    @check "inner holds at tag 1" ki.log_density(tbm_1, true) == 0.0
    @check "complement does not hold at tag 1" ko.log_density(tbm_1, true) == -Inf
    @check "inner does not hold at tag 2" ki.log_density(tbm_2, true) == -Inf
    @check "complement holds at tag 2" ko.log_density(tbm_2, true) == 0.0
end
println()

println("=" ^ 60)
println("TEST 3: Conjunction — both must hold")
println("=" ^ 60)

let
    sp = Interval(0.0, 1.0)
    # Two events sharing the same Space instance (required by Conjunction)
    a = TagSet(sp, Set([1, 2, 3]))
    b = TagSet(sp, Set([2, 3, 4]))
    both = Conjunction(a, b)
    k = indicator_kernel(both)

    tbm(t) = TaggedBetaMeasure(Interval(0.0, 1.0), t, BetaPrevision(1.0, 1.0))
    @check "tag 1 ∈ a, ∉ b → conjunction false" k.log_density(tbm(1), true) == -Inf
    @check "tag 2 ∈ both        → conjunction true"  k.log_density(tbm(2), true) == 0.0
    @check "tag 3 ∈ both        → conjunction true"  k.log_density(tbm(3), true) == 0.0
    @check "tag 4 ∈ b, ∉ a → conjunction false" k.log_density(tbm(4), true) == -Inf
    @check "tag 5 ∉ either      → conjunction false" k.log_density(tbm(5), true) == -Inf
end
println()

println("=" ^ 60)
println("TEST 4: Disjunction — either may hold")
println("=" ^ 60)

let
    sp = Interval(0.0, 1.0)
    a = TagSet(sp, Set([1, 2]))
    b = TagSet(sp, Set([3, 4]))
    either = Disjunction(a, b)
    k = indicator_kernel(either)

    tbm(t) = TaggedBetaMeasure(Interval(0.0, 1.0), t, BetaPrevision(1.0, 1.0))
    @check "tag 1 ∈ a        → disjunction true"  k.log_density(tbm(1), true) == 0.0
    @check "tag 3 ∈ b        → disjunction true"  k.log_density(tbm(3), true) == 0.0
    @check "tag 5 ∉ either  → disjunction false" k.log_density(tbm(5), true) == -Inf
end
println()

println("=" ^ 60)
println("TEST 5: De Morgan — ¬(A ∧ B) == (¬A) ∨ (¬B)")
println("=" ^ 60)

let
    sp = Interval(0.0, 1.0)
    a = TagSet(sp, Set([1, 2, 3]))
    b = TagSet(sp, Set([2, 3, 4]))

    lhs = indicator_kernel(Complement(Conjunction(a, b)))
    rhs = indicator_kernel(Disjunction(Complement(a), Complement(b)))

    tbm(t) = TaggedBetaMeasure(Interval(0.0, 1.0), t, BetaPrevision(1.0, 1.0))
    all_match = all(t -> lhs.log_density(tbm(t), true) == rhs.log_density(tbm(t), true), 0:5)
    @check "De Morgan identity holds on tags 0..5" all_match
end
println()

println("=" ^ 60)
println("TEST 6: Conjunction requires operands over the same Space")
println("=" ^ 60)

let
    sp1 = Interval(0.0, 1.0)
    sp2 = Interval(0.0, 2.0)  # different Space, different .hi
    a = TagSet(sp1, Set([1]))
    b = TagSet(sp2, Set([2]))

    threw = false
    try
        indicator_kernel(Conjunction(a, b))
    catch e
        threw = true
    end
    @check "mismatched Spaces raise at indicator_kernel construction" threw
end
println()

println("=" ^ 60)
println("TEST 7: condition(m, TagSet(…)) drops non-firing components")
println("=" ^ 60)

let
    sp = Interval(0.0, 1.0)
    components = [TaggedBetaMeasure(sp, t, wrap_in_measure(BetaPrevision(2.0, 3.0))) for t in 1:4]
    m = MixtureMeasure(sp, components, Float64[0.0, 0.0, 0.0, 0.0])

    posterior = condition(m, TagSet(sp, Set([1, 3])))
    w = weights(posterior)

    @check "component count preserved" length(posterior.components) == 4
    @check "firing components equally weighted" abs(w[1] - 0.5) < 1e-12 && abs(w[3] - 0.5) < 1e-12
    @check "non-firing components have zero weight" w[2] == 0.0 && w[4] == 0.0
    @check "firing components unchanged (Flat)" posterior.components[1] === components[1]
end
println()

println("=" ^ 60)
println("TEST 8: Empty TagSet (no component fires) raises")
println("=" ^ 60)

let
    sp = Interval(0.0, 1.0)
    components = [TaggedBetaMeasure(sp, t, wrap_in_measure(BetaPrevision(1.0, 1.0))) for t in 1:3]
    m = MixtureMeasure(sp, components, Float64[0.0, 0.0, 0.0])

    threw = false
    try
        condition(m, TagSet(sp, Set{Int}()))
    catch e
        threw = true
    end
    @check "empty TagSet over a mixture raises (zero total mass)" threw
end
println()

println("=" ^ 60)
println("TEST 9: FeatureEquals over a struct hypothesis")
println("=" ^ 60)

let
    # Simple NamedTuple hypotheses — feature_value falls through to index access
    sp = Finite([(cat=:a, urgent=true),
                 (cat=:b, urgent=true),
                 (cat=:a, urgent=false)])
    e = FeatureEquals(sp, :cat, :a)
    k = indicator_kernel(e)

    @check "h.cat == :a → event holds"     k.log_density((cat=:a, urgent=true), true) == 0.0
    @check "h.cat == :b → event does not"  k.log_density((cat=:b, urgent=true), true) == -Inf
    @check "h.cat == :a (variant) still holds" k.log_density((cat=:a, urgent=false), true) == 0.0
end
println()

println("=" ^ 60)
println("TEST 10: FeatureInterval is closed on both ends")
println("=" ^ 60)

let
    sp = Euclidean(1)
    e = FeatureInterval(sp, :v, 0.2, 0.8)
    k = indicator_kernel(e)

    @check "v = 0.1 → below interval"  k.log_density((v=0.1,), true) == -Inf
    @check "v = 0.2 → at lower (closed)" k.log_density((v=0.2,), true) == 0.0
    @check "v = 0.5 → inside"            k.log_density((v=0.5,), true) == 0.0
    @check "v = 0.8 → at upper (closed)" k.log_density((v=0.8,), true) == 0.0
    @check "v = 0.9 → above interval"    k.log_density((v=0.9,), true) == -Inf
end
println()

println("=" ^ 60)
if failed == 0
    println("ALL EVENT TESTS PASSED ($passed checks)")
else
    println("FAILURES: $failed / $(passed + failed)")
    exit(1)
end
println("=" ^ 60)
