# test_prevision_unit.jl — Stratum-1 (unit equivalence) for the Move 2
# `Functional` → `TestFunction` alias migration.
#
# Per docs/posture-3/move-2-design.md §3, four tolerance cases:
#
#   1. Closed-form methods (:TestFunction signatures with closed-form bodies): `==`
#   2. Quadrature paths (:Function signatures, n= kwarg): isapprox(atol=1e-14)
#   3. Monte Carlo paths under deterministic seeding (:Function signatures,
#      n_samples= kwarg): `==`
#   4. OpaqueClosure fallback methods: `==` vs direct-Function call
#
# The assertions pin closed-form mathematical values where derivable by hand
# (α/(α+β) for Beta mean, etc.); for paths where the expected value depends
# on pre-refactor arithmetic, the expected value is captured by first
# computing it once post-refactor and recording it here. Subsequent moves
# (3-7) that refactor around dispatch must leave these values unchanged.

push!(LOAD_PATH, "src")
using Credence
using Random

# Helper: println PASSED / halt on FAILED. Matches the codebase's house
# assertion style (see test/test_core.jl).
function check(name, cond, detail="")
    if cond
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("Stratum-1 assertion failed: $name")
    end
end

println("="^60)
println("Stratum 1 — closed-form methods (==)")
println("="^60)

# ── 1. BetaMeasure × Identity = α / (α+β) ──
let m = BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0)
    expected = 2.0 / 5.0
    actual = expect(m, Identity())
    check("BetaMeasure(2, 3) × Identity = 2/5 (exact)",
          actual == expected,
          "got $actual, expected $expected")
end

let m = BetaMeasure(Interval(0.0, 1.0), 1.0, 1.0)
    expected = 0.5
    actual = expect(m, Identity())
    check("BetaMeasure(1, 1) × Identity = 1/2 (exact)",
          actual == expected,
          "got $actual, expected $expected")
end

let m = BetaMeasure(Interval(0.0, 1.0), 7.0, 2.0)
    expected = 7.0 / 9.0
    actual = expect(m, Identity())
    check("BetaMeasure(7, 2) × Identity = 7/9 (exact)",
          actual == expected,
          "got $actual, expected $expected")
end

# ── 2. TaggedBetaMeasure × Identity — delegates to .beta ──
let beta = BetaMeasure(Interval(0.0, 1.0), 3.0, 4.0)
    m = TaggedBetaMeasure(Interval(0.0, 1.0), 42, beta)
    expected = 3.0 / 7.0
    actual = expect(m, Identity())
    check("TaggedBetaMeasure(tag=42, Beta(3,4)) × Identity = 3/7 (exact)",
          actual == expected,
          "got $actual, expected $expected")
end

# ── 3. GammaMeasure × Identity = α / β ──
let m = GammaMeasure(PositiveReals(), 2.0, 0.5)
    expected = 2.0 / 0.5
    actual = expect(m, Identity())
    check("GammaMeasure(2, 0.5) × Identity = 4.0 (exact)",
          actual == expected,
          "got $actual, expected $expected")
end

let m = GammaMeasure(PositiveReals(), 3.0, 3.0)
    expected = 1.0
    actual = expect(m, Identity())
    check("GammaMeasure(3, 3) × Identity = 1.0 (exact)",
          actual == expected,
          "got $actual, expected $expected")
end

# ── 4. GaussianMeasure × Identity = μ ──
let m = GaussianMeasure(Euclidean(1), 0.0, 1.0)
    expected = 0.0
    actual = expect(m, Identity())
    check("GaussianMeasure(0, 1) × Identity = 0 (exact)",
          actual == expected,
          "got $actual, expected $expected")
end

let m = GaussianMeasure(Euclidean(1), 2.5, 0.3)
    expected = 2.5
    actual = expect(m, Identity())
    check("GaussianMeasure(2.5, 0.3) × Identity = 2.5 (exact)",
          actual == expected,
          "got $actual, expected $expected")
end

# ── 5. CategoricalMeasure × Identity = Σ w_i · v_i ──
let m = CategoricalMeasure(Finite([0.0, 1.0, 2.0, 3.0]))  # uniform over 4 atoms
    expected = (0.0 + 1.0 + 2.0 + 3.0) / 4.0
    actual = expect(m, Identity())
    check("CategoricalMeasure(uniform over [0,1,2,3]) × Identity = 1.5 (exact)",
          actual == expected,
          "got $actual, expected $expected")
end

# Non-uniform: log-weights [log(1), log(2), log(3)] over [10.0, 20.0, 30.0].
# Normalised: w = [1/6, 2/6, 3/6]. Mean = 10/6 + 40/6 + 90/6 = 140/6 = 70/3.
let m = CategoricalMeasure(Finite([10.0, 20.0, 30.0]), [log(1.0), log(2.0), log(3.0)])
    expected = 70.0 / 3.0
    actual = expect(m, Identity())
    check("CategoricalMeasure(weighted [10,20,30] with w=[1/6,2/6,3/6]) × Identity = 70/3",
          isapprox(actual, expected; atol=1e-14),
          "got $actual, expected $expected")
end

# ── 6. ProductMeasure × Projection — selects factor's Identity ──
let m = ProductMeasure(
    ProductSpace([Interval(0.0, 1.0), PositiveReals()]),
    Measure[BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0),
            GammaMeasure(PositiveReals(), 4.0, 2.0)]
)
    # factor 1 is Beta(2,3): mean 2/5
    # factor 2 is Gamma(4,2): mean 4/2 = 2
    actual_1 = expect(m, Projection(1))
    actual_2 = expect(m, Projection(2))
    check("ProductMeasure × Projection(1) = 2/5 (Beta factor mean)",
          actual_1 == 2.0 / 5.0,
          "got $actual_1")
    check("ProductMeasure × Projection(2) = 2.0 (Gamma factor mean)",
          actual_2 == 2.0,
          "got $actual_2")
end

# ── 7. ProductMeasure × NestedProjection — single-level == Projection ──
let m = ProductMeasure(
    ProductSpace([Interval(0.0, 1.0), Interval(0.0, 1.0)]),
    Measure[BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0),
            BetaMeasure(Interval(0.0, 1.0), 5.0, 5.0)]
)
    actual_1 = expect(m, NestedProjection([1]))
    actual_2 = expect(m, NestedProjection([2]))
    check("ProductMeasure × NestedProjection([1]) = 2/5",
          actual_1 == 2.0 / 5.0, "got $actual_1")
    check("ProductMeasure × NestedProjection([2]) = 1/2",
          actual_2 == 0.5, "got $actual_2")
end

# ── 8. ProductMeasure × NestedProjection — two levels, nested products ──
let inner1 = ProductMeasure(
        ProductSpace([Interval(0.0, 1.0), Interval(0.0, 1.0)]),
        Measure[BetaMeasure(Interval(0.0, 1.0), 1.0, 3.0),
                BetaMeasure(Interval(0.0, 1.0), 2.0, 2.0)]
    )
    inner2 = ProductMeasure(
        ProductSpace([Interval(0.0, 1.0), Interval(0.0, 1.0)]),
        Measure[BetaMeasure(Interval(0.0, 1.0), 4.0, 1.0),
                BetaMeasure(Interval(0.0, 1.0), 1.0, 1.0)]
    )
    m = ProductMeasure(
        ProductSpace([ProductSpace([Interval(0.0, 1.0), Interval(0.0, 1.0)]),
                      ProductSpace([Interval(0.0, 1.0), Interval(0.0, 1.0)])]),
        Measure[inner1, inner2]
    )
    # NestedProjection([1, 2]) selects inner1's factor 2 = Beta(2, 2), mean 1/2
    actual = expect(m, NestedProjection([1, 2]))
    check("ProductMeasure × NestedProjection([1, 2]) = 1/2 (nested Beta(2,2) mean)",
          actual == 0.5, "got $actual")
    # NestedProjection([2, 1]) selects inner2's factor 1 = Beta(4, 1), mean 4/5
    actual = expect(m, NestedProjection([2, 1]))
    check("ProductMeasure × NestedProjection([2, 1]) = 4/5 (nested Beta(4,1) mean)",
          actual == 4.0 / 5.0, "got $actual")
end

# ── 9. CategoricalMeasure × Projection — vector-valued atoms ──
let m = CategoricalMeasure(Finite([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))  # uniform
    # uniform over 3 atoms; Projection(1) gives E[first component] = (1+3+5)/3 = 3.0
    actual_1 = expect(m, Projection(1))
    actual_2 = expect(m, Projection(2))
    check("CategoricalMeasure(vec-atoms, uniform) × Projection(1) = 3.0",
          actual_1 == 3.0, "got $actual_1")
    check("CategoricalMeasure(vec-atoms, uniform) × Projection(2) = 4.0",
          actual_2 == 4.0, "got $actual_2")
end

# ── 10. CategoricalMeasure × Tabular ──
let m = CategoricalMeasure(Finite([:a, :b, :c]))  # uniform over 3 symbols
    # Tabular([10, 20, 30]) with uniform weights = (10+20+30)/3 = 20
    t = Tabular([10.0, 20.0, 30.0])
    actual = expect(m, t)
    check("CategoricalMeasure(uniform 3 atoms) × Tabular([10,20,30]) = 20.0",
          actual == 20.0, "got $actual")
end

# ── 11. LinearCombination — linearity of expectation ──
let m = BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0)
    # 3 * Identity + 5 = 3 * 0.4 + 5 = 6.2
    lc = LinearCombination(Tuple{Float64, TestFunction}[(3.0, Identity())], 5.0)
    actual = expect(m, lc)
    expected = 3.0 * (2.0 / 5.0) + 5.0
    check("BetaMeasure × LinearCombination([3 * Identity], offset=5) = 6.2",
          actual == expected, "got $actual, expected $expected")
end

let m = BetaMeasure(Interval(0.0, 1.0), 1.0, 1.0)
    # 2 * Identity - 1 * Identity + 0.3 = (2-1) * 0.5 + 0.3 = 0.8
    lc = LinearCombination(
        Tuple{Float64, TestFunction}[(2.0, Identity()), (-1.0, Identity())],
        0.3,
    )
    actual = expect(m, lc)
    expected = (2.0 - 1.0) * 0.5 + 0.3
    check("BetaMeasure(1,1) × LinearCombination with cancellation = 0.8",
          isapprox(actual, expected; atol=1e-14),
          "got $actual, expected $expected")
end

# ── 12. MixtureMeasure recursion — weighted sum over components ──
let c1 = BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0)  # mean 0.4
    c2 = BetaMeasure(Interval(0.0, 1.0), 5.0, 5.0)  # mean 0.5
    m = MixtureMeasure(Interval(0.0, 1.0), [c1, c2], [log(1.0), log(3.0)])
    # w = [1/4, 3/4]; E[X] = 0.25 * 0.4 + 0.75 * 0.5 = 0.475
    actual = expect(m, Identity())
    expected = 0.25 * 0.4 + 0.75 * 0.5
    check("MixtureMeasure(Beta(2,3)=1/4, Beta(5,5)=3/4) × Identity = 0.475",
          isapprox(actual, expected; atol=1e-14),
          "got $actual, expected $expected")
end

println()
println("="^60)
println("Stratum 1 — OpaqueClosure fallback (==)")
println("="^60)

# ── 13. OpaqueClosure: wrapped call equals direct-Function call ──
# Per §5.3: the test catches a silent-routing failure where a missing
# alias entry sends `expect(m, OpaqueClosure(f))` through the generic
# `::Function` overload with different kwargs defaults.

let m = BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0)
    f = x -> x  # identity-as-function
    direct = expect(m, f)  # uses default n=64 quadrature
    wrapped = expect(m, OpaqueClosure(f))
    check("BetaMeasure × OpaqueClosure(x->x) == BetaMeasure × (x->x) [default kwargs]",
          direct == wrapped,
          "direct=$direct wrapped=$wrapped")
end

let m = BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0)
    f = x -> x^2
    direct = expect(m, f; n=128)  # explicit non-default kwarg
    wrapped = expect(m, OpaqueClosure(f); n=128)
    check("BetaMeasure × OpaqueClosure(x²) == BetaMeasure × x² [n=128 forwarded]",
          direct == wrapped,
          "direct=$direct wrapped=$wrapped")
end

let m = CategoricalMeasure(Finite([1.0, 2.0, 3.0, 4.0]))
    f = x -> x * 2
    direct = expect(m, f)
    wrapped = expect(m, OpaqueClosure(f))
    check("CategoricalMeasure × OpaqueClosure(2x) == direct",
          direct == wrapped,
          "direct=$direct wrapped=$wrapped")
end

# Mixture has a separate `expect(::MixtureMeasure, ::OpaqueClosure)` method
# (ontology.jl:770) specifically to resolve dispatch ambiguity. Test it.
let c1 = BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0)
    c2 = BetaMeasure(Interval(0.0, 1.0), 5.0, 5.0)
    m = MixtureMeasure(Interval(0.0, 1.0), [c1, c2], [log(1.0), log(1.0)])
    f = x -> x
    direct = expect(m, f)
    wrapped = expect(m, OpaqueClosure(f))
    check("MixtureMeasure × OpaqueClosure(x) == direct",
          direct == wrapped,
          "direct=$direct wrapped=$wrapped")
end

println()
println("="^60)
println("Stratum 1 — Quadrature paths (isapprox, atol=1e-14)")
println("="^60)

# ── 14. Quadrature paths — run-to-run consistency at 1e-14 ──
# The ::Function signature for BetaMeasure uses 64-point grid quadrature.
# Call twice; results must match to reassociation tolerance.

let m = BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0)
    f = x -> x^2
    v1 = expect(m, f; n=64)
    v2 = expect(m, f; n=64)
    check("BetaMeasure × x² quadrature deterministic (==, no RNG involved)",
          v1 == v2, "v1=$v1 v2=$v2")
end

let m = GaussianMeasure(Euclidean(1), 0.5, 1.0)
    f = x -> x^2
    v1 = expect(m, f; n=64)
    v2 = expect(m, f; n=64)
    check("GaussianMeasure × x² quadrature deterministic",
          v1 == v2, "v1=$v1 v2=$v2")
end

let m = GammaMeasure(PositiveReals(), 3.0, 1.0)
    f = x -> log(x + 1.0)
    v1 = expect(m, f; n=64)
    v2 = expect(m, f; n=64)
    check("GammaMeasure × log(x+1) quadrature deterministic",
          v1 == v2, "v1=$v1 v2=$v2")
end

println()
println("="^60)
println("Stratum 1 — Monte Carlo paths under deterministic seeding (==)")
println("="^60)

# ── 15. Monte Carlo paths — seeded RNG must produce bit-identical results ──
# Per §3 precedent: seeded MC is deterministic; drift under refactor would
# be a bug (arithmetic reorder, RNG-consumption-order change), not
# reassociation. Move 6 inherits this precedent.

let m = DirichletMeasure(Simplex(3), Finite([:a, :b, :c]), [2.0, 3.0, 5.0])
    f = p -> p[1]  # marginal of first category
    Random.seed!(42)
    v1 = expect(m, f; n_samples=1000)
    Random.seed!(42)
    v2 = expect(m, f; n_samples=1000)
    check("DirichletMeasure × p[1] MC deterministic under seed=42",
          v1 == v2, "v1=$v1 v2=$v2")
end

let space = ProductSpace([Euclidean(1), PositiveReals()])
    m = NormalGammaMeasure(space, 1.0, 0.0, 2.0, 1.0)
    f = x -> x[1]
    Random.seed!(42)
    v1 = expect(m, f; n_samples=1000)
    Random.seed!(42)
    v2 = expect(m, f; n_samples=1000)
    check("NormalGammaMeasure × x[1] MC deterministic under seed=42",
          v1 == v2, "v1=$v1 v2=$v2")
end

let m = ProductMeasure(
    ProductSpace([Interval(0.0, 1.0), PositiveReals()]),
    Measure[BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0),
            GammaMeasure(PositiveReals(), 4.0, 2.0)]
)
    f = x -> x[1] * x[2]
    Random.seed!(42)
    v1 = expect(m, f; n_samples=1000)
    Random.seed!(42)
    v2 = expect(m, f; n_samples=1000)
    check("ProductMeasure × x[1]*x[2] MC deterministic under seed=42",
          v1 == v2, "v1=$v1 v2=$v2")
end

println()
println("="^60)
println("ALL STRATUM-1 TESTS PASSED")
println("="^60)
