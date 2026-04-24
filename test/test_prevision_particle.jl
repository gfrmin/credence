# test_prevision_particle.jl — Stratum-1/Stratum-2 for Move 6's
# particle / quadrature / enumeration refactor.
#
# **Capture-before-refactor discipline.** The canonical values this test
# asserts against were captured from master at SHA 173411b (Move 5's tip,
# pre-Move-6) under Random.seed!(42). They live in
# test/fixtures/particle_canonical_v1.jls. See test/fixtures/README.md for
# the provenance protocol.
#
# This is the tautological-at-capture-moment pattern: at the commit that
# introduces this file, the code under test is master's pre-refactor code,
# and the assertions pass because the code producing the values is the
# code being tested. The tautology is the point — it becomes load-bearing
# the moment the refactor begins. Any subsequent Move 6 commit that
# breaks the == assertion has introduced a seed-consumption reorder or
# arithmetic reassociation; halt, investigate per move-6-design.md §6 R1.
#
# Per precedents.md §4, seeded-MC == is the Stratum-2 tolerance for
# particle paths. Do NOT relax to rtol=1e-12 to make a failing test pass;
# that silently masks the exact class of regression this test exists to
# catch.

push!(LOAD_PATH, "src")
using Credence
using Credence: BetaPrevision, GammaPrevision, GaussianPrevision  # Posture 4 Move 4: Prevision constructors
using Credence.Ontology: wrap_in_measure  # Posture 4 Move 4: Measure-wrap helper for Measure-dispatch consumers
using Random
using Serialization

function check(name, cond, detail="")
    if cond
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("assertion failed: $name")
    end
end

# Load the canonical fixture captured from master SHA 173411b.
const CANONICAL = open(deserialize, joinpath(@__DIR__, "fixtures", "particle_canonical_v1.jls"))

println("="^60)
println("Stratum 2 — particle / quadrature canonical bit-invariance (Move 6)")
println("="^60)
println("Fixture SHA: $(CANONICAL[:source_sha]); Julia $(CANONICAL[:julia_version])")

# ── Case 1: generic importance-sampling fallback (GammaMeasure + PushOnly) ──
#
# Non-conjugate kernel on GammaMeasure. maybe_conjugate returns nothing
# (no (GammaPrevision, PushOnly) pair registered); the generic particle
# fallback fires. 50 particles under seed 42 — samples and log_weights
# are bit-identical pre- and post-Move-6 refactor.

let
    Random.seed!(42)
    m = wrap_in_measure(GammaPrevision(2.0, 3.0))
    k = Kernel(PositiveReals(), Euclidean(1),
               λ -> error("generate not used"),
               (λ, o) -> -0.5 * (o - λ)^2;
               likelihood_family = PushOnly())
    result = condition(m, k, 2.5; n_particles=50)

    check("generic-fallback samples bit-identical to canonical (==)",
          result.space.values == CANONICAL[:gamma_generic_samples],
          "sample-order regression: $(length(setdiff(result.space.values, CANONICAL[:gamma_generic_samples]))) disjoint entries")
    check("generic-fallback log_weights bit-identical to canonical (==)",
          result.logw == CANONICAL[:gamma_generic_logw],
          "log-weight drift: max diff $(maximum(abs.(result.logw .- CANONICAL[:gamma_generic_logw])))")
end

# ── Case 2: grid quadrature on BetaMeasure ──
#
# PushOnly kernel on Beta → _condition_by_grid with n=64 grid points on
# Interval(0, 1). The grid itself is a deterministic range; log_weights
# depend on the kernel's log_density at each grid point.

let
    Random.seed!(42)  # not strictly needed for grid, but keep canonical path identical
    m = wrap_in_measure(BetaPrevision(2.0, 3.0))
    k = Kernel(Interval(0.0, 1.0), Euclidean(1),
               θ -> error("generate not used"),
               (θ, o) -> -0.5 * (o - θ)^2;
               likelihood_family = PushOnly())
    result = condition(m, k, 0.5)

    check("beta-grid values bit-identical to canonical (==)",
          result.space.values == CANONICAL[:beta_grid_values],
          "grid-order regression")
    check("beta-grid log_weights bit-identical to canonical (==)",
          result.logw == CANONICAL[:beta_grid_logw],
          "log-weight drift: max diff $(maximum(abs.(result.logw .- CANONICAL[:beta_grid_logw])))")
end

# ── Case 3: grid quadrature on GaussianMeasure ──
#
# PushOnly kernel on Gaussian → _condition_by_grid with n=64 points over
# μ ± 4σ. Grid is a deterministic range; log_weights pin the arithmetic.

let
    Random.seed!(42)
    m = wrap_in_measure(GaussianPrevision(0.0, 1.0))
    k = Kernel(Euclidean(1), Euclidean(1),
               μ -> error("generate not used"),
               (μ, o) -> log(max(o, 1e-300));
               likelihood_family = PushOnly())
    result = condition(m, k, 1.5)

    check("gaussian-grid values bit-identical to canonical (==)",
          result.space.values == CANONICAL[:gaussian_grid_values],
          "grid-order regression")
    check("gaussian-grid log_weights bit-identical to canonical (==)",
          result.logw == CANONICAL[:gaussian_grid_logw],
          "log-weight drift: max diff $(maximum(abs.(result.logw .- CANONICAL[:gaussian_grid_logw])))")
end

# ── Shared-reference contract tests per precedents.md §2 + §3 ──
#
# Per Move 6 design doc §5.1 Option A narrowed form and §4.5's typed-carrier
# trace: CategoricalMeasure wrapping ParticlePrevision / QuadraturePrevision /
# EnumerationPrevision must preserve reference identity between the wrapper's
# shield-accessed fields and the underlying Prevision's stored Vectors.
# Breaking this contract silently corrupts downstream consumers (skin server
# push! patterns, Move 7 event-conditioning paths, paper §2.3 isa-drilldown).
#
# Contract: cm.logw === pp.log_weights (identity, not equality).
#          cm.space.values === pp.samples / qp.grid / ep.enumerated.
#
# These tests are named in invariant comments on the shield at
# src/ontology.jl:getproperty(::CategoricalMeasure, …). Breaking either
# the shield or the test breaks both — precedent #3 executable-documentation.

function test_particle_shared_reference()
    samples = [1.0, 2.0, 3.0, 4.0]
    log_weights = [0.0, 0.0, 0.0, 0.0]
    pp = ParticlePrevision(samples, log_weights, 42)
    cm = CategoricalMeasure(Finite(pp.samples), pp)

    check("CategoricalMeasure wrapping ParticlePrevision stores pp by ref",
          cm.prevision === pp,
          "shield stored a copy instead of the passed Prevision")
    check("cm.logw === pp.log_weights (reference identity)",
          cm.logw === pp.log_weights,
          "shield defensively copied .log_weights; breaks consumer writes to the vector")
    check("cm.space.values === pp.samples (reference identity)",
          cm.space.values === pp.samples,
          "Finite(pp.samples) lost reference to pp.samples backing")
end

function test_quadrature_shared_reference()
    grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    log_weights = [-1.0, -0.5, 0.0, -0.5, -1.0]
    qp = QuadraturePrevision(grid, log_weights)
    cm = CategoricalMeasure(Finite(qp.grid), qp)

    check("CategoricalMeasure wrapping QuadraturePrevision stores qp by ref",
          cm.prevision === qp, "")
    check("cm.logw === qp.log_weights (reference identity)",
          cm.logw === qp.log_weights, "")
    check("cm.space.values === qp.grid (reference identity)",
          cm.space.values === qp.grid, "")
end

function test_enumeration_shared_reference()
    enumerated = Any[:a, :b, :c]
    log_weights = [log(0.5), log(0.3), log(0.2)]
    ep = EnumerationPrevision(enumerated, log_weights)
    # Enumeration wraps over a Finite{Symbol} space using ep.enumerated
    # as the values backing.
    cm = CategoricalMeasure(Finite(ep.enumerated), ep)

    check("CategoricalMeasure wrapping EnumerationPrevision stores ep by ref",
          cm.prevision === ep, "")
    check("cm.logw === ep.log_weights (reference identity)",
          cm.logw === ep.log_weights, "")
    check("cm.space.values === ep.enumerated (reference identity)",
          cm.space.values === ep.enumerated, "")
end

test_particle_shared_reference()
test_quadrature_shared_reference()
test_enumeration_shared_reference()

# ── _dispatch_path vocabulary pins per §5.3 Option A (uniform `:particle`) ──
#
# Per move-6-design.md §5.3 Option A: all three fallback strategies
# (particle, grid quadrature, enumeration) return `:particle` from
# _dispatch_path. Tests that need to distinguish which fallback fired
# can query the concrete type via isa.

using Credence: _dispatch_path

let k = Kernel(Euclidean(1), Euclidean(1),
               x -> error("unused"),
               (x, o) -> 0.0;
               likelihood_family = PushOnly())

    pp = ParticlePrevision([1.0, 2.0], [log(0.5), log(0.5)], 42)
    qp = QuadraturePrevision([0.1, 0.5, 0.9], [0.0, 0.0, 0.0])
    ep = EnumerationPrevision([:a, :b], [log(0.7), log(0.3)])

    check("ParticlePrevision → :particle (uniform fallback label)",
          _dispatch_path(pp, k) === :particle, "got $(_dispatch_path(pp, k))")
    check("QuadraturePrevision → :particle (uniform fallback label)",
          _dispatch_path(qp, k) === :particle, "got $(_dispatch_path(qp, k))")
    check("EnumerationPrevision → :particle (uniform fallback label)",
          _dispatch_path(ep, k) === :particle, "got $(_dispatch_path(ep, k))")

    # Drilldown via concrete type, per §5.3 Decoupling-from-§5.2:
    # tests that need the strategy distinction use `isa`, not the Symbol.
    check("ParticlePrevision isa ParticlePrevision (type-level drilldown)",
          pp isa ParticlePrevision, "")
    check("QuadraturePrevision isa QuadraturePrevision (type-level drilldown)",
          qp isa QuadraturePrevision, "")
    check("EnumerationPrevision isa EnumerationPrevision (type-level drilldown)",
          ep isa EnumerationPrevision, "")
end

println()
println("="^60)
println("ALL CANONICAL + SHIELD + DISPATCH TESTS PASSED (Move 6)")
println("="^60)
