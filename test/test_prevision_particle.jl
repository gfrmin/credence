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
    m = GammaMeasure(2.0, 3.0)
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
    m = BetaMeasure(2.0, 3.0)
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
    m = GaussianMeasure(Euclidean(1), 0.0, 1.0)
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

println()
println("="^60)
println("ALL CANONICAL BIT-INVARIANCE TESTS PASSED (Move 6 Phase 0)")
println("="^60)
