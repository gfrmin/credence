# test_prevision_conjugate.jl — Stratum-2 (composition equivalence) for the
# Move 4 ConjugatePrevision registry.
#
# Per docs/posture-3/move-4-design.md §3, tolerances:
#   - Closed-form conjugate arithmetic (integer α/β accumulation): `==`
#   - Numerically-sensitive derived (e.g. Gaussian posterior μ via
#     precision-weighted averaging): `rtol=1e-12`
#   - Particle fallback under deterministic seeding: `==`
#
# Test structure per the Move 4 execution guidance: each conjugate-path
# test asserts `_dispatch_path(p, k) == :conjugate` BEFORE the value
# assertion. The dispatch-path assertion is the tripwire for silent
# registry misses; without it, a miss would fall through to particle and
# the value assertion might pass for the wrong reason.
#
# TaggedBetaPrevision specifically returns :particle — transitional
# scaffolding per the pre-committed routing resolution (PR #19, commit
# 6dce5e4). Move 5 moves this to MixturePrevision.

push!(LOAD_PATH, "src")
using Credence
using Credence: ConjugatePrevision, maybe_conjugate, update, _dispatch_path
using Credence: BetaPrevision, TaggedBetaPrevision

function check(name, cond, detail="")
    if cond
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("Stratum-2 assertion failed: $name")
    end
end

println("="^60)
println("Stratum 2 — ConjugatePrevision registry (Move 4)")
println("="^60)

# ── (BetaPrevision, BetaBernoulli) — replaces src/ontology.jl:891-904 ──
#
# Test structure: dispatch path first, then value. Dispatch-path
# assertion is load-bearing; a silent registry miss would pass the
# value assertion for the wrong reason.

let k = Kernel(Interval(0.0, 1.0), Finite([0, 1]),
               h -> CategoricalMeasure(Finite([0, 1])),
               (h, o) -> o == 1 ? log(max(h, 1e-300)) : log(max(1-h, 1e-300));
               likelihood_family = BetaBernoulli())
    p = BetaPrevision(2.0, 3.0)

    # Dispatch path FIRST.
    check("BetaPrevision + BetaBernoulli → :conjugate",
          _dispatch_path(p, k) === :conjugate,
          "got $(_dispatch_path(p, k))")

    # Value on obs=1: α+1, β unchanged. Bit-exact integer increment.
    cp = maybe_conjugate(p, k)
    check("maybe_conjugate returns ConjugatePrevision{BetaPrevision, BetaBernoulli}",
          cp isa ConjugatePrevision{BetaPrevision, BetaBernoulli},
          "got $(typeof(cp))")

    updated_1 = update(cp, 1).prior
    check("BetaPrevision(2, 3) × Bernoulli(obs=1) → BetaPrevision(3, 3) (==)",
          updated_1.alpha == 3.0 && updated_1.beta == 3.0,
          "got α=$(updated_1.alpha), β=$(updated_1.beta)")

    # Value on obs=0: α unchanged, β+1.
    updated_0 = update(cp, 0).prior
    check("BetaPrevision(2, 3) × Bernoulli(obs=0) → BetaPrevision(2, 4) (==)",
          updated_0.alpha == 2.0 && updated_0.beta == 4.0,
          "got α=$(updated_0.alpha), β=$(updated_0.beta)")

    # obs=true/false coerce identically.
    check("BetaBernoulli obs=true ≡ obs=1",
          update(cp, true).prior.alpha == 3.0,
          "got α=$(update(cp, true).prior.alpha)")
    check("BetaBernoulli obs=false ≡ obs=0",
          update(cp, false).prior.beta == 4.0,
          "got β=$(update(cp, false).prior.beta)")
end

# ── (BetaPrevision, Flat) — no-op ──
#
# A Flat kernel's likelihood does not depend on the Beta parameter;
# the posterior equals the prior. The registry handles this as a
# registered conjugate entry (the update is the identity function).

let k = Kernel(Interval(0.0, 1.0), Finite([0, 1]),
               h -> CategoricalMeasure(Finite([0, 1])),
               (h, o) -> 0.0;  # flat log-density
               likelihood_family = Flat())
    p = BetaPrevision(2.0, 3.0)

    check("BetaPrevision + Flat → :conjugate",
          _dispatch_path(p, k) === :conjugate,
          "got $(_dispatch_path(p, k))")

    cp = maybe_conjugate(p, k)
    check("maybe_conjugate returns ConjugatePrevision{BetaPrevision, Flat}",
          cp isa ConjugatePrevision{BetaPrevision, Flat},
          "got $(typeof(cp))")

    updated = update(cp, 1).prior
    check("BetaPrevision(2, 3) × Flat → BetaPrevision(2, 3) (unchanged, ==)",
          updated.alpha == 2.0 && updated.beta == 3.0,
          "got α=$(updated.alpha), β=$(updated.beta)")

    # Flat is observation-agnostic: any obs leaves the prior unchanged.
    updated_anything = update(cp, 0.7).prior
    check("BetaPrevision × Flat is obs-agnostic (arbitrary obs → same posterior)",
          updated_anything.alpha == 2.0 && updated_anything.beta == 3.0,
          "got α=$(updated_anything.alpha), β=$(updated_anything.beta)")
end

# ── TaggedBetaPrevision: must NOT match as conjugate (transitional scaffolding) ──
#
# Per PR #19's Move 1 revision addendum (commit 6dce5e4), TaggedBetaMeasure
# routing stays at the Measure level via the per-tag loop until Move 5's
# MixturePrevision takes over. This test pins that contract — if a future
# change to `maybe_conjugate` accidentally matches TaggedBetaPrevision,
# the per-tag routing gets silently bypassed and mixture-aware tests
# break in hard-to-diagnose ways. The guard here catches it at Stratum-2.

let k = Kernel(Interval(0.0, 1.0), Finite([0, 1]),
               h -> CategoricalMeasure(Finite([0, 1])),
               (h, o) -> 0.0;
               likelihood_family = BetaBernoulli())
    inner = BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0)
    tbp = TaggedBetaPrevision(42, inner)

    check("TaggedBetaPrevision + BetaBernoulli → :particle (transitional scaffolding)",
          _dispatch_path(tbp, k) === :particle,
          "got $(_dispatch_path(tbp, k)); " *
          "if this is :conjugate, the per-tag routing loop is silently bypassed — " *
          "check maybe_conjugate(::TaggedBetaPrevision, ...) method (should not exist until Move 5)")

    check("maybe_conjugate(TaggedBetaPrevision, BernoulliKernel) === nothing",
          maybe_conjugate(tbp, k) === nothing,
          "got $(maybe_conjugate(tbp, k))")
end

# ── Other Prevision types without registered conjugate pairs yet: particle ──
#
# Phase 1 registers BetaBernoulli only. Other previsions fall through
# to :particle; Phases 2-6 add their registry entries. This guard
# asserts the current state — if a future Phase 2 commit accidentally
# over-matches (e.g. maybe_conjugate(::GaussianPrevision, ...) matching
# too eagerly and firing for non-NormalNormal kernels), the value
# assertion below catches it.

let k = Kernel(Interval(0.0, 1.0), Finite([0, 1]),
               h -> CategoricalMeasure(Finite([0, 1])),
               (h, o) -> 0.0;
               likelihood_family = BetaBernoulli())
    using Credence: GaussianPrevision, GammaPrevision, DirichletPrevision, NormalGammaPrevision

    # Gaussian with a Bernoulli-ish kernel — should definitively NOT match.
    check("GaussianPrevision + BetaBernoulli kernel → :particle (no registered pair)",
          _dispatch_path(GaussianPrevision(0.0, 1.0), k) === :particle,
          "got $(_dispatch_path(GaussianPrevision(0.0, 1.0), k))")

    # Gamma with a Bernoulli-ish kernel — should NOT match.
    check("GammaPrevision + BetaBernoulli kernel → :particle",
          _dispatch_path(GammaPrevision(2.0, 1.0), k) === :particle,
          "got $(_dispatch_path(GammaPrevision(2.0, 1.0), k))")
end

println()
println("="^60)
println("ALL STRATUM-2 TESTS PASSED (Phase 1: BetaBernoulli only)")
println("="^60)
