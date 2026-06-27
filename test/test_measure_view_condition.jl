# test_measure_view_condition.jl — Phase 2 of the measure-as-view arc.
#
# Inverts the BACKWARDS DELEGATION in `condition`/`_predictive_ll` for the continuous-scalar families
# (Beta/Gaussian/Gamma) to Prevision-primary, restoring "Measure is a declared view over Prevision"
# (`prevision-not-measure`). Pre-Phase-2 the non-conjugate fallbacks round-tripped `condition(p)` →
# `wrap_in_measure(p)` → `condition(::Measure)` (the Prevision reaching its own behaviour through its
# view). Post-Phase-2: the Prevision owns the space-free posterior (grid → QuadraturePrevision, particle
# → ParticlePrevision); the Measure facade owns the carrier RE-BIND (`Measure binds carrier; Prevision
# does not`).
#
# The 5/4 gate split (design doc §1): the continuous families have a TYPE-RECOVERABLE support
# (Interval(0,1)/Euclidean(1)/PositiveReals, read off the Prevision type), so they invert here. The
# discrete/Product/catch-all sites bind a DATA-valued carrier (CategoricalPrevision has no carrier-free
# `draw`) and defer to Phase 3 (#163).
#
# Capture-before-refactor REUSES test/fixtures/particle_canonical_v1.jls (SHA 173411b, `==` tol) — the
# same canonical vectors test_prevision_particle.jl asserts on the Measure entry points, now asserted on
# the Prevision entry points. No new capture: `draw(p::GammaPrevision)` ≡ `draw(m::GammaMeasure)` in RNG
# ops, so the relocation is bit-identical under fixed seed. Per precedents.md §4 the `==` class is NOT
# relaxable to rtol.
#
# Run from repo root:
#     julia test/test_measure_view_condition.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: BetaPrevision, GaussianPrevision, GammaPrevision, ParticlePrevision,
                QuadraturePrevision, condition, wrap_in_measure
using Credence.Ontology: _predictive_ll
using Random
using Serialization

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

const CANON = open(deserialize, joinpath(@__DIR__, "fixtures", "particle_canonical_v1.jls"))

# Kernels mirror test_prevision_particle.jl EXACTLY so the canonical fixture values apply to the
# Prevision entry points (non-conjugate → grid/particle fallback).
const K_GAMMA = Kernel(PositiveReals(), Euclidean(1),
                       λ -> error("generate not used"), (λ, o) -> -0.5 * (o - λ)^2;
                       likelihood_family = PushOnly())
const K_BETA  = Kernel(Interval(0.0, 1.0), Euclidean(1),
                       θ -> error("generate not used"), (θ, o) -> -0.5 * (o - θ)^2;
                       likelihood_family = PushOnly())
const K_GAUSS = Kernel(Euclidean(1), Euclidean(1),
                       μ -> error("generate not used"), (μ, o) -> log(max(o, 1e-300));
                       likelihood_family = PushOnly())

println("="^64)
println("measure-as-view Phase 2 — condition inversion (scalar, carrier re-bind)")
println("Fixture SHA: $(CANON[:source_sha]); Julia $(CANON[:julia_version])")
println("="^64)

# ── (1) Prevision-entry non-conjugate == canonical (the inversion, bit-exact, no new capture) ──
let
    Random.seed!(42)
    pg = condition(GammaPrevision(2.0, 3.0), K_GAMMA, 2.5; n_particles = 50)
    check("condition(GammaPrevision) non-conjugate → ParticlePrevision (Prevision-primary)",
          pg isa ParticlePrevision, "got $(typeof(pg))")
    check("Gamma particle samples == canonical (Prevision-entry, ==)",
          pg.samples == CANON[:gamma_generic_samples])
    check("Gamma particle log_weights == canonical (Prevision-entry, ==)",
          pg.log_weights == CANON[:gamma_generic_logw])
end
let
    Random.seed!(42)
    pb = condition(BetaPrevision(2.0, 3.0), K_BETA, 0.5)
    check("condition(BetaPrevision) non-conjugate → QuadraturePrevision (Prevision-primary)",
          pb isa QuadraturePrevision, "got $(typeof(pb))")
    check("Beta grid values == canonical (Prevision-entry, ==)", pb.grid == CANON[:beta_grid_values])
    check("Beta grid log_weights == canonical (Prevision-entry, ==)",
          pb.log_weights == CANON[:beta_grid_logw])
end
let
    Random.seed!(42)
    pgs = condition(GaussianPrevision(0.0, 1.0), K_GAUSS, 1.5)
    check("condition(GaussianPrevision) non-conjugate → QuadraturePrevision (Prevision-primary)",
          pgs isa QuadraturePrevision, "got $(typeof(pgs))")
    check("Gaussian grid values == canonical (Prevision-entry, ==)",
          pgs.grid == CANON[:gaussian_grid_values])
    check("Gaussian grid log_weights == canonical (Prevision-entry, ==)",
          pgs.log_weights == CANON[:gaussian_grid_logw])
end

# ── (2) Facade equivalence: Measure-entry still yields the same carrier-bound Measure (== canonical) ──
let
    Random.seed!(42)
    mg = condition(wrap_in_measure(GammaPrevision(2.0, 3.0)), K_GAMMA, 2.5; n_particles = 50)
    check("Gamma Measure facade .space.values == canonical", mg.space.values == CANON[:gamma_generic_samples])
    check("Gamma facade wraps a ParticlePrevision", mg.prevision isa ParticlePrevision)
    check("Gamma facade shared-ref preserved (.prevision.samples === .space.values)",
          mg.prevision.samples === mg.space.values)
end
let
    Random.seed!(42)
    mb = condition(wrap_in_measure(BetaPrevision(2.0, 3.0)), K_BETA, 0.5)
    check("Beta Measure facade .space.values == canonical grid", mb.space.values == CANON[:beta_grid_values])
    check("Beta facade wraps a QuadraturePrevision", mb.prevision isa QuadraturePrevision)
end

# ── (3) Conjugate arm: facade threads m.space (Q3 — preserve the DSL-declared carrier, ===) ──
let
    k_conj = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]), _ -> error("not used"),
                    (h, o) -> o == 1.0 ? log(max(h, 1e-300)) : log(max(1 - h, 1e-300));
                    likelihood_family = BetaBernoulli())
    sp = Interval(0.0, 1.0)
    m = BetaMeasure(sp, 2.0, 3.0)
    post = condition(m, k_conj, 1.0)
    check("Beta conjugate facade → BetaMeasure", post isa BetaMeasure, "got $(typeof(post))")
    check("Beta conjugate exact: α = 3.0 after obs = 1", post.alpha == 3.0, "got $(post.alpha)")
    check("Beta conjugate exact: β = 3.0 unchanged", post.beta == 3.0, "got $(post.beta)")
    check("Q3: m.space threaded (===), not canonicalised", post.space === sp)
    # The Prevision-entry conjugate arm stays Prevision-primary (untouched by Phase 2):
    pc = condition(BetaPrevision(2.0, 3.0), k_conj, 1.0)
    check("condition(BetaPrevision) conjugate → BetaPrevision(3,3) (unchanged)",
          pc isa BetaPrevision && pc.alpha == 3.0 && pc.beta == 3.0)
end

# ── (4) _predictive_ll inversion: Prevision-entry == Measure-entry (Beta expect, Gaussian NormalNormal) ──
let
    pll_p = _predictive_ll(BetaPrevision(2.0, 3.0), K_BETA, 0.5)
    pll_m = _predictive_ll(wrap_in_measure(BetaPrevision(2.0, 3.0)), K_BETA, 0.5)
    check("Beta _predictive_ll: Prevision-entry == Measure-entry (==)", pll_p == pll_m, "p=$pll_p m=$pll_m")

    k_nn = Kernel(Euclidean(1), Euclidean(1), μ -> error("unused"), (μ, o) -> -0.5 * (o - μ)^2;
                  likelihood_family = NormalNormal(1.0))
    gll_p = _predictive_ll(GaussianPrevision(0.0, 1.0), k_nn, 1.5)
    gll_m = _predictive_ll(wrap_in_measure(GaussianPrevision(0.0, 1.0)), k_nn, 1.5)
    check("Gaussian _predictive_ll: Prevision-entry == Measure-entry (NormalNormal, ==)",
          gll_p == gll_m, "p=$gll_p m=$gll_m")
end

println("="^64)
println("ALL CHECKS PASSED — measure-as-view Phase 2")
println("="^64)
