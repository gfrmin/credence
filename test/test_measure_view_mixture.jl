# test_measure_view_mixture.jl — Phase 3 of the measure-as-view arc.
#
# The keystone: `CategoricalPrevision` holds only `log_weights` — it is a distribution over an INDEX set,
# the index→value map living in `CategoricalMeasure.space::Finite`. So `draw(p::CategoricalPrevision)`
# returns the INDEX (carrier-free); `draw(m::CategoricalMeasure) = m.space.values[draw(m.prevision)]` does
# the lookup (index = structure, value = data). With that keystone the MixtureMeasure ↔ MixturePrevision
# twins (condition/prune/truncate/draw) collapse: a MixtureMeasure carries a single shared `space`, so the
# facade is `MixtureMeasure(m.space, op(m.prevision))`.
#
# Capture-before-refactor REUSES test/fixtures/mixture_draw_canonical_v1.jls (SHA 5b0ec51, Julia 1.12.x,
# `==` tol): the seeded draw sequences captured PRE-refactor from draw(::CategoricalMeasure)/(::MixtureMeasure).
# The refactor routes those through the shared `_sample_index` helper + draw(::CategoricalPrevision); the
# sequence is bit-identical under fixed seed (same weights, one rand(), same cumulative loop). Per
# precedents.md §4 the `==` class is NOT relaxable to rtol.
#
# Run from repo root:
#     julia test/test_measure_view_mixture.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: CategoricalPrevision, MixturePrevision, GammaPrevision, QuadraturePrevision,
                condition, draw, prune, truncate, weights, expect
using Credence.Ontology: _predictive_ll
using Random
using Serialization

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

const CANON = open(deserialize, joinpath(@__DIR__, "fixtures", "mixture_draw_canonical_v1.jls"))

# Fixtures mirror the capture script EXACTLY so the canonical sequences apply.
catM   = CategoricalMeasure(Finite([10.0, 20.0, 30.0]), [log(0.2), log(0.3), log(0.5)])
mixM   = MixtureMeasure(Interval(0.0,1.0), Measure[BetaMeasure(2.0,3.0), BetaMeasure(5.0,5.0)], [log(0.4), log(0.6)])
catA   = CategoricalMeasure(Finite([1.0,2.0,3.0]), [log(0.5), log(0.3), log(0.2)])
catB   = CategoricalMeasure(Finite([1.0,2.0,3.0]), [log(0.1), log(0.1), log(0.8)])
mixCat = MixtureMeasure(Finite([1.0,2.0,3.0]), Measure[catA, catB], [log(0.7), log(0.3)])

println("="^64)
println("measure-as-view Phase 3 — keystone (draw index) + mixture-twin collapse")
println("Fixture SHA: $(CANON[:source_sha]); Julia $(CANON[:julia_version])")
println("="^64)

# ── (1) The keystone: draw(::CategoricalPrevision) is the carrier-free INDEX (NEW behaviour, TDD) ──
let
    p = catM.prevision
    Random.seed!(42); i = draw(p)
    check("draw(::CategoricalPrevision) returns an Int index", i isa Int, "got $(typeof(i))")
    check("draw(::CategoricalPrevision) index in 1:n", 1 <= i <= length(weights(p)), "got $i")
    # index = structure, value = data: same seed ⇒ same index ⇒ same value through the carrier
    Random.seed!(42); v_measure = draw(catM)
    Random.seed!(42); v_rebind  = catM.space.values[draw(catM.prevision)]
    check("draw(::CategoricalMeasure) == space.values[draw(::CategoricalPrevision)] (keystone, ==)",
          v_measure == v_rebind, "measure=$v_measure rebind=$v_rebind")
end

# ── (2) Capture-before-refactor: seeded draw sequences == canonical (the _sample_index relocation) ──
let
    Random.seed!(42); cat_seq = [draw(catM) for _ in 1:30]
    check("draw(::CategoricalMeasure) seq == canonical (==)", cat_seq == CANON[:cat_draws])

    Random.seed!(42); mix_seq = [draw(mixM) for _ in 1:20]
    check("draw(::MixtureMeasure) continuous-component seq == canonical (==)", mix_seq == CANON[:mix_draws])

    Random.seed!(42); mixcat_seq = [draw(mixCat) for _ in 1:30]
    check("draw(::MixtureMeasure) categorical-component seq == canonical (==)",
          mixcat_seq == CANON[:mixcat_draws])
end

# ── (3) condition STAYS Measure-resident (the seam boundary); prune/truncate collapse to facades ──
const K_BB = Kernel(Interval(0.0,1.0), Finite([0.0,1.0]), _ -> error("unused"),
                    (h,o) -> o == 1.0 ? log(max(h,1e-300)) : log(max(1-h,1e-300));
                    likelihood_family = BetaBernoulli())
let
    # condition(MixtureMeasure) is NOT collapsed (carrier-bound — the loop hands components to the kernel;
    # see ontology.jl + §5 Q1). For a NON-introspecting conjugate kernel the Measure-entry and Prevision-
    # entry still agree on weights — the boundary the seam draws (they diverge for component-introspecting
    # / carrier-bound kernels: test_flat_mixture TEST 6, test_host TEST 5).
    sp = Interval(0.0,1.0)
    m  = MixtureMeasure(sp, Measure[BetaMeasure(1.0,1.0), BetaMeasure(2.0,3.0)], [log(0.5), log(0.5)])
    post_m = condition(m, K_BB, 1.0)
    check("condition(MixtureMeasure) stays Measure-resident (a MixtureMeasure)", post_m isa MixtureMeasure)
    check("condition(MixtureMeasure) threads m.space (===)", post_m.space === sp)
    check("condition(MixtureMeasure) weights agree with Prevision-entry for a conjugate kernel (==)",
          weights(post_m) == weights(MixtureMeasure(m.space, condition(m.prevision, K_BB, 1.0))))
end
let
    # prune/truncate carry NO kernel — index/weight-only ⇒ genuinely redundant twins ⇒ facades.
    sp = Interval(0.0,1.0)
    comps = Measure[BetaMeasure(Float64(i), 1.0) for i in 1:5]
    m = MixtureMeasure(sp, comps, Float64[0.0, -0.5, -25.0, 0.2, -30.0])    # idx 3,5 below the -20 floor
    pruned = prune(m)
    check("prune(MixtureMeasure) drops the 2 sub-floor components", length(pruned.components) == 3,
          "got $(length(pruned.components))")
    check("prune(MixtureMeasure) threads m.space (===)", pruned.space === sp)
    check("prune facade == Prevision-entry re-bound (==)",
          weights(pruned) == weights(MixtureMeasure(m.space, prune(m.prevision))))
    trunc = truncate(m; max_components=2)
    check("truncate(MixtureMeasure) keeps the top-2 by weight", length(trunc.components) == 2,
          "got $(length(trunc.components))")
    check("truncate(MixtureMeasure) threads m.space (===)", trunc.space === sp)
    check("truncate facade == Prevision-entry re-bound (==)",
          weights(trunc) == weights(MixtureMeasure(m.space, truncate(m.prevision; max_components=2))))
end

# ── (4) The binding sites inverted: Gamma predictive EXACT (Q2); carrier-bound predictive errors (Q1) ──
let
    kg = Kernel(PositiveReals(), Euclidean(1), _ -> error("unused"), (λ, o) -> -0.5 * (o - λ)^2;
                likelihood_family = PushOnly())
    p = GammaPrevision(2.0, 3.0)
    a = _predictive_ll(p, kg, 2.5)
    b = _predictive_ll(p, kg, 2.5)
    check("Gamma _predictive_ll is exact/deterministic (Q2: sampling → exact)", a == b, "a=$a b=$b")
    check("Gamma _predictive_ll == the expect integral (Prevision-primary, exact)",
          a == log(max(expect(p, h -> exp(kg.log_density(h, 2.5))), 1e-300)))
    # Carrier-bound: a bare CategoricalPrevision predictive is a Measure op (no carrier-free expect over a
    # closure) — the Prevision-level call MethodErrors, the boundary asserting itself (Q1 corollary).
    cp = CategoricalPrevision([log(0.4), log(0.6)])
    threw = false
    try
        _predictive_ll(cp, kg, 1.0)
    catch
        threw = true
    end
    check("CategoricalPrevision predictive errors at the Prevision level (Q1: Measure-resident)", threw)
end

println("="^64)
println("ALL CHECKS PASSED — measure-as-view Phase 3")
println("="^64)
