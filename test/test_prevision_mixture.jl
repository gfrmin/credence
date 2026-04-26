# test_prevision_mixture.jl — Stratum-1 / Stratum-2 for Move 5's
# MixturePrevision routing relocation and ExchangeablePrevision decompose.
#
# Per docs/posture-3/move-5-design.md §3, tolerances:
#   - Conjugate-per-component (integer α/β accumulation): `==`
#   - Mixture flattening (logsumexp-normalised weights): `atol=1e-14`
#   - Dirichlet marginal (ExchangeablePrevision.decompose): closed-form `==`
#     on integer-ratio α, `rtol=1e-12` on derived log_weights
#
# The worked example from §4 is the centrepiece: a 3-component mixture
# of TaggedBetaPrevision with distinct tags, a FiringByTag kernel that
# fires on tags 1 and 3 (routing to BetaBernoulli) and does not fire on
# tag 2 (routing to Flat), a Bernoulli observation. Tests verify:
#   (1) mixture-level _dispatch_path rollup == :conjugate (all three
#       components resolve to BetaBernoulli or Flat — both registered).
#   (2) post-condition component α/β are bit-exactly the expected values.
#   (3) non-firing component is bit-exactly unchanged (Flat no-op).

push!(LOAD_PATH, "src")
using Credence
using Credence: _dispatch_path
using Credence.Ontology: _resolve_likelihood_family, _with_resolved_family, wrap_in_measure  # Posture 4 Move 4
using Credence.Previsions: DirichletPrevision
using Credence: BetaPrevision, GaussianPrevision  # Posture 4 Move 4

function check(name, cond, detail="")
    if cond
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("assertion failed: $name")
    end
end

println("="^60)
println("Stratum 1/2 — MixturePrevision + ExchangeablePrevision (Move 5)")
println("="^60)

# ── Worked example §4: 3-component mixture, FiringByTag({1,3}, BB, Flat), obs=1 ──

let
    # Three tagged components with distinct α,β and distinct tags.
    comp1 = TaggedBetaMeasure(Interval(0.0, 1.0), 1, BetaPrevision(2.0, 3.0))
    comp2 = TaggedBetaMeasure(Interval(0.0, 1.0), 2, BetaPrevision(5.0, 5.0))
    comp3 = TaggedBetaMeasure(Interval(0.0, 1.0), 3, BetaPrevision(1.0, 4.0))
    mix = MixtureMeasure(Interval(0.0, 1.0), [comp1, comp2, comp3], [0.0, 0.0, 0.0])

    k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
               θ -> error("generate not used"),
               (θ, o) -> o == 1.0 ? log(max(0.5, 1e-300)) : log(max(0.5, 1e-300));
               likelihood_family = FiringByTag(Set([1, 3]), BetaBernoulli(), Flat()))

    # Dispatch-path rollup: all three resolve to registered conjugate
    # pairs (BetaBernoulli for tags 1,3; Flat for tag 2).
    check("MixturePrevision + FiringByTag → :conjugate (rollup, all comps registered)",
          _dispatch_path(mix.prevision, k) === :conjugate,
          "got $(_dispatch_path(mix.prevision, k))")
    check("_dispatch_path forwards correctly from MixtureMeasure surface",
          _dispatch_path(mix, k) === :conjugate,
          "got $(_dispatch_path(mix, k))")

    # Per-component drilldown via Move 4's per-Prevision hook. Each
    # component's LikelihoodFamily is resolved, then queried at the
    # conjugacy-relevant prevision (inner BetaPrevision).
    fam1 = _resolve_likelihood_family(k.likelihood_family, comp1)
    fam2 = _resolve_likelihood_family(k.likelihood_family, comp2)
    fam3 = _resolve_likelihood_family(k.likelihood_family, comp3)
    check("tag 1 resolves to BetaBernoulli", fam1 isa BetaBernoulli, "got $(typeof(fam1))")
    check("tag 2 resolves to Flat (not in fires)", fam2 isa Flat, "got $(typeof(fam2))")
    check("tag 3 resolves to BetaBernoulli", fam3 isa BetaBernoulli, "got $(typeof(fam3))")

    # Per-component dispatch path. The inner BetaPrevision dispatches
    # through Move 4's registry.
    rk1 = _with_resolved_family(k, fam1)
    check("inner BetaPrevision (tag 1) + resolved BetaBernoulli → :conjugate",
          _dispatch_path(comp1.beta.prevision, rk1) === :conjugate, "")
    rk2 = _with_resolved_family(k, fam2)
    check("inner BetaPrevision (tag 2) + resolved Flat → :conjugate",
          _dispatch_path(comp2.beta.prevision, rk2) === :conjugate, "")

    # Post-condition: condition the mixture on obs=1. Components 1 and 3
    # route to BetaBernoulli → α+1; component 2 routes to Flat → unchanged.
    new_mix = condition(mix, k, 1.0)
    check("post-condition mixture is MixtureMeasure", new_mix isa MixtureMeasure, "")
    check("post-condition has 3 components",
          length(new_mix.components) == 3,
          "got $(length(new_mix.components))")

    # Tag 1 (α=2,β=3) + BetaBernoulli + obs=1 → α=3,β=3.
    c1 = new_mix.components[1]::TaggedBetaMeasure
    check("tag 1 α post = 3 (==)", c1.beta.alpha == 3.0, "got $(c1.beta.alpha)")
    check("tag 1 β post = 3 (==)", c1.beta.beta == 3.0, "got $(c1.beta.beta)")
    check("tag 1 tag preserved", c1.tag == 1, "got $(c1.tag)")

    # Tag 2 (α=5,β=5) + Flat (no-op) → α=5,β=5 unchanged.
    c2 = new_mix.components[2]::TaggedBetaMeasure
    check("tag 2 α unchanged = 5 (==)", c2.beta.alpha == 5.0, "got $(c2.beta.alpha)")
    check("tag 2 β unchanged = 5 (==)", c2.beta.beta == 5.0, "got $(c2.beta.beta)")
    check("tag 2 tag preserved", c2.tag == 2, "got $(c2.tag)")

    # Tag 3 (α=1,β=4) + BetaBernoulli + obs=1 → α=2,β=4.
    c3 = new_mix.components[3]::TaggedBetaMeasure
    check("tag 3 α post = 2 (==)", c3.beta.alpha == 2.0, "got $(c3.beta.alpha)")
    check("tag 3 β unchanged = 4 (==)", c3.beta.beta == 4.0, "got $(c3.beta.beta)")
    check("tag 3 tag preserved", c3.tag == 3, "got $(c3.tag)")
end

# ── Partial-coverage rollup: one component falls through to particle ──

let
    # Mixture with one TaggedBetaMeasure (routes conjugate) and one
    # GaussianMeasure (no registered pair for Bernoulli kernel).
    c1 = TaggedBetaMeasure(Interval(0.0, 1.0), 1, BetaPrevision(2.0, 3.0))
    c2 = wrap_in_measure(GaussianPrevision(0.0, 1.0))
    mix = MixtureMeasure(Interval(0.0, 1.0), Measure[c1, c2], [0.0, 0.0])

    k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
               θ -> error("generate not used"),
               (θ, o) -> 0.0;
               likelihood_family = BetaBernoulli())

    # Component 1 routes conjugate (TaggedBetaMeasure with BB); component 2
    # does not — no (GaussianPrevision, BetaBernoulli) pair. Rollup → :mixed.
    check("MixturePrevision with mixed-type components → :mixed",
          _dispatch_path(mix.prevision, k) === :mixed,
          "got $(_dispatch_path(mix.prevision, k))")
end

# ── Zero-mass guard (Posture 2 gate-3) ──

let
    # All log_weights at -Inf → MixturePrevision constructor errors.
    raised = try
        MixturePrevision(Prevision[BetaPrevision(1.0, 1.0)], [-Inf])
        false
    catch e
        occursin("zero total mass", sprint(showerror, e))
    end
    check("MixturePrevision with all log_weights = -Inf raises on construction", raised, "")
end

# ── ExchangeablePrevision.decompose — simplest Dirichlet case ──

let
    # Dirichlet([2, 3, 5]) over Finite([1,2,3]) → one component per category,
    # weighted by α_i/Σα = [0.2, 0.3, 0.5].
    p = ExchangeablePrevision(Finite([1, 2, 3]), DirichletPrevision([2.0, 3.0, 5.0]))
    mp = decompose(p)
    check("decompose returns MixturePrevision", mp isa MixturePrevision, "got $(typeof(mp))")
    check("decompose has 3 components (one per category)",
          length(mp.components) == 3,
          "got $(length(mp.components))")

    # Components are degenerate CategoricalPrevisions on each category.
    c1, c2, c3 = mp.components
    check("component 1 is CategoricalPrevision", c1 isa CategoricalPrevision, "got $(typeof(c1))")
    w1 = weights(c1)
    check("component 1 concentrated on cat 1 (== [1.0, 0.0, 0.0])",
          w1[1] == 1.0 && w1[2] == 0.0 && w1[3] == 0.0,
          "got $w1")

    # Mixture log_weights match Dirichlet marginal α_i/Σα = [0.2, 0.3, 0.5].
    # α=[2,3,5], Σα=10. Normalised log_weights = log([0.2, 0.3, 0.5]).
    expected = [log(0.2), log(0.3), log(0.5)]
    for i in 1:3
        check("log_weight[$i] == log(α[$i]/Σα) (rtol=1e-12)",
              isapprox(mp.log_weights[i], expected[i]; rtol=1e-12),
              "got $(mp.log_weights[i]) expected $(expected[i])")
    end
end

# ── ExchangeablePrevision.decompose — error paths (§5.1 R3 scoping) ──

let
    # Non-Dirichlet prior → loud error.
    raised = try
        p = ExchangeablePrevision(Finite([1, 2]), Credence.Previsions.BetaPrevision(1.0, 1.0))
        decompose(p)
        false
    catch e
        occursin("only Dirichlet priors supported", sprint(showerror, e))
    end
    check("decompose rejects non-Dirichlet prior with scoping-named error", raised, "")

    # Non-Finite component space → loud error.
    raised = try
        p = ExchangeablePrevision(Interval(0.0, 1.0), DirichletPrevision([1.0, 1.0]))
        decompose(p)
        false
    catch e
        occursin("only Finite component_space supported", sprint(showerror, e))
    end
    check("decompose rejects non-Finite component_space with scoping-named error", raised, "")

    # Mismatched cardinality → loud error.
    raised = try
        p = ExchangeablePrevision(Finite([1, 2, 3]), DirichletPrevision([1.0, 1.0]))  # 3 cats, 2 α
        decompose(p)
        false
    catch e
        occursin("3 categories but Dirichlet prior has 2", sprint(showerror, e))
    end
    check("decompose rejects mismatched Dirichlet/Finite cardinalities", raised, "")
end

# ── Component flattening on nested sub-mixture ──
#
# If a component's condition returns a MixtureMeasure (possible e.g.
# under nested exchangeable decomposition), the outer MixturePrevision
# splices sub-components into the outer mixture with log-weights
# multiplied correctly.

let
    # Construct an outer mixture with two components:
    # - comp1 is a plain BetaMeasure
    # - comp2 is itself a MixtureMeasure of two plain BetaMeasures
    # Condition with a BetaBernoulli kernel. The outer post-condition
    # should have 3 components (1 from comp1 + 2 from flattened comp2).
    #
    # Inner components are plain BetaMeasure (not TaggedBetaMeasure) so
    # the kernel's log_density is called with θ::Float64 samples —
    # keeps the test kernel simple (no m_or_θ isa TaggedBetaMeasure
    # branching). TaggedBetaMeasure flattening is exercised by the
    # master-plan gate-4 MixtureMeasure flattening path, asserted
    # indirectly via test_flat_mixture.jl.
    comp1 = wrap_in_measure(BetaPrevision(2.0, 2.0))
    inner_c1 = wrap_in_measure(BetaPrevision(1.0, 1.0))
    inner_c2 = wrap_in_measure(BetaPrevision(3.0, 2.0))
    comp2 = MixtureMeasure(Interval(0.0, 1.0), Measure[inner_c1, inner_c2], [0.0, 0.0])

    outer = MixtureMeasure(Interval(0.0, 1.0), Measure[comp1, comp2], [0.0, 0.0])

    k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
               θ -> error("generate not used"),
               (θ, o) -> o == 1.0 ? log(max(θ, 1e-300)) : log(max(1.0 - θ, 1e-300));
               likelihood_family = BetaBernoulli())

    new_mix = condition(outer, k, 1.0)
    check("nested-mixture flattening: post-condition has 3 components (1 + 2 flattened)",
          length(new_mix.components) == 3,
          "got $(length(new_mix.components))")
    check("flattened component 1 is BetaMeasure (from comp1)",
          new_mix.components[1] isa BetaMeasure, "got $(typeof(new_mix.components[1]))")
    check("flattened component 2 is BetaMeasure (from comp2 inner_c1)",
          new_mix.components[2] isa BetaMeasure, "got $(typeof(new_mix.components[2]))")
    check("flattened component 3 is BetaMeasure (from comp2 inner_c2)",
          new_mix.components[3] isa BetaMeasure, "got $(typeof(new_mix.components[3]))")
end

println()
println("="^60)
println("ALL MIXTURE + EXCHANGEABLE TESTS PASSED (Move 5)")
println("="^60)
