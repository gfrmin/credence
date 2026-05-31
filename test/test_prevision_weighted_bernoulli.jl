# test_prevision_weighted_bernoulli.jl — Stratum-2 for the
# (BetaPrevision, WeightedBernoulli) conjugate pair (Paper 1, B2c).
#
# WeightedBernoulli is the fractional / soft-evidence Beta update:
# obs = (outcome, weight), α += weight·outcome, β += weight·(1-outcome).
# It is the resource-rational realisation of the exact category-uncertain
# reliability update (docs/paper1/move-2c-design.md). The strict
# unit-count BetaBernoulli is untouched.
#
# Discipline (per test_prevision_conjugate.jl): assert `_dispatch_path ==
# :conjugate` BEFORE the value, so a silent registry miss is caught. The
# degenerate weight=1 case must reduce EXACTLY (==) to the unit-count
# BetaBernoulli update — the substrate-change equivalence guard.
#
# Run from the repo root:
#     julia test/test_prevision_weighted_bernoulli.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: ConjugatePrevision, maybe_conjugate, update, _dispatch_path
using Credence: BetaPrevision
using Credence.Ontology: wrap_in_measure

function check(name, cond, detail="")
    if cond
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("Stratum-2 assertion failed: $name")
    end
end

# Conjugate path is registry-driven; the kernel's generate/log_density are
# not consulted (update() routes through the registry). likelihood_family
# is the load-bearing declaration.
wbern_kernel() = Kernel(Interval(0.0, 1.0), Finite([0, 1]),
    h -> wrap_in_measure(CategoricalPrevision([0.0, 0.0]), Finite([0, 1])),
    (h, o) -> 0.0;
    likelihood_family = WeightedBernoulli())

bbern_kernel() = Kernel(Interval(0.0, 1.0), Finite([0, 1]),
    h -> wrap_in_measure(CategoricalPrevision([0.0, 0.0]), Finite([0, 1])),
    (h, o) -> o == 1 ? log(max(h, 1e-300)) : log(max(1 - h, 1e-300));
    likelihood_family = BetaBernoulli())

println("="^60)
println("Stratum 2 — (BetaPrevision, WeightedBernoulli)")
println("="^60)

let k = wbern_kernel()
    p = BetaPrevision(2.0, 3.0)

    # Dispatch path FIRST.
    check("BetaPrevision + WeightedBernoulli → :conjugate",
          _dispatch_path(p, k) === :conjugate, "got $(_dispatch_path(p, k))")

    cp = maybe_conjugate(p, k)
    check("maybe_conjugate returns ConjugatePrevision{BetaPrevision, WeightedBernoulli}",
          cp isa ConjugatePrevision{BetaPrevision, WeightedBernoulli}, "got $(typeof(cp))")

    # Fractional correct: α += w, β unchanged.
    u = update(cp, (1, 0.7)).prior
    check("Beta(2,3) ×(correct, w=0.7) → Beta(2.7, 3)",
          isapprox(u.alpha, 2.7; rtol=1e-12) && u.beta == 3.0,
          "got α=$(u.alpha), β=$(u.beta)")

    # Fractional wrong: α unchanged, β += w.
    u0 = update(cp, (0, 0.3)).prior
    check("Beta(2,3) ×(wrong, w=0.3) → Beta(2, 3.3)",
          u0.alpha == 2.0 && isapprox(u0.beta, 3.3; rtol=1e-12),
          "got α=$(u0.alpha), β=$(u0.beta)")

    # Zero weight: no-op (a category with π_c = 0 contributes nothing).
    uz = update(cp, (1, 0.0)).prior
    check("Beta(2,3) ×(correct, w=0.0) → Beta(2, 3) (no-op)",
          uz.alpha == 2.0 && uz.beta == 3.0, "got α=$(uz.alpha), β=$(uz.beta)")

    # Bad outcome rejected.
    threw = false
    try; update(cp, (2, 0.5)); catch; threw = true; end
    check("WeightedBernoulli rejects outcome ∉ {0,1}", threw)
end

# ── Degenerate equivalence: weight=1 == unit-count BetaBernoulli (==) ──
let wp = BetaPrevision(2.0, 3.0)
    wcp = maybe_conjugate(wp, wbern_kernel())
    bcp = maybe_conjugate(wp, bbern_kernel())

    w1 = update(wcp, (1, 1.0)).prior
    b1 = update(bcp, 1).prior
    check("(correct, w=1) ≡ BetaBernoulli(correct): Beta(3,3) exactly",
          w1.alpha == b1.alpha && w1.beta == b1.beta && w1.alpha == 3.0 && w1.beta == 3.0,
          "weighted=($(w1.alpha),$(w1.beta)) unit=($(b1.alpha),$(b1.beta))")

    w0 = update(wcp, (0, 1.0)).prior
    b0 = update(bcp, 0).prior
    check("(wrong, w=1) ≡ BetaBernoulli(wrong): Beta(2,4) exactly",
          w0.alpha == b0.alpha && w0.beta == b0.beta && w0.alpha == 2.0 && w0.beta == 4.0,
          "weighted=($(w0.alpha),$(w0.beta)) unit=($(b0.alpha),$(b0.beta))")
end

println("="^60)
println("ALL CHECKS PASSED — (BetaPrevision, WeightedBernoulli)")
println("="^60)
