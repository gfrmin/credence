# test_prevision_soft_bernoulli.jl — Stratum-2 for the
# (BetaPrevision, SoftBernoulli) conjugate pair.
#
# SoftBernoulli is VIRTUAL EVIDENCE on a latent Bernoulli: obs = (r, w) are the
# likelihoods of an indirect signal under the event being true / false
# (Pearl's λ-message), NOT an outcome. The exact posterior under
# L(θ) = r·θ + w·(1−θ) is a 2-component Beta mixture
#     π·Beta(α+1,β) + (1−π)·Beta(α,β+1),   π = r·θ̄ / (r·θ̄ + w·(1−θ̄))
# and the conjugate update here is its MEAN-EXACT ADF collapse Beta(α+π, β+1−π).
#
# Discipline (per test_prevision_weighted_bernoulli.jl): assert
# `_dispatch_path == :conjugate` BEFORE the value, so a silent registry miss is
# caught. Three load-bearing properties:
#   1. MEAN-EXACT — the collapsed posterior mean == the exact 2-component
#      mixture mean (computed independently), so any decision reading E[θ] is
#      exact. (==/rtol 1e-12, not "close enough".)
#   2. UNINFORMATIVE INVARIANCE — a signal equally likely under both (r == w)
#      leaves E[θ] EXACTLY unchanged (it only adds concentration). This is the
#      substrate root of confound-partialling: a non-diagnostic signal moves
#      nothing.
#   3. HARD REDUCTION — (r,w)=(1,0) ≡ BetaBernoulli(1) and (0,1) ≡
#      BetaBernoulli(0), EXACTLY (==) — the substrate-change equivalence guard.
#
# Run from the repo root:
#     julia test/test_prevision_soft_bernoulli.jl

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

bmean(p) = p.alpha / (p.alpha + p.beta)

# Conjugate path is registry-driven; the kernel's generate/log_density are not
# consulted by update() (the predictive log_density is exercised in the brain
# tests). likelihood_family is the load-bearing declaration.
soft_kernel() = Kernel(Interval(0.0, 1.0), Finite([0, 1]),
    h -> wrap_in_measure(CategoricalPrevision([0.0, 0.0]), Finite([0, 1])),
    (h, o) -> 0.0;
    likelihood_family = SoftBernoulli())

bbern_kernel() = Kernel(Interval(0.0, 1.0), Finite([0, 1]),
    h -> wrap_in_measure(CategoricalPrevision([0.0, 0.0]), Finite([0, 1])),
    (h, o) -> o == 1 ? log(max(h, 1e-300)) : log(max(1 - h, 1e-300));
    likelihood_family = BetaBernoulli())

println("="^60)
println("Stratum 2 — (BetaPrevision, SoftBernoulli)")
println("="^60)

let k = soft_kernel()
    p = BetaPrevision(2.0, 3.0)

    # Dispatch path FIRST.
    check("BetaPrevision + SoftBernoulli → :conjugate",
          _dispatch_path(p, k) === :conjugate, "got $(_dispatch_path(p, k))")

    cp = maybe_conjugate(p, k)
    check("maybe_conjugate returns ConjugatePrevision{BetaPrevision, SoftBernoulli}",
          cp isa ConjugatePrevision{BetaPrevision, SoftBernoulli}, "got $(typeof(cp))")

    # ── 1. MEAN-EXACT vs the exact 2-component mixture (independent compute) ──
    α, β = 2.0, 3.0
    θ̄ = α / (α + β)
    r, w = 3.0, 1.0                      # signal 3× more likely if the event is true
    π = r * θ̄ / (r * θ̄ + w * (1.0 - θ̄))
    # Exact posterior is π·Beta(α+1,β) + (1−π)·Beta(α,β+1); its mean:
    exact_mean = π * (α + 1.0) / (α + β + 1.0) + (1.0 - π) * α / (α + β + 1.0)
    u = update(cp, (r, w)).prior
    check("collapsed α = α + π exactly",
          u.alpha == α + π && u.beta == β + (1.0 - π),
          "got α=$(u.alpha), β=$(u.beta); expected ($(α+π), $(β+1-π))")
    check("collapsed mean == exact 2-component mixture mean (rtol 1e-12)",
          isapprox(bmean(u), exact_mean; rtol = 1e-12),
          "collapsed=$(bmean(u)) exact=$(exact_mean)")

    # ── 2. UNINFORMATIVE INVARIANCE: r == w leaves E[θ] exactly unchanged ──
    ui = update(cp, (0.42, 0.42)).prior
    check("uninformative signal (r==w) leaves E[θ] EXACTLY unchanged",
          isapprox(bmean(ui), θ̄; rtol = 1e-12),
          "before=$(θ̄) after=$(bmean(ui))")
    check("uninformative signal still adds concentration mass (+1)",
          isapprox(ui.alpha + ui.beta, α + β + 1.0; rtol = 1e-12),
          "mass=$(ui.alpha + ui.beta)")

    # Zero marginal likelihood rejected (r==w==0 would divide by zero).
    threw = false
    try; update(cp, (0.0, 0.0)); catch; threw = true; end
    check("SoftBernoulli rejects zero-marginal evidence (r=w=0)", threw)
end

# ── 3. HARD REDUCTION: (1,0) ≡ BetaBernoulli(1), (0,1) ≡ BetaBernoulli(0) ──
let wp = BetaPrevision(2.0, 3.0)
    scp = maybe_conjugate(wp, soft_kernel())
    bcp = maybe_conjugate(wp, bbern_kernel())

    s1 = update(scp, (1.0, 0.0)).prior     # signal certain under true, impossible under false ⇒ π=1
    b1 = update(bcp, 1).prior
    check("(r,w)=(1,0) ≡ BetaBernoulli(1): Beta(3,3) exactly",
          s1.alpha == b1.alpha && s1.beta == b1.beta && s1.alpha == 3.0 && s1.beta == 3.0,
          "soft=($(s1.alpha),$(s1.beta)) unit=($(b1.alpha),$(b1.beta))")

    s0 = update(scp, (0.0, 1.0)).prior     # π=0
    b0 = update(bcp, 0).prior
    check("(r,w)=(0,1) ≡ BetaBernoulli(0): Beta(2,4) exactly",
          s0.alpha == b0.alpha && s0.beta == b0.beta && s0.alpha == 2.0 && s0.beta == 4.0,
          "soft=($(s0.alpha),$(s0.beta)) unit=($(b0.alpha),$(b0.beta))")
end

println("="^60)
println("ALL CHECKS PASSED — (BetaPrevision, SoftBernoulli)")
println("="^60)
