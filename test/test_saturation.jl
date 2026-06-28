# test_saturation.jl — the belief-aware saturation signal (exploration-budget Move 2).
# Covers: the ZeroMeanGammaPrevision conjugate primitive (the scale-free :plateaued regime), the
# compression_exhausted prior-side half (≡ perturb_grammar no-op), and the 2-regime BMA plateau signal
# — including the SCALE-FREE slow-improver test (the one a fixed-σ model fails) and the no-hard-gate
# contract (Q3). The reset-clears-history test lives in test_program_space.jl (it needs AgentState).
#
# Run: julia test/test_saturation.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: ZeroMeanGammaPrevision, ZeroMeanGammaLikelihood, condition, log_predictive, params,
                weights, Kernel, Euclidean, PositiveReals, wrap_in_measure, GaussianPrevision,
                Grammar, ProductionRule, SubprogramFrequencyTable, Program, GTExpr, LTExpr, AndExpr,
                IfExpr, ActionExpr, ProgramExpr, analyse_posterior_subtrees, perturb_grammar,
                compression_exhausted, initial_learning_regime, update_learning_regime, plateau_probability

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("saturation — ZeroMeanGamma primitive + compression_exhausted + regime BMA")
println("="^64)

# ── (1) ZeroMeanGammaPrevision conjugate primitive (the scale-free :plateaued component) ──
_zmg_kernel() = Kernel(PositiveReals(), Euclidean(1),
                       h -> wrap_in_measure(GaussianPrevision(0.0, sqrt(h))),
                       (h, o) -> -0.5 * o^2 / h;
                       likelihood_family = ZeroMeanGammaLikelihood())
let
    p = ZeroMeanGammaPrevision(1.0, 1.0)
    k = _zmg_kernel()
    post = condition(p, k, 2.0)
    check("ZeroMeanGamma conjugate update: α+=0.5, β+=r²/2", post.α == 1.5 && post.β == 3.0,
          "got $(params(post))")
    p2 = condition(condition(p, k, 1.0), k, 1.0)
    check("ZeroMeanGamma accumulates: α=2.0, β=1+0.5+0.5=2.0", p2.α == 2.0 && p2.β == 2.0,
          "got $(params(p2))")
    # Zero-mean Student-t PROPERTIES (oracle-free): symmetric about 0 (the defining zero-mean property,
    # which a μ≠0 NormalGamma would break), and peaked at 0 (decreasing in |r|).
    check("log_predictive symmetric about 0 (zero-mean)", log_predictive(p, k, 0.7) ≈ log_predictive(p, k, -0.7))
    check("log_predictive peaked at 0 (decreasing in |r|)", log_predictive(p, k, 0.5) > log_predictive(p, k, 3.0))
end

# ── (2) compression_exhausted ≡ perturb_grammar no-op (the prior-side half, Move-1 reuse) ──
let
    g = Grammar(Set([:red, :green]), ProductionRule[], 1)
    s = AndExpr(GTExpr(:red, 0.7), LTExpr(:green, 0.3))
    progs = Program[Program(IfExpr(s, ActionExpr(:a), ActionExpr(:b)), 6, 1) for _ in 1:3]
    ft = analyse_posterior_subtrees(progs, fill(1 / 3, 3); min_frequency = 0.0, min_complexity = 2)
    check("compressing table ⇒ NOT exhausted", compression_exhausted(g, ft) == false)
    check("not-exhausted ⇒ perturb adds a rule", length(perturb_grammar(g, ft).rules) == 1)

    empty_ft = SubprogramFrequencyTable(ProgramExpr[], Float64[], Vector{Int}[])
    check("empty table ⇒ exhausted", compression_exhausted(g, empty_ft) == true)
    check("exhausted ⇒ perturb is a no-op (same grammar object)", perturb_grammar(g, empty_ft) === g)
    check("compute_cost above payoff·log2 ⇒ exhausted", compression_exhausted(g, ft; compute_cost = 100.0) == true)
end

# ── (3)+(4) the regime BMA — directional, and SCALE-FREE ──
function run_regime(losses)
    regime = initial_learning_regime()
    prev = nothing
    for ℓ in losses
        regime = update_learning_regime(regime, prev, ℓ)
        prev = ℓ
    end
    regime
end
let
    # Cold start: < 2 residuals ⇒ uniform prior (not saturated).
    check("cold start: plateau_probability at uniform prior", plateau_probability(run_regime([2.0])) ≈ 0.5)

    # Phase A — CONSISTENT improvement (steady positive decrements) ⇒ improving ⇒ low plateau prob.
    # (A *decelerating* series — big drops then tiny — correctly reads as transitional/plateauing; the
    # clean "improving" signal is consistency, not mere decrease.)
    phaseA = Float64[3.0]
    for i in 1:8
        push!(phaseA, phaseA[end] - (iseven(i) ? 0.31 : 0.29))   # Δ ≈ 0.3 steady
    end
    pA = plateau_probability(run_regime(phaseA))
    check("Phase A (consistent improvement) ⇒ low plateau probability", pA < 0.3, "got $pA")

    # Phase B — bouncing flat ⇒ plateaued ⇒ high plateau prob.
    phaseB = [0.755, 0.758, 0.752, 0.757, 0.754, 0.756, 0.753, 0.755, 0.754, 0.756]
    pB = plateau_probability(run_regime(phaseB))
    check("Phase B (flat) ⇒ high plateau probability", pB > 0.7, "got $pB")
    check("ordering: plateaued ≫ improving (B ≫ A)", pB > pA, "pA=$pA pB=$pB")

    # SCALE-FREE: a slow-but-CONSISTENT improver (Δ ≈ 0.08, far below Phase A's 0.3–1.3 drops, small
    # bounce) stays improving — its drift is detectable above its OWN inferred noise. A fixed-σ model
    # calibrated to the big drops would falsely call this plateaued; assert it does NOT.
    slow = Float64[3.0]
    for i in 1:11
        push!(slow, slow[end] - (iseven(i) ? 0.09 : 0.07))   # decrements alternate 0.07/0.09 (mean 0.08)
    end
    pslow = plateau_probability(run_regime(slow))
    check("slow-but-consistent improver ⇒ NOT plateaued (scale-free)", pslow < 0.3, "got $pslow")
end

# ── (5) determinism + the no-hard-gate contract (Q3) ──
let
    series = [1.0, 0.8, 0.78, 0.781, 0.779, 0.78]
    check("regime conditioning is deterministic",
          plateau_probability(run_regime(series)) == plateau_probability(run_regime(series)))
    # Q3: Move 2 exposes the COMPONENTS (compression_exhausted, plateau_probability), no hard veto.
    check("no hard-gate `saturated` veto is exported", !isdefined(Credence, :saturated))
end

println("="^64)
println("ALL CHECKS PASSED — saturation")
println("="^64)
