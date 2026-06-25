# test_quadrature_sequential_condition.jl — a continuous latent that absorbs MULTIPLE non-conjugate
# observations (antipattern disposal, Phase A). The utility `u_wrong` latent is a truncated Gaussian
# that is FIRST conditioned by a Gaussian elicitation (`gaussian_known_var`, non-conjugate on the
# truncated support → a QuadraturePrevision) and THEN by a continuous-τ logistic reaction. The
# second condition therefore lands on a QuadraturePrevision, not the truncated prior. Narrative never
# triggered this (BetaBernoulli is conjugate — each update stays a BetaPrevision); the utility fold is
# the first consumer to condition a continuous latent twice non-conjugately. Asserts:
#   (1) the second condition multiplies the new likelihood into the EXISTING grid (no re-gridding) —
#       the engine result is bit-identical to a hand product over the same 64-pt grid;
#   (2) the two-step posterior moments ≈ a dense independent hand-quadrature of the joint model
#       (prior × elicitation-likelihood × continuous-τ reaction) — quadrature convergence;
#   (3) conditioning ORDER is irrelevant (likelihoods commute) — elicit-then-react == react-then-elicit.
#
# Run from repo root:  julia test/test_quadrature_sequential_condition.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: TruncatedGaussianPrevision, condition, expect, mean, variance, Identity, Kernel,
                Euclidean, Finite, PushOnly, QuadraturePrevision, LogisticReaction,
                logistic_reaction_logdensity, log_predictive

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end
approx(a, b; atol = 1e-9) = abs(a - b) <= atol

println("="^64)
println("QuadraturePrevision sequential conditioning — a continuous latent, two observations")
println("="^64)

# u_wrong ~ TruncatedNormal(-3, 4; [-10, 0]); the support encodes u(wrong) ≤ 0.
prior_mu, prior_sigma, lo, hi = -3.0, 4.0, -10.0, 0.0
p = TruncatedGaussianPrevision(prior_mu, prior_sigma, lo, hi)

# Elicitation: gaussian_known_var(variance=4), stated value -8.
noise_var, stated = 4.0, -8.0
elicit = Kernel(Euclidean(1), Euclidean(1), μ -> error("ng"), (μ, o) -> -0.5 * (o - μ)^2 / noise_var;
                likelihood_family = PushOnly())
# Reaction: continuous-τ logistic, sign -1, threshold 0, τ ~ TruncatedNormal(1, 0.5; [0.5, 2]).
fam = LogisticReaction(-1.0, 0.0, 1.0, 0.5, 0.5, 2.0)
react = Kernel(Euclidean(1), Finite([0.0, 1.0]), x -> error("ng"),
               (x, o) -> logistic_reaction_logdensity(fam, x, o); likelihood_family = fam)

p1 = condition(p, elicit, stated)           # truncated → QuadraturePrevision
check("first (elicitation) condition yields a QuadraturePrevision", p1 isa QuadraturePrevision,
      string(typeof(p1)))
p2 = condition(p1, react, 1.0)              # QuadraturePrevision → QuadraturePrevision (the new path)
check("second (reaction) condition on a QuadraturePrevision yields a QuadraturePrevision",
      p2 isa QuadraturePrevision, string(typeof(p2)))
check("the second condition keeps the same grid (no re-gridding)", p2.grid == p1.grid,
      "grid changed under a sequential condition")

# the SECOND condition's log_marginal: log ∫ p1(x)·P(react|x) dx — the skin needs this on a
# QuadraturePrevision (the generic fallback routes through wrap_in_measure, which it has no Space
# for). Bit-exact to a hand quadrature over p1's grid.
lp = log_predictive(p1, react, 1.0)
w1 = exp.(p1.log_weights .- maximum(p1.log_weights)); w1 ./= sum(w1)
hand_lp = log(sum(w1[i] * exp(logistic_reaction_logdensity(fam, p1.grid[i], 1.0))
                  for i in eachindex(p1.grid)))
check("second-condition log_predictive is bit-exact to a hand quadrature over the grid",
      approx(lp, hand_lp; atol = 1e-12), "$lp vs $hand_lp")

m2, v2 = mean(p2), variance(p2)

# ── (1) bit-exact to a hand product over the engine's OWN 64-pt midpoint grid ──
grid = p1.grid
logw = [(-0.5 * ((x - prior_mu) / prior_sigma)^2) +              # truncated-normal prior kernel
        (-0.5 * (stated - x)^2 / noise_var) +                    # gaussian elicitation
        logistic_reaction_logdensity(fam, x, 1.0)                # continuous-τ reaction
        for x in grid]
wm = maximum(logw); w = exp.(logw .- wm); w ./= sum(w)
hand_mean = sum(w[i] * grid[i] for i in eachindex(grid))
hand_var = sum(w[i] * (grid[i] - hand_mean)^2 for i in eachindex(grid))
check("two-step mean is bit-exact to a hand product over the SAME grid",
      approx(m2, hand_mean; atol = 1e-12), "$m2 vs $hand_mean")
check("two-step variance is bit-exact to a hand product over the SAME grid",
      approx(v2, hand_var; atol = 1e-12), "$v2 vs $hand_var")

# ── (2) ≈ a dense independent hand-quadrature of the joint model (convergence) ──
xs = range(lo, hi, length = 40001)
dlw = [(-0.5 * ((x - prior_mu) / prior_sigma)^2) + (-0.5 * (stated - x)^2 / noise_var) +
       logistic_reaction_logdensity(fam, x, 1.0) for x in xs]
dm = maximum(dlw); dw = exp.(dlw .- dm); dw ./= sum(dw)
dense_mean = sum(dw[i] * xs[i] for i in eachindex(xs))
dense_var = sum(dw[i] * (xs[i] - dense_mean)^2 for i in eachindex(xs))
check("two-step mean ≈ dense independent quadrature (64-pt convergence)",
      approx(m2, dense_mean; atol = 1e-2), "$m2 vs $dense_mean")
check("two-step variance ≈ dense independent quadrature", approx(v2, dense_var; atol = 1e-2),
      "$v2 vs $dense_var")

# the evidence is informative: elicitation -8 + a reaction that fires for low x pulls the mean below
# the prior mean and the elicitation respects the bound.
check("evidence moves the mean below the prior mean", m2 < mean(p), "$m2 vs $(mean(p))")
check("posterior stays within the support", lo <= m2 <= hi, string(m2))

# ── (3) conditioning order is irrelevant — likelihoods commute on the shared grid ──
q1 = condition(p, react, 1.0)               # react first (truncated → quadrature)
q2 = condition(q1, elicit, stated)          # then elicit (quadrature → quadrature)
check("order independence: react-then-elicit mean == elicit-then-react mean",
      approx(mean(q2), m2; atol = 1e-12), "$(mean(q2)) vs $m2")
check("order independence: react-then-elicit variance == elicit-then-react variance",
      approx(variance(q2), v2; atol = 1e-12), "$(variance(q2)) vs $v2")

println("="^64)
println("ALL PASSED")
println("="^64)
