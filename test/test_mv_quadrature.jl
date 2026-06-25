# test_mv_quadrature.jl — the multivariate product-grid posterior (antipattern disposal, Phase B).
# The COUPLED utility latents (e.g. {u_wrong, κ_att} joined by a narrative MarginReaction) have a
# joint prior that is an INDEPENDENT product of truncated Gaussians; the likelihood (a margin
# reaction) couples them. The body declares `{type:truncated_mv_gaussian, mu, sigma, lo, hi}` (the
# box is the model SUPPORT, not a grid) + a `margin_reaction` kernel, and the engine integrates the
# joint box and τ by an internal product-grid quadrature — invisible to the declared model. This
# disposes the last host product grid (`_fold_joint`). Asserts:
#   (1) the independent prior's coordinate marginals match per-dim truncated moments;
#   (2) `marginal(p,i)` is bit-exact to a hand product-grid sum over the engine's OWN grid;
#   (3) a margin reaction FLAT in coordinate i leaves coord-i's marginal == its standalone 1-D
#       truncated fold (the product factorises — the coupling is the whole point of getting this
#       right) — bit-exact on the shared grid;
#   (4) a coupling margin moves BOTH coordinates off their priors in the modelled directions;
#   (5) conditioning order is irrelevant (likelihoods commute on the shared grid);
#   (6) the second condition keeps the same product grid (sequential, no re-gridding).
#
# Run from repo root:  julia test/test_mv_quadrature.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: TruncatedGaussianPrevision, MvQuadraturePrevision, truncated_mv_quadrature,
                condition, expect, mean, variance, Identity, marginal, Kernel, Euclidean, Finite,
                PushOnly, QuadraturePrevision, LogisticReaction, logistic_reaction_logdensity,
                MarginReaction, margin_reaction_logdensity, log_predictive, _mv_points

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end
approx(a, b; atol = 1e-9) = abs(a - b) <= atol

println("="^64)
println("MvQuadraturePrevision — coupled continuous latents, the engine owns the grid")
println("="^64)

# Two latents: x₁ ~ TruncatedNormal(-3, 4; [-10, 0])  (u_wrong-like, ≤ 0)
#              x₂ ~ TruncatedNormal( 0.05, 0.3; [-0.2, 1])  (κ_att-like)
mu  = [-3.0, 0.05]
sig = [4.0, 0.3]
lo  = [-10.0, -0.2]
hi  = [0.0, 1.0]
prior = truncated_mv_quadrature(mu, sig, lo, hi)
check("the truncated-mv prior is an MvQuadraturePrevision", prior isa MvQuadraturePrevision,
      string(typeof(prior)))
np = length(prior.axes[1])
check("product grid is n_per² (independent box quadrature)",
      length(prior.log_weights) == np^2 && length(prior.axes) == 2, "np=$np")

# ── (1) prior coordinate marginals ≈ per-dim truncated moments ──
for (i, name) in [(1, "x1"), (2, "x2")]
    sp = TruncatedGaussianPrevision(mu[i], sig[i], lo[i], hi[i])
    mi = marginal(prior, i)
    check("prior marginal[$name] mean ≈ the 1-D truncated mean",
          approx(mean(mi), mean(sp); atol = 2e-3), "$(mean(mi)) vs $(mean(sp))")
    check("prior marginal[$name] variance ≈ the 1-D truncated variance",
          approx(variance(mi), variance(sp); atol = 2e-3), "$(variance(mi)) vs $(variance(sp))")
end

# ── reaction kernels ──
# A FLAT-in-x₂ reaction (coeffs = [1, 0]) — i.e. a single-latent reaction on x₁ only, lifted to the
# joint. τ ~ TruncatedNormal(1, 0.5; [0.5, 2]).
flat = MarginReaction([1.0, 0.0], 0.0, -1.0, 0.0, 1.0, 0.5, 0.5, 2.0)
kflat = Kernel(Euclidean(2), Finite([0.0, 1.0]), x -> error("ng"),
               (x, o) -> margin_reaction_logdensity(flat, x, o); likelihood_family = flat)
# A COUPLING margin (a narrative "good-on-abstain"): margin = -x₂ + 0.36·x₁ - 0.36, sign -1.
couple = MarginReaction([0.36, -1.0], 0.36, -1.0, 0.0, 1.0, 0.5, 0.5, 2.0)
kcouple = Kernel(Euclidean(2), Finite([0.0, 1.0]), x -> error("ng"),
                 (x, o) -> margin_reaction_logdensity(couple, x, o); likelihood_family = couple)

# ── (6) sequential conditioning keeps the same product grid ──
post_flat = condition(prior, kflat, 1.0)
check("conditioning yields an MvQuadraturePrevision", post_flat isa MvQuadraturePrevision,
      string(typeof(post_flat)))
check("the conditioned grid is the SAME product grid (no re-gridding)",
      post_flat.axes == prior.axes, "axes changed under condition")

# ── (2) marginal(p,1) bit-exact to a hand product-grid sum over the engine's own grid ──
lw = post_flat.log_weights
w = exp.(lw .- maximum(lw)); w ./= sum(w)
hand1 = zeros(np)
for (k, x) in enumerate(_mv_points(post_flat.axes))
    # row-major: point k's axis-1 index = (k-1) ÷ np (np points per axis, last axis fastest)
    hand1[((k - 1) ÷ np) + 1] += w[k]
end
hand_mean1 = sum(hand1[i] * post_flat.axes[1][i] for i in 1:np)
m1 = marginal(post_flat, 1)
check("marginal(post,1) mean is bit-exact to the hand product-grid sum",
      approx(mean(m1), hand_mean1; atol = 1e-12), "$(mean(m1)) vs $hand_mean1")

# ── (3) a reaction FLAT in x₂ leaves x₂'s marginal == the prior x₂ marginal (factorisation);
#        and x₁'s joint marginal == the standalone 1-D truncated fold (bit-exact, shared grid) ──
m2_flat = marginal(post_flat, 2)
m2_prior = marginal(prior, 2)
check("a reaction flat in x₂ leaves x₂'s marginal at its prior (the joint factorises)",
      approx(mean(m2_flat), mean(m2_prior); atol = 1e-12) &&
      approx(variance(m2_flat), variance(m2_prior); atol = 1e-12),
      "$(mean(m2_flat)) vs $(mean(m2_prior))")
# the standalone 1-D fold: TruncatedGaussian(x₁) conditioned by the SCALAR logistic reaction
sp1 = TruncatedGaussianPrevision(mu[1], sig[1], lo[1], hi[1])
react1 = LogisticReaction(-1.0, 0.0, 1.0, 0.5, 0.5, 2.0)   # == flat margin restricted to x₁
k1 = Kernel(Euclidean(1), Finite([0.0, 1.0]), x -> error("ng"),
            (x, o) -> logistic_reaction_logdensity(react1, x, o); likelihood_family = react1)
post1d = condition(sp1, k1, 1.0)
check("x₁'s joint marginal == the standalone 1-D truncated fold (bit-exact, shared grid)",
      approx(mean(marginal(post_flat, 1)), mean(post1d); atol = 1e-12) &&
      approx(variance(marginal(post_flat, 1)), variance(post1d); atol = 1e-12),
      "$(mean(marginal(post_flat,1))) vs $(mean(post1d))")

# ── (4) a COUPLING margin moves BOTH coordinates off their priors (good-on-abstain: x₁↓, x₂↑) ──
post_c = condition(prior, kcouple, 1.0)
mc1, mc2 = mean(marginal(post_c, 1)), mean(marginal(post_c, 2))
check("coupling margin moves x₁ DOWN off its prior", mc1 < mean(marginal(prior, 1)),
      "$mc1 vs $(mean(marginal(prior,1)))")
check("coupling margin moves x₂ UP off its prior", mc2 > mean(marginal(prior, 2)),
      "$mc2 vs $(mean(marginal(prior,2)))")

# ── (5) conditioning order is irrelevant (likelihoods commute on the shared grid) ──
ab = condition(condition(prior, kflat, 1.0), kcouple, 1.0)
ba = condition(condition(prior, kcouple, 1.0), kflat, 1.0)
check("order independence: marginal means match for both coordinates",
      approx(mean(marginal(ab, 1)), mean(marginal(ba, 1)); atol = 1e-12) &&
      approx(mean(marginal(ab, 2)), mean(marginal(ba, 2)); atol = 1e-12),
      "$(mean(marginal(ab,1))) vs $(mean(marginal(ba,1)))")

# ── log_predictive on a joint posterior (the condition verb's log_marginal) is finite ──
lp = log_predictive(prior, kflat, 1.0)
check("log_predictive over the joint grid is finite", isfinite(lp), string(lp))

# ── (7) METAREASONED FIDELITY: the grid resolution falls out of the compute budget and degrades
#        SMOOTHLY with coupling depth — no cliff, no error; the total stays within budget (no OOM). ──
check("d=2 (the dominant consumer) is exact at 64/dim", length(prior.axes[1]) == 64,
      "d=2 should keep the 1-D resolution: $(length(prior.axes[1]))")
for dd in [3, 5, 8, 12]
    p = truncated_mv_quadrature(fill(0.0, dd), fill(1.0, dd), fill(-1.0, dd), fill(1.0, dd))
    total = length(p.log_weights)
    check("d=$dd builds within the compute budget (graceful degradation, no error)",
          p isa MvQuadraturePrevision && total <= 65536, "d=$dd total=$total")
end

println("="^64)
println("ALL PASSED")
println("="^64)
