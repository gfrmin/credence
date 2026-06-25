# test_truncated_gaussian.jl — a CONTINUOUS bounded prior (antipattern disposal, Phase A). The
# utility latents are Gaussians on a stated support [lo,hi] (a sign/range constraint, e.g.
# u_wrong ≤ 0); the body declares `{type:truncated_gaussian, mu, sigma, lo, hi}` — the bounds are
# model SUPPORT, not a discretisation grid — and the engine integrates over [lo,hi] by an internal
# midpoint quadrature (invisible to the model). Asserts:
#   (1) the truncated prior mean/variance match a dense hand-quadrature (the engine's 64-pt grid);
#   (2) conditioning is non-conjugate → a QuadraturePrevision whose support stays within [lo,hi];
#   (3) a Gaussian elicitation pulls the posterior toward the observation but respects the bound;
#   (4) the bound BINDS — truncating N(-4,3) to [-10,0] (≈9% mass cut above 0) moves the mean below
#       the untruncated -4.
#
# Run from repo root:  julia test/test_truncated_gaussian.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: TruncatedGaussianPrevision, GaussianPrevision, condition, expect, Identity, mean,
                variance, Kernel, Euclidean, PushOnly, QuadraturePrevision, weights

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end
approx(a, b; atol = 1e-9) = abs(a - b) <= atol

println("="^64)
println("TruncatedGaussianPrevision — a continuous bounded prior (Phase A)")
println("="^64)

p = TruncatedGaussianPrevision(-4.0, 3.0, -10.0, 0.0)

# ── (1) prior moments ≈ a dense hand-quadrature over [lo,hi] ──
xs = range(-10.0, 0.0, length = 200001)
lw = [-0.5 * ((x + 4.0) / 3.0)^2 for x in xs]
w = exp.(lw); w ./= sum(w)
ref_mean = sum(w[i] * xs[i] for i in eachindex(xs))
ref_var = sum(w[i] * (xs[i] - ref_mean)^2 for i in eachindex(xs))
check("truncated prior mean ≈ dense hand-quadrature", approx(mean(p), ref_mean; atol = 1e-3),
      "$(mean(p)) vs $ref_mean")
check("truncated prior variance ≈ dense hand-quadrature", approx(variance(p), ref_var; atol = 1e-2),
      "$(variance(p)) vs $ref_var")

# ── (4) the bound BINDS: the truncated mean is below the untruncated -4 (upper tail cut) ──
check("truncation binds — mean < the untruncated -4", mean(p) < -4.0, string(mean(p)))

# ── (2)+(3) Gaussian elicitation (gaussian_known_var): non-conjugate → quadrature, bounded ──
elicit = Kernel(Euclidean(1), Euclidean(1), μ -> error("ng"), (μ, o) -> -0.5 * (o - μ)^2 / 4.0;
                params = Dict{Symbol, Any}(:sigma_obs => 2.0), likelihood_family = PushOnly())
post = condition(p, elicit, -6.0)
check("elicitation posterior is a QuadraturePrevision (non-conjugate, truncation breaks NormalNormal)",
      post isa QuadraturePrevision, string(typeof(post)))
m_post = expect(post, Identity())
check("elicitation pulls the mean toward the observation (−6), past the prior", m_post < mean(p),
      "$m_post vs prior $(mean(p))")
check("elicitation posterior stays within the support [lo,hi]", m_post > -10.0 && m_post <= 0.0,
      string(m_post))
check("posterior support ⊆ [lo, hi] (the quadrature grid is on the bound)",
      all(-10.0 <= g <= 0.0 for g in post.grid), "grid spans $(extrema(post.grid))")

println("="^64)
println("ALL PASSED")
println("="^64)
