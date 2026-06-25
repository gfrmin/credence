# test_continuous_reaction.jl — the engine owns the grid (antipattern disposal, Phase A).
# A continuous Gaussian latent conditioned on (a) a continuous-τ logistic reaction — the engine
# integrates τ internally AND auto-quadratures x via `_condition_by_grid` (no grid in the declared
# model); (b) a Gaussian elicitation (`gaussian_known_var` → NormalNormal conjugate, exact). The
# body would declare only `{type: gaussian, mu, sigma}` + these kernels. Asserts:
#   (1) the reaction posterior mean matches a dense hand-integral (the engine's 64-pt quadrature);
#   (2) react=1 / react=0 shift the latent symmetrically (sign·x − threshold);
#   (3) the reaction posterior is a QuadraturePrevision (non-conjugate → engine grid), and
#       expect(::QuadraturePrevision, Identity) works (the apply(::Identity) path);
#   (4) the elicitation is EXACT (a GaussianPrevision posterior, NormalNormal closed form).
#
# Run from repo root:  julia test/test_continuous_reaction.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: GaussianPrevision, LogisticReaction, logistic_reaction_logdensity, Kernel,
                Euclidean, Finite, PushOnly, condition, expect, Identity, QuadraturePrevision

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end
approx(a, b; atol = 1e-9) = abs(a - b) <= atol

println("="^64)
println("engine owns the grid — continuous-τ reaction + Gaussian elicitation (Phase A)")
println("="^64)

# ── continuous-τ logistic reaction: τ ~ TruncatedNormal(1.0, 0.5; [0.5, 2.0]), engine-integrated ──
react = LogisticReaction(1.0, 0.0, 1.0, 0.5, 0.5, 2.0)
react_k = Kernel(Euclidean(1), Finite([0.0, 1.0]), x -> error("ng"),
                 (x, o) -> logistic_reaction_logdensity(react, x, o); likelihood_family = react)
prior = GaussianPrevision(0.0, 1.0)

post1 = condition(prior, react_k, 1.0)
post0 = condition(prior, react_k, 0.0)
check("reaction posterior is a QuadraturePrevision (engine grid, non-conjugate)",
      post1 isa QuadraturePrevision, string(typeof(post1)))
m1 = expect(post1, Identity())   # exercises apply(::Identity, ::Float64) on the grid
m0 = expect(post0, Identity())
check("react=1 shifts the latent positive", m1 > 0.0, "mean=$m1")
check("react=0 shifts the latent negative", m0 < 0.0, "mean=$m0")
check("react 0/1 are symmetric (threshold 0)", approx(m1, -m0; atol = 1e-9), "$m1 vs $(-m0)")

# dense hand-integral reference (the engine's 64-pt quadrature must match to quadrature tolerance)
function ref_mean(r)
    xs = range(-8, 8, length = 8001)
    num = 0.0; den = 0.0
    for x in xs
        w = exp(-0.5 * x^2) * exp(logistic_reaction_logdensity(react, x, r))
        num += x * w; den += w
    end
    num / den
end
check("reaction posterior mean ≈ hand-integral (react=1)", approx(m1, ref_mean(1.0); atol = 5e-3),
      "$m1 vs $(ref_mean(1.0))")

# ── Gaussian elicitation (gaussian_known_var → NormalNormal, EXACT) ──
elicit = Kernel(Euclidean(1), Euclidean(1), μ -> error("ng"), (μ, o) -> -0.5 * (o - μ)^2 / 4.0;
                params = Dict{Symbol, Any}(:sigma_obs => 2.0), likelihood_family = PushOnly())
post_e = condition(GaussianPrevision(0.0, 1.0), elicit, -2.0)
check("elicitation posterior is EXACT (a GaussianPrevision, NormalNormal)", post_e isa GaussianPrevision,
      string(typeof(post_e)))
# τ_prior=1, τ_obs=1/4 → μ_post = (1·0 + 0.25·(−2)) / 1.25 = −0.4, σ_post = 1/√1.25
check("elicitation NormalNormal mean exact", approx(expect(post_e, Identity()), -0.4), string(expect(post_e, Identity())))

println("="^64)
println("ALL PASSED")
println("="^64)
