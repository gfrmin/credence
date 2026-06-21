# test_linear_gaussian_conjugate.jl — the (MvGaussianPrevision, LinearGaussian)
# conjugate: the exact Bayesian-linear-regression / Kalman measurement update.
#
# Tolerances (mirroring docs/posture-3/move-4-design.md §3):
#   - the diagonal-form identity vᵢ' = vᵢ − (aᵢvᵢ)²/s holds by exact algebra: `==`
#   - the full posterior μ/Σ vs an independent hand oracle: `rtol=1e-12`
#
# Each conjugate-path test asserts `_dispatch_path(p, k) == :conjugate` BEFORE
# the value assertion — the tripwire for a silent registry miss (without it a
# miss would error in `condition` and the value test never runs).
#
# The headline property is exactness: conditioning two independent weights on a
# noisy measurement of their sum induces a NONZERO off-diagonal covariance (the
# explaining-away signal). A diagonal/mean-field collapse would zero it; we keep
# it. See docs/linear-gaussian-conjugate.md.

push!(LOAD_PATH, "src")
using Credence
using Credence: MvGaussianPrevision, LinearGaussian, Kernel, Euclidean,
                maybe_conjugate, update, ConjugatePrevision, _dispatch_path

function check(name, cond, detail="")
    if cond
        println("  ✓ $name")
    else
        println("  ✗ $name  $detail")
        error("FAILED: $name")
    end
end

approx(a, b; rtol=1e-12) = abs(a - b) <= rtol * max(abs(a), abs(b), 1.0)
throws(f) = try; f(); false; catch; true; end

println("test_linear_gaussian_conjugate:")

# ── Fixture: one diffuse weight (v=1.0), one confident weight (v=0.25) ──
# observed jointly through y ~ N(a'w, σ²), a = [1,1], σ² = 0.5, y = −1.
m  = [0.0, 0.0]
Σ  = [1.0 0.0; 0.0 0.25]
a  = [1.0, 1.0]
σ²  = 0.5
y  = -1.0
prior = MvGaussianPrevision(copy(m), copy(Σ))
k = Kernel(Euclidean(2), Euclidean(1),
           w -> error("generate not used"),
           (w, o) -> -0.5 * (o - (a[1]*w[1] + a[2]*w[2]))^2 / σ²;
           likelihood_family = LinearGaussian(a, sqrt(σ²)))

# ── Independent hand oracle (scalar arithmetic, d = 2) ──
Σa1, Σa2 = Σ[1,1]*a[1], Σ[2,2]*a[2]          # Σ is diagonal here: Σa = [v1·a1, v2·a2]
ŷ  = a[1]*m[1] + a[2]*m[2]
s  = σ² + a[1]*Σa1 + a[2]*Σa2
k1, k2 = Σa1/s, Σa2/s
m1ʹ, m2ʹ = m[1] + k1*(y - ŷ), m[2] + k2*(y - ŷ)
Σ11ʹ = Σ[1,1] - Σa1*Σa1/s
Σ22ʹ = Σ[2,2] - Σa2*Σa2/s
Σ12ʹ = Σ[1,2] - Σa1*Σa2/s

# ── dispatch tripwire ──
check("dispatch is :conjugate", _dispatch_path(prior, k) == :conjugate,
      "got $(_dispatch_path(prior, k)) — registry miss")

# ── update() matches the oracle (rtol 1e-12) ──
post = update(ConjugatePrevision(prior, k.likelihood_family), y).prior
check("post.mu[1]", approx(post.mu[1], m1ʹ), "$(post.mu[1]) vs $m1ʹ")
check("post.mu[2]", approx(post.mu[2], m2ʹ), "$(post.mu[2]) vs $m2ʹ")
check("post.Sigma[1,1]", approx(post.Sigma[1,1], Σ11ʹ))
check("post.Sigma[2,2]", approx(post.Sigma[2,2], Σ22ʹ))
check("post.Sigma[1,2]", approx(post.Sigma[1,2], Σ12ʹ))
check("post.Sigma symmetric", post.Sigma[1,2] == post.Sigma[2,1])

# ── EXACTNESS: induced off-diagonal is nonzero and negative (explaining away) ──
check("off-diagonal induced (nonzero)", post.Sigma[1,2] != 0.0,
      "a diagonal/mean-field collapse would zero this")
check("off-diagonal negative (anti-correlation)", post.Sigma[1,2] < 0.0)

# ── explaining-away: the confident weight (idx 2) moves LESS than diffuse (idx 1) ──
check("confident weight moves less", abs(post.mu[2]) < abs(post.mu[1]),
      "confident $(post.mu[2]) vs diffuse $(post.mu[1])")

# ── ZERO COST to a marginals-only consumer: diag(Σ') equals the diagonal form ──
# vᵢ' = vᵢ − (aᵢ·vᵢ)²/s   (the request's "diagonal" update) is EXACTLY diag(Σ').
check("diag matches diagonal-form v1'", post.Sigma[1,1] == Σ[1,1] - (a[1]*Σ[1,1])^2/s)
check("diag matches diagonal-form v2'", post.Sigma[2,2] == Σ[2,2] - (a[2]*Σ[2,2])^2/s)

# ── public condition() path matches update() path ──
post2 = condition(prior, k, y)
check("condition == update (mu)", post2.mu == post.mu)
check("condition == update (Sigma)", post2.Sigma == post.Sigma)

# ── read surface: mean / variance / marginal / draw ──
check("mean is mu", mean(post) == post.mu)
check("variance is diag(Sigma)", variance(post) == [post.Sigma[1,1], post.Sigma[2,2]])
mg1 = Credence.marginal(post, 1)
check("marginal(1) mu", mg1.mu == post.mu[1])
check("marginal(1) sigma", mg1.sigma == sqrt(post.Sigma[1,1]))
d = draw(post)
check("draw length", length(d) == 2)

# ── error paths ──
bad_k = Kernel(Euclidean(2), Euclidean(1), w -> 0.0, (w, o) -> 0.0;
               likelihood_family = Credence.NormalNormal(1.0))
check("wrong family errors", throws(() -> condition(prior, bad_k, y)))
check("dim mismatch errors",
      throws(() -> MvGaussianPrevision([0.0, 0.0], [1.0 0.0 0.0; 0.0 1.0 0.0])))

# ── multi-step: fold a second observation; off-diagonal carries forward ──
post3 = condition(post, k, 0.5)
check("two-step stays symmetric", post3.Sigma[1,2] == post3.Sigma[2,1])
check("two-step variances shrink further", post3.Sigma[1,1] < post.Sigma[1,1])

println("test_linear_gaussian_conjugate: all passed")
