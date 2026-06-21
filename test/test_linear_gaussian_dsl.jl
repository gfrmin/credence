# test_linear_gaussian_dsl.jl — the BDSL `:family linear-gaussian` surface.
#
# The thin-brain (decouple Move 2) shape: the consumer's pure BDSL `observe`
# conditions a JOINT MvGaussian weight belief on a noisy measurement of aᵀw via a
# declared `:family linear-gaussian xs sigma` kernel — the joint generalisation of
# maut_demo.bdsl's per-weight `:family normal`. The coefficient vector `xs` is a
# RUNTIME list (per article), which is why the `:family` parser now evaluates its
# args. The belief crosses back as a {type:mv_gaussian, mu, sigma} spec (params),
# exactly as the wire serialises it. Asserts the posterior against an independent
# Kalman oracle at rtol 1e-12. See docs/linear-gaussian-conjugate.md.

push!(LOAD_PATH, "src")
using Credence
using Credence: MvGaussianMeasure, MvGaussianPrevision, Euclidean

function check(name, cond, detail="")
    cond ? println("  ✓ $name") : (println("  ✗ $name  $detail"); error("FAILED: $name"))
end
approx(a, b; rtol=1e-12) = abs(a - b) <= rtol * max(abs(a), abs(b), 1.0)

println("test_linear_gaussian_dsl:")

# The consumer's pure BDSL model: weights in, joint-conditioned weights out.
# `xs` is the article's feature values (the LinearGaussian coefficients).
src = """
(define observe
  (lambda (weights xs y sigma)
    (condition weights
      (kernel (space :euclidean 2) (space :euclidean 1)
              (lambda (w) (lambda (o) o))
              :family linear-gaussian xs sigma)
      y)))
"""
env = load_dsl(src)
observe = env[Symbol("observe")]

# Prior: one diffuse weight (v=1.0), one confident (v=0.25). The wire would
# reconstruct this MvGaussianMeasure from a {type:mv_gaussian, ...} spec.
prior = MvGaussianMeasure(Euclidean(2), MvGaussianPrevision([0.0, 0.0], [1.0 0.0; 0.0 0.25]))
xs = [1.0, 1.0]
y  = -1.0
sigma = sqrt(0.5)

post = observe(prior, xs, y, sigma)
check("observe returns MvGaussianMeasure", post isa MvGaussianMeasure)

# Independent Kalman oracle (Σ diagonal here).
Σa1, Σa2 = 1.0, 0.25
s = 0.5 + (1.0*Σa1 + 1.0*Σa2)
m1ʹ, m2ʹ = (Σa1/s)*(y), (Σa2/s)*(y)
check("dsl post.mu[1]", approx(post.mu[1], m1ʹ), "$(post.mu[1]) vs $m1ʹ")
check("dsl post.mu[2]", approx(post.mu[2], m2ʹ), "$(post.mu[2]) vs $m2ʹ")
check("dsl off-diagonal induced", post.Sigma[1,2] != 0.0 && post.Sigma[1,2] < 0.0,
      "joint update keeps the explaining-away covariance")
check("dsl confident moves less", abs(post.mu[2]) < abs(post.mu[1]))

# Read-back: the conditioned belief serialises to a readable belief-spec — the
# decouple Move-2 call_dsl path (no read_params verb needed).
spec = params(post)
check("belief-spec is mv_gaussian", spec.type == :mv_gaussian)
check("belief-spec mu matches", spec.mu == post.mu)
check("belief-spec sigma is rows", spec.sigma == [[post.Sigma[1,1], post.Sigma[1,2]],
                                                   [post.Sigma[2,1], post.Sigma[2,2]]])

# Backward-compat: a literal-arg family (`:family normal 0.5`) still parses.
env2 = load_dsl("""
(define obs1
  (lambda (w y)
    (condition w
      (kernel (space :euclidean 1) (space :euclidean 1)
              (lambda (mu) (lambda (o) o))
              :family normal 0.5)
      y)))
""")
g = env2[Symbol("obs1")](Credence.GaussianMeasure(Euclidean(1), 0.0, 1.0), 1.0)
check(":family normal still works", g isa Credence.GaussianMeasure)

println("test_linear_gaussian_dsl: all passed")
