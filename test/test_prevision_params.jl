# test_prevision_params.jl — the `params` serialization protocol (decouple Move 2).
# `params(p)` emits a type tag + sufficient statistics; reconstruction via the
# constructor round-trips bit-exact. Covers the Prevision methods and the Measure
# facade forwarding (`params(m) = params(m.prevision)`). This is the Invariant-3
# serialization view, distinct from mean/expect.

push!(LOAD_PATH, "src")
using Credence
using Credence: BetaPrevision, GaussianPrevision, GammaPrevision, DirichletPrevision,
                CategoricalPrevision
using Credence: BetaMeasure, GaussianMeasure, GammaMeasure

function check(name, cond, detail="")
    if cond
        println("  ✓ $name")
    else
        println("  ✗ $name  $detail")
        error("FAILED: $name")
    end
end

println("test_prevision_params:")

# ── per-subtype emission ──
check("beta", params(BetaPrevision(2.0, 3.0)) == (type=:beta, alpha=2.0, beta=3.0))  # credence-lint: allow — precedent:test-oracle — sufficient statistics emitted verbatim
check("gaussian", params(GaussianPrevision(0.5, 1.5)) == (type=:gaussian, mu=0.5, sigma=1.5))  # credence-lint: allow — precedent:test-oracle — sufficient statistics emitted verbatim
check("gamma", params(GammaPrevision(4.0, 0.5)) == (type=:gamma, alpha=4.0, beta=0.5))  # credence-lint: allow — precedent:test-oracle — sufficient statistics emitted verbatim
check("dirichlet", params(DirichletPrevision([1.0, 2.0, 3.0])) == (type=:dirichlet, alpha=[1.0, 2.0, 3.0]))  # credence-lint: allow — precedent:test-oracle — sufficient statistics emitted verbatim
check("categorical", params(CategoricalPrevision([0.0, -1.0])).type == :categorical)

# ── vectors are copied, not aliased (snapshot, not shared-reference) ──
α = [1.0, 2.0]
p = DirichletPrevision(α)
emitted = params(p).alpha
emitted[1] = 99.0
check("dirichlet alpha is a copy", params(p).alpha == [1.0, 2.0], "snapshot must not alias")

# ── round-trip: params → constructor → params is bit-exact ──
b = BetaPrevision(2.0, 3.0)
nt = params(b)
check("beta round-trip", params(BetaPrevision(nt.alpha, nt.beta)) == nt)  # credence-lint: allow — precedent:test-oracle — reconstruct∘serialize is identity
g = GaussianPrevision(0.25, 0.8)
ntg = params(g)
check("gaussian round-trip", params(GaussianPrevision(ntg.mu, ntg.sigma)) == ntg)  # credence-lint: allow — precedent:test-oracle — reconstruct∘serialize is identity

# ── Measure facade forwards to its wrapped Prevision ──
check("BetaMeasure forwards", params(BetaMeasure(2.0, 3.0)) == (type=:beta, alpha=2.0, beta=3.0))  # credence-lint: allow — precedent:test-oracle — facade serializes via its prevision
check("GaussianMeasure forwards",
      params(GaussianMeasure(Credence.Euclidean(1), 0.5, 1.5)) == (type=:gaussian, mu=0.5, sigma=1.5))  # credence-lint: allow — precedent:test-oracle — facade serializes via its prevision

println("test_prevision_params: all passed")
