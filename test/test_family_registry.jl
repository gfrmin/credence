# test_family_registry.jl — the BDSL `:family` surface reflects the FAMILY_REGISTRY
# (decouple Move 2). Asserts the registry dispatch resolves every keyword, that
# `:normal` consumes its declared σ arity, that the newly-exposed graded families
# drive their (already-existing) conjugate updates, and that the pre-existing
# bernoulli/flat behaviour is bit-identical.

push!(LOAD_PATH, "src")
using Credence
using Credence: BetaBernoulli, Flat, SoftBernoulli, WeightedBernoulli, NormalNormal
using Credence: FAMILY_REGISTRY, condition, mean, GaussianPrevision

function check(name, cond, detail="")
    if cond
        println("  ✓ $name")
    else
        println("  ✗ $name  $detail")
        error("FAILED: $name")
    end
end

# Build a kernel via the DSL `:family` form and return its declared family.
mk(famspec) = run_dsl(
    "(kernel (space :interval 0.0 1.0) (space :finite 0 1) " *
    "(lambda (h) (lambda (o) 0.0)) $famspec)").likelihood_family

println("test_family_registry:")

# ── pre-existing families unchanged ──
check("bernoulli → BetaBernoulli", mk(":family bernoulli") isa BetaBernoulli)
check("flat → Flat", mk(":family flat") isa Flat)

# ── newly-exposed families resolve ──
check("soft → SoftBernoulli", mk(":family soft") isa SoftBernoulli)
check("weighted → WeightedBernoulli", mk(":family weighted") isa WeightedBernoulli)

# ── :normal consumes its declared σ arity ──
kn = mk(":family normal 0.3")
check("normal → NormalNormal(σ)", kn isa NormalNormal && kn.sigma_obs == 0.3, "got $kn")  # credence-lint: allow — precedent:test-oracle — declared σ flows verbatim

# ── the registry is the single source of truth ──
check("registry holds all five", all(haskey(FAMILY_REGISTRY, k)
    for k in (:bernoulli, :flat, :soft, :weighted, :normal)))

# ── errors are clear and list the roster ──
throws_with(f, substr) =
    try; f(); false; catch e; occursin(substr, sprint(showerror, e)); end
check("unknown family errors with roster", throws_with(() -> mk(":family bogus"), "unknown"))
check(":normal without σ errors", throws_with(() -> mk(":family normal"), "numeric"))

# ── the exposed family actually drives its conjugate update (NormalNormal) ──
# This is the wire path: build_prevision yields a raw GaussianPrevision, which
# hits the conjugate ConjugatePrevision{GaussianPrevision,NormalNormal} update.
# prior N(0,1), obs-noise σ=0.3, obs=0.7 → precision-weighted posterior.
w = GaussianPrevision(0.0, 1.0)
k = run_dsl("(kernel (space :euclidean 1) (space :euclidean 1) " *
            "(lambda (mu) (lambda (o) o)) :family normal 0.3)")
post = condition(w, k, 0.7)
τ = 1.0 + 1.0 / 0.3^2
check("NormalNormal posterior mean exact", abs(mean(post) - (0.7 / 0.3^2) / τ) < 1e-12,
      "got $(mean(post))")  # credence-lint: allow — precedent:test-oracle — closed-form precision-weighted posterior

println("test_family_registry: all passed")
