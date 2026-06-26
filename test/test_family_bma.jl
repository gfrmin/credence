# test_family_bma.jl — Family-BMA: a posterior over likelihood families (collapse-towers Phase 2).
# The complexity log-prior (Phase 1) pointed at a *family* index, conditioned by the existing
# chain-rule marginal-likelihood reweighting. The agent AVERAGES over the family posterior; it never
# selects a family (`average-not-collapse`). Worked candidate set: NormalNormal (known σ) vs
# NormalGamma (unknown σ), both over ℝ.
#
# Run from repo root:
#     julia test/test_family_bma.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: GaussianPrevision, NormalGammaPrevision, MixturePrevision, NormalNormal,
                NormalGammaLikelihood, BetaBernoulli, Euclidean, Finite, ProductSpace, PositiveReals,
                Kernel, condition, weights, mean, variance, params, log_predictive, wrap_in_measure
using Credence: FamilyCandidate, build_family_model, build_family_prior, family_observe, family_posterior

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end
raises(f) = try (f(); false) catch; true end

println("="^64)
println("Family-BMA — posterior over likelihood families (Phase 2)")
println("="^64)

# Candidate families over ℝ: NormalNormal(σ=1, known variance) vs NormalGamma (unknown variance).
nn = FamilyCandidate(NormalNormal(1.0), GaussianPrevision(0.0, 2.0), Euclidean(1), 0.0)
ng = FamilyCandidate(NormalGammaLikelihood(), NormalGammaPrevision(1.0, 0.0, 2.0, 2.0), Euclidean(1), 0.0)
model = build_family_model([nn, ng])
prior = build_family_prior(model)

# ── (0) prior is the 2-family mixture, uniform (λ_family=0 ⇒ complexity prior is a no-op) ──
check("prior is a 2-family mixture, uniform weights", weights(prior) == [0.5, 0.5], "got $(weights(prior))")  # credence-lint: allow — precedent:test-oracle — uniform prior weights are the hand value [0.5,0.5]

# ── (1) generative recovery (directional, no thresholds) ──
# High-spread data (|obs| ≫ σ=1) is implausible under the fixed-variance NormalNormal but fits the
# adaptive-variance NormalGamma ⇒ NormalGamma's posterior mass rises and the gap WIDENS with data.
let m = prior, gaps = Float64[]
    for o in (5.0, -5.0, 4.5, -6.0, 5.5)
        m = family_observe(model, m, o)
        push!(gaps, abs(weights(m)[1] - weights(m)[2]))  # credence-lint: allow — precedent:test-oracle — test-side gap metric for the directional recovery assertion
    end
    check("conditioning reweights families away from uniform", weights(m) != [0.5, 0.5])  # credence-lint: allow — precedent:test-oracle — asserts the posterior moved off the uniform prior
    check("adaptive-variance family (NormalGamma) wins on high-spread data",
          weights(m)[2] > weights(m)[1], "got $(weights(m))")  # credence-lint: allow — precedent:test-oracle — directional: the adaptive-variance family wins on high-spread data
    check("family posterior gap widens with data (directional)", gaps[end] > gaps[1], "gaps=$gaps")
end

# ── (2) degenerate reduction: a SINGLETON candidate set ≡ the plain conjugate posterior (bit-exact) ──
let single = build_family_model([nn])
    post = family_observe(single, build_family_prior(single), 1.5)
    # oracle: condition the NormalNormal prior directly through a plain NormalNormal kernel.
    oracle_k = Kernel(Euclidean(1), Euclidean(1),
                      h -> wrap_in_measure(GaussianPrevision(h, 1.0)), (h, o) -> -0.5 * (o - h)^2;
                      likelihood_family = NormalNormal(1.0))
    oracle = condition(GaussianPrevision(0.0, 2.0), oracle_k, 1.5)
    check("singleton family-BMA weight is a point mass", weights(post) == [1.0], "got $(weights(post))")  # credence-lint: allow — precedent:test-oracle — a singleton mixture is the hand value [1.0]
    check("singleton family-BMA component ≡ direct conjugate posterior (bit-exact)",
          params(post.components[1]) == params(oracle),                 # credence-lint: allow — precedent:test-oracle — direct conjugate is the independent oracle
          "got $(params(post.components[1])) vs $(params(oracle))")
end

# ── (3) commensurability guard: candidates over DIFFERENT obs spaces error at construction ──
# (distinct prevision types, so this isolates the obs-space check, not the distinct-type guard)
let off_space = FamilyCandidate(NormalGammaLikelihood(), NormalGammaPrevision(1.0, 0.0, 2.0, 2.0),
                                Finite([0.0, 1.0]), 0.0)
    check("commensurability: different obs spaces ⇒ construction error",
          raises(() -> build_family_model([nn, off_space])))
end

# ── (4) distinct-type guard: two candidates with the SAME prevision type error at construction ──
let dup = FamilyCandidate(NormalNormal(2.0), GaussianPrevision(1.0, 1.0), Euclidean(1), 0.0)
    check("distinct-type: duplicate prevision type ⇒ construction error",
          raises(() -> build_family_model([nn, dup])))
end

# ── (5) conjugate-recognised guard: a non-conjugate (prior, family) pair errors at construction ──
let notconj = FamilyCandidate(BetaBernoulli(), GaussianPrevision(0.0, 1.0), Euclidean(1), 0.0)
    check("conjugate-recognised: Gaussian prior + Bernoulli family ⇒ construction error",
          raises(() -> build_family_model([notconj])))
end

# ── (6) deferral guard (Q4): the mixture PREDICTIVE over a family mixture raises today ──
# The unresolved DispatchByComponent kernel falls through to the nominal kernel's error-stub
# log_density — the predictive deferral is guarded by this assertion, not an assumption.
check("family-mixture log_predictive raises (deferral pinned)",
      raises(() -> log_predictive(prior, model.kernel, 1.5)))  # credence-lint: allow — precedent:test-oracle — asserts the family-mixture predictive throws (the Q4 deferral guard)

# ── (7) average-not-collapse: the readout is the FULL family posterior, never an argmax/index ──
let post = family_observe(model, prior, 1.5)
    fp = family_posterior(post)
    check("average-not-collapse: readout is the full posterior vector",
          fp isa AbstractVector{<:Real} && length(fp) == 2, "got $fp")
    check("no `select_family` collapse function exists", !isdefined(Credence, :select_family))
end

println("="^64)
println("ALL CHECKS PASSED — Family-BMA")
println("="^64)
