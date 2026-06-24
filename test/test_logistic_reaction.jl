# test_logistic_reaction.jl — the LogisticReaction kernel (decouple Move 1, commit 3). A binary
# reaction to a latent x under the choice model marginalised over a declared τ-prior:
#     P(react=1 | x) = Σ_t w_t · σ((sign·x − threshold)/t),  σ(z) = 1/(1+e^{−z})
# It moves life-agent's host-side `reaction_probability` (utility.py:155-166, shipped as a
# `tabular_log_density`) into the engine. Asserts:
#   (1) PARITY — the kernel log-density reproduces the host `reaction_probability` expression
#       BIT-FOR-BIT across an x-grid, for both reaction outcomes (the repointed body must get
#       identical results);
#   (2) a hand-computed value (σ(0)=0.5 at the threshold) as an independent sanity;
#   (3) conditioning a categorical over the x-grid by a reaction reweights toward the x the
#       reaction favours, exact vs a hand posterior.
#
# Run from repo root:
#     julia test/test_logistic_reaction.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: LogisticReaction, logistic_reaction_logdensity, Kernel, Finite,
                CategoricalMeasure, condition, weights

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("LogisticReaction kernel — τ-marginalised reaction (Move 1)")
println("="^64)

sign, thr = 1.0, 0.0
tvals, twts = [0.5, 2.0], [0.5, 0.5]            # a 2-temperature τ-prior
fam = LogisticReaction(sign, thr, tvals, twts)

# The host expression (utility.py `reaction_probability`), reproduced for the parity oracle.
host_p(x) = sum(twts[k] / (1.0 + exp(-(sign * x - thr) / tvals[k])) for k in eachindex(tvals))

# ── (1) parity: the kernel log-density ≡ log of the host reaction probability, bit-for-bit ──
for x in [-3.0, -1.0, -0.25, 0.0, 0.5, 2.0, 4.0]
    check("logdensity(x=$x, react=1) ≡ log host_p(x) (bit-exact parity)",
          logistic_reaction_logdensity(fam, x, 1) == log(max(host_p(x), 1e-300)))
    check("logdensity(x=$x, react=0) ≡ log(1−host_p(x)) (bit-exact parity)",
          logistic_reaction_logdensity(fam, x, 0) == log(max(1.0 - host_p(x), 1e-300)))
end

# ── (2) hand value: at the threshold every σ is 0.5, so P(react=1)=0.5 ──
check("at the threshold P(react=1)=0.5 (single τ, hand value)",
      logistic_reaction_logdensity(LogisticReaction(1.0, 0.0, [1.0], [1.0]), 0.0, 1) == log(0.5))

# ── (3) conditioning the x-grid by a reaction reweights toward the favoured x ──
grid = [-2.0, -1.0, 0.0, 1.0, 2.0]
space = Finite(grid)
kernel = Kernel(Finite(grid), Finite([0.0, 1.0]),
                x -> error("generate unused"),
                (x, o) -> logistic_reaction_logdensity(fam, x, o);
                likelihood_family = fam)
prior = CategoricalMeasure(space)               # uniform over the grid
post1 = condition(prior, kernel, 1.0)           # react = 1 (sign>0) ⇒ higher x favoured
# Hand posterior: ∝ prior · P(react=1|x). Replicates the engine path (prior's normalised logw +
# density, rebuilt through CategoricalMeasure) but reaches the density via the host `host_p`,
# so it is bit-exact AND independent of the helper under test.
ref1 = weights(CategoricalMeasure(space,
                  [prior.logw[i] + log(max(host_p(grid[i]), 1e-300)) for i in eachindex(grid)]))
check("posterior after react=1 ≡ prior·P(react=1|x) (bit-exact)", weights(post1) == ref1,
      "$(weights(post1)) vs $ref1")
check("react=1 favours higher x (monotone for sign>0)",
      weights(post1)[5] > weights(post1)[1], "$(weights(post1))")
check("react=0 favours lower x (the mirror)",
      weights(condition(prior, kernel, 0.0))[1] > weights(condition(prior, kernel, 0.0))[5])

println("="^64)
println("ALL CHECKS PASSED — LogisticReaction exact")
println("="^64)
