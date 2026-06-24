# test_group_noisy_channel.jl — the exact correlated-evidence kernel (decouple Move 1,
# src/kernels.jl `GroupNoisyChannel`). A document of `m` correlated chunk-extractions of one
# source, read as evidence about a categorical hypothesis V (which candidate is the truth, or
# NONE). It replaces life-agent's §4.2 "tempering" (raising each chunk's log-likelihood to a
# power to fake non-independence) with the exact marginalisation of a binary document latent.
# Asserts:
#   (1) m=1 reduces BIT-FOR-BIT to the single-obs noisy channel (match r_d+(1−r_d)/A,
#       miss (1−r_d)/A) — the degenerate case tempering also reproduced at scale 1;
#   (2) m>1 corroboration UNDER-COUNTS: two chunks of ONE document move the V posterior LESS
#       than two INDEPENDENT documents (the correction tempering crudely faked), the group
#       likelihood-ratio matching the closed form 1 + r_d·A²/(1−r_d), strictly below the
#       independent model's [1 + r_d·A/(1−r_d)]².
#
# Run from repo root:
#     julia test/test_group_noisy_channel.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: CategoricalMeasure, Finite, Kernel, GroupNoisyChannel,
                group_noisy_channel_logdensity, condition, weights

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("group-noisy-channel kernel — exact correlated evidence (Move 1)")
println("="^64)

# 3 candidates (atoms 0,1,2) + NONE (atom 3); A alternatives in the noise channel.
A = 5
atoms = [0.0, 1.0, 2.0, 3.0]              # last atom = NONE
space = Finite(atoms)
ρ, cov = 0.9, 0.8
r_d = ρ * cov                            # document reliability 0.72
fam = GroupNoisyChannel(cov, ρ, A)
gnc = Kernel(space, Finite([0.0, 1.0, 2.0]), v -> error("generate unused"),
             (v, reports) -> group_noisy_channel_logdensity(fam, v, reports);
             likelihood_family = fam)

# ── (1) m=1 ≡ single-obs noisy channel, bit-for-bit ──
log_match = log(r_d + (1.0 - r_d) / Float64(A))   # canonical single-obs match
log_miss  = log((1.0 - r_d) / Float64(A))         # canonical single-obs miss
# One chunk reporting candidate 1 (atom 1.0): atom 1 matches, every other atom (incl. NONE) misses.
for (atom, expected, label) in [(0.0, log_miss, "miss"), (1.0, log_match, "match"),
                                (2.0, log_miss, "miss"), (3.0, log_miss, "miss (NONE)")]
    d = group_noisy_channel_logdensity(fam, atom, [1.0])
    check("m=1 density at V=$atom == single-obs $label (exact)", d == expected, "got $d, want $expected")
end

# The `condition` posterior (uniform prior) is the normalised exp of those densities.
prior = CategoricalMeasure(space)
post1 = condition(prior, gnc, [1.0])
ref1 = let d = [log_miss, log_match, log_miss, log_miss]
    e = exp.(d .- maximum(d)); e ./ sum(e)
end
check("m=1 posterior == single-obs noisy-channel posterior (bit-exact)",
      weights(post1) == ref1, "$(weights(post1)) vs $ref1")

# ── (2) m=2 corroboration under-counts ──
# Both chunks report candidate 1. Group: P(reports|V=1) ∝ r_d + (1−r_d)/A²; P(|V≠1) ∝ (1−r_d)/A².
post2_group = condition(prior, gnc, [1.0, 1.0])
post2_indep = condition(condition(prior, gnc, [1.0]), gnc, [1.0])   # WRONG: two independent conditions
wg, wi = weights(post2_group), weights(post2_indep)
check("corroborating chunks move the posterior LESS than independent docs",
      wg[2] < wi[2], "group P(V=1)=$(wg[2]) should be < independent $(wi[2])")

# Closed-form group likelihood-ratio P(reports|V=1)/P(reports|V≠1).
lr_group  = exp(group_noisy_channel_logdensity(fam, 1.0, [1.0, 1.0]) -
                group_noisy_channel_logdensity(fam, 0.0, [1.0, 1.0]))
lr_closed = 1.0 + r_d * Float64(A)^2 / (1.0 - r_d)
check("m=2 group likelihood-ratio == 1 + r_d·A²/(1−r_d)", isapprox(lr_group, lr_closed; rtol = 1e-12),
      "got $lr_group, want $lr_closed")
lr_indep = (1.0 + r_d * Float64(A) / (1.0 - r_d))^2   # the independent-docs LR
check("group LR strictly below independent LR (the corroboration discount)",
      lr_group < lr_indep, "$lr_group vs $lr_indep")

println("="^64)
println("ALL CHECKS PASSED — group-noisy-channel exact")
println("="^64)
