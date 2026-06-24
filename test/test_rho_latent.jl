# test_rho_latent.jl — the carried shared ρ-latent (decouple Move 1, commit 2). The joint
# (ρ, V) posterior is a `MixturePrevision` over a ρ-grid; each component is a
# `LabelledCategoricalPrevision` (label = ρ) over the candidate hypotheses V. A
# `DispatchByComponent` group-noisy-channel kernel routes per component — each ρ sets its own
# `r_d = ρ·covariate` — so conditioning learns ρ jointly with V, and ρ is SHARED across
# documents (the s_mod coupling): conditioning doc 1 reweights the ρ-mixture that bears on doc 2.
# Asserts:
#   (1) L=1 (single ρ) ≡ the commit-1 group-noisy-channel posterior, bit-for-bit;
#   (2) per-ρ routing — in a 2-ρ mixture each component is conditioned with ITS OWN ρ (the
#       closure reads `comp.label`), bit-for-bit vs an independent single-ρ condition;
#   (3) ρ-learning — corroborating reports reweight the mixture toward the ρ with the higher
#       document marginal likelihood (`_predictive_ll`), exact values + direction;
#   (4) `expect`/`optimise` on the mixture integrate ρ for free (Σ_ρ w_ρ·P(V|ρ)), no collapse.
#
# Run from repo root:
#     julia test/test_rho_latent.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: MixturePrevision, LabelledCategoricalPrevision, CategoricalPrevision,
                GroupNoisyChannel, Kernel, Finite, Tabular, group_noisy_channel_logdensity,
                condition, weights, logsumexp, DispatchByComponent
using Credence.Ontology: expect, optimise

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("ρ-latent — mixture + labelled categorical (Move 1, commit 2)")
println("="^64)

# 2 candidates (positions 1,2) + NONE (position 3); A alternatives; document covariate cov.
A, cov = 5, 0.8
prior_lw = [0.0, 0.0, 0.0]                                    # uniform V prior over the 3 positions
labelled(ρ) = LabelledCategoricalPrevision(ρ, CategoricalPrevision(copy(prior_lw)))
gnc = Kernel(Finite([1.0, 2.0, 3.0]), Finite([1.0, 2.0]),
             v -> error("generate unused"),
             (h, o) -> error("group-noisy-channel routes via categorical_logdensity");
             likelihood_family = DispatchByComponent(c -> GroupNoisyChannel(cov, c.label, A)))

# References replicate the EXACT engine ops (the component stores the NORMALISED prior, and
# `condition` rebuilds through CategoricalPrevision / MixturePrevision — so the oracle must
# normalise the same way to be bit-exact), but reach the likelihood through the commit-1
# `group_noisy_channel_logdensity` directly — NOT through the labelled-categorical `condition`
# under test.
base = CategoricalPrevision(copy(prior_lw)).log_weights       # the normalised prior the component holds
gd(ρ, i, reports) = group_noisy_channel_logdensity(GroupNoisyChannel(cov, ρ, A), i, reports)
comp_ref(ρ, reports) = weights(CategoricalPrevision([base[i] + gd(ρ, i, reports) for i in 1:3]))
predll(ρ, reports)   = logsumexp([base[i] + gd(ρ, i, reports) for i in 1:3])

# ── (1) L=1 (single ρ) ≡ commit-1 group-noisy-channel posterior ──
let post = condition(MixturePrevision([labelled(0.9)], [0.0]), gnc, [1])
    check("L=1 component posterior ≡ single-ρ group-channel (bit-exact)",
          weights(post.components[1]) == comp_ref(0.9, [1]),
          "$(weights(post.components[1])) vs $(comp_ref(0.9, [1]))")
end

# ── (2) per-ρ routing: each component conditioned with ITS OWN ρ ──
ρ_grid = [0.5, 0.9]
prior_mix = MixturePrevision([labelled(ρ) for ρ in ρ_grid], [0.0, 0.0])
post = condition(prior_mix, gnc, [1, 1])                      # two corroborating chunks reporting position 1
for (r, ρ) in enumerate(ρ_grid)
    check("component ρ=$ρ conditioned with its own ρ (bit-exact routing)",
          weights(post.components[r]) == comp_ref(ρ, [1, 1]),
          "$(weights(post.components[r])) vs $(comp_ref(ρ, [1, 1]))")
end

# ── (3) ρ-learning: the mixture reweights toward the better-explaining ρ ──
# Exact: new mixture logw[r] = prior_mix.logw[r] + _predictive_ll(comp_r); rebuild + normalise the same.
ref_mix = weights(MixturePrevision([labelled(ρ) for ρ in ρ_grid],
                  [prior_mix.log_weights[r] + predll(ρ_grid[r], [1, 1]) for r in eachindex(ρ_grid)]))
check("ρ-mixture weights ≡ Bayes reweight by document marginal likelihood (bit-exact)",
      weights(post) == ref_mix, "$(weights(post)) vs $ref_mix")
check("corroborating reports shift mass toward higher ρ (s_mod learning)",
      weights(post)[2] > weights(post)[1],
      "P(ρ=0.9)=$(weights(post)[2]) should exceed P(ρ=0.5)=$(weights(post)[1])")

# ── (4) expect / optimise on the mixture integrate ρ for free (no collapse) ──
u = Tabular([1.0, 0.0, 0.0])                                  # utility 1 iff V = position 1
v_marginal_1 = sum(weights(post)[r] * weights(post.components[r].categorical)[1]
                   for r in eachindex(ρ_grid))               # Σ_ρ w_ρ · P(V=1|ρ)
check("expect(mixture, Tabular) == the ρ-integrated V-marginal (exact, no collapse)",
      expect(post, u) == v_marginal_1, "$(expect(post, u)) vs $v_marginal_1")
check("optimise over the mixture picks the corroborated candidate",
      optimise(post, [:report1, :abstain],
               Dict(:report1 => u, :abstain => Tabular([0.4, 0.4, 0.4]))) === :report1)

println("="^64)
println("ALL CHECKS PASSED — ρ-latent exact")
println("="^64)
