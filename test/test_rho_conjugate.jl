# test_rho_conjugate.jl — the EXACT continuous-ρ group-channel join (antipattern disposal, Phase C).
# The lookup carries a Beta reliability latent ρ and a categorical V; the group-noisy-channel
# likelihood P(reports|v,ρ) = a + ρ·covariate·(1{all match v} − a) is LINEAR in ρ, so a Beta prior
# stays a polynomial-in-ρ × Beta under conditioning and the V-marginal is an EXACT sum of Beta
# moments — no ρ-grid, no quadrature (replacing the `labelled_mixture` ρ-grid). Asserts:
#   (1) the closed-form V-marginal == a dense ρ-grid reference (the engine is exact, not discretised);
#   (2) corroboration sharpens ρ exactly across groups (two agreeing docs concentrate more than one);
#   (3) a positional Tabular reads the ρ-integrated V-marginal EU (the optimise path);
#   (4) condition's log_marginal is the exact predictive; (5) an impossible atom gets weight 0.
#
# Run from repo root:  julia test/test_rho_conjugate.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: RhoCategoricalPrevision, RhoGroupChannel, rho_group_channel_factor, condition,
                weights, expect, log_predictive, Kernel, Finite, Tabular

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end
approx(a, b; atol = 1e-9) = abs(a - b) <= atol

println("="^64)
println("RhoCategoricalPrevision — exact continuous-ρ group-channel (Phase C)")
println("="^64)

A = 5
alpha, beta = 4.0, 4.0
v_prior = [0.45, 0.45, 0.10]          # atoms 1, 2, 3(=NONE)
groups = [(0.8, [1.0, 1.0]), (0.5, [2.0])]   # (covariate, reports)

mk(cov) = Kernel(Finite([0.0]), Finite([0.0]), h -> error("ng"),
                 (h, o) -> error("routes via condition"); likelihood_family = RhoGroupChannel(cov, A))

p = RhoCategoricalPrevision(log.(v_prior), alpha, beta, [[1.0], [1.0], [1.0]])
post = p
for (cov, reports) in groups
    global post = condition(post, mk(cov), reports)
end
w = weights(post)

# ── (1) closed form == dense ρ-grid reference ──
function gnc_like(v, cov, reports, rho)
    r_d = rho * cov
    m = length(reports)
    reliable = all(==(v), reports) ? r_d : 0.0
    (reliable + (1.0 - r_d) / A^m)
end
N = 400_001
ref = zeros(3)
for kk in 1:N
    rho = (kk - 0.5) / N
    bw = rho^(alpha - 1) * (1 - rho)^(beta - 1)       # unnormalised Beta kernel
    for vi in 1:3
        lik = 1.0
        for (cov, reports) in groups
            lik *= gnc_like(Float64(vi), cov, reports, rho)
        end
        ref[vi] += bw * v_prior[vi] * lik
    end
end
ref ./= sum(ref)
check("closed-form V-marginal == dense ρ-grid reference",
      maximum(abs.(w .- ref)) < 1e-6, "$w vs $ref")
check("V-marginal normalised", approx(sum(w), 1.0), string(sum(w)))

# ── (2) corroboration sharpens: two agreeing docs concentrate atom 1 more than one ──
one_doc = condition(p, mk(0.8), [1.0, 1.0])
two_doc = condition(one_doc, mk(0.8), [1.0, 1.0])
check("two corroborating docs concentrate atom 1 more than one",
      weights(two_doc)[1] > weights(one_doc)[1] > weights(p)[1],
      "$(weights(p)[1]) → $(weights(one_doc)[1]) → $(weights(two_doc)[1])")

# ── (3) positional Tabular reads the ρ-integrated V-marginal EU (the optimise path) ──
check("expect(one-hot atom 1) == P(V=1) (the ρ-integrated EU of report_1)",
      approx(expect(post, Tabular([1.0, 0.0, 0.0])), w[1]; atol = 1e-12), "")
u = [2.0, -1.0, -1.0]   # a utility vector over atoms
check("expect(Tabular) == Σ_v w_v·u_v (linearity)",
      approx(expect(post, Tabular(u)), sum(w[i] * u[i] for i in 1:3); atol = 1e-12), "")

# ── (4) condition's log_marginal is the exact predictive ──
# P(doc2 reports=[2] | doc1) computed independently: Σ_v P(v|doc1)·E_{ρ|v,doc1}[a+ρ b_v]. Easiest
# independent check: mass_after/mass_before with a hand fine-grid predictive.
lp = log_predictive(one_doc, mk(0.5), [2.0])
# fine-grid predictive of doc2 given the doc1 posterior
pred_ref = let num = 0.0, den = 0.0
    for kk in 1:N
        rho = (kk - 0.5) / N
        bw = rho^(alpha - 1) * (1 - rho)^(beta - 1)
        for vi in 1:3
            post1 = v_prior[vi] * gnc_like(Float64(vi), 0.8, [1.0, 1.0], rho)   # ∝ P(v,ρ | doc1)
            den += bw * post1
            num += bw * post1 * gnc_like(Float64(vi), 0.5, [2.0], rho)
        end
    end
    num / den
end
check("condition log_marginal == the exact predictive (fine-grid)",
      approx(exp(lp), pred_ref; atol = 1e-5), "$(exp(lp)) vs $pred_ref")

# ── (5) an impossible atom gets weight 0 ──
# Atom 2 with a fully-reliable corroboration of atom 1 and ρ pinned high stays possible (the noise
# floor keeps every atom > 0); but an atom whose prior is 0 stays 0.
p0 = RhoCategoricalPrevision(log.([0.5, 0.5, 0.0] .+ [0.0, 0.0, 1e-300]), alpha, beta,
                             [[1.0], [1.0], [1.0]])
check("a zero-prior atom keeps weight ≈ 0", weights(condition(p0, mk(0.8), [1.0]))[3] < 1e-200,
      string(weights(condition(p0, mk(0.8), [1.0]))[3]))

# ── (6) guards (wire-surface hardening): covariate must keep the likelihood ≥ 0, and an all-impossible
#        V posterior fails loud (never a silent NaN through expect/optimise). ──
check("covariate > 1 is rejected (likelihood a+ρb would go negative for high ρ)",
      try; RhoGroupChannel(1.5, A); false; catch e; occursin("covariate", sprint(showerror, e)); end)
check("covariate = 1 (the boundary) is allowed", RhoGroupChannel(1.0, A) isa RhoGroupChannel)
check("n_alternatives < 1 is rejected", try; RhoGroupChannel(0.5, 0); false; catch; true; end)
bad = RhoCategoricalPrevision(log.([0.5, 0.5]), 4.0, 4.0, [[-1.0], [-1.0]])  # negative integrals
check("an all-zero-mass V posterior → loud error (not a silent NaN)",
      try; weights(bad); false; catch e; occursin("zero integrated mass", sprint(showerror, e)); end)

println("="^64)
println("ALL PASSED")
println("="^64)
