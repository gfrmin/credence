# test_complexity.jl — the single structural complexity log-prior `complexity_logprior`
# (collapse-towers Phase 1, src/complexity.jl), the operational form of SPEC §1.3.
#
# Asserts: the basic −λ·L + offset contract; λ=0 ⇒ uniform; the program axis reproduces the
# existing `-g.complexity*log(2) - p.complexity*log(2)` literal BIT-FOR-BIT (so test_program_space.jl
# stays green); the edge axis reproduces `k·log(p)+(n−k)·log(1−p)` up to the shared offset (assert
# DIFFERENCES between structures, never absolutes — the offset cancels under the mixture renorm);
# and the directional monotonicity in parent-count.
#
# Run from repo root:
#     julia test/test_complexity.jl

push!(LOAD_PATH, "src")
using Credence
using Credence.Ontology: complexity_logprior

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("complexity_logprior — single structural complexity log-prior (Phase 1)")
println("="^64)

# ── (1) basic contract: −λ·L + offset ──
check("−λ·L: complexity_logprior(5; λ=2.0) == -10.0",
      complexity_logprior(5; λ = 2.0) == -10.0,                       # credence-lint: allow — precedent:test-oracle — hand value −2·5
      "got $(complexity_logprior(5; λ=2.0))")
check("offset adds: complexity_logprior(5; λ=2.0, offset=3.0) == -7.0",
      complexity_logprior(5; λ = 2.0, offset = 3.0) == -7.0,          # credence-lint: allow — precedent:test-oracle — −10+3
      "got $(complexity_logprior(5; λ=2.0, offset=3.0))")

# ── (2) λ=0 ⇒ uniform (constant in L) ──
check("λ=0 ⇒ constant in L",
      complexity_logprior(3; λ = 0.0) == complexity_logprior(99; λ = 0.0) == 0.0)

# ── (3) program axis: two-call sum is BIT-IDENTICAL to the live literal ──
# enumeration.jl:170 / agent_state.jl:131 use `-g.complexity*log(2) - p.complexity*log(2)`.
# Phase 1 rewires them to the two-call sum below; this asserts the substitution is bit-exact.
for (g, p) in [(1, 1), (3, 2), (5, 4), (12, 7)]
    new_lw = complexity_logprior(g; λ = log(2)) + complexity_logprior(p; λ = log(2))
    old_lw = -g * log(2) - p * log(2)                                 # credence-lint: allow — precedent:test-oracle — the pre-refactor literal
    check("program axis bit-exact (g=$g,p=$p): two-call sum == literal", new_lw == old_lw,
          "new=$new_lw old=$old_lw")
end

# ── (4) edge axis: differences between structures match the old hand-formula (offset cancels) ──
# structure_bma.jl:96-99 uses `k·log(p) + (n−k)·log(1−p)`. Phase 1 rewires it to
# complexity_logprior(k; λ=log((1−p)/p), offset=n·log(1−p)). Assert DIFFERENCES (never absolutes):
# the per-structure offset is shared and the mixture renormalises it away.
let n = 2
    for p in (0.3, 0.5, 0.7)
        λ = log((1 - p) / p)
        offset = n * log(1 - p)
        new_lw(k) = complexity_logprior(k; λ = λ, offset = offset)
        old_lw(k) = k * log(p) + (n - k) * log(1 - p)                 # credence-lint: allow — precedent:test-oracle — the pre-refactor edge formula
        for k1 in 0:n, k2 in 0:n
            dnew = new_lw(k1) - new_lw(k2)
            dold = old_lw(k1) - old_lw(k2)
            check("edge diff p=$p (k$k1−k$k2) matches old",
                  isapprox(dnew, dold; atol = 1e-12, rtol = 1e-12), "Δnew=$dnew Δold=$dold")
        end
    end
end

# ── (5) p_edge=0.5 ⇒ λ=0 ⇒ all structures equal weight (uniform) ──
let n = 2, p = 0.5
    λ = log((1 - p) / p)
    check("p_edge=0.5 ⇒ λ==0 exactly", λ == 0.0, "λ=$λ")
    offset = n * log(1 - p)
    check("p_edge=0.5 ⇒ uniform over parent-counts",
          complexity_logprior(0; λ = λ, offset = offset) ==
          complexity_logprior(2; λ = λ, offset = offset))
end

# ── (6) monotonicity (directional): sparser-favouring iff p_edge<0.5 ──
let n = 2
    λlo = log((1 - 0.3) / 0.3)   # p<0.5 ⇒ λ>0 ⇒ more parents cost more
    check("p_edge<0.5 ⇒ more parents ⇒ strictly lower weight",
          complexity_logprior(2; λ = λlo) < complexity_logprior(1; λ = λlo) <
          complexity_logprior(0; λ = λlo))
    λhi = log((1 - 0.7) / 0.7)   # p>0.5 ⇒ λ<0 ⇒ more parents favoured
    check("p_edge>0.5 ⇒ more parents ⇒ strictly higher weight",
          complexity_logprior(2; λ = λhi) > complexity_logprior(1; λ = λhi) >
          complexity_logprior(0; λ = λhi))
end

println("="^64)
println("ALL CHECKS PASSED — complexity_logprior")
println("="^64)
