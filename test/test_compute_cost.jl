# test_compute_cost.jl — the compute-cost coordinate on decide_with_voi (collapse-towers Phase 4).
# compute_cost is the agent's FORWARD inference spend, priced on :ask (the only action that commits to
# further inference — interrupt/await/condition/re-decide; :proceed/:block terminate). It rides as a
# constant subtraction from the :ask EU, in the one currency with interrupt_cost. Asserts: degenerate
# reduction at compute_cost=0 (bit-for-bit), the directional shift off :ask, and that compute_cost and
# interrupt_cost are DISTINCT currencies that SUM (the decision depends only on their total).
#
# Run from repo root:
#     julia test/test_compute_cost.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: MixturePrevision, TaggedBetaPrevision, BetaPrevision, structure_decision_kernel,
                Identity, LinearCombination, TestFunction
using Credence.Ontology: decide_with_voi, voi

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

cell(α, β) = MixturePrevision([TaggedBetaPrevision(1, BetaPrevision(α, β))], [0.0])
k = structure_decision_kernel()
lo, hi, unc = cell(2.0, 8.0), cell(8.0, 2.0), cell(1.0, 1.0)
unsafe = cell(9.0, 1.0)

println("="^64)
println("compute_cost — forward inference coordinate on :ask (Phase 4)")
println("="^64)

# ── (1) degenerate reduction: compute_cost = 0 ≡ omitting it, bit-for-bit, for ANY belief ──
for b in (lo, hi, unc)
    base = decide_with_voi(b, k; cost = 1.0, aversion = 1.0, interrupt_cost = 0.5)
    z    = decide_with_voi(b, k; cost = 1.0, aversion = 1.0, interrupt_cost = 0.5, compute_cost = 0.0)
    check("compute_cost=0 ≡ omitted (single-outcome, bit-exact)", base === z, "base=$base z=$z")
    baseh = decide_with_voi(b, k; cost = 1.0, aversion = 1.0, interrupt_cost = 0.5,
                            harm_belief = unsafe, harm_cost = 1.0)
    zh    = decide_with_voi(b, k; cost = 1.0, aversion = 1.0, interrupt_cost = 0.5,
                            harm_belief = unsafe, harm_cost = 1.0, compute_cost = 0.0)
    check("compute_cost=0 ≡ omitted (multi-outcome, bit-exact)", baseh === zh, "base=$baseh z=$zh")
end

# ── (2) directional: large compute_cost prices :ask's forward inference out of contention ──
check("uncertain + free ask + no compute_cost ⇒ :ask",
      decide_with_voi(unc, k; cost = 1.0, aversion = 1.0, interrupt_cost = 0.0, compute_cost = 0.0) === :ask)
check("uncertain + free ask + dear compute ⇒ not :ask (forward inference priced out)",
      decide_with_voi(unc, k; cost = 1.0, aversion = 1.0, interrupt_cost = 0.0, compute_cost = 1.0e9) !== :ask)

# ── (3) distinct-sum: compute_cost (inference) + interrupt_cost (attention) sum; decision depends only
# on the total, and the two are interchangeable (no double-count). Reconstruct the internal voi `v`:
# decide_with_voi's eu_ask = voi(...) − interrupt_cost − compute_cost (base value = 0 for unc), so
# :ask wins iff interrupt_cost + compute_cost < v, regardless of the split.
const0 = LinearCombination(Tuple{Float64, TestFunction}[], 0.0)
ask_block = LinearCombination(Tuple{Float64, TestFunction}[(-2.0, Identity())], 1.0)  # _lin(-cost·(tf+λ), cost·tf): cost=λ=1, tf=1
v = voi(unc, k, [:proceed, :block], Dict(:proceed => const0, :block => ask_block), [0, 1])
check("reconstructed voi > 0", v > 1e-6, "got $v")   # credence-lint: allow — precedent:test-oracle — v is the internal EVPI the sum is measured against
for (i, c) in [(0.5v, 0.0), (0.0, 0.5v), (0.25v, 0.25v)]   # credence-lint: allow — precedent:test-oracle — total 0.5v < v ⇒ :ask for every split
    check("total 0.5v ⇒ :ask (split-invariant: i,c interchangeable)",
          decide_with_voi(unc, k; cost = 1.0, aversion = 1.0, interrupt_cost = i, compute_cost = c) === :ask,
          "i=$i c=$c")
end
for (i, c) in [(1.5v, 0.0), (0.0, 1.5v), (0.75v, 0.75v)]   # credence-lint: allow — precedent:test-oracle — total 1.5v > v ⇒ not :ask for every split
    check("total 1.5v ⇒ not :ask (split-invariant: costs sum, no double-count)",
          decide_with_voi(unc, k; cost = 1.0, aversion = 1.0, interrupt_cost = i, compute_cost = c) !== :ask,
          "i=$i c=$c")
end

println("="^64)
println("ALL CHECKS PASSED — compute_cost")
println("="^64)
