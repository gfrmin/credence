# test_decide_with_voi.jl — the proceed/block/ask EU decision template lifted into engine
# stdlib (decouple Move 3, src/stdlib.jl `decide_with_voi`). It is the engine-side home of
# the EU coefficient assembly that used to live in the credence-pi app brain (`decide`/
# `decide_multi`); lifting it here is what lets a non-embedding consumer drive the decision
# over the wire (the body ships utility scalars, the engine does all the arithmetic).
# Asserts:
#   (1) the single-outcome block/proceed cutoff against an independent hand oracle (`:ask`
#       suppressed by a huge interrupt cost), AND that VOI actually flows — `:ask` wins when a
#       maximally-uncertain belief makes one observation worth more than a free gate;
#   (2) the OPTIONAL harm coordinate is additive (a high-harm belief flips a would-be
#       `:proceed` to `:block`) AND degenerate-reduces: `harm_cost = 0` collapses bit-for-bit
#       to the single-outcome decision for ANY harm belief (the "one op, degenerate case"
#       property that justified a single template over two).
#
# End-to-end behaviour preservation through the credence-pi shim (`decide`/`decide_multi`
# now delegate here) is additionally pinned by the credence-openclaw repo's tests/julia/test_feature_brain.jl
# (unchanged, passes through the shim).
#
# Run from repo root:
#     julia test/test_decide_with_voi.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: BetaPrevision, TaggedBetaPrevision, MixturePrevision, structure_decision_kernel
using Credence.Ontology: decide_with_voi

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

# The decision belief shape is exactly what `belief_at_context` returns: a one-cell
# `MixturePrevision` over a `TaggedBetaPrevision` (E[θ] = α/(α+β)). This carries the
# `LinearCombination`/`Projection` Functionals the EU template integrates against.
cell(α, β) = MixturePrevision([TaggedBetaPrevision(1, BetaPrevision(α, β))], [0.0])

println("="^64)
println("decide_with_voi — proceed/block/ask EU template (Move 3)")
println("="^64)

k = structure_decision_kernel()
HUGE = 1.0e9   # an interrupt cost that removes :ask from contention (EU(ask) = voi − HUGE ≪ 0)

# ── (1) single-outcome cutoff: at m=0, λ=1, block beats proceed iff E[θ] < 1/(1+λ) = 0.5 ──
# EU(block) = c·tf − c·(tf+λ)·E[θ] with tf = 1 ⇒ c − c·(1+λ)·E[θ]; EU(proceed) = 0.
lo = cell(2.0, 8.0)   # E[θ] = 0.2 < 0.5 ⇒ EU(block) = 1 − 2·0.2 = +0.6 ⇒ :block
hi = cell(8.0, 2.0)   # E[θ] = 0.8 > 0.5 ⇒ EU(block) = 1 − 2·0.8 = −0.6 ⇒ :proceed
check("low approve-prob ⇒ :block (hand cutoff 1/(1+λ))",
      decide_with_voi(lo, k; cost = 1.0, aversion = 1.0, interrupt_cost = HUGE) === :block)
check("high approve-prob ⇒ :proceed",
      decide_with_voi(hi, k; cost = 1.0, aversion = 1.0, interrupt_cost = HUGE) === :proceed)

# VOI flows: a maximally-uncertain belief makes one observation worth the (free) gate.
# Beta(1,1): E[θ] = 0.5 ⇒ EU(block) = 0 = EU(proceed); voi = 0.5·max(0, EU(block|θ⁻)) > 0 ⇒ :ask.
unc = cell(1.0, 1.0)
check("uncertain belief + free ask ⇒ :ask (VOI > 0 flows through net_voi)",
      decide_with_voi(unc, k; cost = 1.0, aversion = 1.0, interrupt_cost = 0.0) === :ask)

# ── (2a) harm coordinate is additive: a high-harm belief flips :proceed → :block ──
# approve E[θ_a] = 0.8 alone ⇒ :proceed (EU(block) = −0.6). Add unsafe E[θ_u] = 0.9, H = 2 ⇒
# EU(block) = −0.6 + H·E[θ_u] = −0.6 + 1.8 = +1.2 ⇒ :block.
unsafe = cell(9.0, 1.0)   # E[θ_u] = 0.9
check("harm belief flips :proceed → :block (additive H·θ_u)",
      decide_with_voi(hi, k; cost = 1.0, aversion = 1.0, interrupt_cost = HUGE,
                      harm_belief = unsafe, harm_cost = 2.0) === :block)

# ── (2b) degenerate reduction: harm_cost = 0 ≡ single-outcome, for ANY harm belief ──
# The harm Projection coefficient is 0 ⇒ the joint EU(block) and the ask-gate VOI (always on
# the approve belief) match the single-outcome triple exactly ⇒ identical argmax.
for (bθ, hb) in [(lo, unsafe), (hi, cell(1.0, 1.0)), (unc, cell(3.0, 4.0))]
    single = decide_with_voi(bθ, k; cost = 1.0, aversion = 1.0, interrupt_cost = 0.5)
    degen  = decide_with_voi(bθ, k; cost = 1.0, aversion = 1.0, interrupt_cost = 0.5,
                             harm_belief = hb, harm_cost = 0.0)
    check("harm_cost=0 ≡ single-outcome (bit-exact, any harm belief)", single === degen,
          "single=$single degen=$degen")
end

println("="^64)
println("ALL CHECKS PASSED — decide_with_voi exact")
println("="^64)
