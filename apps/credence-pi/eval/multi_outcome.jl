# Role: eval
#
# multi_outcome.jl — the capstone: a SINGLE expected-utility decision that trades off
# harm and waste in ONE currency, defeating not just a regex but any OR of independent
# thresholds. Built brain-side from EXISTING Tier-1 ops (ProductMeasure + Projection +
# LinearCombination + optimise) — NO src/ changes, the constitution's "declare data, call
# primitives" pattern. This is the production shape for the multi-outcome `decide`.
#
# Two posteriors at decision time:
#   θ_a = P(approve|X)  — the waste brain (high = the user wants this call)
#   θ_u = P(unsafe|X)   — the harm brain (high = the action is unsafe)
# The OpenClaw user's per-action EU (proceed is the baseline = 0; H=0 recovers the existing
# waste-only decide exactly):
#   EU(proceed) = 0
#   EU(block)   = c·[1 − (1+λ)·θ_a] + H·θ_u      (block avoids BOTH waste and harm)
#   EU(ask)     = voi − q                        (VOI of the user resolving "did you want this?")
# expressed as LinearCombinations over Projections of the JOINT belief and maximised by the
# ONE canonical `optimise`.
#
# Block beats proceed iff  c·[1−(1+λ)θ_a] + H·θ_u > 0, i.e.  θ_a < 1/(1+λ) + H·θ_u/((1+λ)c).
# THE WASTE THRESHOLD ON θ_a RISES WITH THE HARM BELIEF θ_u — the outcomes are COUPLED.
# An OR of two fixed thresholds (block if θ_a<a* OR θ_u>u*) has a θ_a cutoff independent of
# θ_u; it cannot express the coupling. That is the regex/OR-impossible behaviour.
#
# Run from repo root:  julia --project=. apps/credence-pi/eval/multi_outcome.jl

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using Credence
using Credence: Identity, Projection, LinearCombination, TestFunction, expect, wrap_in_measure
import Main.Credence.Ontology: optimise, net_voi, ProductMeasure, Measure
include(joinpath(@__DIR__, "brain_env.jl"))
include(joinpath(@__DIR__, "..", "brain", "feature_brain.jl"))
using .FeatureBrain: build_model, build_prior, observe, belief_at_context

const M = build_model(["ctx"], [["c"]])     # 1-cell model; beliefs are made by feeding counts
const X = ["c"]
# Build a belief with posterior mean ≈ target by feeding (approve, deny) counts from Beta(2,2).
function belief_with(a::Int, d::Int)
    t = build_prior(M)
    for _ in 1:a; t = observe(M, t, X, 1); end
    for _ in 1:d; t = observe(M, t, X, 0); end
    belief_at_context(M, t, X)
end
mean_of(b) = expect(b, Identity())

const _ID = Identity()
_lin(coeff, off) = LinearCombination(Tuple{Float64, TestFunction}[(coeff, _ID)], off)
_const(off) = LinearCombination(Tuple{Float64, TestFunction}[], off)
_projlin(idx, coeff, off) = LinearCombination(Tuple{Float64, TestFunction}[(coeff, Projection(idx))], off)
_decision_kernel() = FeatureBrain._decision_kernel()

"""
    decide_multi(bxa, bxu; c, H, λ, q) -> Symbol

Multi-outcome EU decision over the joint of the approve-belief `bxa` and the unsafe-belief
`bxu`, via the single canonical `optimise`. Brain-side composition of Tier-1 ops only.
"""
function decide_multi(bxa, bxu; c::Float64, H::Float64, λ::Float64, q::Float64)
    joint = ProductMeasure(Measure[wrap_in_measure(bxa), wrap_in_measure(bxu)])
    proceed_fn = _const(0.0)                            # baseline
    # EU(block) = c·[1 − (1+λ)θ_a] + H·θ_u  (avoids waste AND harm); one Functional over the joint.
    block_fn = LinearCombination(Tuple{Float64, TestFunction}[(-c * (1.0 + λ), Projection(1)),
                                                              (H, Projection(2))], c)
    # ask EU: VOI of the user resolving the approve question (single-posterior, as the
    # existing decide), entered as a constant so all three compare through one optimise.
    eu_ask = net_voi(bxa, _decision_kernel(), [:proceed, :block],
                     Dict(:proceed => _const(0.0), :block => _lin(-c * (1.0 + λ), c)),
                     [0, 1], q)
    fpa = Dict(:proceed => proceed_fn, :block => block_fn, :ask => _const(eu_ask))
    optimise(joint, [:proceed, :block, :ask], fpa)
end

# The strongest fixed-rule foil: an OR of two independent thresholds.
or_foil(θa, θu; a_star = 0.5, u_star = 0.5) = (θa < a_star || θu > u_star) ? :block : :proceed

function main()
    println("="^80)
    println("  multi-outcome EU: one currency, coupled outcomes — beyond any OR of thresholds")
    println("="^80)
    println("EU(proceed)=−H·θ_u   EU(block)=c·[1−(1+λ)θ_a]   EU(ask)=voi−q")
    println("block beats proceed iff θ_a < 1/(1+λ) + H·θ_u/((1+λ)c)  ⇒ the θ_a cutoff MOVES with θ_u.\n")

    # beliefs at chosen posterior means (counts feeding Beta(2,2))
    ba_hi  = belief_with(16, 2)   # θ_a ≈ 0.82  (user clearly wants it)
    ba_mid = belief_with(2, 1)    # θ_a ≈ 0.57  (mildly wanted)
    bu_lo  = belief_with(0, 8)    # θ_u ≈ 0.17  (low harm)
    bu_mid = belief_with(2, 4)    # θ_u ≈ 0.40  (moderate harm)
    bu_hi  = belief_with(8, 2)    # θ_u ≈ 0.83  (high harm)

    scen = [("clearly-wanted, safe",        ba_hi,  bu_lo),
            ("clearly-wanted, HIGH harm",   ba_hi,  bu_hi),
            ("mildly-wanted, moderate harm", ba_mid, bu_mid),
            ("clearly-wanted, moderate harm",ba_hi,  bu_mid)]
    c, λ, q, H = 0.5, 1.0, 0.02, 1.0
    println("dial: c=\$$c  λ=$λ  q=\$$q  H=$H\n")
    println(rpad("scenario", 32), rpad("θ_a", 7), rpad("θ_u", 7), rpad("EU decision", 13), "OR-of-thresholds")
    for (name, bxa, bxu) in scen
        θa = round(mean_of(bxa); digits=2); θu = round(mean_of(bxu); digits=2)
        d = decide_multi(bxa, bxu; c=c, H=H, λ=λ, q=q)
        o = or_foil(θa, θu)
        flag = d == o ? "" : "   ← DIVERGES (coupling)"
        println(rpad(name, 32), rpad(θa, 7), rpad(θu, 7), rpad(d, 13), string(o), flag)
    end

    println("\n── the coupling, made explicit: hold θ_a≈0.82 fixed, raise harm θ_u ──")
    println(rpad("θ_u", 8), rpad("EU decision (H=$H)", 22), "why")
    for bu in [bu_lo, bu_mid, bu_hi]
        θu = round(mean_of(bu); digits=2)
        d = decide_multi(ba_hi, bu; c=c, H=H, λ=λ, q=q)
        why = d == :proceed ? "value outweighs harm" : "harm now outweighs the wanted call"
        println(rpad(θu, 8), rpad(d, 22), why)
    end

    println("\n", "="^80)
    println("Why no OR of thresholds can match this: the EU block-cutoff on θ_a is")
    println("1/(1+λ) + H·θ_u/((1+λ)c) — it SLIDES with θ_u. A fixed θ_a threshold OR a fixed")
    println("θ_u threshold cannot express a cutoff on one axis that depends on the other.")
    println("Sub-threshold harm + sub-threshold waste can SUM past the bar (neither rule")
    println("alone fires); and a high-enough θ_a can OUTWEIGH a harm that a fixed harm-rule")
    println("would block. That is evidence integration in one currency — EU-max, not rules.")
    println("And the same H/V dial is the user's risk preference, turned at will.")
    println("="^80)
end

main()
