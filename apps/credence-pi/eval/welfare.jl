# Role: eval
"""
    welfare.jl — realized-welfare primitives for the credence-pi proof harnesses.

The unifying frame (docs: the welfare-MVP plan): a human is a utility PROFILE over
cost coordinates — money, time, attention, risk. credence-pi is a proxy
EU-maximiser for whichever human it serves: ONE shared learned posterior (beliefs),
MANY per-human utilities (preferences), the SAME EU-max mechanism. A "user type" is
a point in profile space.

These functions SCORE already-decided actions against objective labels. They are
NON-CAUSAL MEASUREMENT (the constitution's eval / test-oracle carve-out, CLAUDE.md
"display formatting, diagnostic telemetry … out of scope"): they call neither
`condition` nor `expect`, never feed a decision or a belief update, and only price
outcomes the brain has already chosen. The brain's `decide`/`decide_multi` are what
MAXIMISE the expectation of the welfare scored here; this module is the realized
read-out, the way `savings.jl` is for the live log.

Scope of THIS file (the offline corpus witness): the money + attention axes, the two
the offline corpora support (ClawsBench carries no per-call tokens or duration, so
money is a flat per-call unit and time has no separate signal; risk needs ATBench's
`is_safe` label). The time and risk axes are scored by the live report and the
ATBench face respectively — see the welfare-MVP plan's data-availability matrix.
"""
module Welfare

export Profile, realized_welfare, welfare_breakdown, WelfareTotals, COST_HAWK, FLOW_GUARD

"""
    Profile(name; w_money, w_time, w_attn, w_risk, λ, c, q, H)

A human's utility profile. The four `w_*` are the relative weights on the cost
coordinates (the human's preferences); `λ` is false-block aversion (the opportunity
cost of blocking a call the user actually wanted, in multiples of a call's cost);
`c` is the per-call money stake, `q` the per-ask attention price, `H` the per-unit
harm price. The offline witness varies `(λ, q)` at a fixed unit `c` — the absolute
`c` cancels in the argmax, so adaptivity is driven by the ratios `q/c` and `λ`.

The brain consumes a profile as the kwargs of `FeatureBrain.decide`
(`cost=c, aversion=λ, interrupt_cost=q`); live, a profile is a `utility.bdsl`. Same
numbers, two surfaces.
"""
struct Profile
    name::String
    w_money::Float64
    w_time::Float64
    w_attn::Float64
    w_risk::Float64
    λ::Float64
    c::Float64
    q::Float64
    H::Float64
end

function Profile(name::AbstractString; w_money::Real = 1.0, w_time::Real = 0.0,
                w_attn::Real = 1.0, w_risk::Real = 0.0,
                λ::Real, c::Real = 1.0, q::Real, H::Real = 0.0)
    Profile(String(name), Float64(w_money), Float64(w_time), Float64(w_attn),
            Float64(w_risk), Float64(λ), Float64(c), Float64(q), Float64(H))
end

# Two contrasting anchors spanning the money⟷attention trade-off (the corners the
# offline corpora can witness). cost-hawk: money precious, attention cheap → low λ
# (block readily; threshold 1/(1+λ)=0.8) + low q (ask freely to save spend).
# flow-guard: attention precious, money less so → high λ (block only when very
# sure; threshold 0.2) + high q (asking costs ~a whole call → almost never asks).
const COST_HAWK = Profile("cost-hawk"; w_money = 1.0, w_attn = 1.0, λ = 0.25, c = 1.0, q = 0.05)
const FLOW_GUARD = Profile("flow-guard"; w_money = 1.0, w_attn = 1.0, λ = 4.0, c = 1.0, q = 1.0)

"""
    realized_welfare(p::Profile, decision::Symbol, is_loop::Bool) -> Float64

The realized utility (higher is better; proceed-on-a-wanted-call is the 0 reference)
of one decided call under profile `p`, given the objective label `is_loop`
(true = waste / the user would deny, false = wanted). This is the realized form of
the brain's per-action EU:

    proceed: waste → −c        (you burned the call)          ; wanted → 0
    block:   waste → 0         (correctly stopped the waste)  ; wanted → −λ·c  (wrongly blocked a wanted call)
    ask:     either → −q       (right outcome at the cost of one interruption)

`ask` idealises the human as resolving correctly (yes on wanted, no on waste) at
attention price `q` — the same assumption the brain's VOI term makes. The money and
attention coordinates are weighted by `p.w_money` / `p.w_attn`.
"""
function realized_welfare(p::Profile, decision::Symbol, is_loop::Bool)::Float64
    if decision === :proceed
        return p.w_money * (is_loop ? -p.c : 0.0)
    elseif decision === :block
        return p.w_money * (is_loop ? 0.0 : -p.λ * p.c)
    elseif decision === :ask
        return p.w_attn * (-p.q)
    else
        error("realized_welfare: unknown decision $decision")
    end
end

"Per-axis realized-cost accumulator for a policy under one profile (costs ≥ 0; welfare = −total)."
struct WelfareTotals
    waste_cost::Float64        # money lost proceeding on loops              (money axis)
    falseblock_cost::Float64   # opportunity money-equiv lost blocking wanted (money axis)
    attention_cost::Float64    # attention spent asking                      (attention axis)
    n_proceed::Int
    n_block::Int
    n_ask::Int
end

total_welfare(t::WelfareTotals) = -(t.waste_cost + t.falseblock_cost + t.attention_cost)

"""
    welfare_breakdown(p::Profile, decisions, labels) -> WelfareTotals

Score a whole policy: `decisions[i]` is the action taken on held-out call `i`,
`labels[i]` its `is_loop` ground truth. Returns the per-axis cost decomposition
under profile `p` (so callers can see WHERE a policy wins or loses, not just the
scalar). `total_welfare(::WelfareTotals)` collapses it to the single number.
"""
function welfare_breakdown(p::Profile, decisions, labels)::WelfareTotals
    waste = 0.0; fblock = 0.0; attn = 0.0; np = 0; nb = 0; na = 0
    for (d, isloop) in zip(decisions, labels)
        if d === :proceed
            np += 1
            isloop && (waste += p.w_money * p.c)
        elseif d === :block
            nb += 1
            isloop || (fblock += p.w_money * p.λ * p.c)
        elseif d === :ask
            na += 1
            attn += p.w_attn * p.q
        else
            error("welfare_breakdown: unknown decision $d")
        end
    end
    WelfareTotals(waste, fblock, attn, np, nb, na)
end

end # module Welfare
