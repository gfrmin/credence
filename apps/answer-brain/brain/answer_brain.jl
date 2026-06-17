# Role: brain
"""
    answer_brain.jl — the answer-brain's belief + decision core (Stage 1).

A native-Julia port of `life-agent`'s validated Stage-0 answerer
(`src/life_agent/core/lookup.py`, `core/gather.py`): the tempered candidate posterior and
the EU decision over the terminal effectors, plus the `net_voi`-priced gather/ask gate. See
`docs/answer-brain/master-plan.md` and `docs/answer-brain/move-1-design.md`.

The parity boundary (move-1-design §1): the brain reasons over ABSTRACT candidates and
evidence groups. Candidate identity (string canon) and covariate projection are the body's
job; an `Obs` here is pure numbers — which candidate index it reports, which ancestry group
it belongs to, and its already-projected reliability covariates (authority / subject / time).

Belief: a `CategoricalMeasure` over K candidate atoms `0..K-1` plus an explicit NONE atom `K`
("the truth is not among the retrieved candidates"). Each observation conditions the
categorical through a tempered `tabular_log_density` (PushOnly) kernel — the same construction
`life-agent` ships to the skin (`apps/skin/server.jl` `build_kernel`), now built natively.

Invariant 1 (single reasoner): this module DECLARES data (the kernel's log-density matrix, the
`Tabular` utility vectors) and CALLS the Tier-1 primitives (`condition`, `optimise`, `value`,
`net_voi`). It never reads `weights` to select behaviour — the argmax over candidates lives in
`optimise` via K explicit `report_j` actions (move-1-design Open-Q1), not a hand-coded
`argmax(weights)`.
"""
module AnswerBrain

using Main.Credence: CategoricalMeasure, Finite, Kernel, PushOnly, condition, weights,
                     Tabular, Functional
import Main.Credence.Ontology: optimise, value, net_voi

export Obs, ChannelParams, CANONICAL_CHANNEL,
       candidate_posterior, terminal_decide, decide_full, decision_fpa, voi_gather,
       provisional_leader, gather_decide

# ── Stated channel parameters (the §4.2/§4.1 priors; calibration moves them) ────────────
# Mirror of life-agent's `lookup.py` constants and `bdsl/utility.bdsl`. A `ChannelParams`
# value travels with every call so the parity test can pin these against the fixture's
# `channel_params` (drift guard) rather than trusting a silent default.
struct ChannelParams
    a_alternatives::Float64   # effective number of wrong values a misreport spreads over
    beta_ancestry::Float64    # within-document ancestry temper exponent
    beta_model::Float64       # across-document (shared extractor) temper exponent
    p_none_prior::Float64     # prior mass on none-of-the-retrieved
    oracle_p::Float64         # owner-as-oracle reliability, pricing ask_clarify
    prob_eps::Float64         # log-domain floor
end

const CANONICAL_CHANNEL = ChannelParams(10.0, 0.3, 0.7, 0.5, 0.9, 1e-12)

# ── One observation, abstracted to numbers (the parity boundary) ────────────────────────
struct Obs
    reports::Int          # candidate index in 0..K-1 this observation asserts
    group::Int            # ancestry-group id (chunks of one document share a group)
    authority::Float64    # §4.1 source-authority prior
    subject_factor::Float64   # §4.1 doc_subject covariate on aᵢ (1.0 = no covariate)
    time_factor::Float64      # §4.1 doc_date covariate on aᵢ (1.0 = no covariate)
end

# ── The §4.2 lineage temper (pure; mirrors lookup.temper_scales, keyed on group) ────────
"""
    temper_scales(obs, cp) -> Vector{Float64}

Per observation: within an ancestry group of size m the group counts as
`1 + β_anc·(m-1)` effective observations; across the G groups (one shared extractor) the
groups count as `1 + β_mod·(G-1)`. A single observation ⇒ scale 1.
"""
function temper_scales(obs::Vector{Obs}, cp::ChannelParams)::Vector{Float64}
    counts = Dict{Int, Int}()
    for o in obs
        counts[o.group] = get(counts, o.group, 0) + 1
    end
    n_groups = length(counts)
    s_mod = n_groups == 0 ? 1.0 : (1.0 + cp.beta_model * (n_groups - 1)) / n_groups
    [begin
         m = counts[o.group]
         s_anc = (1.0 + cp.beta_ancestry * (m - 1)) / m
         s_anc * s_mod
     end for o in obs]
end

# ── The tempered noisy-channel log-density matrix (pure; mirrors observation_densities) ──
"""
    observation_densities(o, k, rho, scale, cp) -> Vector{Vector{Float64}}

Rows are the K candidate hypotheses + NONE (last); columns the K reported-candidate atoms.
With reliability `r = rho · authority · subject · time`, a match carries
`scale·log(r + (1-r)/A)` and any miss `scale·log((1-r)/A)`; NONE misses everything.
"""
function observation_densities(o::Obs, k::Int, rho::Float64, scale::Float64,
                               cp::ChannelParams)::Vector{Vector{Float64}}
    r = rho * o.authority * o.subject_factor * o.time_factor
    log_match = scale * log(max(r + (1.0 - r) / cp.a_alternatives, cp.prob_eps))
    log_miss  = scale * log(max((1.0 - r) / cp.a_alternatives, cp.prob_eps))
    rows = [[t == j ? log_match : log_miss for t in 0:(k - 1)] for j in 0:(k - 1)]
    push!(rows, fill(log_miss, k))   # NONE row: every report is a misreport
    rows
end

# ── The posterior: condition the candidate+NONE categorical on every observation ────────
"""
    candidate_posterior(k, obs, rho; cp=CANONICAL_CHANNEL) -> CategoricalMeasure

The tempered posterior over K candidates + NONE. Same construction life-agent ships to the
skin: a `categorical` prior (`p_none` on NONE, the rest uniform over candidates) conditioned
on each observation through a `tabular_log_density` PushOnly kernel, in observation order.
"""
function candidate_posterior(k::Int, obs::Vector{Obs}, rho::Float64;
                             cp::ChannelParams = CANONICAL_CHANNEL)::CategoricalMeasure
    atoms = collect(Float64, 0:k)                       # 0..k-1 candidates, k = NONE
    prior = vcat(fill((1.0 - cp.p_none_prior) / k, k), [cp.p_none_prior])
    state = CategoricalMeasure(Finite(atoms), log.(prior))
    scales = temper_scales(obs, cp)
    src, tgt = Finite(atoms), Finite(collect(Float64, 0:(k - 1)))
    for (o, scale) in zip(obs, scales)
        dens = observation_densities(o, k, rho, scale, cp)
        kern = Kernel(src, tgt, _ -> error("generate not used"),
                      (h, ob) -> dens[Int(round(h)) + 1][Int(round(ob)) + 1];
                      likelihood_family = PushOnly())  # credence-lint: allow — precedent:declarative-construction — tabular_log_density kernel, mirrors apps/skin/server.jl build_kernel
        state = condition(state, kern, Float64(o.reports))
    end
    state
end

# ── The decision: optimise over {report_j × K, hedge, ask_clarify, abstain} ──────────────
# action keys (deterministic numeric order = the skin's _sorted_action_keys semantics):
#   1..k       report_j        reward candidate (j-1), every other atom + NONE = u_wrong
#   k+1        hedge           the named-set value; misleads only when the truth is NONE
#   k+2        ask_clarify     the oracle price (NOT a u_assert outcome)
#   k+3        abstain         the gauge zero
"""
    decision_fpa(k, u_bar; cp=CANONICAL_CHANNEL) -> (order, fpa)

The terminal preference: an ordered action-key vector and a `Dict` of `Tabular` utility
vectors over the K+1 atoms. Pure declarative construction of the §4.4 utility (the argmax
over candidates is left to `optimise` via the K `report_j` actions — Invariant 1).
`u_bar` is the owner's utility posterior mean Ū, supplied by the body.
"""
function decision_fpa(k::Int, u_bar::AbstractDict;
                      cp::ChannelParams = CANONICAL_CHANNEL)
    u_c = Float64(u_bar["u_correct"]); u_w = Float64(u_bar["u_wrong"])
    u_h = Float64(u_bar["u_hedged"]);  u_ab = Float64(u_bar["u_abstain"])
    lam = Float64(u_bar["lambda_int"])
    order = Int[]
    fpa = Dict{Int, Functional}()
    for j in 1:k                                   # report_j: reward candidate (j-1)
        vals = fill(u_w, k + 1); vals[j] = u_c
        fpa[j] = Tabular(vals); push!(order, j)
    end
    fpa[k + 1] = Tabular(vcat(fill(u_h, k), [u_w]));           push!(order, k + 1)  # hedge
    fpa[k + 2] = Tabular(fill(cp.oracle_p * u_c - lam, k + 1)); push!(order, k + 2)  # ask
    fpa[k + 3] = Tabular(fill(u_ab, k + 1));                    push!(order, k + 3)  # abstain
    (order, fpa)
end

_action_name(act::Int, k::Int)::String =
    act <= k ? "report" : act == k + 1 ? "hedge" : act == k + 2 ? "ask_clarify" : "abstain"

# Shared decision: `optimise` over the terminal action set; returns the chosen action KEY + its EU.
# The argmax is the single decision mechanism — no `weights` are read to select behaviour.
function _decide(state::CategoricalMeasure, k::Int, u_bar::AbstractDict, cp::ChannelParams)
    order, fpa = decision_fpa(k, u_bar; cp = cp)
    act = optimise(state, order, fpa)          # argmax_a expect(state, fpa[a]) (single decision mech.)
    eu  = value(state, order, fpa)             # = EU of the chosen action
    (act, Float64(eu))
end

"""
    terminal_decide(state, k, u_bar; cp=CANONICAL_CHANNEL) -> (action::String, eu::Float64)

`optimise` over the terminal action set on the live posterior; the chosen `report_j` maps to
`"report"`. Returns the Stage-0 action vocabulary so parity compares like-for-like.
"""
function terminal_decide(state::CategoricalMeasure, k::Int, u_bar::AbstractDict;
                         cp::ChannelParams = CANONICAL_CHANNEL)
    act, eu = _decide(state, k, u_bar, cp)
    (_action_name(act, k), eu)
end

"""
    decide_full(state, k, u_bar; cp=CANONICAL_CHANNEL)
        -> (action::String, report_index::Union{Int,Nothing}, eu::Float64)

As `terminal_decide`, but also returns the **0-based candidate index** `optimise` chose when the
action is a report (`report_{j*}` ⇒ index `j*−1`), `nothing` otherwise. That index is the decision
mechanism's own choice, not an `argmax(weights)` — the wire surface (`daemon/server.jl`) needs the
reported value while keeping the Invariant-1 promise that no caller reads `weights` to pick an action.
"""
function decide_full(state::CategoricalMeasure, k::Int, u_bar::AbstractDict;
                     cp::ChannelParams = CANONICAL_CHANNEL)
    act, eu = _decide(state, k, u_bar, cp)
    report_index = act <= k ? act - 1 : nothing
    (_action_name(act, k), report_index, eu)
end

# ── The forward capability: VOI-priced gather/ask (NEW; no Stage-0 parity counterpart) ──
"""
    voi_gather(state, k, u_bar, probe_kernel, possible_obs, cost; cp=CANONICAL_CHANNEL)

`net_voi` of one probe for the terminal decision: the expected gain in `value` from
conditioning on the probe's observation, net of `cost`. Positive ⇒ the probe earns its
keep. The forward gather/ask gate the Stage-0 loop lacked (it gathered unconditionally);
priced against the SAME terminal preference, so gather competes with answer/abstain in one EU.
"""
function voi_gather(state::CategoricalMeasure, k::Int, u_bar::AbstractDict,
                    probe_kernel::Kernel, possible_obs, cost::Float64;
                    cp::ChannelParams = CANONICAL_CHANNEL)::Float64
    order, fpa = decision_fpa(k, u_bar; cp = cp)
    Float64(net_voi(state, probe_kernel, order, fpa, possible_obs, cost))
end

# ── The forward gather steer: an operator-set feature-policy (Move 4, move-4-design §2C) ──
"""
    provisional_leader(state, k, u_bar; cp=CANONICAL_CHANNEL) -> Int

The 0-based candidate index the decision mechanism would report **if forced to report** —
`optimise` over the K `report_j` actions ALONE. Invariant 1: the leader comes from the decision
mechanism, never `argmax(weights)`. Used as the `gather` target when the terminal decision
withholds, so the steer names the candidate whose support the probe will test.
"""
function provisional_leader(state::CategoricalMeasure, k::Int, u_bar::AbstractDict;
                            cp::ChannelParams = CANONICAL_CHANNEL)::Int
    order, fpa = decision_fpa(k, u_bar; cp = cp)
    optimise(state, order[1:k], fpa) - 1        # report keys 1..k only; key j ⇒ index j-1
end

"""
    gather_decide(state, k, u_bar; era_split=false, applied_probes=String[], cp=CANONICAL_CHANNEL)
        -> (effector, report_index, probe, target, eu)

The Move-4 feature-policy (move-4-design §2C, §5 Q2). If the terminal decision is a confident
`report`, take it — the leader cleared the EU bar (the bar already integrates `u_wrong` under Ū).
Otherwise, if a **class-valid discriminating probe** is available-and-unapplied, emit
`gather(probe, target)` to defer a below-bar answer; else the terminal decision (hedge / ask /
abstain) stands. v0's only class-valid probe is `recency` on an `era_split` — the validated lever
(`master-plan.md` §"probe library": stale leader 0.605 → ~0.09). `applied_probes` (body-held,
resent each step) guarantees termination. corroborate / subject / dispersion-gating are the
declared-but-deferred refinements (§5 Q2 named successor); their thresholds are the §5 Q5 params.
`report_index`/`probe`/`target` are `nothing` where inapplicable (a report carries no probe; a
gather carries no report index).
"""
function gather_decide(state::CategoricalMeasure, k::Int, u_bar::AbstractDict;
                       era_split::Bool = false,
                       applied_probes::AbstractVector{<:AbstractString} = String[],
                       cp::ChannelParams = CANONICAL_CHANNEL)
    action, report_index, eu = decide_full(state, k, u_bar; cp = cp)
    action == "report" && return (action, report_index, nothing, nothing, eu)
    if era_split && !("recency" in applied_probes)
        return ("gather", nothing, "recency", provisional_leader(state, k, u_bar; cp = cp), eu)
    end
    (action, report_index, nothing, nothing, eu)
end

end # module AnswerBrain
