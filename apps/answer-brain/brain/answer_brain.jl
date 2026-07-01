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
# Gather VOI is engine stdlib (src/gather_voi.jl — upstreamed); imported so the app
# surface (exports below) stays stable for the daemon/tests.
import Main.Credence: grow_value, best_grow

export Obs, ChannelParams, CANONICAL_CHANNEL,
       candidate_posterior, terminal_decide, decide_full, decision_fpa, voi_gather,
       grow_value, best_grow,
       provisional_leader, gather_decide,
       Transform, ScheduleCtx, default_registry, registry_from_wire, schedule_decide

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

# The corroborate probe KERNEL (Slice 3): the tempered noisy-channel at the re-read's reliability
# `gather_rho` over a NEUTRAL source (authority=subject=time=1) — `observation_densities`, the same
# matrix `candidate_posterior` conditions on. `net_voi` integrates it against the current posterior
# to PRICE "would a high-reliability re-read move the decision?" WITHOUT running it (Plan §1: the
# in-set corroboration VOI is well-defined over the fixed candidate set; discovery is not).
function _corroborate_kernel(k::Int, gather_rho::Float64, cp::ChannelParams)::Kernel
    dens = observation_densities(Obs(0, 0, 1.0, 1.0, 1.0), k, gather_rho, 1.0, cp)
    Kernel(Finite(collect(Float64, 0:k)), Finite(collect(Float64, 0:(k - 1))),
           _ -> error("generate not used"),
           (h, o) -> dens[Int(round(h)) + 1][Int(round(o)) + 1];
           likelihood_family = PushOnly())  # credence-lint: allow — precedent:declarative-construction — corroborate probe, mirrors observation_densities
end

# ── The transformation registry: the menu becomes DATA (the VOI scheduler) ───────────────
# The owner's directive: one rule — schedule whichever transformation maximises VOI−cost on this
# data, cheap or expensive alike. `net_voi` is the principled mechanism; hardcoding the menu (the
# old 3-branch cascade) was the unprincipled part. A `Transform` is one menu entry; `schedule`
# prices the whole menu in one uniform loop. The honest boundary (three named kinds — uniformity
# claimed only where it holds): a transform is VOI-schedulable iff its OUTPUT is modelled as a
# likelihood (a kernel) over the belief's space.
#   :voi   — VOI-priced refine. Output is a kernel over the fixed candidate set; `net_voi` integrates
#            it against the posterior and gathers the argmax if it clears its cost. (corroborate)
#   :guard — mandatory non-VOI refine. Defends an OUT-OF-MODEL risk the candidate belief cannot price
#            (the belief has no atom for "misattributed to the owner"), so VOI over it is blind and
#            the guard must fire unconditionally. (recency-on-era_split; the owner-scoped corroborate)
# `:grow` (discovery/recall — enlarges K) cannot be priced by `net_voi` over the closed categorical
# (Plan §1) — but it IS priced (the conferred gather half, life-agent docs/ask-as-connection.md §4/§7):
# by the engine's gather VOI (`grow_value`/`best_grow`/`recovery_g`, src/gather_voi.jl) against a
# SEPARATE structure-BMA recovery belief `g = P(recover | sensors)`. Grow actuators ride their own
# `grows` lane in `schedule_decide` (per-actuator `(probe, g, cost)` — no kernel over the candidate
# space, so not a registry `Transform`), self-gating on the terminal EU. The body enacts; the agent
# decides.
struct Transform
    name::String        # registry id (unique within a registry)
    probe::String       # emitted wire-probe name (the body's capability; the `applied_probes` dedup key)
    kind::Symbol        # :voi | :guard
    applies::Function   # (action::String, ctx::ScheduleCtx) -> Bool — eligibility on this decision
    kernel_fn::Function # (k::Int, cp::ChannelParams) -> Kernel — the output-likelihood model (:voi only)
    cost::Float64       # cost-in-utility, commensurate with `value` (:voi only; guards: 0.0)
end

# Request-level gates the registry's `applies` predicates read (the per-question context). Transform
# kernels/costs are baked into the Transform itself (so model tiers each carry their own rho/cost —
# Slice 2); `ctx` carries only what gates eligibility and the termination set.
struct ScheduleCtx
    era_split::Bool
    owner_scoped::Bool
    gather_rho::Float64
    gather_cost::Float64
    applied_probes::Vector{String}
end

_no_kernel(::Int, ::ChannelParams)::Kernel = error("a :guard transform has no VOI kernel")

"""
    default_registry(; gather_rho=0.0, gather_cost=0.0) -> Vector{Transform}

The v0 menu, reproducing the validated behaviour exactly:
- `recency` (`:guard`): rule out staleness before any terminal report — a count-led stale leader can
  sit ABOVE the EU bar, and reporting it without the recency re-weight is a confident-wrong
  (`gather.py` applies recency pre-decision; the daemon ports that). Fires whenever `era_split` holds.
- `corroborate_owner` (`:guard`): the owner-scoped ATTRIBUTION guard — an owner-scoped question about
  to REPORT an in-set leader must first corroborate with a subject-aware re-read, because the cheap
  per-chunk extractor is owner-CENTRIC (it reports the owner's OWN value for a relative's question).
  The risk lives outside the candidate belief, so it is mandatory, not VOI-priced.
- `corroborate_voi` (`:voi`, only when `gather_rho > 0`): the §2-A rescue — a WITHHOLDING leader may be
  rescued by a `net_voi`-gated re-read, gathered only if re-reading is expected to move `value` by more
  than its cost. Omitted when `gather_rho == 0` (no re-read budget) ⇒ the default is byte-identical.
Both corroborate entries emit the same wire probe `"corroborate"`, so `applied_probes` disables both
once either fires (the loop terminates).
"""
function default_registry(; gather_rho::Float64 = 0.0, gather_cost::Float64 = 0.0)::Vector{Transform}
    reg = Transform[
        Transform("recency", "recency", :guard,
                  (action, ctx) -> ctx.era_split, _no_kernel, 0.0),
        Transform("corroborate_owner", "corroborate", :guard,
                  (action, ctx) -> ctx.owner_scoped && action == "report", _no_kernel, 0.0),
    ]
    if gather_rho > 0.0
        push!(reg, Transform("corroborate_voi", "corroborate", :voi,
                  (action, ctx) -> action != "report",
                  (k, cp) -> _corroborate_kernel(k, gather_rho, cp), gather_cost))
    end
    reg
end

# The built-in eligibility predicates a wire trigger names. The MENU is data (which transforms, at
# what rho/cost — the body declares them per question), but the `applies` predicates and the kernel
# stay in the brain: the wire ships a `trigger` STRING, not a Julia closure.
const _TRIGGERS = Dict{String, Function}(
    "era_split"    => (action, ctx) -> ctx.era_split,                          # recency guard
    "owner_report" => (action, ctx) -> ctx.owner_scoped && action == "report", # attribution guard
    "below_bar"    => (action, ctx) -> action != "report",                     # the §2-A rescue gate
)

"""
    registry_from_wire(descriptors; cp=CANONICAL_CHANNEL) -> Vector{Transform}

Build a registry from the body's declared menu (Slice 2 — model-tier escalation). Each descriptor is
`{name, probe, kind, trigger, rho, cost}`: `kind` ∈ {"voi","guard"}, `trigger` ∈ keys(`_TRIGGERS`),
and a `:voi` entry's `kernel_fn` is the corroborate noisy-channel at its own `rho` (so a haiku /
sonnet / opus tier each carry their own reliability + cost, and `schedule_decide` gathers the
net_voi ARGMAX among them — the cost-efficient tier wins; sequential escalation emerges as the loop
re-prices the remaining tiers after the chosen one is applied). The body enacts the chosen tier by
its `probe` name.
"""
function registry_from_wire(descriptors; cp::ChannelParams = CANONICAL_CHANNEL)::Vector{Transform}
    map(descriptors) do d
        name    = String(d["name"])
        probe   = String(get(d, "probe", name))
        kind    = Symbol(d["kind"])
        applies = _TRIGGERS[String(d["trigger"])]
        rho     = Float64(get(d, "rho", 0.0))
        cost    = Float64(get(d, "cost", 0.0))
        kfn     = kind === :voi ? ((k, cpp) -> _corroborate_kernel(k, rho, cpp)) : _no_kernel
        Transform(name, probe, kind, applies, kfn, cost)
    end
end

"""
    schedule(state, k, u_bar, registry, ctx; cp=CANONICAL_CHANNEL, grows=[])
        -> (effector, report_index, probe, target, eu)

The uniform VOI scheduler. Decide terminally once, then: every eligible, unapplied **guard** fires
first (mandatory, registry order — it defends a risk the belief can't price); then the eligible,
unapplied **:voi** transforms are priced by `net_voi` and the **grow** actuators by `grow_value`
(the engine gather VOI — `g` per actuator, already read from the gather structure-BMA), and the
overall argmax is gathered iff it clears 0. None fire ⇒ the terminal decision stands.
`applied_probes` (keyed on the emitted `probe`, body-held and resent) makes each probe — registry
and grow alike — fire at most once ⇒ the loop terminates. Parity: an empty / fully-applied registry
with no `grows` returns exactly `decide_full`'s terminal tuple. Grow self-gates on the terminal EU
(`grow_value(g, u_correct, eu, cost)`): a confident report prices ≈ −cost, so no `p_none` branch.
"""
function schedule_decide(state::CategoricalMeasure, k::Int, u_bar::AbstractDict,
                  registry::Vector{Transform}, ctx::ScheduleCtx;
                  cp::ChannelParams = CANONICAL_CHANNEL,
                  grows::AbstractVector = Tuple{String, Float64, Float64}[])
    action, report_index, eu = decide_full(state, k, u_bar; cp = cp)
    leader() = provisional_leader(state, k, u_bar; cp = cp)
    # Guards first: mandatory, registry order. A guard prices an out-of-model risk VOI is blind to.
    for t in registry
        t.kind === :guard || continue
        t.probe in ctx.applied_probes && continue
        t.applies(action, ctx) && return ("gather", nothing, t.probe, leader(), eu)
    end
    # :voi transforms: price each eligible, unapplied one; keep the net_voi argmax if > 0.
    best = nothing; best_nv = 0.0
    for t in registry
        t.kind === :voi || continue
        t.probe in ctx.applied_probes && continue
        t.applies(action, ctx) || continue
        nv = voi_gather(state, k, u_bar, t.kernel_fn(k, cp), collect(Float64, 0:(k - 1)), t.cost; cp = cp)
        nv > best_nv && (best_nv = nv; best = t)
    end
    # Grow actuators: the engine gather VOI over the unapplied ones; one EU comparison vs :voi.
    unapplied = [(p, g, c) for (p, g, c) in grows if !(String(p) in ctx.applied_probes)]
    grow_probe, grow_v = best_grow(unapplied, Float64(u_bar["u_correct"]), eu)
    if grow_probe !== nothing && grow_v > best_nv
        return ("gather", nothing, grow_probe, leader(), eu)
    end
    best === nothing && return (action, report_index, nothing, nothing, eu)
    ("gather", nothing, best.probe, leader(), eu)
end

"""
    gather_decide(state, k, u_bar; era_split=false, owner_scoped=false, gather_rho=0.0,
                  gather_cost=0.0, applied_probes=String[], cp=CANONICAL_CHANNEL)
        -> (effector, report_index, probe, target, eu)

Thin wrapper: build the `default_registry` from the request flags and run `schedule`. Kept as the
daemon's entry point so every existing caller/test is unchanged; the menu-as-data refactor lives in
`schedule`. `report_index`/`probe`/`target` are `nothing` where inapplicable.
"""
function gather_decide(state::CategoricalMeasure, k::Int, u_bar::AbstractDict;
                       era_split::Bool = false,
                       owner_scoped::Bool = false,
                       gather_rho::Float64 = 0.0,
                       gather_cost::Float64 = 0.0,
                       applied_probes::AbstractVector{<:AbstractString} = String[],
                       cp::ChannelParams = CANONICAL_CHANNEL,
                       grows::AbstractVector = Tuple{String, Float64, Float64}[])
    reg = default_registry(; gather_rho = gather_rho, gather_cost = gather_cost)
    ctx = ScheduleCtx(era_split, owner_scoped, gather_rho, gather_cost,
                      collect(String, applied_probes))
    schedule_decide(state, k, u_bar, reg, ctx; cp = cp, grows = grows)
end

end # module AnswerBrain
