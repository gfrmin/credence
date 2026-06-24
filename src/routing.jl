# routing.jl — EU-max model routing, lifted into the engine (decouple Move 4).
#
# The probabilistic body of what was apps/credence-pi/brain/routing_brain.jl: the routing
# DECISIONS (route / route_eu / escalation_next over a ProductMeasure of per-model
# belief-at-context views) and the ONLINE confound-learning (decode_correctness + the
# coupled-EM route_outcome! over ρ/σ emission Betas + the Gamma latency belief). It REUSES
# the structure-BMA substrate already lifted in structure_bma.jl
# (build_structure_prior / structure_observe / structure_observe_soft / belief_at_context /
# context_from_features) — one shared feature schema, K trained posteriors `tops[a]` (label
# "model a was correct"). Composes only Tier-1 ops; no new frozen type, no new
# axiom-constrained function. Included inside `module Ontology` after structure_bma.jl, so
# every symbol is in scope (no Main.Credence / ..FeatureBrain qualifiers).
#
# The decision: given a request's features X and a roster of candidate models, pick the model
# maximising expected welfare EU(a|X) = reward·E[θ_a|X] − cost_a — the SAME EU-max the
# governance brain runs (decide_with_voi), only the action set is the model roster and the
# payoff is reward·accuracy − cost. The argmax is the single canonical `optimise` (Invariant 1).

# ── Latency belief: E[time | model, X] = E[turns|X]·s̄, the TIME coordinate ──────────────
struct LatencyBelief
    time_mean::Dict{Tuple{String, String}, Float64}   # (model_id, ctx_key) -> E[time] seconds
end

# Reconstruct E[time] per (model, context) from already-parsed counts data (the skin/shim
# does the JSON parsing; the engine never reads the host FS). Shape:
# { "turns_prior":[α0,β0], "per_model":[ {"model_id":..,"rate_s":..,
#   "contexts":[ {"ctx":["short"],"sum_turns":412,"n_obs":30} ]} ] }. The posterior
# Gamma(α0+Σt, β0+n) is order-independent, so reconstruction is exact (no Serialization).
function reconstruct_latency_from_data(data)::LatencyBelief
    α0, β0 = Float64(data["turns_prior"][1]), Float64(data["turns_prior"][2])
    tm = Dict{Tuple{String, String}, Float64}()
    for pm in data["per_model"]
        id = String(pm["model_id"]); rate = Float64(pm["rate_s"])
        for ctx in pm["contexts"]
            g = GammaPrevision(α0 + Float64(ctx["sum_turns"]), β0 + Float64(ctx["n_obs"]))
            etime = Float64(expect(g, Identity())) * rate     # E[turns]=α/β, ×s̄ ⇒ E[time]
            tm[(id, _ctx_key([String(c) for c in ctx["ctx"]]))] = etime
        end
    end
    LatencyBelief(tm)
end

# E[time|model,X] seconds, or 0.0 for an unknown (model, context) ⇒ that candidate carries no
# time term (conservative: time never penalises a model we have no latency belief for).
latency_at(lb::LatencyBelief, model_id::AbstractString, X::AbstractVector) =
    get(lb.time_mean, (String(model_id), _ctx_key(X)), 0.0)
latency_at(::Nothing, ::AbstractString, ::AbstractVector) = 0.0

# Per-action EU functional: reward·θ_a − cost_a, over the joint belief's a-th component
# (Projection(a) = θ_a). reward/cost are declared utility DATA; multiplying them into
# LinearCombination coefficients is coefficient construction, not probability arithmetic.
# `time_cost` = w_time·E[time|a,X] folds into the SAME offset as −cost (E[time] is a known
# scalar at decision time, not co-varying with θ). time_cost=0 ⇒ −(cost+0)==−cost.
_eu_functional(a::Int, reward::Float64, cost::Float64, time_cost::Float64 = 0.0) =
    LinearCombination(Tuple{Float64, TestFunction}[(reward, Projection(a))], -(cost + time_cost))

# Joint belief over all K models at context X: a ProductMeasure of each model's
# belief-at-context view (independence-across-models made explicit). Action a's EU reads its
# own component via Projection(a); the others integrate out.
function _joint_at(model::StructureBMA, tops::AbstractVector, X::AbstractVector)
    K = length(tops)
    cells = Measure[wrap_in_measure(belief_at_context(model, tops[a], X)) for a in 1:K]
    ProductMeasure(cells), K
end

function _fpa(K::Int, reward::Real, costs::AbstractVector,
             w_time::Real = 0.0, times = nothing)
    r = Float64(reward); wt = Float64(w_time)
    Dict{Int, LinearCombination}(
        a => _eu_functional(a, r, Float64(costs[a]),
                            times === nothing ? 0.0 : wt * Float64(times[a])) for a in 1:K)
end

"""
    route(model, tops, X, costs, reward) -> Int

Return the 1-based index of the EU-max model for a request with context `X`. `tops[a]` is
model a's `StructureBMA` posterior over P(correct|·); `costs[a]` its per-call cost; `reward`
the profile's dollar value of a correct answer. The argmax is the single canonical `optimise`
over the joint per-model belief (Invariant 1).
"""
function route(model::StructureBMA, tops::AbstractVector, X::AbstractVector,
               costs::AbstractVector, reward::Real; w_time::Real = 0.0, times = nothing)
    length(tops) == length(costs) ||
        error("route: tops/costs length mismatch ($(length(tops)) vs $(length(costs)))")
    (times === nothing || length(times) == length(tops)) ||
        error("route: tops/times length mismatch ($(length(tops)) vs $(length(times)))")
    joint, K = _joint_at(model, tops, X)
    optimise(joint, collect(1:K), _fpa(K, reward, costs, w_time, times))
end

"""
    route_eu(model, tops, X, costs, reward) -> (Int, Float64)

`route` plus the EU of the chosen model (reward·E[θ_a|X] − cost_a), re-read through `expect`
of the chosen action's functional — never recomputed by hand.
"""
function route_eu(model::StructureBMA, tops::AbstractVector, X::AbstractVector,
                  costs::AbstractVector, reward::Real; w_time::Real = 0.0, times = nothing)
    joint, K = _joint_at(model, tops, X)
    fpa = _fpa(K, reward, costs, w_time, times)
    a = optimise(joint, collect(1:K), fpa)
    a, Float64(expect(joint, fpa[a]))
end

"""
    posterior_accuracy(model, top, X) -> Float64

E[θ | X] = the posterior-mean accuracy of one model at context X, read through `expect` of
`Identity` over its belief-at-context view. Inspection only; never used to decide.
"""
posterior_accuracy(model::StructureBMA, top::MixturePrevision, X::AbstractVector) =
    Float64(expect(belief_at_context(model, top, X), Identity()))

# Stop action = the zero functional (EU 0: decline to spend). Peer of `_eu_functional`.
const _STOP_FUNCTIONAL = LinearCombination(Tuple{Float64, TestFunction}[], 0.0)

"""
    escalation_next(model, tops, X, costs_X, reward, tried) -> Int

Observe-then-escalate routing. Among tiers not yet `tried`, cheapest first (`costs_X`
ascending), return the cheapest whose single-step EU to try is at least the EU of stopping
(`optimise` prefers "try" to "stop"), else 0 (STOP). The {try, stop} choice is the single
canonical `optimise` over the tier's belief-at-context. MYOPIC (one rung at a time).
"""
function escalation_next(model::StructureBMA, tops::AbstractVector, X::AbstractVector,
                         costs_X::AbstractVector, reward::Real, tried;
                         w_time::Real = 0.0, times_X = nothing)
    r = Float64(reward); wt = Float64(w_time)
    for a in eachindex(tops)                       # costs_X ascending ⇒ cheapest-first
        a in tried && continue
        tc = times_X === nothing ? 0.0 : wt * Float64(times_X[a])   # w_time·E[time|a,X]
        belief = ProductMeasure(Measure[wrap_in_measure(belief_at_context(model, tops[a], X))])
        tryf = LinearCombination(Tuple{Float64, TestFunction}[(r, Projection(1))], -(Float64(costs_X[a]) + tc))
        return optimise(belief, [1, 2], Dict(1 => tryf, 2 => _STOP_FUNCTIONAL)) == 1 ? a : 0
    end
    0
end

# ── Warm routing belief: per-model COUNTS, reconstructed via structure_observe ──
# K posteriors (one per model) from one shared schema. Bayesian updating is order-independent,
# so the warm belief depends only on each (model, context)'s correct/incorrect counts; the
# skin/shim ships the parsed counts and the engine replays structure_observe (version-stable).
# JSON shape: { "per_model": [ { "contexts": [ {"ctx":["short"], "n1":44, "n0":6} ] }, … ] }
# where n1 = correct, n0 = incorrect. `data === nothing` ⇒ K cold priors.
function reconstruct_routing_tops_from_data(model::StructureBMA, K::Int, data)
    cold() = MixturePrevision[build_structure_prior(model) for _ in 1:K]
    data === nothing && return cold()
    pm = data["per_model"]
    length(pm) == K ||
        error("routing warm brain has $(length(pm)) models but the roster has $K")
    tops = MixturePrevision[]
    for entry in pm
        top = build_structure_prior(model)
        for c in entry["contexts"]
            ctx = String[String(v) for v in c["ctx"]]
            for _ in 1:Int(c["n1"]); top = structure_observe(model, top, ctx, 1); end
            for _ in 1:Int(c["n0"]); top = structure_observe(model, top, ctx, 0); end
        end
        push!(tops, top)
    end
    tops
end

# ── Online correctness learning: latent per-turn correctness + learned confounds ──
# We never observe model correctness directly; we observe a per-turn exec signal e (did the
# proposed call execute cleanly), a NOISY emission of the latent C = "the turn was correct",
# confounded by TOOL RELIABILITY. We model the confound explicitly and learn it:
#   θ_a(X) = P(C=1 | model a, context X)   — the routing belief (its own prior for C)
#   ρ_X    = P(e=1 | C=1)                  — tool reliability
#   σ_X    = P(e=1 | C=0)                  — false-success
# ρ_X, σ_X are SHARED across models at a context (the identification lever) — LATENT Beta
# beliefs, updated only through `condition`. Per turn: decode π = P(C=1|e) with θ_a as prior,
# then ONE coupled coordinate step (the EM the constitution treats as a computational
# strategy): routing belief soft-counted by (r,w); emissions M-stepped by (e, π) / (e, 1−π).

# Default weakly-informative DIRECTIONAL emission prior: E[ρ0]=2/3 > E[σ0]=1/3. Only the
# inequality is load-bearing (it breaks the EM label-symmetry); magnitudes are weak.
const _DEFAULT_EMISSION_PRIOR = (2.0, 1.0, 1.0, 2.0)  # (ρα, ρβ, σα, σβ)

mutable struct EmissionBelief
    rho_cells::Dict{String, BetaPrevision}     # P(e=1|C=1) per context-key, lazily populated
    sigma_cells::Dict{String, BetaPrevision}   # P(e=1|C=0) per context-key
    rho0::BetaPrevision                        # directional prior, E[ρ0] > E[σ0]
    sigma0::BetaPrevision
end

EmissionBelief(prior::NTuple{4, Float64} = _DEFAULT_EMISSION_PRIOR) =
    EmissionBelief(Dict{String, BetaPrevision}(), Dict{String, BetaPrevision}(),
                   BetaPrevision(prior[1], prior[2]), BetaPrevision(prior[3], prior[4]))

# The live routing belief: per-model posteriors `tops` (MUTATED by route_outcome!) and the
# shared emission belief. An immutable StructureBMA descriptor + mutable beliefs as SEPARATE
# objects (Invariant 3 holds at representation granularity; the bundling is handle-level).
mutable struct RoutingState
    model::StructureBMA
    tops::Vector{MixturePrevision}              # warm beliefs for the DEFAULT roster (positional)
    extra_tops::Dict{String, MixturePrevision}  # beliefs for models NOT in the default roster —
                                                # the user's OWN models: cold prior on first sight,
                                                # then learned online; keyed by model id
    emission::EmissionBelief
    names::Vector{String}
    providers::Vector{String}
    model_ids::Vector{String}
    costs::Vector{Float64}
    reward::Float64
    w_time::Float64                             # profile weight on time ($/sec of wall-clock)
    latency::Union{LatencyBelief, Nothing}      # learned E[time|model,X]; nothing ⇒ time-blind
end

# Fetch model_id's belief plus a setter to write an update back. KNOWN models (the default
# roster) live in the positional `tops`; the user's OWN models live in `extra_tops` (cold
# build_structure_prior on first sight, then learned). No error on an unknown model.
function _belief_slot(rt::RoutingState, model_id::AbstractString)
    a = findfirst(==(String(model_id)), rt.model_ids)
    a !== nothing && return (rt.tops[a], top -> (rt.tops[a] = top))
    id = String(model_id)
    cur = get!(() -> build_structure_prior(rt.model), rt.extra_tops, id)
    (cur, top -> (rt.extra_tops[id] = top))
end

_ctx_key(X::AbstractVector) = join(string.(X), "|")

# WeightedBernoulli kernel for the emission Beta updates (fractional pseudo-counts). The
# conjugate update keys on likelihood_family; generate/log_density are not consulted.
_emission_kernel() = Kernel(Interval(0.0, 1.0), Finite([0, 1]), theta -> theta,
                            (h, o) -> 0.0; likelihood_family = WeightedBernoulli())

# Emission likelihoods of the exec signal e under each correctness hypothesis:
# r = P(e | C=1), w = P(e | C=0), from the emission belief means (mean = the integrated
# likelihood, since P(e|C) is linear in ρ/σ).
function _emission_likelihoods(em::EmissionBelief, key::AbstractString, e::Bool)
    ρ̄ = mean(get(em.rho_cells,   key, em.rho0))
    σ̄ = mean(get(em.sigma_cells, key, em.sigma0))
    e ? (ρ̄, σ̄) : (1.0 - ρ̄, 1.0 - σ̄)
end

"""
    decode_correctness(em, θ̄, key, e) -> (r, w, π)

The signal→correctness likelihood: r = P(e|C=1), w = P(e|C=0) from the emission belief, and
π = P(C=1 | e) = r·θ̄/(r·θ̄ + w·(1−θ̄)) with the routing belief θ̄ as the coherent prior. π is
the soft correctness label; (r,w) is the virtual evidence the routing belief conditions on
(SoftBernoulli). Pure readout — `mean` and Bayes, no mutation. `θ̄` is the MIXTURE-level
E[θ|X] (`posterior_accuracy`), not a per-cell α/(α+β) — the closed form is what keeps the
emission M-step weighted by the coherent per-turn correctness (and keeps replay bit-exact).
"""
function decode_correctness(em::EmissionBelief, θ̄::Float64, key::AbstractString, e::Bool)
    r, w = _emission_likelihoods(em, key, e)
    denom = r * θ̄ + w * (1.0 - θ̄)
    π = denom > 0.0 ? r * θ̄ / denom : θ̄
    (r, w, π)
end

# Update one reliability cell. `into_rho` selects ρ (the C=1 cell) vs σ (the C=0 cell); the
# outcome is the exec signal e, credited `weight` pseudo-counts (WeightedBernoulli).
function _update_emission!(em::EmissionBelief, key::AbstractString, into_rho::Bool,
                           e::Bool, weight::Float64)
    weight > 0.0 || return em
    d  = into_rho ? em.rho_cells : em.sigma_cells
    p0 = into_rho ? em.rho0 : em.sigma0
    cur = get(d, key, p0)
    d[key] = condition(cur, _emission_kernel(), (e ? 1 : 0, weight))
    em
end

"""
    route_outcome!(rt, model_id, features, success; human=nothing) -> RoutingState

Learn from one routed turn's per-turn outcome. `features` → context via
`context_from_features`. `success` = the proposed call executed cleanly (the exec signal e).
With no `human` label, decode the latent correctness from e (confound-aware) and take one
coupled coordinate step: the routing belief conditions on the virtual evidence (mean-exact
soft-count via `structure_observe_soft`); the shared emission belief takes the EM M-step
(ρ←(e,π), σ←(e,1−π) through `condition`). A `human` approve/reject makes C known — a hard
`structure_observe` and a unit-weight emission update (the gold anchor). The single learning
mechanism is `condition` throughout; the θ↔emission coupling is coordinate computation
(a strategy). Mutates `rt` in place.
"""
function route_outcome!(rt::RoutingState, model_id::AbstractString, features::AbstractDict,
                        success::Bool; human::Union{Nothing, Bool} = nothing)
    X = context_from_features(rt.model, features)
    key = _ctx_key(X)
    e = success
    top, setby = _belief_slot(rt, model_id)                 # known slot or the user's own model
    if human !== nothing
        C = human ? 1 : 0                                   # known correctness
        setby(structure_observe(rt.model, top, X, C))
        _update_emission!(rt.emission, key, C == 1, e, 1.0)
    else
        θ̄ = posterior_accuracy(rt.model, top, X)            # E[θ|X] — the decode prior
        r, w, π = decode_correctness(rt.emission, θ̄, key, e)
        setby(structure_observe_soft(rt.model, top, X, r, w))
        _update_emission!(rt.emission, key, true,  e, π)        # ρ gets weight π
        _update_emission!(rt.emission, key, false, e, 1.0 - π)  # σ gets weight 1−π
    end
    rt
end

# Per-request profile override: the user's utility weights, shipped by the body per request.
# Returns the override for `key` if present, else the wired default. Preference DATA only.
_pget(profile, key::AbstractString, default::Float64)::Float64 =
    (profile isa AbstractDict && haskey(profile, key) && profile[key] !== nothing) ?
        Float64(profile[key]) : default

"""
    route_decide(rt, features, roster, profile) -> Dict | nothing

Resolve a routing decision over the LIVE roster (the user's actual models, sent per request)
or the declared default roster. Returns `nothing` — body keeps OpenClaw's model — when fewer
than 2 candidates exist. Per-request profile overrides reward/w_time with no daemon restart.
"""
function route_decide(rt::RoutingState, features, roster, profile = nothing)
    names, providers, ids, costs, tops = _resolve_roster(rt, roster)
    length(ids) >= 2 || return nothing
    reward = _pget(profile, "reward", rt.reward)        # quality coordinate (per-request override)
    w_time = _pget(profile, "w_time", rt.w_time)        # time coordinate (per-request override)
    X = context_from_features(rt.model, features)
    times = Float64[latency_at(rt.latency, ids[a], X) for a in eachindex(ids)]   # E[time|a,X], 0 if unknown
    a = route(rt.model, tops, X, costs, reward; w_time = w_time, times = times)
    Dict{String, Any}("model" => ids[a], "provider" => providers[a], "name" => names[a])
end

"""
    escalate_decide(rt, features, roster, tried, reward, profile) -> Dict | nothing

One escalation step over the LIVE roster: the cheapest not-yet-`tried` rung whose myopic
try-EU clears the stop gate, else `nothing` (STOP). `tried` carries cost-ascending indices;
the returned `tier_index` is in that cost-ascending space, so the host pushes it straight
back into `tried` on a failure.
"""
function escalate_decide(rt::RoutingState, features, roster, tried, reward, profile = nothing)
    names, providers, ids, costs, tops = _resolve_roster(rt, roster)
    length(ids) >= 2 || return nothing
    reward = _pget(profile, "reward", Float64(reward))   # per-request quality override
    w_time = _pget(profile, "w_time", rt.w_time)         # per-request time override
    order = sortperm(costs)                              # cheapest → dearest
    X = context_from_features(rt.model, features)
    triedset = Set{Int}(Int(t) for t in tried)          # indices in cost-ascending space
    times = Float64[latency_at(rt.latency, ids[i], X) for i in order]   # E[time], aligned to tops[order]
    a = escalation_next(rt.model, tops[order], X, costs[order], reward, triedset;
                        w_time = w_time, times_X = times)
    a == 0 && return nothing                             # no positive-EU rung ⇒ STOP
    j = order[a]
    Dict{String, Any}("model" => ids[j], "provider" => providers[j], "name" => names[j],
                      "tier_index" => a)
end

# Aligned (names, providers, ids, costs, tops) for the decision. No live roster ⇒ the declared
# default (warm beliefs). A live roster ⇒ each entry is (name provider model-id cost): a known
# model reuses its warm/learned belief, an unknown one its cold/learned belief from extra_tops.
function _resolve_roster(rt::RoutingState, roster)
    (roster === nothing || (roster isa AbstractVector && isempty(roster))) &&
        return (rt.names, rt.providers, rt.model_ids, rt.costs, rt.tops)
    roster isa AbstractVector ||
        error("route-decide: roster must be a list of (name provider model-id cost), got $(typeof(roster))")
    names = String[]; providers = String[]; ids = String[]; costs = Float64[]
    tops = MixturePrevision[]
    for m in roster
        name, provider, id, cost = _roster_entry(m)
        push!(names, name); push!(providers, provider); push!(ids, id); push!(costs, cost)
        push!(tops, _belief_slot(rt, id)[1])
    end
    (names, providers, ids, costs, tops)
end

# One roster entry off the wire: a 4-vector (name provider id cost) or a dict carrying those.
function _roster_entry(m)
    if m isa AbstractDict
        id = String(get(m, "model", get(m, "model_id", get(m, "id", ""))))
        return (String(get(m, "name", id)), String(get(m, "provider", "")), id,
                Float64(get(m, "cost", 0.0)))
    elseif m isa AbstractVector && length(m) == 4
        return (string(m[1]), string(m[2]), string(m[3]), Float64(m[4]))
    end
    error("route-decide: roster entry must be (name provider model-id cost) or {model,provider,cost}, got $(m)")
end
