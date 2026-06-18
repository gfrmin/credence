# Role: brain
"""
    routing_brain.jl — EU-max model routing over the credence-pi feature brain.

The credence-proxy decision: given a request's features X and a set of candidate
models, pick the model that maximises expected welfare

    EU(model a | X) = reward · E[θ_a | X] − cost_a

where θ_a = P(model a answers correctly | X) is the per-model belief and `reward`
is the profile's dollar value of a correct answer. This is the SAME EU-max the
governance brain runs — only the action set changes from {proceed, block, ask} to
the model roster, and the per-action payoff is reward·accuracy − cost instead of the
waste/harm payoff.

REUSE, not reimplementation. The belief is `FeatureBrain.StructureBMA`: one shared
feature schema, K trained posteriors `tops[a]` (one per model, label = "model a was
correct"), each a structure-BMA that auto-discovers which features matter. The
decision is built EXACTLY as `FeatureBrain.decide_multi` builds its multi-outcome
EU — a joint `ProductMeasure` over the per-model `belief_at_context` views, a
per-action `LinearCombination` over `Projection`s, maximised by the ONE canonical
`optimise`. No new axiom op; no raw probability arithmetic here (the `credence-lint`
brain/ rule enforces it — the EU is closed-form inside `expect`/`optimise`, and the
only arithmetic below builds LinearCombination coefficients out of declared utility
data: reward and per-model cost).

Why per-model posteriors and not one joint belief: each "is model a correct?" is its
own Bernoulli-labelled prediction problem, so each model gets its own StructureBMA
posterior (mirroring credence-pi's per-outcome brains: waste vs harm). At decision
time the K views are assembled into the joint the argmax integrates over — the
independence across models is the ProductMeasure, exactly as the waste⊗harm joint in
`decide_multi`.
"""
module RoutingBrain

using Main.Credence: MixturePrevision, BetaPrevision, GammaPrevision, Projection,
    LinearCombination, TestFunction, Identity, Kernel, Interval, Finite, WeightedBernoulli,
    mean, condition, expect, wrap_in_measure
import Main.Credence.Ontology: optimise, ProductMeasure, Measure
# `..FeatureBrain` (the sibling submodule), NOT `Main.FeatureBrain`: this resolves both
# when the eval includes both brains at Main (siblings of Main) and when the daemon
# includes both inside `module Server` (siblings of Server). FeatureBrain itself reaches
# Credence via the always-Main `Main.Credence`, so only this sibling ref needs to be relative.
using ..FeatureBrain: StructureBMA, belief_at_context, context_from_features,
    build_model_from_decls, build_prior, observe, observe_soft
using JSON3

export route, route_eu, escalation_next, posterior_accuracy, wire_routing!,
    RoutingState, EmissionBelief, route_outcome!, decode_correctness,
    LatencyBelief, reconstruct_latency, latency_at

# ── Latency belief: E[time | model, X] = E[turns|X]·s̄, the TIME coordinate ─────────
#
# Time is the profile coordinate that lets a user trade wall-clock against money/quality.
# It is a LEARNED belief (you cannot know a call's duration before making it): the SAME
# Poisson-Gamma "turns" belief the dominance eval already uses for E[cost] (tb_dominance.jl
# `CostBelief`), reused for E[time]. E[time|model,X] = E[turns|model,X]·s̄_model, where the
# per-model turns posterior is a Gamma (reconstructed below) read by `expect(·,Identity)`=α/β,
# and s̄_model is the measured seconds/turn. The decision folds w_time·E[time] into the EU
# offset (route/escalation_next), so a slow-but-cheap model loses to a fast one exactly when
# the user values their seconds enough — the time/money trade-off, via the one `optimise`.
#
# Version-stable like the other warm artifacts: ship the Gamma sufficient statistic
# (sum_turns, n_obs) per (model, context) + the measured rate_s per model + the prior; the
# posterior Gamma(α0+Σt, β0+n) is order-independent, so reconstruction is exact (no
# Serialization). E[time] is computed once at load through `expect` and cached per cell.
struct LatencyBelief
    time_mean::Dict{Tuple{String, String}, Float64}   # (model_id, ctx_key) -> E[time] seconds
end

# Reconstruct E[time] per (model, context) from the counts-JSON. Shape:
# { "turns_prior":[α0,β0], "per_model":[ {"model_id":..., "rate_s":..,
#   "contexts":[ {"ctx":["short"], "sum_turns":412, "n_obs":30 } ]} ] }
function reconstruct_latency(path)::LatencyBelief
    data = JSON3.read(read(String(path), String))
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

# Per-action EU functional: reward·θ_a − cost_a, expressed over the joint belief's
# a-th component (Projection(a) = θ_a). `reward` and `cost` are declared utility DATA;
# multiplying them into LinearCombination coefficients is coefficient construction, not
# probability arithmetic (mirrors decide_multi's `(-cost*(tf+aversion), Projection(1))`).
# `time_cost` = w_time·E[time|a,X] (declared weight × learned latency, prepared upstream); it
# folds into the SAME constant offset as `-cost` — E[time] is a known scalar at decision time
# (it does not co-vary with the routing posterior θ), so it belongs in the offset, not a new
# Projection. time_cost=0 ⇒ `-(cost+0.0)`==`-cost` ⇒ bit-identical to the pre-time functional.
_eu_functional(a::Int, reward::Float64, cost::Float64, time_cost::Float64 = 0.0) =
    LinearCombination(Tuple{Float64, TestFunction}[(reward, Projection(a))], -(cost + time_cost))

# Build the joint belief over all K models at context X: a ProductMeasure of each
# model's belief-at-context view (the independence-across-models assumption made
# explicit, exactly as decide_multi joins the waste and harm beliefs). The action's
# EU then reads its own component via Projection(a) — the other components integrate
# out, so EU(a) depends only on model a's belief, as it must.
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

Return the 1-based index of the EU-max model for a request with context `X`.
`tops[a]` is model a's `StructureBMA` posterior over P(correct|·); `costs[a]` its
per-call cost; `reward` the profile's dollar value of a correct answer. The argmax
is the single canonical `optimise` over the joint per-model belief (Invariant 1).

Routing is non-degenerate for k ≥ 2 models: the per-profile argmax over
reward·E[θ_a|X] − cost_a is a different model for different reward, so no single
fixed table is the Bayes rule for more than one profile — the Wald complete-class
core of the dominance proof.
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

`route` plus the EU of the chosen model (reward·E[θ_a|X] − cost_a), for reporting.
The EU is `expect` of the chosen action's functional — the same number `optimise`
maximised — so this re-reads through the canalised path, never recomputing it by hand.
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

E[θ | X] = the posterior-mean accuracy of one model at context X, read through
`expect` of `Identity` over its belief-at-context view. The public way to inspect a
model's learned accuracy (for reporting / the structure-posterior diagnostic); never
used to make a routing decision — that is `route`'s job, through `optimise`.
"""
posterior_accuracy(model::StructureBMA, top::MixturePrevision, X::AbstractVector) =
    Float64(expect(belief_at_context(model, top, X), Identity()))

# Stop action = the zero functional (EU 0: decline to spend). Peer of `_eu_functional`.
const _STOP_FUNCTIONAL = LinearCombination(Tuple{Float64, TestFunction}[], 0.0)

"""
    escalation_next(model, tops, X, costs_X, reward, tried) -> Int

Observe-then-escalate routing (the deployable strategy that wins the dominance eval when
up-front prediction can't — features don't determine the capability boundary, but observing
a failure does). Among tiers not yet `tried`, cheapest first (`costs_X` ascending), return
the cheapest whose single-step EU to try is at least the EU of stopping — i.e. `optimise`
prefers "try" to "stop" — else 0 (STOP). The host runs the returned tier, observes success
via its verifier, and on failure calls again with that tier added to `tried`.

The {try, stop} choice is the SINGLE canonical `optimise` (Invariant 1) over the tier's
belief-at-context: try-functional = reward·E[θ_a|X] − cost (an `Identity` LinearCombination,
the per-tier analogue of `route`'s Projection one), stop-functional = constant 0. Cost is
context-dependent prepared utility data (`costs_X[a]` = E[cost|a,X]; see `route`). MYOPIC —
one rung at a time, ignoring the option value of still-dearer rungs (conservative); the
exact sequential value is a future refinement. This is the ONE escalation decision; the eval
calls it rather than reimplementing the gate (no host-side decision mechanism).
"""
function escalation_next(model::StructureBMA, tops::AbstractVector, X::AbstractVector,
                         costs_X::AbstractVector, reward::Real, tried;
                         w_time::Real = 0.0, times_X = nothing)
    r = Float64(reward); wt = Float64(w_time)
    for a in eachindex(tops)                       # costs_X ascending ⇒ cheapest-first
        a in tried && continue
        tc = times_X === nothing ? 0.0 : wt * Float64(times_X[a])   # w_time·E[time|a,X]
        # Single-tier joint + Projection(1) — mirror `route`'s ProductMeasure path so
        # `expect` resolves to Measure×LinearCombination (a bare MixtureMeasure×TestFunction
        # is ambiguous against the LinearCombination method).
        belief = ProductMeasure(Measure[wrap_in_measure(belief_at_context(model, tops[a], X))])
        tryf = LinearCombination(Tuple{Float64, TestFunction}[(r, Projection(1))], -(Float64(costs_X[a]) + tc))
        return optimise(belief, [1, 2], Dict(1 => tryf, 2 => _STOP_FUNCTIONAL)) == 1 ? a : 0
    end
    0
end

# ── Warm routing belief: per-model COUNTS, reconstructed via `observe` ──
#
# Mirrors FeatureBrain.reconstruct_harm_posterior, but yields K posteriors (one per
# model) from one shared schema. Bayesian updating is order-independent, so the warm
# belief depends only on each (model, context) pair's correct/incorrect counts; we ship
# those as JSON and replay `observe` — version-stable (unlike Serialization). Any load
# failure falls back LOUDLY to the cold prior (a stale warm belief must not mis-route).
# JSON shape: { "per_model": [ { "contexts": [ {"ctx":["short"], "n1":44, "n0":6} ] }, … ] }
# where n1 = correct, n0 = incorrect, ctx = the prompt-length bucket.
function _reconstruct_routing_tops(model::StructureBMA, K::Int, warm_path)
    cold() = MixturePrevision[build_prior(model) for _ in 1:K]
    (warm_path === nothing || isempty(string(warm_path))) && return cold()
    isfile(string(warm_path)) ||
        (@warn "routing warm brain not found; cold start" path=string(warm_path); return cold())
    try
        data = JSON3.read(read(string(warm_path), String))
        pm = data.per_model
        length(pm) == K ||
            error("routing warm brain has $(length(pm)) models but the roster has $K")
        tops = MixturePrevision[]
        for entry in pm
            top = build_prior(model)
            for c in entry.contexts
                ctx = String[String(v) for v in c.ctx]
                for _ in 1:Int(c.n1); top = observe(model, top, ctx, 1); end
                for _ in 1:Int(c.n0); top = observe(model, top, ctx, 0); end
            end
            push!(tops, top)
        end
        @info "routing warm brain reconstructed" path=string(warm_path) models=K
        tops
    catch e
        @warn "routing warm brain failed to load; cold start" path=string(warm_path) error=e
        cold()
    end
end

# ── Online correctness learning: latent per-turn correctness + learned confounds ──
#
# The deferred online signal (ROUTING_DOMINANCE.md). We never observe model correctness
# directly; we observe a per-turn signal — did the proposed call execute cleanly (e). e is
# a NOISY emission of the latent C = "the model's turn was correct", confounded by TOOL
# RELIABILITY: a correct call can still error on a flaky tool, a wrong call can still
# execute. We model the confound explicitly and learn it, so a flaky tool is absorbed by
# the reliability latent, NOT mis-attributed to the model.
#
#   θ_a(X) = P(C=1 | model a, context X)   — the routing belief (its own prior for C)
#   ρ_X    = P(e=1 | C=1)                  — tool reliability (correct ⇒ clean exec)
#   σ_X    = P(e=1 | C=0)                  — false-success (wrong ⇒ clean exec anyway)
#
# ρ_X, σ_X are SHARED across models at a context (the environment doesn't know which model
# proposed the call) — the identification lever: the cross-model spread in observed e-rate,
# against the (oracle-anchored) θ_a, pins ρ and σ. They are LATENT Beta beliefs
# (weakly-informative directional prior E[ρ]>E[σ], refined by data — not fixed constants),
# updated only through `condition`.
#
# Per turn (no human label): decode π = P(C=1 | e) with θ_a(X) as prior and the emission
# means as likelihoods (emission uncertainty integrated via `expect`/`mean` — P(e|C) is
# linear in ρ,σ, so the mean IS the integrated likelihood). Then ONE coupled coordinate
# step (the EM the constitution treats as a computational strategy, invisible to the DSL):
#   * routing belief: observe_soft(model, top_a, X, r, w)  — mean-exact soft-count
#   * emissions (M-step): ρ_X ← (e, weight π),  σ_X ← (e, weight 1−π)   (WeightedBernoulli)
# A human approve/reject, when present, makes C KNOWN — a clean hard `observe` on θ_a and a
# unit-weight emission update — the gold anchor (no human-emission constant).

# Default weakly-informative DIRECTIONAL emission prior: E[ρ0]=2/3 > E[σ0]=1/3. Only the
# inequality is load-bearing (it breaks the EM label-symmetry); the magnitudes are weak
# (pseudo-count 3) and washed out by data. Overridable via the declared `emission-prior`
# (routing.bdsl) — auditable data, not a buried constant.
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
# shared emission belief. Captured by the `route-decide` closure, so routing reads the live
# belief — un-frozen vs v1's closure-captured constant.
mutable struct RoutingState
    model::StructureBMA
    tops::Vector{MixturePrevision}              # warm beliefs for the DEFAULT roster (positional)
    extra_tops::Dict{String, MixturePrevision}  # beliefs for models NOT in the default roster —
                                                # the user's OWN models: cold prior on first sight,
                                                # then learned online; keyed by model id
    emission::EmissionBelief
    # The DEFAULT roster (declared in routing.bdsl): the warm/known models, and the fallback
    # used when a route-request carries no live roster. The LIVE roster (the user's actual
    # OpenClaw models) arrives PER REQUEST — that is what makes routing roster-aware.
    names::Vector{String}
    providers::Vector{String}
    model_ids::Vector{String}
    costs::Vector{Float64}
    reward::Float64
    w_time::Float64                             # profile weight on time ($/sec of the user's wall-clock)
    latency::Union{LatencyBelief, Nothing}      # learned E[time|model,X]; nothing ⇒ time-blind (default)
end

# Fetch model_id's belief plus a setter to write an update back. KNOWN models (the default
# roster) live in the positional `tops`; the user's OWN models live in `extra_tops` (cold
# `build_prior` on first sight, then learned). No error on an unknown model — routing is
# roster-aware, so any model the body routes to gets a coherent belief.
function _belief_slot(rt::RoutingState, model_id::AbstractString)
    a = findfirst(==(String(model_id)), rt.model_ids)
    a !== nothing && return (rt.tops[a], top -> (rt.tops[a] = top))
    id = String(model_id)
    cur = get!(() -> build_prior(rt.model), rt.extra_tops, id)
    (cur, top -> (rt.extra_tops[id] = top))
end

_ctx_key(X::AbstractVector) = join(string.(X), "|")

# WeightedBernoulli kernel for the emission Beta updates (fractional pseudo-counts). The
# conjugate update keys on likelihood_family; generate/log_density are not consulted.
_emission_kernel() = Kernel(Interval(0.0, 1.0), Finite([0, 1]), theta -> theta,
                            (h, o) -> 0.0; likelihood_family = WeightedBernoulli())

# Emission likelihoods of the exec signal e under each correctness hypothesis:
# r = P(e | C=1), w = P(e | C=0), from the emission belief means (`mean` = the integrated
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
(SoftBernoulli). Pure readout — `mean` and Bayes, no mutation.
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

Learn from one routed turn's per-turn outcome. `features` is the route-request feature dict
(converted to the context via `context_from_features`). `success` = the proposed call executed
cleanly (the exec signal e). With no `human` label, decode the latent correctness from e
(confound-aware) and take one coupled coordinate step: the routing belief conditions on the
virtual evidence (mean-exact soft-count via `observe_soft`); the shared emission belief
takes the EM M-step (ρ←(e,π), σ←(e,1−π) through `condition`). A `human` approve/reject makes
C known: a hard `observe` on the routed model and a unit-weight emission update — the gold
anchor. The single learning mechanism is `condition` throughout; the θ↔emission coupling is
coordinate computation (a strategy), invisible to the DSL. Mutates `rt` in place.
"""
function route_outcome!(rt::RoutingState, model_id::AbstractString, features::AbstractDict,
                        success::Bool; human::Union{Nothing, Bool} = nothing)
    X = context_from_features(rt.model, features)
    key = _ctx_key(X)
    e = success
    top, setby = _belief_slot(rt, model_id)                 # known slot or the user's own model
    if human !== nothing
        C = human ? 1 : 0                                   # known correctness
        setby(observe(rt.model, top, X, C))
        _update_emission!(rt.emission, key, C == 1, e, 1.0)
    else
        θ̄ = posterior_accuracy(rt.model, top, X)            # E[θ|X] — the decode prior
        r, w, π = decode_correctness(rt.emission, θ̄, key, e)
        setby(observe_soft(rt.model, top, X, r, w))
        _update_emission!(rt.emission, key, true,  e, π)        # ρ gets weight π
        _update_emission!(rt.emission, key, false, e, 1.0 - π)  # σ gets weight 1−π
    end
    rt
end

"""
    wire_routing!(env; warm_path) -> RoutingState | nothing

Inject `env[:route-decide]` — the daemon's model-routing closure — from the declared
routing manifest (routing.bdsl) and the per-profile reward (utility.bdsl). Mirrors
`FeatureBrain.wire_brain!`: the `.bdsl` carries the DECLARED DATA (roster, costs,
feature spaces, reward); this builds the typed belief and installs the closure, so the
daemon call-site (`env[:route-decide](features)`) is unchanged in shape.

INERT (returns `nothing`, installs no closure) unless `routing-models` is declared —
so a governance-only install with no routing.bdsl is unaffected. The K per-model
`StructureBMA` posteriors are warm-seeded from `warm_path` (measured per-model accuracy);
they are held LIVE in the returned `RoutingState` and the `route-decide` closure reads them,
so `route_outcome!` updates flow into subsequent routing decisions (the deferred online
signal, now landed — per-turn decoded correctness with learned confounds; see
ROUTING_DOMINANCE.md). `route-decide` returns `Dict("model"=>id, "provider"=>provider,
"name"=>name)` for the daemon's route signal. The daemon stores the returned `RoutingState`
and drives learning via `route_outcome!`.
"""
function wire_routing!(env;
                       warm_path = get(env, Symbol("routing-brain-path"),
                                       joinpath(@__DIR__, "routing_brain.counts.json")))
    roster = get(env, Symbol("routing-models"), nothing)
    # Inert unless a roster is declared (no routing.bdsl ⇒ governance-only, unchanged).
    (roster isa AbstractVector && !isempty(roster)) || return nothing

    names = String[]; providers = String[]; model_ids = String[]; costs = Float64[]
    for m in roster
        (m isa AbstractVector && length(m) == 4) ||
            error("routing-models: each entry must be (name provider model-id cost), got $(m)")
        push!(names, string(m[1])); push!(providers, string(m[2]))
        push!(model_ids, string(m[3])); push!(costs, Float64(m[4]))
    end
    K = length(model_ids)

    reward = Float64(get(env, Symbol("correct-answer-value"), 0.02))

    # TIME coordinate (profile dial): w-time = $/sec of the user's wall-clock. 0.0 ⇒ time-blind
    # (bit-identical to the pre-time router). The latency belief is opt-in by file presence
    # (like the tail belief): absent ⇒ nothing ⇒ no time term. routing-latency-path overrides.
    w_time = Float64(get(env, Symbol("w-time"), 0.0))
    latency = let p = get(env, Symbol("routing-latency-path"),
                          joinpath(@__DIR__, "routing_latency.counts.json"))
        (p === nothing || isempty(string(p)) || !isfile(string(p))) ? nothing :
            try
                reconstruct_latency(p)
            catch e
                @warn "routing latency belief failed to load; time-blind" path = string(p) error = e
                nothing
            end
    end

    # Emission prior: declared (`emission-prior`) auditable data, else the directional default.
    ep = get(env, Symbol("emission-prior"), nothing)
    emission_prior = (ep isa AbstractVector && length(ep) == 4) ?
        (Float64(ep[1]), Float64(ep[2]), Float64(ep[3]), Float64(ep[4])) :
        _DEFAULT_EMISSION_PRIOR

    rfeat = get(env, Symbol("routing-features"), nothing)
    rfeat isa AbstractVector ||
        error("routing: routing-models declared but routing-features missing (declare it in routing.bdsl)")
    rmodel = build_model_from_decls(rfeat)
    tops = _reconstruct_routing_tops(rmodel, K, warm_path)

    rt = RoutingState(rmodel, tops, Dict{String, MixturePrevision}(),
                      EmissionBelief(emission_prior), names, providers,
                      model_ids, costs, reward, w_time, latency)

    # The daemon's routing call-site: (feature dict[, live roster]) → the chosen model's wire
    # fields, or `nothing` (inert ⇒ body keeps OpenClaw's model). Reads rt's beliefs LIVE
    # (route_outcome! mutates them); the argmax is the single canonical `optimise` inside
    # `route` (Invariant 1); the only arithmetic is reward/cost coefficient construction.
    env[Symbol("route-decide")] = (features, roster = nothing, profile = nothing) ->
        _route_decide(rt, features, roster, profile)

    # Observe-then-escalate call-site (the dominance proof's WINNING policy, now wired live):
    # (feature dict, live roster, tried-set, reward) → the next rung's wire fields + its
    # cost-ascending `tier_index` (the host adds it to `tried` on an observed failure), or
    # `nothing` (STOP — no positive-EU rung left ⇒ body keeps the current model). The gate is
    # the single canonical `optimise` inside `escalation_next` (Invariant 1); the host drives
    # the try→observe→escalate loop. Reads rt's beliefs LIVE, same as route-decide.
    env[Symbol("escalate-decide")] =
        (features, roster = nothing, tried = Int[], reward = nothing, profile = nothing) ->
            _escalate_decide(rt, features, roster, tried,
                             reward === nothing ? rt.reward : Float64(reward), profile)

    rt
end

# Resolve a routing decision over the LIVE roster (the user's actual models, sent per request)
# or the declared default roster. Returns `nothing` — body keeps OpenClaw's model — when fewer
# than 2 candidates exist (routing is a no-op with nothing to choose between), so routing-on-
# by-default is safe for a single-model install.
# Per-request profile override: the user's utility weights, sent by the body PER REQUEST (like
# the live roster), so a user switches their cost/time/quality trade-off with NO daemon restart.
# Returns the override value for `key` if the profile carries it, else the wired default. The
# profile is preference DATA the body ships; the brain still does all the EU-max (body math-free).
_pget(profile, key::AbstractString, default::Float64)::Float64 =
    (profile isa AbstractDict && haskey(profile, key) && profile[key] !== nothing) ?
        Float64(profile[key]) : default

function _route_decide(rt::RoutingState, features, roster, profile = nothing)
    names, providers, ids, costs, tops = _resolve_roster(rt, roster)
    length(ids) >= 2 || return nothing
    reward = _pget(profile, "reward", rt.reward)        # quality coordinate (per-request override)
    w_time = _pget(profile, "w_time", rt.w_time)        # time coordinate (per-request override)
    X = context_from_features(rt.model, features)
    times = Float64[latency_at(rt.latency, ids[a], X) for a in eachindex(ids)]   # E[time|a,X], 0 if unknown
    a = route(rt.model, tops, X, costs, reward; w_time = w_time, times = times)
    Dict{String, Any}("model" => ids[a], "provider" => providers[a], "name" => names[a])
end

# One escalation step over the LIVE roster: the cheapest not-yet-`tried` rung whose myopic
# try-EU clears the stop gate, else `nothing` (STOP). `tried` carries cost-ascending indices
# the host accumulated from observed failures; `reward` is the per-call profile value (defaults
# to the wired reward). The gate is `escalation_next` (the canonical {try,stop} `optimise`);
# rungs are ordered cheapest-first because `escalation_next` scans `eachindex(tops)` ascending.
# The returned `tier_index` is in that cost-ascending space, so the host pushes it straight
# back into `tried` on a failure (mirrors the eval's `push!(tried, a)` loop).
function _escalate_decide(rt::RoutingState, features, roster, tried, reward, profile = nothing)
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
# model reuses its warm/learned belief, an unknown one its cold/learned belief from extra_tops
# (instantiated on first sight). This is how the user's OWN models get routed.
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

end # module RoutingBrain
