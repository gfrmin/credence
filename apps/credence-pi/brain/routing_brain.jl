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

using Main.Credence: MixturePrevision, Projection, LinearCombination, TestFunction,
    Identity, expect, wrap_in_measure
import Main.Credence.Ontology: optimise, ProductMeasure, Measure
# `..FeatureBrain` (the sibling submodule), NOT `Main.FeatureBrain`: this resolves both
# when the eval includes both brains at Main (siblings of Main) and when the daemon
# includes both inside `module Server` (siblings of Server). FeatureBrain itself reaches
# Credence via the always-Main `Main.Credence`, so only this sibling ref needs to be relative.
using ..FeatureBrain: StructureBMA, belief_at_context, context_from_features,
    build_model_from_decls, build_prior, observe
using JSON3

export route, route_eu, posterior_accuracy, wire_routing!

# Per-action EU functional: reward·θ_a − cost_a, expressed over the joint belief's
# a-th component (Projection(a) = θ_a). `reward` and `cost` are declared utility DATA;
# multiplying them into LinearCombination coefficients is coefficient construction, not
# probability arithmetic (mirrors decide_multi's `(-cost*(tf+aversion), Projection(1))`).
_eu_functional(a::Int, reward::Float64, cost::Float64) =
    LinearCombination(Tuple{Float64, TestFunction}[(reward, Projection(a))], -cost)

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

function _fpa(K::Int, reward::Real, costs::AbstractVector)
    r = Float64(reward)
    Dict{Int, LinearCombination}(a => _eu_functional(a, r, Float64(costs[a])) for a in 1:K)
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
               costs::AbstractVector, reward::Real)
    length(tops) == length(costs) ||
        error("route: tops/costs length mismatch ($(length(tops)) vs $(length(costs)))")
    joint, K = _joint_at(model, tops, X)
    optimise(joint, collect(1:K), _fpa(K, reward, costs))
end

"""
    route_eu(model, tops, X, costs, reward) -> (Int, Float64)

`route` plus the EU of the chosen model (reward·E[θ_a|X] − cost_a), for reporting.
The EU is `expect` of the chosen action's functional — the same number `optimise`
maximised — so this re-reads through the canalised path, never recomputing it by hand.
"""
function route_eu(model::StructureBMA, tops::AbstractVector, X::AbstractVector,
                  costs::AbstractVector, reward::Real)
    joint, K = _joint_at(model, tops, X)
    fpa = _fpa(K, reward, costs)
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

"""
    wire_routing!(env; warm_path) -> NamedTuple | nothing

Inject `env[:route-decide]` — the daemon's model-routing closure — from the declared
routing manifest (routing.bdsl) and the per-profile reward (utility.bdsl). Mirrors
`FeatureBrain.wire_brain!`: the `.bdsl` carries the DECLARED DATA (roster, costs,
feature spaces, reward); this builds the typed belief and installs the closure, so the
daemon call-site (`env[:route-decide](features)`) is unchanged in shape.

INERT (returns `nothing`, installs no closure) unless `routing-models` is declared —
so a governance-only install with no routing.bdsl is unaffected. The K per-model
`StructureBMA` posteriors are warm-seeded from `warm_path` (measured per-model accuracy)
and FROZEN: v1 ships the measured belief; online learning of a correctness signal is
deferred (it needs credit assignment — see ROUTING_DOMINANCE.md). `route-decide` returns
`Dict("model"=>id, "provider"=>provider, "name"=>name)` for the daemon's route signal.
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

    rfeat = get(env, Symbol("routing-features"), nothing)
    rfeat isa AbstractVector ||
        error("routing: routing-models declared but routing-features missing (declare it in routing.bdsl)")
    rmodel = build_model_from_decls(rfeat)
    tops = _reconstruct_routing_tops(rmodel, K, warm_path)

    # The daemon's routing call-site: a feature dict → the chosen model's wire fields.
    # The argmax is the single canonical `optimise` inside `route` (Invariant 1); the
    # only arithmetic is reward/cost coefficient construction from declared utility data.
    env[Symbol("route-decide")] = (features) -> begin
        X = context_from_features(rmodel, features)
        a = route(rmodel, tops, X, costs, reward)
        Dict{String, Any}("model" => model_ids[a], "provider" => providers[a],
                          "name" => names[a])
    end

    (model = rmodel, tops = tops, names = names, providers = providers,
     model_ids = model_ids, costs = costs, reward = reward)
end

end # module RoutingBrain
