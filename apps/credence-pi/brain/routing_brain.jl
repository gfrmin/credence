# Role: brain (thin shim)
"""
    routing_brain.jl — credence-pi's thin shim over the engine's routing stdlib.

After decouple Move 4 the routing brain (decisions + the online confound-learning EM) lives
in the engine (`src/routing.jl`: `RoutingState` / `route` / `route_eu` / `escalation_next` /
`posterior_accuracy` / `route_outcome!` / `decode_correctness` / `route_decide` /
`escalate_decide` / `reconstruct_*_from_data`). This module re-exports them under the names
the daemon, tests, and eval use, and keeps ONLY the genuinely consumer-side wiring:
`wire_routing!` (reads the BDSL env + installs the route/escalate closures), and the
path-reading reconstruction wrappers (the embedding daemon reads its committed counts files;
the wire path ships the counts as inline data to `routing_init`). NO `optimise` /
`LinearCombination` / `Projection` / `ProductMeasure` import remains — the leak-closed
witness, mirroring `feature_brain.jl`.
"""
module RoutingBrain

using Main.Credence: RoutingState, EmissionBelief, LatencyBelief, route, route_eu,
    escalation_next, posterior_accuracy, route_outcome!, decode_correctness, latency_at,
    route_decide, escalate_decide, reconstruct_latency_from_data,
    reconstruct_routing_tops_from_data, _ctx_key, MixturePrevision
# `..FeatureBrain` (the sibling submodule), NOT `Main.FeatureBrain`: resolves both when the
# eval includes both brains at Main and when the daemon includes both inside `module Server`.
using ..FeatureBrain: build_model_from_decls
using JSON3

export route, route_eu, escalation_next, posterior_accuracy, wire_routing!,
    RoutingState, EmissionBelief, route_outcome!, decode_correctness,
    LatencyBelief, reconstruct_latency, latency_at, _ctx_key

# ── Path-reading reconstruction wrappers (co-released-image-only) ──────────────────────────
# The embedding daemon reads its committed counts files; the wire consumer ships the parsed
# content to the engine's `reconstruct_*_from_data` instead (the skin never touches host FS).

reconstruct_latency(path) = reconstruct_latency_from_data(JSON3.read(read(String(path), String)))

function _reconstruct_routing_tops(model, K, warm_path)
    (warm_path === nothing || isempty(string(warm_path))) &&
        return reconstruct_routing_tops_from_data(model, K, nothing)
    isfile(string(warm_path)) ||
        (@warn "routing warm brain not found; cold start" path = string(warm_path);
         return reconstruct_routing_tops_from_data(model, K, nothing))
    try
        data = JSON3.read(read(string(warm_path), String))
        tops = reconstruct_routing_tops_from_data(model, K, data)
        @info "routing warm brain reconstructed" path = string(warm_path) models = K
        tops
    catch e
        @warn "routing warm brain failed to load; cold start" path = string(warm_path) error = e
        reconstruct_routing_tops_from_data(model, K, nothing)
    end
end

"""
    wire_routing!(env; warm_path) -> RoutingState | nothing

Inject `env[:route-decide]`/`[:escalate-decide]` from the declared routing manifest
(routing.bdsl) + per-profile reward (utility.bdsl). The `.bdsl` carries the DECLARED DATA
(roster, costs, feature spaces, reward); this builds the typed `RoutingState` (engine struct)
and installs closures over the lifted `route_decide`/`escalate_decide`, so the daemon
call-sites are unchanged in shape. INERT (returns `nothing`) unless `routing-models` is
declared. The K per-model `StructureBMA` posteriors are warm-seeded from `warm_path`; held
LIVE in the returned `RoutingState` so `route_outcome!` updates flow into later decisions.
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

    # TIME coordinate (profile dial): w-time = $/sec of wall-clock. 0.0 ⇒ time-blind. Latency
    # belief is opt-in by file presence; absent ⇒ nothing ⇒ no time term.
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

    # Emission prior: declared (`emission-prior`) auditable data, else the engine's directional
    # default (EmissionBelief() with no arg).
    ep = get(env, Symbol("emission-prior"), nothing)
    emission = (ep isa AbstractVector && length(ep) == 4) ?
        EmissionBelief((Float64(ep[1]), Float64(ep[2]), Float64(ep[3]), Float64(ep[4]))) :
        EmissionBelief()

    rfeat = get(env, Symbol("routing-features"), nothing)
    rfeat isa AbstractVector ||
        error("routing: routing-models declared but routing-features missing (declare it in routing.bdsl)")
    rmodel = build_model_from_decls(rfeat)
    tops = _reconstruct_routing_tops(rmodel, K, warm_path)

    rt = RoutingState(rmodel, tops, Dict{String, MixturePrevision}(),
                      emission, names, providers,
                      model_ids, costs, reward, w_time, latency)

    # The daemon's routing call-sites: closures over the LIVE RoutingState (route_outcome!
    # mutates it). All arithmetic is engine-side inside route_decide/escalate_decide.
    env[Symbol("route-decide")] = (features, roster = nothing, profile = nothing) ->
        route_decide(rt, features, roster, profile)

    env[Symbol("escalate-decide")] =
        (features, roster = nothing, tried = Int[], reward = nothing, profile = nothing) ->
            escalate_decide(rt, features, roster, tried,
                            reward === nothing ? rt.reward : Float64(reward), profile)

    rt
end

end # module RoutingBrain
