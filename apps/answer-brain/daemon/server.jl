# Role: brain
"""
    server.jl — the answer-brain daemon's HTTP decision surface (Stage 2a, move-2-design §2).

A **stateless** request/response wire over the Stage-1 brain (`brain/answer_brain.jl`). The body
(pi-mono, Stage 2b) gathers evidence, projects the parity-boundary covariates Python-side, and posts
the accumulated *abstract* observations + the owner's Ū; the daemon rebuilds the candidate posterior
and returns the chosen effector. It holds **no** per-question state — `candidate_posterior` is rebuilt
from the request's full observation vector each call (a handful of `condition`s), so isolation is by
construction and there is no belief to leak across questions (move-2-design §5 Resolution).

Invariant 1 (single reasoner): this module **transports** the brain's outputs; it computes no
probability. The effector and the reported candidate index both come from `AnswerBrain.decide_full`
(i.e. from `optimise`), never from reading `weights` here. `credences`/`p_none` are returned for the
body to *display* the withheld leader, not to select an action.

Routes:
  POST /decide   {question_id?, observations[], candidates[], rho, u_bar, channel?,
                  era_split?, owner_scoped?, applied_probes?, transforms?,
                  sensors?, grow?}
             →   {effector, report_index, value, probe, target, credences[], p_none, eu,
                  grow_g?}

The grow lane (slice 5): `sensors` is the body's bucketed feature dict; `grow` carries the shared
feature vocabulary (`features: {names, values}`) and per-actuator descriptors
(`actuators: [{probe, cost, alpha0?, beta0?, warm_counts?}]`). The daemon rebuilds each actuator's
outcome belief from the body-persisted `warm_counts` (exact — order-independence), reads the learned
`g = recovery_g(model, top, sensors)` (cold ⇒ the declared Beta prior mean), and the scheduler
prices the grow argmax against the :voi argmax, self-gating on the terminal EU. `grow_g` echoes the
per-actuator g for the body's gather-outcome log.
  GET  /manifest → {effectors:[{name,parameters:[{name,type}]}], features:[{name,spaceName}]}
  GET  /ready    → "ok"

`GET /manifest` serves the body's effector/feature vocabulary, parsed from `bdsl/*.bdsl` by the ONE
library parser (`manifest.jl`); the body verifies its registered impls against it and has no parser
of its own (move-4-design §0, §5 Q1).

The `era_split?`/`owner_scoped?`/`applied_probes?` fields drive the Move-4 gather branch
(`AnswerBrain.gather_decide`); absent ⇒ the Stage-2a terminal decision, unchanged. `effector` may
now be `gather`, with `probe`/`target` naming the forward steer (null on a terminal decision).

`decide_response(req::AbstractDict) -> Dict` is the pure handler the tests drive directly (no socket);
`start_daemon`/`stop_daemon` wrap it in HTTP for the body + a curl smoke.
"""
module Server

using Main.AnswerBrain: Obs, ChannelParams, CANONICAL_CHANNEL, candidate_posterior, gather_decide,
                        ScheduleCtx, registry_from_wire, schedule_decide
using Main.Credence: weights, build_structure_model, reconstruct_structure_prior_from_data,
                     recovery_g
using HTTP
using JSON3

include(joinpath(@__DIR__, "manifest.jl"))
using .Manifest: manifest_dict

export decide_response, manifest_dict, start_daemon, stop_daemon

# The bdsl the daemon serves at GET /manifest — the body's effector/feature vocabulary, parsed by
# the ONE library parser (move-4-design §2C). Located relative to this file (daemon/ → ../bdsl).
const _BDSL_DIR = abspath(joinpath(@__DIR__, "..", "bdsl"))
const CAPABILITIES_PATH = joinpath(_BDSL_DIR, "capabilities.bdsl")
const FEATURES_PATH = joinpath(_BDSL_DIR, "features.bdsl")

# ── Request parsing (the wire → the brain's abstract types) ──────────────────────────────

_obs(o::AbstractDict)::Obs = Obs(
    Int(round(Float64(o["reports"]))),
    Int(round(Float64(o["group"]))),
    Float64(o["authority"]),
    Float64(o["subject_factor"]),
    Float64(o["time_factor"]),
)

# Optional per-request channel override; defaults to the operator-set CANONICAL_CHANNEL. The body
# ships the canonical params; the parity test relies on the default + a drift guard (move-1 §0).
function _channel(req::AbstractDict)::ChannelParams
    haskey(req, "channel") || return CANONICAL_CHANNEL
    c = req["channel"]
    ChannelParams(Float64(c["a_alternatives"]), Float64(c["beta_ancestry"]),
                  Float64(c["beta_model"]), Float64(c["p_none_prior"]),
                  Float64(c["oracle_p"]), Float64(c["prob_eps"]))
end

"""
    decide_response(req) -> Dict

The decision for one question, given the evidence so far: the terminal effector (report / hedge /
ask_clarify / abstain), or — when the body supplies `era_split` (the candidates span eras) and
`recency` is unapplied — a forward `gather(recency, target)` steer that re-weights by recency BEFORE
any terminal report, ruling out a stale leader first (`AnswerBrain.gather_decide`, move-4-design
§2C). Pure: same inputs → same Dict. Throws on a malformed request (the HTTP layer turns that into a
400).

Optional request fields (all absent ⇒ the Stage-2a terminal decision, unchanged):
  `era_split::Bool`           body-projected feature — candidates split across eras (recency discriminates)
  `owner_scoped::Bool`        body-projected — an owner-scoped report corroborates first (the
                              attribution guard, `AnswerBrain.gather_decide`): the body re-extracts
                              with the subject-aware model and re-posts before any in-set report
  `applied_probes::[String]`  probes already applied this question (body-held, resent) — guarantees termination
"""
function decide_response(req::AbstractDict)::Dict{String, Any}
    candidates = String.(req["candidates"])
    k = length(candidates)
    k >= 1 || error("`candidates` must be non-empty")
    rho = Float64(req["rho"])
    cp = _channel(req)
    obs = Obs[_obs(o) for o in req["observations"]]
    u_bar = Dict{String, Float64}(String(kk) => Float64(vv) for (kk, vv) in req["u_bar"])
    era_split = Bool(get(req, "era_split", false))
    owner_scoped = Bool(get(req, "owner_scoped", false))
    gather_rho = Float64(get(req, "gather_rho", 0.0))    # the corroborate re-read's reliability …
    gather_cost = Float64(get(req, "gather_cost", 0.0))  # … and its cost (utility units) — 0 ⇒ off
    applied = String[String(p) for p in get(req, "applied_probes", String[])]

    # The grow lane (slice 5): bucketed `sensors` + a `grow` block — a shared feature vocabulary
    # and per-actuator {probe, cost, alpha0?, beta0?, warm_counts?}. The stateless daemon rebuilds
    # each actuator's outcome belief from the body-persisted counts (warm reconstruction is exact —
    # order-independence) and reads the learned `g = recovery_g(model, top, sensors)`; a cold
    # actuator's g is its declared Beta(alpha0, beta0) prior mean (the demoted hand-set g-prior).
    grows = Tuple{String, Float64, Float64}[]
    grow_g = Dict{String, Float64}()
    if haskey(req, "grow")
        haskey(req, "sensors") || error("`grow` requires `sensors` (the bucketed feature dict)")
        gr = req["grow"]
        feats = gr["features"]
        names = String[String(n) for n in feats["names"]]
        values = Vector{String}[String[String(v) for v in vs] for vs in feats["values"]]
        sensors = Dict{String, String}(String(kk) => String(vv) for (kk, vv) in req["sensors"])
        for a in gr["actuators"]
            aprobe = String(a["probe"])
            acost = Float64(get(a, "cost", 0.0))
            alpha0 = Float64(get(a, "alpha0", 2.0))
            beta0 = Float64(get(a, "beta0", 2.0))
            model = build_structure_model(names, values; alpha0 = alpha0, beta0 = beta0)
            wc = get(a, "warm_counts", nothing)
            top = reconstruct_structure_prior_from_data(model, wc)
            g = recovery_g(model, top, sensors)
            push!(grows, (aprobe, g, acost))
            grow_g[aprobe] = g
        end
    end

    post = candidate_posterior(k, obs, rho; cp = cp)
    w = weights(post)                                  # length k+1: candidates then NONE
    # The menu is data: when the body declares a `transforms` list (Slice 2 — e.g. the corroborate
    # model-tier ladder), build the registry from it and run the uniform scheduler; absent ⇒ the
    # default registry built from the flags (byte-identical to the Stage-2a path). The grows lane
    # rides both paths.
    effector, report_index, probe, target, eu =
        if haskey(req, "transforms")
            reg = registry_from_wire(req["transforms"]; cp = cp)
            ctx = ScheduleCtx(era_split, owner_scoped, gather_rho, gather_cost, applied)
            schedule_decide(post, k, u_bar, reg, ctx; cp = cp, grows = grows)
        else
            gather_decide(post, k, u_bar; era_split = era_split, owner_scoped = owner_scoped,
                          gather_rho = gather_rho, gather_cost = gather_cost,
                          applied_probes = applied, cp = cp, grows = grows)
        end

    resp = Dict{String, Any}(
        "effector"     => effector,                    # report | hedge | ask_clarify | abstain | gather
        "report_index" => report_index,                # 0-based candidate idx (report), or nothing → null
        "value"        => report_index === nothing ? nothing : candidates[report_index + 1],
        "probe"        => probe,                        # the gather probe (e.g. "recency"), or null
        "target"       => target === nothing ? nothing : candidates[target + 1],  # candidate to test, or null
        "credences"    => w[1:k],                       # candidate weights (NONE excluded)
        "p_none"       => w[k + 1],
        "eu"           => eu,
    )
    # Per-actuator g, exposed only when the body sent a grow block — for the body's
    # gather-outcome log (Invariant 1: display/logging, never selection — the daemon already chose).
    haskey(req, "grow") && (resp["grow_g"] = grow_g)
    resp
end

# ── HTTP transport (adapted from apps/credence-pi/daemon/server.jl) ──────────────────────

function _write_json_response(stream::HTTP.Stream, status::Int, payload::AbstractDict)
    body = JSON3.write(payload)
    HTTP.setstatus(stream, status)
    HTTP.setheader(stream, "Content-Type" => "application/json")
    HTTP.setheader(stream, "Content-Length" => string(sizeof(body)))
    HTTP.startwrite(stream)
    write(stream, body)
end

function _handle_decide(stream::HTTP.Stream)
    req = try
        JSON3.read(String(read(stream)), Dict{String, Any})
    catch e
        return _write_json_response(stream, 400,
            Dict("error" => "malformed JSON: $(sprint(showerror, e))"))
    end
    resp = try
        decide_response(req)
    catch e
        return _write_json_response(stream, 400,
            Dict("error" => sprint(showerror, e)))
    end
    _write_json_response(stream, 200, resp)
end

function _handle_manifest(stream::HTTP.Stream)
    resp = try
        manifest_dict(; capabilities_path = CAPABILITIES_PATH, features_path = FEATURES_PATH)
    catch e
        # A malformed/absent bdsl is an operator error, not a bad request → 500.
        return _write_json_response(stream, 500,
            Dict("error" => "manifest: $(sprint(showerror, e))"))
    end
    _write_json_response(stream, 200, resp)
end

"""
    start_daemon(; port, host="127.0.0.1") -> server

Serve the decision surface. Stateless, so there is nothing to initialise or replay — unlike
credence-pi's daemon this takes no `log_path`/`bdsl_dir` (the brain's params are compiled-in
`CANONICAL_CHANNEL`; Ū is per-request). `stop_daemon(server)` closes the listener.
"""
function start_daemon(; port::Int, host::AbstractString = "127.0.0.1")
    function dispatch(stream::HTTP.Stream)
        method = stream.message.method
        target = stream.message.target
        if method == "POST" && target == "/decide"
            _handle_decide(stream)
        elseif method == "GET" && target == "/manifest"
            _handle_manifest(stream)
        elseif method == "GET" && target == "/ready"
            HTTP.setstatus(stream, 200)
            HTTP.setheader(stream, "Content-Type" => "text/plain")
            HTTP.startwrite(stream)
            write(stream, "ok\n")
        else
            HTTP.setstatus(stream, 404)
            HTTP.startwrite(stream)
            write(stream, "Not Found\n")
        end
    end
    # `stream=true` requires HTTP 1.11.x (pinned in Project.toml [compat]) — same as credence-pi.
    HTTP.serve!(dispatch, host, port; stream = true)
end

stop_daemon(server) = close(server)

end # module Server
