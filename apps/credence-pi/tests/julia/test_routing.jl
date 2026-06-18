#!/usr/bin/env julia
# Role: tests
"""
    test_routing.jl — live model-routing wiring (brain + daemon transport).

Section A (brain): wire_routing! reads the declared roster + reward + features and
installs env[:route-decide]; the warm belief reconstructs the measured per-model
accuracy (Beta-shrunk); route-decide returns the EU-max model and FLIPS with the profile
reward — cost-hawk → cheap, quality-hawk → best — the Wald per-profile divergence, live,
through the one canonical `optimise`.

Section B (daemon transport): a route-request sensor event emits a `route` effector
signal carrying the chosen model; routing is INERT without a declared roster; a
route-request never touches the governance posterior (separate belief).
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "src"))
using Credence
using Credence: Eval, Parse, Identity, weights, expect, mean
using JSON3
using Random: MersenneTwister

include(joinpath(@__DIR__, "..", "..", "brain", "feature_brain.jl"))
using .FeatureBrain: build_model, observe, observe_soft
include(joinpath(@__DIR__, "..", "..", "brain", "routing_brain.jl"))
using .RoutingBrain: wire_routing!, route, escalation_next, posterior_accuracy,
    route_outcome!, _ctx_key, reconstruct_latency, latency_at, LatencyBelief

const BDSL_DIR = joinpath(@__DIR__, "..", "..", "bdsl")
const TOL = 1e-12
const PASSED = String[]
ok(name) = (push!(PASSED, name); println("PASSED: ", name))

# An env carrying the declared routing data (routing.bdsl + utility.bdsl). reward is
# overridable to exercise profiles. routing.bdsl uses only core forms, so no stdlib.
function routing_env(; reward = nothing)
    env = Eval.default_env(); env[:__toplevel__] = true
    for f in ("utility.bdsl", "routing.bdsl")
        for expr in Parse.parse_all(read(joinpath(BDSL_DIR, f), String))
            Eval.eval_dsl(expr, env)
        end
    end
    reward === nothing || (env[Symbol("correct-answer-value")] = reward)
    env
end

# ── A. wire_routing! + route-decide ─────────────────────────────────────

let env = routing_env()
    rt = wire_routing!(env)
    @assert rt !== nothing
    @assert rt.model_ids == ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"]
    @assert rt.providers == ["anthropic", "anthropic", "anthropic"]
    @assert length(rt.costs) == 3 && rt.costs[1] < rt.costs[2] < rt.costs[3]
    ok("wire_routing! parses the declared roster (3 models, ascending cost)")

    # Warm belief (AGENTIC, from the tb matrix): posterior-mean accuracy at a short prompt is
    # ordered haiku < opus < sonnet — sonnet is the best-believed on short agentic tasks. (The
    # superseded MCQ warm wrongly ordered opus top; eval/calibration.jl quantified the
    # mis-rank.) Beta(2,2)-shrunk from the measured short-bucket rates (haiku 0.33 / sonnet
    # 0.70 / opus 0.57).
    θ = [posterior_accuracy(rt.model, rt.tops[i], ["short"]) for i in 1:3]
    @assert θ[1] < θ[3] < θ[2]          # haiku < opus < sonnet
    @assert 0.30 < θ[1] < 0.42 && 0.62 < θ[2] < 0.72 && 0.50 < θ[3] < 0.62
    ok("warm belief: P(correct|short) ordered haiku<opus<sonnet, agentic-warmed ($(round.(θ, digits = 3)))")

    @assert haskey(env, Symbol("route-decide"))
    ok("wire_routing! installed env[:route-decide]")
end

# Cost-hawk (default reward 0.02): route the CHEAP model on every prompt-length (the
# correct answer is worth ≈ one call, so the small accuracy gain never justifies paying).
let env = routing_env()
    decide = (wire_routing!(env); env[Symbol("route-decide")])
    for len in ("short", "long")
        choice = decide(Dict("prompt-length" => len))
        @assert choice["model"] == "claude-haiku-4-5"  # cost-hawk/$len
    end
    ok("cost-hawk (reward 0.02): routes the cheap model (haiku) at short AND long")
end

# Quality-hawk (reward 1.0): route the BEST-BELIEVED model where the belief is warm — now
# sonnet (the agentic warm's top tier at short), which is also CHEAPER than opus: better+cheaper.
let env = routing_env(reward = 1.0)
    wire_routing!(env)
    choice = env[Symbol("route-decide")](Dict("prompt-length" => "short"))
    @assert choice["model"] == "claude-sonnet-4-6"
    @assert choice["provider"] == "anthropic" && choice["name"] == "sonnet"
    ok("quality-hawk (reward 1.0): routes the best-believed model (sonnet) on a short prompt — better AND cheaper than opus")
end

# Per-profile divergence on ONE shared belief is the Wald core: same warm belief, the
# routed model flips with the reward alone.
let
    c_hawk = (e = routing_env(reward = 0.02); wire_routing!(e); e[Symbol("route-decide")](Dict("prompt-length" => "short")))
    q_hawk = (e = routing_env(reward = 1.0);  wire_routing!(e); e[Symbol("route-decide")](Dict("prompt-length" => "short")))
    @assert c_hawk["model"] != q_hawk["model"]
    ok("Wald divergence, live: same belief, reward 0.02 → $(c_hawk["name"]), reward 1.0 → $(q_hawk["name"])")
end

# Inert when no roster is declared (utility.bdsl only): wire_routing! returns nothing and
# installs nothing — a governance-only install is unaffected.
let
    env = Eval.default_env(); env[:__toplevel__] = true
    for expr in Parse.parse_all(read(joinpath(BDSL_DIR, "utility.bdsl"), String))
        Eval.eval_dsl(expr, env)
    end
    @assert wire_routing!(env) === nothing
    @assert !haskey(env, Symbol("route-decide"))
    ok("inert: no routing-models declared ⇒ wire_routing! installs no route-decide")
end

# Cold fallback: a missing warm file falls back to uniform priors (loudly), and an
# uninformative belief routes the cheapest model even under a quality reward (it cannot
# justify paying more for accuracy it has no evidence of).
let env = routing_env(reward = 1.0)
    wire_routing!(env; warm_path = "")
    choice = env[Symbol("route-decide")](Dict("prompt-length" => "short"))
    @assert choice["model"] == "claude-haiku-4-5"
    ok("cold fallback: uniform belief routes the cheapest model even at reward 1.0")
end

# ── A'. Observe-then-escalate: escalation_next, the gate via the one optimise ──
# The deployable strategy that wins the dominance eval: try cheapest, escalate on
# observed failure only when the EU justifies it. Same warm belief; the decision is
# `optimise` over {try, stop}, fed context-dependent cost.

# High reward: escalate cheapest→dearest as cheaper rungs are observed to fail, then stop.
let env = routing_env(reward = 1.0)
    rt = wire_routing!(env); X = ["short"]; costs = [0.01, 0.05, 0.20]  # ascending E[cost|tier]
    @assert escalation_next(rt.model, rt.tops, X, costs, 1.0, Set{Int}()) == 1
    @assert escalation_next(rt.model, rt.tops, X, costs, 1.0, Set([1])) == 2
    @assert escalation_next(rt.model, rt.tops, X, costs, 1.0, Set([1, 2])) == 3
    @assert escalation_next(rt.model, rt.tops, X, costs, 1.0, Set([1, 2, 3])) == 0
    ok("escalation_next: high reward escalates cheapest-first 1→2→3→stop")
end

# The EU gate's value: a solve worth too little ⇒ STOP without spending (what the
# gateless cascade can't do).
let env = routing_env(reward = 1.0)
    rt = wire_routing!(env)
    @assert escalation_next(rt.model, rt.tops, ["short"], [0.01, 0.05, 0.20], 0.001, Set{Int}()) == 0
    ok("escalation_next: tiny reward ⇒ gate STOPS (no rung worth trying)")
end

# Tightest invariant: the gate opens IFF reward·E[θ_a|X] ≥ cost_a — the boundary, exact,
# through `optimise` (try the cheapest tier with cost just below vs just above its EU break).
let env = routing_env(reward = 1.0)
    rt = wire_routing!(env); X = ["short"]; reward = 1.0
    θ1 = posterior_accuracy(rt.model, rt.tops[1], X)
    @assert escalation_next(rt.model, rt.tops, X, [reward * θ1 - 1e-6, 9.9, 9.9], reward, Set{Int}()) == 1
    @assert escalation_next(rt.model, rt.tops, X, [reward * θ1 + 1e-6, 9.9, 9.9], reward, Set{Int}()) == 0
    ok("escalation_next gate ≡ reward·E[θ|X] ≥ cost (boundary exact, via optimise)")
end

# escalate-decide: the closure the DAEMON calls (env[:escalate-decide]) must be a faithful
# wrapper of escalation_next — same rung on identical inputs. This is the brain half of
# closing EXPERIMENT.md honest-limit (c) (the gate wired into the live loop, not just the eval).
let env = routing_env(reward = 1.0)
    rt = wire_routing!(env); decide = env[Symbol("escalate-decide")]
    costs = [0.01, 0.05, 0.20]                                   # ascending E[cost|tier]
    roster = [["haiku",  "anthropic", "claude-haiku-4-5",  costs[1]],   # KNOWN ids ⇒ warm tops
              ["sonnet", "anthropic", "claude-sonnet-4-6", costs[2]],
              ["opus",   "anthropic", "claude-opus-4-8",   costs[3]]]
    for tried in (Int[], [1], [1, 2], [1, 2, 3])
        ref = escalation_next(rt.model, rt.tops, ["short"], costs, 1.0, Set{Int}(tried))
        got = decide(Dict("prompt-length" => "short"), roster, tried)
        @assert (got === nothing ? 0 : got["tier_index"]) == ref
    end
    ok("escalate-decide == in-process escalation_next on identical inputs (the daemon's gate is faithful)")

    # Sort-robustness: escalation_next scans cheapest-first, so escalate-decide must order the
    # roster by cost itself — a dearest-first roster still yields the cheapest rung at tried=[].
    g = decide(Dict("prompt-length" => "short"), reverse(roster), Int[])
    @assert g["model"] == "claude-haiku-4-5" && g["tier_index"] == 1
    ok("escalate-decide sorts by cost: a dearest-first roster still escalates cheapest-first")
end

# ── A'''. The TIME coordinate: w_time·E[time] in the routing EU (time vs money/quality) ──
# Time is the profile dial that lets a user trade wall-clock against money/quality. E[time] is a
# learned Poisson-Gamma latency belief; w_time the declared $/sec weight; the EU folds w_time·
# E[time] into the offset. w_time=0 ⇒ bit-identical (the GeometricTail-rollout discipline).
const LATENCY_JSON = joinpath(@__DIR__, "..", "..", "brain", "routing_latency.counts.json")

# The latency belief reconstructs E[time|model,X] from version-stable counts; opus is slower
# than haiku on the measured matrix (the time signal); unknown (model/ctx) ⇒ 0 (no time term).
let lb = reconstruct_latency(LATENCY_JSON)
    t_opus = latency_at(lb, "claude-opus-4-8", ["short"])
    t_haiku = latency_at(lb, "claude-haiku-4-5", ["short"])
    @assert t_opus > 0.0 && t_haiku > 0.0
    @assert t_opus > t_haiku                                   # opus slower/turn ⇒ more wall-clock
    @assert latency_at(lb, "unknown-model", ["short"]) == 0.0  # conservative: no belief ⇒ no penalty
    @assert latency_at(nothing, "claude-opus-4-8", ["short"]) == 0.0
    # credence-lint: allow — precedent:test-oracle — exact reconstruction-determinism oracle
    @assert latency_at(reconstruct_latency(LATENCY_JSON), "claude-opus-4-8", ["short"]) == t_opus
    ok("latency belief: E[time] reconstructs (opus $(round(Int, t_opus))s > haiku $(round(Int, t_haiku))s short), 0 for unknown, deterministic")
end

# w_time=0 ⇒ the time term is exactly 0 ⇒ route is bit-identical regardless of the times vector.
let env = routing_env(reward = 1.0)
    rt = wire_routing!(env); X = ["short"]; costs = [0.01, 0.05, 0.20]; times = [99.0, 5.0, 50.0]
    base = route(rt.model, rt.tops, X, costs, 1.0)
    # credence-lint: allow — precedent:test-oracle — w_time=0 bit-identical equality oracle
    @assert route(rt.model, rt.tops, X, costs, 1.0; w_time = 0.0, times = times) == base
    @assert route(rt.model, rt.tops, X, costs, 1.0; w_time = 0.0, times = nothing) == base
    ok("route: w_time=0 is bit-identical regardless of times (time-blind ⇒ unchanged)")
end

# Time/quality trade-off, LIVE through route-decide (the wired path: w-time read + latency
# loaded). Same belief + reward; the TIME dial alone flips the choice from the accuracy winner
# (sonnet) to the FASTER model (haiku) — the "time vs money/quality" the MVP must express.
let q0 = (e = routing_env(reward = 1.0); wire_routing!(e); e[Symbol("route-decide")](Dict("prompt-length" => "short"))),
    qt = (e = routing_env(reward = 1.0); e[Symbol("w-time")] = 0.03; wire_routing!(e);
          e[Symbol("route-decide")](Dict("prompt-length" => "short")))
    @assert q0["model"] == "claude-sonnet-4-6"    # w-time 0: the accuracy winner
    @assert qt["model"] == "claude-haiku-4-5"     # w-time 0.03: the faster model
    @assert q0["model"] != qt["model"]
    ok("time/quality trade-off (live): reward 1.0 + w-time 0 → sonnet (accurate); + w-time 0.03 → haiku (fast)")
end

# ── A''. Roster-aware routing: the LIVE roster (the user's actual models) ────────
# route-decide takes an optional per-request roster — the models OpenClaw actually has +
# their costs. Known models reuse the warm belief; the user's OWN (unknown) models get a
# cold prior + online learning; fewer than 2 candidates ⇒ inert (keep OpenClaw's model).
# This is what makes routing safe to turn ON by default for any roster.

# Known models in a live roster reuse the WARM belief: at equal cost, quality-hawk picks the
# best-believed (opus) — the warm seed is used, not ignored, for a roster sent per request.
let env = routing_env(reward = 1.0)
    wire_routing!(env); decide = env[Symbol("route-decide")]
    roster = [["haiku", "anthropic", "claude-haiku-4-5", 0.001],
              ["opus",  "anthropic", "claude-opus-4-8",  0.001]]
    @assert decide(Dict("prompt-length" => "short"), roster)["model"] == "claude-opus-4-8"
    ok("roster-aware: known models in a live roster reuse the warm belief (quality-hawk → opus)")
end

# The user's OWN models (unknown to the warm seed) get a COLD prior (θ=0.5 each) ⇒ EU-max
# reduces to CHEAPEST — the safe cost-saving default for an unseen roster — then learns up.
let env = routing_env()   # cost-hawk reward 0.02
    wire_routing!(env); decide = env[Symbol("route-decide")]
    c1 = decide(Dict("prompt-length" => "short"),
                [["cheapo", "x", "cheapo-1", 0.001], ["premium", "y", "premium-1", 0.5]])
    @assert c1["model"] == "cheapo-1" && c1["provider"] == "x"
    c2 = decide(Dict("prompt-length" => "short"),   # flip the costs ⇒ the other model
                [["cheapo", "x", "cheapo-1", 0.5], ["premium", "y", "premium-1", 0.001]])
    @assert c2["model"] == "premium-1"
    ok("roster-aware: unknown models get a cold prior ⇒ EU-max picks the cheapest (cost-saving default)")
end

# Inert: a single-model roster has nothing to route between ⇒ route-decide returns nothing
# (the body keeps OpenClaw's model). An EMPTY roster falls back to the declared default (≥2).
let env = routing_env()
    decide = (wire_routing!(env); env[Symbol("route-decide")])
    @assert decide(Dict("prompt-length" => "short"), [["solo", "x", "solo-1", 0.01]]) === nothing
    @assert decide(Dict("prompt-length" => "short"), Any[]) !== nothing
    ok("roster-aware: <2 candidate models ⇒ route-decide is inert (returns nothing)")
end

# The user's own model LEARNS online and persists in extra_tops (keyed by id), exactly like a
# default-roster model — so an unknown roster gets calibrated by real traffic over time.
let rt = wire_routing!(routing_env())
    @assert !haskey(rt.extra_tops, "mymodel-1")
    for _ in 1:30; route_outcome!(rt, "mymodel-1", Dict("prompt-length" => "short"), true); end
    @assert haskey(rt.extra_tops, "mymodel-1")
    after = posterior_accuracy(rt.model, rt.extra_tops["mymodel-1"], ["short"])
    @assert after > 0.5    # cold prior mean 0.5, raised by clean successes
    ok("roster-aware: an unknown model learns online + persists in extra_tops (θ → $(round(after, digits = 3)))")
end

# ── B. Daemon transport: route-request → route signal ───────────────────

include(joinpath(@__DIR__, "..", "..", "daemon", "server.jl"))
using .Server: init_state, handle_sensor_event, snapshot

const DAEMON_BDSL = joinpath(@__DIR__, "..", "..", "bdsl")

# init_state loads routing.bdsl and runs wire_routing! (warm belief from the committed
# counts), so the daemon routes out of the box.
let path = tempname() * ".jsonl"
    state = init_state(; bdsl_dir = DAEMON_BDSL, log_path = path)
    @assert haskey(state.env, Symbol("route-decide"))
    ok("daemon init_state wires routing from bdsl/routing.bdsl (route-decide installed)")

    # A route-request emits exactly one `route` signal carrying the chosen model. The
    # default profile is cost-hawk (correct-answer-value 0.02) ⇒ the cheap model.
    w_before = weights(state.posterior[])
    ack = handle_sensor_event(state, Dict{String, Any}(
        "event_type" => "route-request", "event_id" => "rt_1",
        "features" => Dict("prompt-length" => "short")))
    @assert ack["ack"] == true && ack["event_id"] == "rt_1"
    sigs = snapshot(state.signal_queue)
    @assert length(sigs) == 1
    @assert sigs[1]["effector"] == "route"
    @assert sigs[1]["in_response_to"] == "rt_1"
    @assert sigs[1]["parameters"]["model"] == "claude-haiku-4-5"
    @assert sigs[1]["parameters"]["provider"] == "anthropic"
    ok("route-request → single `route` signal with the EU-max model (cost-hawk → haiku)")

    # Routing is a SEPARATE belief: a route-request must not touch the governance posterior.
    # credence-lint: allow — precedent:test-oracle — structure-posterior equality oracle (routing must not learn governance)
    @assert weights(state.posterior[]) == w_before
    ok("route-request leaves the governance posterior untouched (separate belief)")
    rm(path; force = true)
end

# Inert: with routing unconfigured (no route-decide), a route-request emits no signal —
# the body times out and fails open to OpenClaw's default model.
let path = tempname() * ".jsonl"
    state = init_state(; bdsl_dir = DAEMON_BDSL, log_path = path)
    delete!(state.env, Symbol("route-decide"))
    handle_sensor_event(state, Dict{String, Any}(
        "event_type" => "route-request", "event_id" => "rt_inert",
        "features" => Dict("prompt-length" => "short")))
    @assert isempty(snapshot(state.signal_queue))
    ok("route-request with routing unconfigured emits no signal (body fails open)")
    rm(path; force = true)
end

# Roster-aware via the daemon: a route-request carrying the user's live `models` routes over
# THAT roster (the user's own models, unknown to the warm seed), and a single-model roster is
# inert (no signal). This is the on-by-default safety path end-to-end through the transport.
let path = tempname() * ".jsonl"
    state = init_state(; bdsl_dir = DAEMON_BDSL, log_path = path)
    handle_sensor_event(state, Dict{String, Any}(
        "event_type" => "route-request", "event_id" => "rr_dyn", "session_id" => "Sx",
        "features" => Dict("prompt-length" => "short"),
        "models" => Any[Dict("name" => "a", "provider" => "p", "model" => "my-cheap", "cost" => 0.001),
                        Dict("name" => "b", "provider" => "p", "model" => "my-dear",  "cost" => 0.5)]))
    sigs = snapshot(state.signal_queue)
    @assert length(sigs) == 1 && sigs[1]["parameters"]["model"] == "my-cheap"
    ok("daemon roster-aware: route-request with a live `models` roster routes the user's own models (cost-hawk → cheapest)")

    state2 = init_state(; bdsl_dir = DAEMON_BDSL, log_path = tempname() * ".jsonl")
    handle_sensor_event(state2, Dict{String, Any}(
        "event_type" => "route-request", "event_id" => "rr_solo",
        "features" => Dict("prompt-length" => "short"),
        "models" => Any[Dict("name" => "a", "provider" => "p", "model" => "only-one", "cost" => 0.01)]))
    @assert isempty(snapshot(state2.signal_queue))
    ok("daemon roster-aware: single-model roster ⇒ no route signal (inert, fail open)")
    rm(path; force = true)
end

# ── B'. Observe-then-escalate THROUGH the daemon (escalate-request) ──────────────
# Closes EXPERIMENT.md honest-limit (c): the EU escalation gate now runs as live daemon code.
# The host drives try→observe→escalate by re-POSTing with the failed rungs in `tried`; the
# decision rides back synchronously in the ack (`route`/`stop`). The live-A/B harness
# (eval/live_ab/escalation_live.jl) scores realized welfare over exactly this path.
let path = tempname() * ".jsonl"
    state = init_state(; bdsl_dir = DAEMON_BDSL, log_path = path)
    @assert haskey(state.env, Symbol("escalate-decide"))
    w_before = weights(state.posterior[])
    costs = [0.01, 0.05, 0.20]
    roster = Any[Dict("name" => "haiku",  "provider" => "anthropic", "model" => "claude-haiku-4-5",  "cost" => costs[1]),
                 Dict("name" => "sonnet", "provider" => "anthropic", "model" => "claude-sonnet-4-6", "cost" => costs[2]),
                 Dict("name" => "opus",   "provider" => "anthropic", "model" => "claude-opus-4-8",   "cost" => costs[3])]
    esc(tried) = handle_sensor_event(state, Dict{String, Any}("event_type" => "escalate-request",
        "event_id" => "e", "features" => Dict("prompt-length" => "short"),
        "models" => roster, "tried" => tried, "reward" => 1.0))
    model_of(ack) = haskey(ack, "route") ? ack["route"]["model"] : (get(ack, "stop", false) ? "STOP" : "?")
    @assert model_of(esc(Int[]))   == "claude-haiku-4-5"
    @assert model_of(esc([1]))     == "claude-sonnet-4-6"
    @assert model_of(esc([1, 2]))  == "claude-opus-4-8"
    @assert model_of(esc([1, 2, 3])) == "STOP"
    ok("daemon escalate-request: try→escalate→STOP ladder over the live roster (haiku→sonnet→opus→STOP)")

    # Full-path equivalence: the daemon's escalate-request decision == in-process escalation_next
    # (via Server.RoutingBrain — the daemon's own module instance, per the section-D note).
    for tried in (Int[], [1], [1, 2], [1, 2, 3])
        got = let a = esc(tried); haskey(a, "route") ? a["route"]["tier_index"] : 0 end
        ref = Server.RoutingBrain.escalation_next(state.routing.model, state.routing.tops,
                                                  ["short"], costs, 1.0, Set{Int}(tried))
        @assert got == ref
    end
    ok("daemon escalate-request decision == in-process escalation_next (gate wired faithfully — closes EXPERIMENT.md limit (c))")

    # Routing is a SEPARATE belief: an escalate-request must not touch the governance posterior.
    # credence-lint: allow — precedent:test-oracle — governance-posterior equality oracle (escalation must not learn governance)
    @assert weights(state.posterior[]) == w_before
    ok("daemon escalate-request leaves the governance posterior untouched (separate belief)")
    rm(path; force = true)
end

# ── B''. Per-request PROFILE override: one daemon, many users' trade-offs, no restart ──
# The user's utility weights ride in the route-request `profile` (like the live roster), so a
# user switches cost/time/quality WITHOUT a daemon restart. The SAME daemon + belief routes
# differently per profile: speed-first (high w_time) → the faster model; quality-first (high
# reward, no time pressure) → a stronger one. This is the MVP's "handle different trade-offs".
let path = tempname() * ".jsonl"
    state = init_state(; bdsl_dir = DAEMON_BDSL, log_path = path)
    w_before2 = weights(state.posterior[])
    route_with(profile) = begin
        handle_sensor_event(state, Dict{String, Any}("event_type" => "route-request",
            "event_id" => "rp", "features" => Dict("prompt-length" => "short"), "profile" => profile))
        snapshot(state.signal_queue)[end]["parameters"]["model"]
    end
    speed   = route_with(Dict("reward" => 1.0, "w_time" => 0.05))   # time precious
    quality = route_with(Dict("reward" => 5.0, "w_time" => 0.0))    # quality precious, time free
    @assert speed == "claude-haiku-4-5"        # speed-first ⇒ the FASTER model
    @assert quality != "claude-haiku-4-5"      # quality-first ⇒ a stronger model
    @assert speed != quality
    ok("per-request profile: same daemon, speed-first → $speed, quality-first → $quality (no restart)")

    # A profile override is PREFERENCE, never learning: a route-request with a profile must not
    # touch the governance posterior (separate belief, like the un-profiled path).
    # credence-lint: allow — precedent:test-oracle — governance-posterior equality oracle (profile is preference, not learning)
    @assert weights(state.posterior[]) == w_before2
    ok("per-request profile: routing with a profile leaves the governance posterior untouched")
    rm(path; force = true)
end

# ── C. Online correctness learning (brain): soft evidence + learned confounds ───
#
# The deferred online signal, landed. We never see model correctness; we see whether the
# proposed call executed cleanly (e), a NOISY emission of latent correctness confounded by
# tool reliability. route_outcome! decodes e against a LEARNED confound (ρ,σ) and conditions
# the routing belief on the resulting virtual evidence — so a flaky tool is absorbed by ρ,
# not blamed on the model. All updates are `condition`; the decode is `expect`.

const FEAT_SHORT = Dict("prompt-length" => "short")
θat(rt, a) = posterior_accuracy(rt.model, rt.tops[a], ["short"])
ρ̄of(rt) = mean(get(rt.emission.rho_cells, _ctx_key(["short"]), rt.emission.rho0))

# C1. Reduces EXACTLY to the hard observe at the certain-signal corners: (r,w)=(1,0) is a
# hard 1, (0,1) is a hard 0 — observe_soft generalises observe (the substrate-equivalence guard).
let rt = wire_routing!(routing_env())
    top = rt.tops[1]
    h1 = posterior_accuracy(rt.model, observe(rt.model, top, ["short"], 1), ["short"])
    s1 = posterior_accuracy(rt.model, observe_soft(rt.model, top, ["short"], 1.0, 0.0), ["short"])
    h0 = posterior_accuracy(rt.model, observe(rt.model, top, ["short"], 0), ["short"])
    s0 = posterior_accuracy(rt.model, observe_soft(rt.model, top, ["short"], 0.0, 1.0), ["short"])
    @assert s1 == h1 && s0 == h0
    ok("soft evidence reduces EXACTLY to the hard observe at (1,0)/(0,1)")
end

# C2. Direction: clean per-turn successes raise the routed model's accuracy belief.
let rt = wire_routing!(routing_env())
    before = θat(rt, 1)
    for _ in 1:30; route_outcome!(rt, rt.model_ids[1], FEAT_SHORT, true); end
    @assert θat(rt, 1) > before
    ok("clean successes raise θ ($(round(before, digits = 3)) → $(round(θat(rt, 1), digits = 3)))")
end

# C3. CONFOUND-PARTIALLING — the property the deferral was about. A pure tool-flakiness
# stream (every call fails for EVERY model, true accuracy unchanged) must NOT collapse θ:
# the failures are absorbed by the learned reliability ρ, not blamed on the models. Contrast
# a NAIVE hard "fail ⇒ incorrect" rule, which craters the same belief — the mis-attribution
# the decode exists to avoid.
let rt = wire_routing!(routing_env())
    before = [θat(rt, a) for a in 1:3]
    for _ in 1:60, a in 1:3; route_outcome!(rt, rt.model_ids[a], FEAT_SHORT, false); end
    drift = maximum(abs.([θat(rt, a) for a in 1:3] .- before))
    @assert drift < 0.03          # θ essentially flat
    @assert ρ̄of(rt) < 0.2         # ρ learned LOW: correct calls failed too ⇒ the tool is flaky
    ok("confound-partialling: pure flakiness leaves θ flat (max drift $(round(drift, digits = 4))), absorbed by ρ̄=$(round(ρ̄of(rt), digits = 3)))")

    naive = wire_routing!(routing_env())   # baseline: same failures as hard "incorrect" labels
    # credence-lint: allow — precedent:baseline-comparison — naive hard-label credit rule, the conflation the decode avoids
    for _ in 1:60, a in 1:3; naive.tops[a] = observe(naive.model, naive.tops[a], ["short"], 0); end
    collapsed = maximum(before .- [θat(naive, a) for a in 1:3])
    @assert collapsed > 0.3
    ok("contrast: naive hard fail⇒incorrect collapses θ by $(round(collapsed, digits = 3))")
end

# C4. Human approve/reject is the gold anchor: a KNOWN label drives θ hard.
let rt = wire_routing!(routing_env())
    before = θat(rt, 1)
    for _ in 1:40; route_outcome!(rt, rt.model_ids[1], FEAT_SHORT, false; human = false); end
    @assert θat(rt, 1) < before - 0.15      # sharp drop from the agentic warm start (~0.39)
    ok("human reject is the gold anchor: θ falls sharply ($(round(before, digits = 3)) → $(round(θat(rt, 1), digits = 3)))")
end

# C5. Identifiability (seeded synthetic stream). With the AGENTIC warm (θ≈0.4–0.7, not the
# MCQ regime's ≈0.9), C=1 no longer dominates, so ρ = P(exec-clean | correct) is only
# PARTIALLY identified: it recovers from the 0.67 prior toward 0.9 but lands ~0.78 (fewer C=1
# samples than the high-accuracy regime — an honest limit of warming on harder tasks). θ
# stays anchored-stable; σ stays weakly identified. Seeded ⇒ reproducible; bands carry
# comfortable margin around the recovered values.
let rt = wire_routing!(routing_env())
    ρ_true, σ_true = 0.9, 0.25
    θtrue = [θat(rt, a) for a in 1:3]            # the warm anchors — data is consistent with them
    σ̄of(r) = mean(get(r.emission.sigma_cells, _ctx_key(["short"]), r.emission.sigma0))
    rng = MersenneTwister(20260616)
    for _ in 1:1500
        a = rand(rng, 1:3)
        C = rand(rng) < θtrue[a]
        e = rand(rng) < (C ? ρ_true : σ_true)
        route_outcome!(rt, rt.model_ids[a], FEAT_SHORT, e)
    end
    Δθ = maximum(abs.([θat(rt, a) for a in 1:3] .- θtrue))
    @assert ρ̄of(rt) > 0.74 && abs(ρ̄of(rt) - ρ_true) < 0.15   # partially identified: 0.67 prior → ~0.78 → 0.9
    @assert Δθ < 0.05                            # θ anchored-stable under consistent data
    @assert abs(σ̄of(rt) - σ_true) < 0.25         # σ weakly identified — honest loose band
    ok("identifiability: ρ̄=$(round(ρ̄of(rt), digits = 3)) (0.67 prior → 0.9, partially identified), θ stable (Δ$(round(Δθ, digits = 3))); σ weak")
end

# ── D. Online learning through the daemon: route-outcome → live tops, exact replay ──
#
# Cross-module note: `state.routing` is built by the daemon's nested `Server.RoutingBrain`,
# so its readouts go through `Server.RoutingBrain.posterior_accuracy` (the top-level
# RoutingBrain is a different module instance with different types).

let path = tempname() * ".jsonl"
    state = init_state(; bdsl_dir = DAEMON_BDSL, log_path = path)
    @assert state.routing !== nothing
    θd(a) = Server.RoutingBrain.posterior_accuracy(state.routing.model, state.routing.tops[a], ["short"])
    w_before = weights(state.posterior[])
    before = θd(1)
    # A routed turn (session S1) whose proposed call executes cleanly, ×20.
    for i in 1:20
        handle_sensor_event(state, Dict{String, Any}("event_type" => "route-request",
            "event_id" => "rq$i", "session_id" => "S1", "features" => Dict("prompt-length" => "short")))
        handle_sensor_event(state, Dict{String, Any}("event_type" => "tool-completed",
            "event_id" => "tc$i", "session_id" => "S1", "outcome" => Dict("success" => true)))
    end
    @assert θd(1) > before
    ok("daemon: route-request→tool-completed(success) raises the routed model's θ live ($(round(before, digits = 3)) → $(round(θd(1), digits = 3)))")

    # Routing is a SEPARATE belief: learning to route never touches the governance posterior.
    # credence-lint: allow — precedent:test-oracle — governance-posterior equality oracle (routing must not learn governance)
    @assert weights(state.posterior[]) == w_before
    ok("daemon: route learning leaves the governance posterior untouched (separate belief)")

    # Durability: a fresh daemon replays the route-outcome log and reconstructs θ EXACTLY.
    learned = θd(1)
    state2 = init_state(; bdsl_dir = DAEMON_BDSL, log_path = path)
    # credence-lint: allow — precedent:test-oracle — exact-replay equality oracle
    @assert Server.RoutingBrain.posterior_accuracy(state2.routing.model, state2.routing.tops[1], ["short"]) == learned
    ok("daemon: restart replays route-outcomes and reconstructs the routing belief EXACTLY")
    rm(path; force = true)
end

println()
println("=" ^ 60)
println("ALL ", length(PASSED), " ASSERTIONS PASSED")
println("=" ^ 60)
