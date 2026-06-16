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
using .RoutingBrain: wire_routing!, route, posterior_accuracy, route_outcome!, _ctx_key

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

    # Warm belief: posterior-mean accuracy at a short prompt is ordered cheap<mid<exp,
    # shrunk from the measured 0.88/0.96/0.98 toward the Beta(2,2) prior.
    θ = [posterior_accuracy(rt.model, rt.tops[i], ["short"]) for i in 1:3]
    @assert θ[1] < θ[2] < θ[3]
    @assert 0.80 < θ[1] < 0.90 && 0.90 < θ[3] < 0.96
    ok("warm belief: P(correct|short) ordered haiku<sonnet<opus, Beta-shrunk ($(round.(θ, digits = 3)))")

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

# Quality-hawk (reward 1.0): route the BEST-BELIEVED model where the belief is warm.
let env = routing_env(reward = 1.0)
    wire_routing!(env)
    choice = env[Symbol("route-decide")](Dict("prompt-length" => "short"))
    @assert choice["model"] == "claude-opus-4-8"
    @assert choice["provider"] == "anthropic" && choice["name"] == "opus"
    ok("quality-hawk (reward 1.0): routes the best-believed model (opus) on a short prompt")
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
    for _ in 1:25; route_outcome!(rt, rt.model_ids[1], FEAT_SHORT, false; human = false); end
    @assert θat(rt, 1) < before - 0.2
    ok("human reject is the gold anchor: θ falls sharply ($(round(before, digits = 3)) → $(round(θat(rt, 1), digits = 3)))")
end

# C5. Identifiability (seeded synthetic stream). ρ (correct ⇒ clean-exec, the dominant
# confound for high-accuracy models) recovers tightly; θ stays anchored-stable. σ (wrong ⇒
# clean-exec) is only WEAKLY identified — C=0 is rare when models are mostly correct, so
# little data bears on it (an honest limit; its decode influence is proportionally small).
# Seeded ⇒ reproducible; bands carry comfortable margin.
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
    @assert abs(ρ̄of(rt) - ρ_true) < 0.06        # ρ tightly identified
    @assert Δθ < 0.05                            # θ anchored-stable under consistent data
    @assert abs(σ̄of(rt) - σ_true) < 0.25         # σ weakly identified (rare C=0) — honest loose band
    ok("identifiability: ρ̄=$(round(ρ̄of(rt), digits = 3))≈$(ρ_true) (tight), θ stable (Δ$(round(Δθ, digits = 3))); σ weakly identified")
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
