# Role: eval
#
# oc_welfare_score.jl — score net realized WELFARE per user profile over the LIVE welfare matrix
# (oc_welfare_matrix.py output: each easy task × {qwen, haiku, sonnet} run through real OpenClaw,
# with solved/money/time measured). For each profile this:
#   1. asks the brain's route-decide (the daemon's exact routing closure — test_routing.jl §B
#      proves the live daemon's route-request emits the identical model) which model the profile
#      gets, over the live roster + the measured latency belief;
#   2. computes that profile's realized welfare for credence-pi's routing vs every FIXED router
#      (always-qwen / always-haiku / always-sonnet), as
#          welfare = Σ_task [ reward·solved − money − w_time·time ]
#      — the routing coordinates (quality, money, time) in the user's own $ units.
#
# The thesis (MVP-D): credence-pi adapts to the user's trade-off — no single fixed router is
# welfare-optimal for ALL profiles, but credence-pi's per-profile routing is. The time⟷money
# divergence is the batch-saver (latency-indifferent ⇒ free-slow qwen) vs speed-first
# (time-precious ⇒ paid-fast haiku) contrast; the broad quality⟷cost divergence is
# cost-saver/speed-first (haiku) vs quality-first (sonnet).
#
# NON-CAUSAL read-out (Role: eval): scores already-realized outcomes; no belief here drives a
# real action (the welfare.jl / escalation_live.jl test-oracle carve-out).
#
# Run (daemon need not be up — routing is computed in-process via the same RoutingBrain the
# daemon loads, with the latency belief that includes the measured qwen tier):
#   julia --project=. apps/credence-pi/eval/live_ab/oc_welfare_score.jl [matrix.jsonl]

using JSON3, Printf
using Statistics: mean

push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..", "..", "..", "src")))
using Credence
using Credence: Eval, Parse

include(joinpath(@__DIR__, "..", "..", "brain", "feature_brain.jl"))
include(joinpath(@__DIR__, "..", "..", "brain", "routing_brain.jl"))
using .RoutingBrain: wire_routing!

const BDSL_DIR = normpath(joinpath(@__DIR__, "..", "..", "bdsl"))
const MATRIX = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "results", "welfare_matrix.jsonl")

# tier -> (belief id used by the routing/latency beliefs, per-call cost $ = out-$/Mtok × 700).
# Matches roster.ts NOMINAL_OUTPUT_TOKENS=700 so the routing decision sees the same cost scale
# the live plugin sends. qwen is free.
const TIERS  = ["qwen", "haiku", "sonnet"]
const BELIEF = Dict("qwen" => "qwen2.5:7b-instruct",
                    "haiku" => "claude-haiku-4-5", "sonnet" => "claude-sonnet-4-6")
const COST   = Dict("qwen" => 0.0, "haiku" => 5.0 * 700 / 1e6, "sonnet" => 15.0 * 700 / 1e6)

# Profiles = points in (reward, w_time) space, calibrated to defensible absolute $ (no arbitrary
# constants): reward = $ value of a solved task — a correct easy task ≈ 3 min of a $60/hr dev's
# time ≈ $3 (constant: these easy tasks have the same value to every user); w_time = the user's
# own $ per SECOND of wall-clock, the dial that distinguishes the profiles. This calibrates the
# shipped presets (which encode relative shape) to the absolute units a welfare number needs.
# The time⟷money axis: a latency-indifferent batch user vs a time-precious one.
const REWARD = 3.0
const PROFILES = [
    ("batch-saver", REWARD, 0.0),      # $0/hr — latency genuinely unvalued (overnight/batch)
    ("cost-saver",  REWARD, 0.003),    # ~$11/hr — time cheap
    ("balanced",    REWARD, 0.017),    # ~$60/hr — a typical dev
    ("speed-first", REWARD, 0.042),    # ~$150/hr — time precious
]

# ── load the matrix: (tier, task) -> per-rep (solved, cost, time_s) ──────────────────────
function load_matrix(path)
    rows = [JSON3.read(ln) for ln in eachline(path) if !isempty(strip(ln))]
    cells = Dict{Tuple{String,String}, Vector{NTuple{3,Float64}}}()
    tasks = String[]
    for r in rows
        task = String(r.task); tier = String(r.tier)
        task in tasks || push!(tasks, task)
        # time the user waits: wall-clock, consistent across tiers (qwen has no model-only
        # durationMs; the ~constant startup overhead cancels in cross-tier comparison).
        t_s = Float64(r.wall_s)
        push!(get!(cells, (tier, task), NTuple{3,Float64}[]),
              (r.solved === true ? 1.0 : 0.0, Float64(r.cost_usd), t_s))
    end
    # mean over reps
    agg = Dict{Tuple{String,String}, NTuple{3,Float64}}()
    for (k, v) in cells
        agg[k] = (mean(getindex.(v, 1)), mean(getindex.(v, 2)), mean(getindex.(v, 3)))
    end
    (agg, tasks)
end

# in-process route-decide (the daemon's exact closure; test_routing.jl §B ≡ live route-request)
function make_router()
    env = Eval.default_env(); env[:__toplevel__] = true
    for f in ("utility.bdsl", "routing.bdsl")
        for expr in Parse.parse_all(read(joinpath(BDSL_DIR, f), String))
            Eval.eval_dsl(expr, env)
        end
    end
    # Point routing at the eval latency belief (measured per-tier wall-time incl. the free qwen
    # tier; the shipped TB belief has no qwen entry ⇒ would read qwen as 0s ⇒ speed-first would
    # wrongly pick it). Same env key the daemon honours (routing_brain.jl wire_routing!).
    env[Symbol("routing-latency-path")] =
        normpath(joinpath(@__DIR__, "results", "welfare_latency.counts.json"))
    # COLD accuracy belief (env routing-brain-path=""): the shipped warm seed is from HARD agentic
    # TB tasks (θ haiku 0.39, sonnet 0.70) and mis-ranks this EASY regime, where every tier solves
    # equally (measured in the matrix — all 6/6). Cold ⇒ equal θ ⇒ routing is COST/TIME-driven,
    # the welfare-optimal call when quality is equal, and what online learning converges to here.
    # The quality⟷cost trade (a stronger model earning its cost) is the separate HARD-task result
    # (eval/results/ROUTING_DOMINANCE.md), not demonstrable on tasks every model solves.
    env[Symbol("routing-brain-path")] = ""
    wire_routing!(env)
    env[Symbol("route-decide")]
end

# the live roster the body would send (belief ids + per-call costs), cost-ascending.
const ROSTER = Any[[t == "qwen" ? "qwen" : t, t == "qwen" ? "ollama" : "anthropic",
                    BELIEF[t], COST[t]] for t in TIERS]

route_tier(decide, reward, w_time) = begin
    choice = decide(Dict("prompt-length" => "short"), ROSTER,
                    Dict("reward" => reward, "w_time" => w_time))
    choice === nothing && return nothing
    m = choice["model"]
    for (t, b) in BELIEF; b == m && return t; end
    nothing
end

welfare(agg, tasks, tier, reward, w_time) =
    sum(tasks; init = 0.0) do task
        haskey(agg, (tier, task)) || return 0.0
        s, c, t = agg[(tier, task)]
        reward * s - c - w_time * t
    end

function main()
    isfile(MATRIX) || error("matrix not found: $MATRIX (run oc_welfare_matrix.py first)")
    agg, tasks = load_matrix(MATRIX)
    # tasks with all three tiers present (fair policy comparison)
    full = [t for t in tasks if all(haskey(agg, (tr, t)) for tr in TIERS)]
    decide = make_router()

    println("MVP-D welfare A/B — credence-pi per-profile routing vs fixed routers")
    println("  matrix=", MATRIX, "  tasks(full)=", length(full), "/", length(tasks), "\n")

    # per-tier realized means (the raw matrix — the money⟷time spread to point at)
    @printf("  %-7s %7s %9s %8s\n", "tier", "solved", "cost\$", "time_s")
    for tr in TIERS
        cells = [agg[(tr, t)] for t in full]
        @printf("  %-7s %7.2f %9.4f %8.1f\n", tr,
                mean(getindex.(cells, 1)), mean(getindex.(cells, 2)), mean(getindex.(cells, 3)))
    end
    println()

    policies = ["credence-pi"; ["always-$t" for t in TIERS]]
    println("per-profile welfare (Σ reward·solved − money − w_time·time, the user's \$):")
    for (pname, reward, wt) in PROFILES
        rt = route_tier(decide, reward, wt)
        W = Dict(p => (p == "credence-pi" ? welfare(agg, full, rt, reward, wt) :
                       welfare(agg, full, replace(p, "always-" => ""), reward, wt))
                 for p in policies)
        best = argmax(p -> W[p], policies)
        cp_optimal = W["credence-pi"] ≈ maximum(values(W)) || W["credence-pi"] >= W[best] - 1e-9
        @printf("\nprofile=%-13s reward=%.2f w_time=%.3f  → credence-pi routes: %s\n",
                pname, reward, wt, rt === nothing ? "(inert)" : uppercase(rt))
        for p in policies
            @printf("  %-14s welfare=%+8.4f%s\n", p, W[p],
                    p == "credence-pi" ? "   <<<" : (p == "always-$(rt)" ? "  (= credence-pi's pick)" : ""))
        end
        @printf("  -> credence-pi is %s for this profile\n",
                cp_optimal ? "WELFARE-OPTIMAL" : "NOT optimal (best=$best)")
    end

    # The headline divergence, explicit.
    println("\n", "="^64)
    bt = route_tier(decide, 0.02, 0.0); st = route_tier(decide, 1.0, 0.05)
    @printf("time⟷money divergence: batch-saver → %s (free, slow) ; speed-first → %s (paid, fast)\n",
            uppercase(something(bt, "?")), uppercase(something(st, "?")))
    println("="^64)
end

main()
