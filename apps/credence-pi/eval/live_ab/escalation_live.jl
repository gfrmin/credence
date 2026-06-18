# Role: eval
#
# escalation_live.jl — drive the OBSERVE-THEN-ESCALATE gate THROUGH the live daemon over the
# Terminal-Bench matrix, scoring realized welfare vs every fixed single-model policy.
#
# WHY: EXPERIMENT.md honest-limit (c) said the EU escalation gate "lives in the brain
# (RoutingBrain.escalation_next) and the eval calls it, but is not yet wired into the live
# daemon loop (offline eval result)." This closes it: every {try,stop} decision here is a
# round-trip to the running daemon's `escalate-request` handler (real daemon code, over HTTP),
# not an in-process call. The companion unit test (tests/julia/test_routing.jl) asserts the
# daemon decision equals in-process `escalation_next` on identical inputs, so the proof's
# welfare transfers to the live system.
#
# LEAKAGE-FREE BY CONSTRUCTION:
#   • Accuracy belief is COLD (run the daemon with CREDENCE_PI_ROUTING_BRAIN=""), so the
#     shipped warm belief — which was trained on THIS matrix — never sees the scored tasks.
#     Cold θ ⇒ the gate is purely "is the next rung worth its cost vs reward·0.5"; the
#     observed `resolved` (ground truth) drives escalation. This is a CONSERVATIVE lower
#     bound: the deployed warm belief can only sharpen the gate.
#   • The gate's only matrix-derived input is per-(tier,difficulty) EXPECTED cost — and it is
#     computed LEAVE-ONE-TASK-OUT, so the scored task never informs its own gate. Cost is not
#     the label (`resolved`) in any case.
#
# NON-CAUSAL read-out (Role: eval): scores realized outcomes; no belief here drives a real
# action. Run (daemon must be up COLD on $CREDENCE_PI_DAEMON):
#   CREDENCE_PI_ROUTING_BRAIN="" julia --project=. apps/credence-pi/daemon/main.jl &   # cold
#   julia --project=. apps/credence-pi/eval/live_ab/escalation_live.jl

using HTTP, JSON3, Printf
using Statistics: mean

const DAEMON  = get(ENV, "CREDENCE_PI_DAEMON", "http://127.0.0.1:8787")
const MATRIX  = normpath(joinpath(@__DIR__, "results", "tb_matrix_rep3.jsonl"))
const TIERS   = ["haiku", "sonnet", "opus"]                               # cost-ascending
const IDS     = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"]
# Profiles = the per-call dollar value of a correct answer, identical to tb_dominance.jl.
const PROFILES = [("cost-hawk", 0.25), ("balanced", 1.0), ("quality-hawk", 5.0)]

# prompt-length bucket — MUST match routing-features.ts / routing.bdsl (short ≤400, ≤1000, >).
length_bucket(n) = n <= 400 ? "short" : (n <= 1000 ? "medium" : "long")

# One {try,stop} decision THROUGH the daemon: returns the cost-ascending tier index (1..K) of
# the next rung to try, or 0 for STOP. `tried` = the rungs already failed this attempt.
function ask_escalate(features, roster, tried, reward)
    body = JSON3.write(Dict("event_type" => "escalate-request", "event_id" => "esc",
        "features" => features, "models" => roster, "tried" => tried, "reward" => reward))
    r = HTTP.post("$DAEMON/sensor", ["Content-Type" => "application/json"], body; retry = false)
    a = JSON3.read(String(r.body))
    haskey(a, :route) ? Int(a.route.tier_index) : 0
end

function main()
    rows = [JSON3.read(ln) for ln in eachline(MATRIX) if !isempty(strip(ln))]

    # task -> (difficulty, instr_len, tier -> [(resolved, cost) per rep])
    tasks = Dict{String, Any}()
    for r in rows
        d = get!(tasks, String(r.task), Dict{String, Any}(
            "difficulty" => String(r.difficulty), "instr_len" => Int(r.instr_len),
            "tiers" => Dict{String, Vector{Tuple{Bool, Float64}}}()))
        push!(get!(d["tiers"], String(r.tier), Tuple{Bool, Float64}[]),
              (Bool(r.resolved), Float64(r.cost_usd)))
    end

    # Leave-one-task-out expected cost: E[cost | tier, difficulty] over OTHER tasks.
    function ecost(task, tier, diff)
        cs = Float64[Float64(r.cost_usd) for r in rows
                     if String(r.tier) == tier && String(r.difficulty) == diff &&
                        String(r.task) != task]
        isempty(cs) ? 0.0 : mean(cs)
    end

    arms = ["escalation"; ["always-$t" for t in TIERS]]
    W = Dict(p[1] => Dict(a => 0.0 for a in arms) for p in PROFILES)   # welfare
    S = Dict(p[1] => Dict(a => 0.0 for a in arms) for p in PROFILES)   # solves
    C = Dict(p[1] => Dict(a => 0.0 for a in arms) for p in PROFILES)   # cost
    nworlds = 0

    for (task, d) in tasks
        tl = d["tiers"]
        all(haskey(tl, t) for t in TIERS) || continue                 # need all 3 tiers
        diff = d["difficulty"]
        feats = Dict("prompt-length" => length_bucket(d["instr_len"]))
        roster = [Dict("name" => TIERS[i], "provider" => "anthropic", "model" => IDS[i],
                       "cost" => ecost(task, TIERS[i], diff)) for i in 1:length(TIERS)]
        R = minimum(length(tl[t]) for t in TIERS)
        for rep in 1:R
            resolved = Bool[tl[t][rep][1] for t in TIERS]
            realcost = Float64[tl[t][rep][2] for t in TIERS]
            nworlds += 1
            for (pname, reward) in PROFILES
                tried = Int[]; paid = 0.0; solved = false
                while true                                            # try → observe → escalate
                    a = ask_escalate(feats, roster, tried, reward)
                    a == 0 && break                                   # STOP (no positive-EU rung)
                    paid += realcost[a]                               # pay the realized cost
                    resolved[a] && (solved = true; break)             # observed solve → stop
                    push!(tried, a)                                   # failed → next rung
                end
                W[pname]["escalation"] += reward * (solved ? 1 : 0) - paid
                S[pname]["escalation"] += solved ? 1 : 0
                C[pname]["escalation"] += paid
                for (i, t) in enumerate(TIERS)                        # fixed single-model arms
                    W[pname]["always-$t"] += reward * (resolved[i] ? 1 : 0) - realcost[i]
                    S[pname]["always-$t"] += resolved[i] ? 1 : 0
                    C[pname]["always-$t"] += realcost[i]
                end
            end
        end
    end

    println("Live-daemon observe-then-escalate vs fixed policies")
    println("  daemon=", DAEMON, "  worlds(tasks×reps)=", nworlds,
            "  belief=COLD (leakage-free, conservative)\n")
    for (pname, _) in PROFILES
        println("profile=", pname, "  (per-world means)")
        @printf("  %-14s %9s %8s %9s\n", "arm", "welfare", "solves", "cost")
        for a in arms
            @printf("  %-14s %9.4f %8.3f %9.4f%s\n", a,
                    W[pname][a] / nworlds, S[pname][a] / nworlds, C[pname][a] / nworlds,
                    a == "escalation" ? "  <<<" : "")
        end
        # Explicit A/B vs the chosen baseline (always-sonnet).
        dw = (W[pname]["escalation"] - W[pname]["always-sonnet"]) / nworlds
        dc = (C[pname]["escalation"] - C[pname]["always-sonnet"]) / nworlds
        ds = (S[pname]["escalation"] - S[pname]["always-sonnet"]) / nworlds
        @printf("  escalation vs always-sonnet: Δwelfare=%+.4f  Δcost=%+.4f  Δsolves=%+.3f\n\n",
                dw, dc, ds)
    end
end

main()
