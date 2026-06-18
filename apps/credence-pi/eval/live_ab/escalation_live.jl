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
using Statistics: mean, quantile
using Random: MersenneTwister

const DAEMON  = get(ENV, "CREDENCE_PI_DAEMON", "http://127.0.0.1:8787")
const BOOT_SEED = 20260618        # seeded ⇒ reproducible CIs (eval, non-causal)
const BOOT_B    = 10_000          # bootstrap resamples
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

# Cluster bootstrap. Resample the K scored TASKS with replacement (carrying ALL reps of a drawn
# task together); the SAME drawn-task set is applied to every arm, so the paired escalation−fixed
# difference stays valid. Returns (per-arm 95% CI, escalation−best_fixed 95% CI, one-sided
# p(esc ≤ best)). CAVEAT: K≈17 clusters ⇒ the percentile interval UNDER-COVERS — read as
# indicative; BCa is the rigorous upgrade. (Non-causal eval arithmetic over realized welfare.)
function cluster_bootstrap(Wv_p, arms, task_ids, best_fixed)
    rng = MersenneTwister(BOOT_SEED)
    K = length(task_ids)
    arm_boot = Dict(a => Vector{Float64}(undef, BOOT_B) for a in arms)
    diff_boot = Vector{Float64}(undef, BOOT_B)
    for b in 1:BOOT_B
        idx = rand(rng, 1:K, K)                       # resample tasks with replacement
        for a in arms
            s = 0.0; n = 0
            for j in idx, w in Wv_p[a][task_ids[j]]   # concatenate all reps of each drawn task
                s += w; n += 1
            end
            arm_boot[a][b] = n == 0 ? 0.0 : s / n
        end
        diff_boot[b] = arm_boot["escalation"][b] - arm_boot[best_fixed][b]
    end
    arm_ci = Dict(a => quantile(arm_boot[a], [0.025, 0.975]) for a in arms)
    (arm_ci, quantile(diff_boot, [0.025, 0.975]), count(<=(0.0), diff_boot) / BOOT_B)
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
    # Per-world welfare GROUPED BY TASK (task -> [welfare per rep]) for the cluster bootstrap:
    # reps of a task are correlated (a hard task is hard in all reps), so the bootstrap resamples
    # TASKS — not the 51 (task,rep) rows i.i.d., which would be pseudo-replication and shrink the
    # CIs (a too-narrow interval is how a within-noise margin spuriously excludes zero).
    Wv = Dict(p[1] => Dict(a => Dict{String, Vector{Float64}}() for a in arms) for p in PROFILES)
    # Per-task SOLVE indicators — the capability-union signal (escalation captures the union of
    # tier capabilities). Solve-rate has lower variance than welfare (no cost term), so it is the
    # more robust comparison; welfare variance is what washes out the quality-hawk margin.
    Sv = Dict(p[1] => Dict(a => Dict{String, Vector{Float64}}() for a in arms) for p in PROFILES)
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
                w_esc = reward * (solved ? 1 : 0) - paid
                W[pname]["escalation"] += w_esc
                S[pname]["escalation"] += solved ? 1 : 0
                C[pname]["escalation"] += paid
                push!(get!(Wv[pname]["escalation"], task, Float64[]), w_esc)
                push!(get!(Sv[pname]["escalation"], task, Float64[]), solved ? 1.0 : 0.0)
                for (i, t) in enumerate(TIERS)                        # fixed single-model arms
                    w_fix = reward * (resolved[i] ? 1 : 0) - realcost[i]
                    W[pname]["always-$t"] += w_fix
                    S[pname]["always-$t"] += resolved[i] ? 1 : 0
                    C[pname]["always-$t"] += realcost[i]
                    push!(get!(Wv[pname]["always-$t"], task, Float64[]), w_fix)
                    push!(get!(Sv[pname]["always-$t"], task, Float64[]), resolved[i] ? 1.0 : 0.0)
                end
            end
        end
    end

    task_ids = sort(collect(keys(Wv[PROFILES[1][1]]["escalation"])))

    println("Live-daemon observe-then-escalate vs fixed policies")
    println("  daemon=", DAEMON, "  worlds(tasks×reps)=", nworlds,
            "  tasks=", length(task_ids), "  belief=COLD (leakage-free, conservative)")
    println("  CIs: cluster bootstrap B=", BOOT_B, " resampling ", length(task_ids),
            " tasks (seed ", BOOT_SEED, "); percentile under-covers at K=", length(task_ids),
            " — indicative, BCa is the rigorous upgrade\n")
    fixed_arms = ["always-$t" for t in TIERS]
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
        @printf("  escalation vs always-sonnet: Δwelfare=%+.4f  Δcost=%+.4f  Δsolves=%+.3f\n",
                dw, dc, ds)
        # Cluster-bootstrap CIs + the decision-relevant escalation−best_fixed comparison.
        best_fixed = argmax(a -> W[pname][a], fixed_arms)
        arm_ci, diff_ci, p1 = cluster_bootstrap(Wv[pname], arms, task_ids, best_fixed)
        println("  95% CI (cluster bootstrap):")
        for a in arms
            @printf("    %-14s welfare %+.4f  [%+.4f, %+.4f]%s\n", a, W[pname][a] / nworlds,
                    arm_ci[a][1], arm_ci[a][2], a == best_fixed ? "  (best fixed)" : "")
        end
        dbest = (W[pname]["escalation"] - W[pname][best_fixed]) / nworlds
        @printf("  escalation − best_fixed (%s) WELFARE: Δ=%+.4f  CI [%+.4f, %+.4f]  one-sided p(esc≤best)=%.3f\n",
                best_fixed, dbest, diff_ci[1], diff_ci[2], p1)
        # Capability-union: solve-rate vs the best-SOLVING fixed tier (lower-variance signal).
        best_solver = argmax(a -> S[pname][a], fixed_arms)
        _, sdiff_ci, sp1 = cluster_bootstrap(Sv[pname], arms, task_ids, best_solver)
        dsolve = (S[pname]["escalation"] - S[pname][best_solver]) / nworlds
        @printf("  escalation − best_solver (%s) SOLVE-RATE: Δ=%+.3f  CI [%+.3f, %+.3f]  one-sided p(esc≤best)=%.3f\n\n",
                best_solver, dsolve, sdiff_ci[1], sdiff_ci[2], sp1)
    end
end

main()
