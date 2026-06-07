# Role: eval
"""
    replay.jl — offline replay of normalized tool-call events through the
    credence-pi feature-conditioned brain.

Reads the NormalizedEvent JSONL produced by extract.ts and answers, on real
recorded sessions and at zero spend:

  * COLD baseline — a fresh brain (per-session `make-prior`) decides each call.
    Cold-start asks almost everything (the Pass-1 problem the warm brain fixes).
  * WARM — the brain is trained on a training split (learning from a loop-waste
    feedback proxy through the canonical `observe`/`condition`), the posterior is
    FROZEN, and it decides the held-out test split. This is the deployed shape.

Both decisions go through the SAME wired env closures the daemon uses
(`make-prior`, `decide-action`, `observe-response`) — no reimplemented axiom op,
no host-side arithmetic on probabilities. The objective waste label (an exact
repeated `(tool, command)` within a session) is computed for CALIBRATION only and
is never fed to a decision.

"Record everything": every test-split call is written to a per-call JSONL with
its decision, both arms, the objective label, and provenance.

Run:
    julia --project=<repo-root> apps/credence-pi/eval/replay.jl \
        --events data/credence_pi_eval/clawsbench_openclaw.events.jsonl \
        --out    data/credence_pi_eval/clawsbench_openclaw.replay.jsonl \
        [--train-frac 0.7] [--seed 0] [--call-cost 0.01]
"""

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using Credence
using Credence: Eval, Parse
using JSON3
using Random: MersenneTwister, shuffle!
using Statistics: mean, std

include(joinpath(@__DIR__, "..", "brain", "feature_brain.jl"))
using .FeatureBrain: wire_brain!

# ── Env: the daemon's exact program load (stdlib + declared bdsl + wire) ──
function build_env()
    env = Eval.default_env()
    env[:__toplevel__] = true
    stdlib = joinpath(@__DIR__, "..", "..", "..", "src", "stdlib.bdsl")
    for expr in Parse.parse_all(read(stdlib, String))
        Eval.eval_dsl(expr, env)
    end
    bdsl = joinpath(@__DIR__, "..", "bdsl")
    for f in ("capabilities.bdsl", "features.bdsl", "utility.bdsl")
        for expr in Parse.parse_all(read(joinpath(bdsl, f), String))
            Eval.eval_dsl(expr, env)
        end
    end
    wire_brain!(env)
    env
end

# ── Args ──
function parse_args(argv)
    a = Dict{String,Any}("train-frac" => 0.7, "seed" => 0, "call-cost" => 0.01,
                         "events" => "", "out" => "")
    i = 1
    while i <= length(argv)
        t = argv[i]
        if t == "--events"; a["events"] = argv[i+1]; i += 2
        elseif t == "--out"; a["out"] = argv[i+1]; i += 2
        elseif t == "--train-frac"; a["train-frac"] = parse(Float64, argv[i+1]); i += 2
        elseif t == "--seed"; a["seed"] = parse(Int, argv[i+1]); i += 2
        elseif t == "--call-cost"; a["call-cost"] = parse(Float64, argv[i+1]); i += 2
        else; error("unknown arg $t"); end
    end
    isempty(a["events"]) && error("--events required")
    a
end

# ── Load events grouped by session, in file order ──
function load_sessions(path)
    sessions = Vector{Tuple{String,Vector{Dict{String,Any}}}}()
    index = Dict{String,Int}()
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            e = JSON3.read(line, Dict{String,Any})
            sid = String(e["session_id"])
            if !haskey(index, sid)
                push!(sessions, (sid, Dict{String,Any}[]))
                index[sid] = length(sessions)
            end
            push!(sessions[index[sid]][2], e)
        end
    end
    sessions
end

# Objective waste label (calibration ground truth, never fed to a decision): an
# exact repeat of a previous (tool, command) within the same session is a loop.
# A call counts ONLY when its command is genuinely distinguishable (present and
# ≠ the tool name) — some corpora collapse a tool's args into command==toolname
# (e.g. ClawsBench `read`), and treating those repeats as loops would label
# every legitimate sequential read a loop. We only claim a loop when we can
# actually see the args repeat.
distinguishable(tool, cmd) = !isempty(cmd) && cmd != tool
function mark_loops!(events::Vector{Dict{String,Any}})
    seen = Set{Tuple{String,String}}()
    for e in events
        tool = String(get(e, "tool_name", ""))
        cmdv = get(get(e, "meta", Dict()), "command", nothing)
        cmd = cmdv === nothing ? "" : String(cmdv)
        if distinguishable(tool, cmd)
            key = (tool, cmd)
            e["is_loop"] = key in seen
            push!(seen, key)
        else
            e["is_loop"] = false
        end
    end
end

features_of(e) = Dict{String,String}(string(k) => string(v) for (k, v) in e["features"])

# ── Main ──
function main()
    args = parse_args(ARGS)
    env = build_env()
    make_prior = env[Symbol("make-prior")]
    decide     = env[Symbol("decide-action")]
    observe    = env[Symbol("observe-response")]
    call_cost  = args["call-cost"]

    sessions = load_sessions(args["events"])
    for (_, evs) in sessions; mark_loops!(evs); end

    rng = MersenneTwister(args["seed"])
    order = collect(1:length(sessions)); shuffle!(rng, order)
    n_train = round(Int, args["train-frac"] * length(sessions))
    train_ids = Set(order[1:n_train])

    # WARM: one global posterior, trained on the train split via the loop-waste
    # feedback proxy (loop -> deny obs=0, else approve obs=1) through `observe`.
    warm = make_prior()
    train_obs = 0
    for (i, (_, evs)) in enumerate(sessions)
        i in train_ids || continue
        for e in evs
            X = features_of(e)
            obs = e["is_loop"] ? 0 : 1
            warm = observe(warm, X, obs)
            train_obs += 1
        end
    end

    # TEST: decide each held-out call with the frozen WARM brain and with a
    # fresh COLD brain. Record everything.
    out = open(args["out"], "w")
    cold_prior = make_prior()
    n = 0
    counts = Dict("warm" => Dict("proceed"=>0,"block"=>0,"ask"=>0),
                  "cold" => Dict("proceed"=>0,"block"=>0,"ask"=>0))
    # calibration tallies for WARM block-vs-loop
    tp = fp = tn = fn = 0
    loops = 0
    for (i, (sid, evs)) in enumerate(sessions)
        i in train_ids && continue
        for e in evs
            X = features_of(e)
            dw = string(decide(warm, X, call_cost))
            dc = string(decide(cold_prior, X, call_cost))
            counts["warm"][dw] += 1
            counts["cold"][dc] += 1
            isloop = e["is_loop"]::Bool
            loops += isloop ? 1 : 0
            blocked = dw == "block"
            if isloop && blocked; tp += 1
            elseif !isloop && blocked; fp += 1
            elseif !isloop && !blocked; tn += 1
            else; fn += 1; end
            rec = Dict{String,Any}(
                "session_id" => sid, "idx" => e["idx"],
                "tool_name" => get(e, "tool_name", ""),
                "features" => e["features"], "is_loop" => isloop,
                "failed" => get(e, "failed", false),
                "decision_warm" => dw, "decision_cold" => dc,
                "model" => get(get(e, "meta", Dict()), "model", nothing),
                "task_name" => get(get(e, "meta", Dict()), "task_name", nothing),
            )
            println(out, JSON3.write(rec))
            n += 1
        end
    end
    close(out)

    # ── Summary ──
    prec = tp + fp == 0 ? NaN : tp / (tp + fp)
    rec  = tp + fn == 0 ? NaN : tp / (tp + fn)
    warm_block = counts["warm"]["block"]; cold_block = counts["cold"]["block"]
    println("── credence-pi replay summary ──")
    println("sessions: $(length(sessions))  (train $(n_train), test $(length(sessions)-n_train))")
    println("train observations: $train_obs")
    println("test calls: $n   loops (objective waste): $loops ($(round(100loops/n;digits=1))%)")
    println()
    println("WARM (trained, frozen):  proceed=$(counts["warm"]["proceed"])  block=$warm_block  ask=$(counts["warm"]["ask"])")
    println("COLD (fresh prior):      proceed=$(counts["cold"]["proceed"])  block=$cold_block  ask=$(counts["cold"]["ask"])")
    println()
    println("WARM block-vs-loop calibration:")
    println("  precision (blocked & loop / all blocked) = $(round(prec;digits=3))   [doesn't-block-useful-calls]")
    println("  recall    (blocked & loop / all loops)   = $(round(rec;digits=3))   [catches-waste]")
    println("  tp=$tp fp=$fp tn=$tn fn=$fn")
    println()
    println("prevented calls (WARM auto-block): $warm_block")
    println("prevented spend @ \$$(call_cost)/call (LABELLED ESTIMATE, no per-call tokens in corpus): \$$(round(warm_block*call_cost;digits=2))")
    println()
    println("per-call records: $(args["out"])")
end

main()
