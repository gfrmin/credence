# Role: eval
#
# train_warm_brain.jl — distil the shippable WARM WASTE prior P(approve|waste-features).
#
# Trains the waste structure-BMA on real-session loop-waste feedback through the
# canonical `observe`/`condition` path, then writes the posterior as VERSION-STABLE
# per-context COUNTS (JSON) — the daemon reconstructs it by replaying `observe`
# (order-independent, so identical), avoiding the Julia-version fragility of
# `Serialization` (CI/image pin 1.11; dev may be newer — this repo runs 1.12). It is
# aggregate per-context Beta counts (n1=approvals, n0=loop-denials), no raw session
# data, so it is safe to ship.
#
# Unlike the FROZEN harm posterior, the warm waste brain is a PRIOR: each user's own
# usage keeps updating it live via `condition`. Shipping it (vs cold Beta(2,2)) is
# what lets the daemon AUTO-BLOCK known waste from install — unattended, no cold-start
# ask-everything — which is what the live A/B needs to govern without a human.
#
# Run from repo root:
#   julia --project=. apps/credence-pi/eval/train_warm_brain.jl \
#       --events data/credence_pi_eval/clawsbench_openclaw.events.jsonl \
#       --out-counts apps/credence-pi/brain/warm_brain.counts.json \
#       --corpus "benchflow/ClawsBench@e7c45cc9 (openclaw)" --call-cost 0.01

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using Credence
using JSON3
using Dates: now
include(joinpath(@__DIR__, "brain_env.jl"))
using .FeatureBrain: build_model_from_env, build_prior, observe, context_from_features,
    belief_at_context, decide, reconstruct_posterior

read_events(p) = [JSON3.read(l, Dict{String,Any}) for l in eachline(p) if !isempty(strip(l))]
features_of(e) = Dict{String,String}(string(k) => string(v) for (k, v) in e["features"])

function parse_args(argv)
    a = Dict{String,Any}("events"=>"", "out-counts"=>"", "out-provenance"=>"",
                         "corpus"=>"unknown", "call-cost"=>0.01)
    i = 1
    while i <= length(argv)
        t = argv[i]
        if t == "--events"; a["events"]=argv[i+1]; i+=2
        elseif t == "--out-counts"; a["out-counts"]=argv[i+1]; i+=2
        elseif t == "--out-provenance"; a["out-provenance"]=argv[i+1]; i+=2
        elseif t == "--corpus"; a["corpus"]=argv[i+1]; i+=2
        elseif t == "--call-cost"; a["call-cost"]=parse(Float64, argv[i+1]); i+=2
        else; error("unknown arg $t"); end
    end
    (isempty(a["events"]) || isempty(a["out-counts"])) && error("--events and --out-counts required")
    a
end

function main()
    a = parse_args(ARGS)
    env = build_env()
    model = build_model_from_env(env)
    λ = Float64(get(env, Symbol("false-block-aversion"), 1.0))
    q = Float64(get(env, Symbol("interrupt-cost"), 0.02))

    # loop-waste feedback proxy: an exact (tool,command) repeat WITHIN a session →
    # deny (obs=0); else approve (obs=1). `seen` MUST reset per session_id — a
    # cross-session set saturates and mislabels ~everything a loop. A loop counts
    # ONLY when the command is genuinely distinguishable (present and ≠ tool name)
    # — see replay.jl — so collapsed-arg tools (ClawsBench `read`) are never
    # loop-denied and the brain doesn't learn to distrust legitimate reads.
    top = build_prior(model)
    counts = Dict{Vector{String}, Vector{Int}}()   # context → [n1 (approve), n0 (loop-deny)]
    seen = Set{Tuple{String,String}}(); cur = nothing; nloop = 0; ncalls = 0
    for e in read_events(a["events"])
        sid = String(get(e, "session_id", ""))
        if sid != cur; seen = Set{Tuple{String,String}}(); cur = sid; end
        tool = String(get(e, "tool_name", ""))
        cmdv = get(get(e, "meta", Dict()), "command", nothing)
        cmd = cmdv === nothing ? "" : String(cmdv)
        isloop = (!isempty(cmd) && cmd != tool) ? ((tool, cmd) in seen) : false
        (!isempty(cmd) && cmd != tool) && push!(seen, (tool, cmd))
        obs = isloop ? 0 : 1
        ctx = context_from_features(model, features_of(e))
        top = observe(model, top, ctx, obs); ncalls += 1; nloop += isloop
        c = get!(counts, ctx, [0, 0]); c[obs == 1 ? 1 : 2] += 1
    end

    # Auditable provenance via PUBLIC decide() only (no private param reads): for each
    # distinct trained context, what does the warm brain now decide (default profile)?
    cc = a["call-cost"]
    dcounts = Dict("proceed"=>0, "block"=>0, "ask"=>0)
    for ctx in keys(counts)
        d = string(decide(model, top, ctx, cc; aversion = λ, interrupt_cost = q))
        dcounts[d] += 1
    end

    contexts = [Dict("ctx" => k, "n1" => v[1], "n0" => v[2]) for (k, v) in counts]
    out = Dict(
        "artifact" => "credence-pi warm waste posterior P(approve|waste-features) — per-context counts",
        "corpus" => a["corpus"], "features" => model.feature_names, "feature_values" => model.feature_values,
        "n_calls" => ncalls, "loop_denials" => nloop, "approvals" => ncalls - nloop,
        "n_contexts" => length(contexts), "trained_at" => string(now()), "frozen" => false,
        "call_cost_assumption" => cc,
        "warm_decision_over_distinct_contexts" => dcounts,
        "note" => "A PRIOR: each user's own usage continues to update it via condition(). " *
                  "Daemon reconstructs by replaying these counts via observe " *
                  "(order-independent ⇒ identical; version-stable, unlike Serialization).",
        "contexts" => contexts)
    open(a["out-counts"], "w") do io; JSON3.pretty(io, out); end
    println("warm brain → $(a["out-counts"])  ($(length(contexts)) contexts, $ncalls calls, $nloop loop-denials)")
    println("warm decision over distinct contexts: $dcounts")

    if !isempty(a["out-provenance"])
        open(a["out-provenance"], "w") do io; JSON3.pretty(io, out); end
    end

    # Verify the reconstruction reproduces the directly-trained posterior exactly.
    rt = reconstruct_posterior(model, a["out-counts"])
    using_credence_identity = Credence.Identity()
    maxdiff = 0.0
    for ctx in collect(keys(counts))[1:min(50, length(counts))]
        maxdiff = max(maxdiff, abs(
            Credence.expect(belief_at_context(model, top, ctx), using_credence_identity) -
            Credence.expect(belief_at_context(model, rt, ctx), using_credence_identity)))
    end
    println("reconstruction vs direct training: max θ_a diff = $maxdiff",
            maxdiff < 1e-9 ? "  ✓ exact" : "  ✗ MISMATCH")
end

main()
