# Role: eval
#
# train_warm_brain.jl — distil a shippable WARM prior from corpus replay.
#
# Trains the brain on real-session loop-waste feedback through the canonical
# `observe`/`condition` path, then SERIALIZES the resulting posterior so the
# daemon can load it as its starting belief (instead of cold Beta(2,2)) and
# auto-block known waste from install. The warm brain is aggregate per-context
# Beta counts — no raw session data — so it is safe to ship.
#
# The artifact is the serialized posterior (binary, deserialized by the daemon's
# Credence types). An auditable provenance JSON is written alongside via PUBLIC
# brain ops only (no private parameter reads → no expect-through-accessor issue).
#
# Run:
#   julia --project=<repo-root> apps/credence-pi/eval/train_warm_brain.jl \
#       --events data/credence_pi_eval/clawsbench_openclaw.events.jsonl \
#       --out-brain apps/credence-pi/brain/warm_brain.jls \
#       --out-provenance apps/credence-pi/brain/warm_brain.provenance.json \
#       --corpus "benchflow/ClawsBench@e7c45cc9 (openclaw)" --call-cost 0.01

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using Credence
using JSON3
using Serialization: serialize
using Dates: now

include(joinpath(@__DIR__, "brain_env.jl"))

function parse_args(argv)
    a = Dict{String,Any}("events"=>"", "out-brain"=>"", "out-provenance"=>"",
                         "corpus"=>"unknown", "call-cost"=>0.01)
    i = 1
    while i <= length(argv)
        t = argv[i]
        if t == "--events"; a["events"]=argv[i+1]; i+=2
        elseif t == "--out-brain"; a["out-brain"]=argv[i+1]; i+=2
        elseif t == "--out-provenance"; a["out-provenance"]=argv[i+1]; i+=2
        elseif t == "--corpus"; a["corpus"]=argv[i+1]; i+=2
        elseif t == "--call-cost"; a["call-cost"]=parse(Float64, argv[i+1]); i+=2
        else; error("unknown arg $t"); end
    end
    (isempty(a["events"]) || isempty(a["out-brain"])) && error("--events and --out-brain required")
    a
end

read_events(p) = [JSON3.read(l, Dict{String,Any}) for l in eachline(p) if !isempty(strip(l))]
features_of(e) = Dict{String,String}(string(k)=>string(v) for (k,v) in e["features"])

function main()
    args = parse_args(ARGS)
    env = build_env()
    make_prior = env[Symbol("make-prior")]
    decide     = env[Symbol("decide-action")]
    observe    = env[Symbol("observe-response")]

    events = read_events(args["events"])
    # loop-waste feedback proxy (exact (tool,command) repeat WITHIN a session ->
    # deny). `seen` MUST reset per session_id — a cross-session set saturates and
    # labels nearly everything a loop (39% vs the true ~4.5%).
    # A loop counts ONLY when the command is genuinely distinguishable (present
    # and ≠ the tool name) — see replay.jl. Collapsed-arg tools (ClawsBench
    # `read`) are never loop-denied, so the warm brain does not learn to distrust
    # legitimate sequential reads.
    seen = Set{Tuple{String,String}}(); cur = nothing; n_loop = 0
    top = make_prior()
    for e in events
        sid = String(get(e,"session_id",""))
        if sid != cur; seen = Set{Tuple{String,String}}(); cur = sid; end
        tool = String(get(e,"tool_name",""))
        cmdv = get(get(e,"meta",Dict()), "command", nothing)
        cmd = cmdv===nothing ? "" : String(cmdv)
        isloop = (!isempty(cmd) && cmd != tool) ? ((tool,cmd) in seen) : false
        (!isempty(cmd) && cmd != tool) && push!(seen, (tool,cmd))
        n_loop += isloop
        top = observe(top, features_of(e), isloop ? 0 : 1)
    end

    # serialize the shippable warm posterior
    mkpath(dirname(args["out-brain"]))
    serialize(args["out-brain"], top)
    sz = stat(args["out-brain"]).size

    # auditable provenance via PUBLIC decide() only: for the distinct contexts
    # observed, what does the warm brain now decide? (no raw param reads)
    distinct = Dict{Tuple,String}()
    cc = args["call-cost"]
    for e in events
        X = features_of(e)
        k = Tuple(X[name] for name in sort(collect(keys(X))))
        haskey(distinct, k) && continue
        distinct[k] = string(decide(top, X, cc))
    end
    dcounts = Dict("proceed"=>0,"block"=>0,"ask"=>0)
    for d in values(distinct); dcounts[d] += 1; end

    prov = Dict(
        "artifact" => basename(args["out-brain"]),
        "bytes" => sz,
        "corpus" => args["corpus"],
        "trained_observations" => length(events),
        "loop_denials" => n_loop,
        "approvals" => length(events) - n_loop,
        "distinct_contexts" => length(distinct),
        "warm_decision_over_distinct_contexts" => dcounts,
        "call_cost_assumption" => cc,
        "trained_at" => string(now()),
        "note" => "Aggregate per-context Beta counts only; no raw session data. " *
                  "A PRIOR: each user's own usage continues to update it via condition().",
    )
    if !isempty(args["out-provenance"])
        open(args["out-provenance"], "w") do io; JSON3.pretty(io, prov); end
    end
    println("warm brain → $(args["out-brain"]) ($(round(sz/1024;digits=1)) KiB)")
    println("trained on $(length(events)) obs ($n_loop loop-denials); " *
            "distinct contexts $(length(distinct)): $dcounts")
end

main()
