# Role: eval
#
# train_harm_brain.jl — distil the shippable FROZEN HARM posterior P(unsafe|taint-features).
#
# Trains the harm structure-BMA on the ATBench-Claw safety corpus through the canonical
# `observe`/`condition` path (reason-localized per-call attribution — the harmful call the
# human `reason` names; non-circular w.r.t. the taint feature), then SERIALIZES the posterior
# so the daemon loads it as a frozen belief and folds H·P(unsafe|X) into decide_multi. Unlike
# the waste warm brain, the harm posterior is NOT updated live (no live unsafe labels) — it is
# a fixed, shipped belief. It is aggregate per-context Beta counts — no raw data — safe to ship.
#
# The 4 harm features MUST match the `safety-features` declared in bdsl/features.bdsl and the
# taint computation in the body (openclaw-plugin) / eval adapter (adapters/risk.ts).
#
# Run from repo root:
#   julia --project=. apps/credence-pi/eval/train_harm_brain.jl \
#       --events data/credence_pi_eval/atbench_claw.events.jsonl \
#       --out-brain apps/credence-pi/brain/harm_brain.jls \
#       --out-provenance apps/credence-pi/brain/harm_brain.provenance.json \
#       --corpus "AI45Research/ATBench-Claw (test.json)"

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using Credence
using Credence: Identity, expect
using JSON3
using Serialization: serialize
using Dates: now
include(joinpath(@__DIR__, "brain_env.jl"))
include(joinpath(@__DIR__, "..", "brain", "feature_brain.jl"))
using .FeatureBrain: build_model, build_prior, observe, belief_at_context

# The harm feature vocabulary — must equal bdsl/features.bdsl `safety-features`.
const FEATS = ["action-class", "taint-flow", "injected-imperative", "cred-exfil-chain"]
const VALS = [
    ["read-only","local-write","delete","exec","external-send","credential-access","cross-boundary","other"],
    ["none","tainted-sink","tainted-external-target"],
    ["no","yes"],
    ["no","yes"],
]
context(e) = String[String(e["features"][f]) for f in FEATS]

# Reason-localized per-call attribution (copied from safety_eval.jl — the honest label).
const SINK_J = r"send|forward|post|publish|tweet|webhook|curl|wget|upload|message|write|edit|create|apply_patch|append|insert|move|copy|cp|mv|rm|unlink|delete|drop|remove|exec|process|bash|sh|tmux|run|api|gh "i
const READ_J = r"read|cat|head|tail|ls|grep|find|list|view|envelope|capture"i
is_read_j(n) = occursin(READ_J, n) || lowercase(n) == "get"
is_sink_j(n) = occursin(SINK_J, n) && !is_read_j(n)
const TOKRX = [r"[\w.+\-]+@[\w.\-]+\.\w+", r"https?://[\w./?=&%+#~\-]+", r"(?:/[\w.+\-]+){2,}/?"]
function arg_tokens_j(s)
    toks = String[]
    for rx in TOKRX, m in eachmatch(rx, lowercase(s))
        t = rstrip(m.match, ['/','.',',',')',';','\'','"']); length(t) >= 4 && push!(toks, t)
    end
    toks
end
function attr_reason(evs)
    labels = zeros(Int, length(evs))
    any(String(e["label"])=="unsafe" for e in evs) || return labels
    reason = ""
    m = get(evs[1], "meta", nothing)
    if m !== nothing; r = get(m, "reason", nothing); r !== nothing && (reason = lowercase(String(r))); end
    found = false
    if !isempty(reason)
        for (i,e) in enumerate(evs)
            name = lowercase(String(e["tool_name"])); is_sink_j(name) || continue
            nmtoks = [t for t in split(name, r"[\s_\-]+") if length(t) >= 3]
            name_match = any(occursin(t, reason) for t in nmtoks)
            arg_match = any(occursin(t, reason) for t in arg_tokens_j(String(get(e,"input_summary",""))))
            (name_match || arg_match) && (labels[i] = 1; found = true)
        end
    end
    if !found
        last = 0
        for (i,e) in enumerate(evs); is_sink_j(lowercase(String(e["tool_name"]))) && (last = i); end
        last > 0 && (labels[last] = 1)
    end
    labels
end

function load_by_session(path)
    sess = Dict{String,Vector{Any}}(); order = String[]
    for l in eachline(path)
        isempty(strip(l)) && continue
        e = JSON3.read(l, Dict{String,Any}); sid = String(e["session_id"])
        haskey(sess, sid) || (push!(order, sid); sess[sid]=[]); push!(sess[sid], e)
    end
    [(sid, sess[sid]) for sid in order]
end

function parse_args(argv)
    a = Dict{String,Any}("events"=>"", "out-brain"=>"", "out-provenance"=>"", "corpus"=>"unknown")
    i = 1
    while i <= length(argv)
        t = argv[i]
        t=="--events" ? (a["events"]=argv[i+1]; i+=2) :
        t=="--out-brain" ? (a["out-brain"]=argv[i+1]; i+=2) :
        t=="--out-provenance" ? (a["out-provenance"]=argv[i+1]; i+=2) :
        t=="--corpus" ? (a["corpus"]=argv[i+1]; i+=2) :
        error("unknown arg $t")
    end
    a
end

function main()
    a = parse_args(ARGS)
    isempty(a["events"]) && error("--events required")
    sessions = load_by_session(a["events"])
    model = build_model(FEATS, VALS; p_edge=0.5)
    top = build_prior(model)
    ncalls = npos = 0
    for (sid, evs) in sessions
        for (e, y) in zip(evs, attr_reason(evs))
            top = observe(model, top, context(e), y); ncalls += 1; npos += y
        end
    end

    # Report the learned θ_u for the discriminating cells (calibration aid for harm-cost).
    println("trained harm posterior on $(length(sessions)) trajectories, $ncalls calls ($npos harm-labelled)")
    println("learned P(unsafe|X) at key contexts (action-class × taint-flow):")
    for (ac, tf) in [("external-send","tainted-external-target"), ("external-send","none"),
                     ("exec","tainted-sink"), ("read-only","none"), ("other","none")]
        X = [ac, tf, "no", "no"]
        θ = expect(belief_at_context(model, top, X), Identity())
        println("  ", rpad("$ac × $tf", 44), "θ_u=", round(θ; digits=3))
    end

    if !isempty(a["out-brain"])
        serialize(a["out-brain"], top)
        println("serialized harm posterior → ", a["out-brain"])
    end
    if !isempty(a["out-provenance"])
        prov = Dict("artifact"=>"credence-pi harm posterior P(unsafe|taint-features)",
                    "corpus"=>a["corpus"], "features"=>FEATS, "feature_values"=>VALS,
                    "n_trajectories"=>length(sessions), "n_calls"=>ncalls, "n_harm_labelled"=>npos,
                    "attribution"=>"reason-localized (the harmful call the human reason names)",
                    "trained_at"=>string(now()), "frozen"=>true,
                    "note"=>"Not updated live; folds H·P(unsafe|X) into decide_multi when harm-cost>0.")
        open(a["out-provenance"], "w") do io; JSON3.pretty(io, prov); end
        println("wrote provenance → ", a["out-provenance"])
    end
end

main()
