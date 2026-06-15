# Role: eval
#
# train_tail_brain.jl — distil the shippable TAIL (continuation) prior
# P(another identical call follows | features), the multi-turn look-ahead belief.
#
# The tail brain answers a different question from the waste brain over the SAME feature
# model: not "would the user approve this call?" but "if this call runs, will the agent
# re-issue an IDENTICAL call later in the session?" — the per-step loop-continuation
# probability ρ. A call's continuation label is forward-looking WITHIN its session:
#   continue (1)  if an identical (tool, command) recurs later in the same session,
#   stop     (0)  if this is the last occurrence (or the call is not loop-distinguishable).
# Tallied per feature-context and trained through the canonical `observe`/`condition`
# path, the posterior Beta(α,β) per context gives the closed-form expected remaining
# repeats  m = E[ρ/(1−ρ)] = α/(β−1)  via  expect(belief, GeometricTail())  — the
# infinite-horizon look-ahead the daemon multiplies the waste-block stakes by.
#
# Shipped as VERSION-STABLE per-context COUNTS (JSON), reconstructed by replaying
# `observe` (order-independent ⇒ identical; robust across Julia versions), exactly like
# the warm waste and frozen harm posteriors. Aggregate Beta counts only (n1=continue,
# n0=stop) — no raw session data, safe to ship.
#
# Scope: FROZEN (a corpus prior). The live daemon currently observes the waste label, not
# continuation events, so it does not update the tail belief online — wiring live
# continuation-learning (the daemon already sees the call sequence) is the natural,
# recoverable extension. Until then this corpus prior supplies m.
#
# Run from repo root:
#   julia --project=. apps/credence-pi/eval/train_tail_brain.jl \
#       --events data/credence_pi_eval/clawsbench_openclaw.events.jsonl \
#       --out-counts apps/credence-pi/brain/tail_brain.counts.json \
#       --corpus "benchflow/ClawsBench@e7c45cc9 (openclaw)"

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using Credence
using JSON3
using Dates: now
include(joinpath(@__DIR__, "brain_env.jl"))
using .FeatureBrain: build_model_from_env, build_prior, observe, context_from_features,
    belief_at_context, reconstruct_posterior

read_events(p) = [JSON3.read(l, Dict{String,Any}) for l in eachline(p) if !isempty(strip(l))]
features_of(e) = Dict{String,String}(string(k) => string(v) for (k, v) in e["features"])

# The loop-distinguishable key for a call: (tool, command) only when the command is
# present and ≠ the tool name (collapsed-arg tools like ClawsBench `read` cannot loop —
# matches the waste trainer + replay.jl). `nothing` ⇒ not loop-distinguishable.
function loop_key(e)
    tool = String(get(e, "tool_name", ""))
    cmdv = get(get(e, "meta", Dict()), "command", nothing)
    cmd = cmdv === nothing ? "" : String(cmdv)
    (!isempty(cmd) && cmd != tool) ? (tool, cmd) : nothing
end

function parse_args(argv)
    a = Dict{String,Any}("events"=>"", "out-counts"=>"", "corpus"=>"unknown")
    i = 1
    while i <= length(argv)
        t = argv[i]
        if t == "--events"; a["events"]=argv[i+1]; i+=2
        elseif t == "--out-counts"; a["out-counts"]=argv[i+1]; i+=2
        elseif t == "--corpus"; a["corpus"]=argv[i+1]; i+=2
        else; error("unknown arg $t"); end
    end
    (isempty(a["events"]) || isempty(a["out-counts"])) && error("--events and --out-counts required")
    a
end

function main()
    a = parse_args(ARGS)
    env = build_env()
    model = build_model_from_env(env)

    # Group events by session (continuation is a forward-looking, within-session property);
    # observe is order-independent, so the per-(ctx,label) multiset is all that matters.
    sessions = Dict{String, Vector{Dict{String,Any}}}()
    order = String[]
    for e in read_events(a["events"])
        sid = String(get(e, "session_id", ""))
        haskey(sessions, sid) || (sessions[sid] = Dict{String,Any}[]; push!(order, sid))
        push!(sessions[sid], e)
    end

    top = build_prior(model)
    counts = Dict{Vector{String}, Vector{Int}}()   # context → [n1 (continue), n0 (stop)]
    ncalls = 0; ncont = 0
    for sid in order
        evs = sessions[sid]
        total = Dict{Tuple{String,String},Int}()   # occurrences of each loop-key in this session
        for e in evs
            k = loop_key(e); k === nothing || (total[k] = get(total, k, 0) + 1)
        end
        seen = Dict{Tuple{String,String},Int}()
        for e in evs
            k = loop_key(e)
            cont = 0
            if k !== nothing
                seen[k] = get(seen, k, 0) + 1
                cont = seen[k] < total[k] ? 1 : 0   # an identical call follows later ⇒ continue
            end
            ctx = context_from_features(model, features_of(e))
            top = observe(model, top, ctx, cont); ncalls += 1; ncont += cont
            c = get!(counts, ctx, [0, 0]); c[cont == 1 ? 1 : 2] += 1
        end
    end

    # Auditable provenance via the PUBLIC functional path only (no private param reads):
    # the m = E[ρ/(1−ρ)] distribution over distinct trained contexts, and the top tails.
    gt = Credence.GeometricTail()
    ms = Tuple{Float64, Vector{String}}[]
    for ctx in keys(counts)
        push!(ms, (Credence.expect(belief_at_context(model, top, ctx), gt), ctx))
    end
    sort!(ms; by = first)
    mvals = first.(ms)
    msummary = Dict("min"=>mvals[1], "median"=>mvals[cld(length(mvals), 2)],
                    "max"=>mvals[end], "mean"=>sum(mvals)/length(mvals))
    top_tails = [Dict("ctx"=>c, "m"=>round(m; digits=3)) for (m, c) in reverse(ms)[1:min(8, length(ms))]]

    contexts = [Dict("ctx" => k, "n1" => v[1], "n0" => v[2]) for (k, v) in counts]
    out = Dict(
        "artifact" => "credence-pi tail (continuation) posterior P(continue|features) — per-context counts",
        "corpus" => a["corpus"], "features" => model.feature_names, "feature_values" => model.feature_values,
        "n_calls" => ncalls, "continue_events" => ncont, "stop_events" => ncalls - ncont,
        "n_contexts" => length(contexts), "trained_at" => string(now()), "frozen" => true,
        "expected_repeats_summary" => msummary, "top_tail_contexts" => top_tails,
        "note" => "m = E[ρ/(1−ρ)] = expect(belief, GeometricTail()) is the expected remaining " *
                  "identical repeats a block prevents (the multi-turn look-ahead). FROZEN corpus " *
                  "prior: the daemon supplies m from this; live continuation-learning is future work. " *
                  "Reconstructed by replaying these counts via observe (order-independent ⇒ identical).",
        "contexts" => contexts)
    open(a["out-counts"], "w") do io; JSON3.pretty(io, out); end
    println("tail brain → $(a["out-counts"])  ($(length(contexts)) contexts, $ncalls calls, $ncont continue-events)")
    println("expected-repeats m over distinct contexts: ", msummary)
    println("longest learned tails (m): ", [(round(m; digits=2), c) for (m, c) in reverse(ms)[1:min(4, length(ms))]])

    # Verify the reconstruction reproduces the directly-trained posterior EXACTLY (the m
    # the daemon will compute equals the m trained here — bit-exact, not merely close).
    rt = reconstruct_posterior(model, a["out-counts"])
    maxdiff = 0.0
    for (_, ctx) in ms[1:min(50, length(ms))]
        maxdiff = max(maxdiff, abs(
            Credence.expect(belief_at_context(model, top, ctx), gt) -
            Credence.expect(belief_at_context(model, rt, ctx), gt)))
    end
    println("reconstruction vs direct training: max m diff = $maxdiff",
            maxdiff < 1e-12 ? "  ✓ exact" : "  ✗ MISMATCH")
end

main()
