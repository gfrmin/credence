# Role: eval
#
# train_latency_brain.jl — warm the routing LATENCY belief E[time | model, prompt-length] from
# the agentic Terminal-Bench matrix, the TIME-coordinate sibling of train_routing_from_matrix.jl
# (which warms the accuracy belief). Time is the profile coordinate that lets a user trade
# wall-clock against money/quality (speed-first vs cost-saver); it is a LEARNED belief because a
# call's duration is unknown until after the call.
#
# MODEL (reused, not reinvented): the Poisson-Gamma "turns" belief from tb_dominance.jl. Per
# (tier, prompt-length bucket) we ship the Gamma sufficient statistic (sum_turns, n_obs) — the
# posterior Gamma(α0+Σt, β0+n) is order-independent, so reconstruction is exact and version-
# stable (no Serialization). Per tier we ship the measured rate_s = Σduration_s / Σnum_turns
# (seconds/turn). The daemon's RoutingBrain.reconstruct_latency reads these and computes
# E[time|model,X] = expect(Gamma, Identity)·rate_s = (α/β)·s̄ at load.
#
# Buckets MUST match routing-features.ts / routing.bdsl (short ≤400, medium ≤1000, long >1000
# instruction chars) — the train/live synchronisation point, same as the accuracy trainer.
#
# NON-CAUSAL (Role: eval): reads measured data, tallies sufficient statistics, writes + verifies
# the fixture. No belief drives a decision here.
#   julia --project=. apps/credence-pi/eval/train_latency_brain.jl

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using JSON3

const MATRIX = normpath(joinpath(@__DIR__, "live_ab", "results", "tb_matrix_rep3.jsonl"))
const OUT = normpath(joinpath(@__DIR__, "..", "brain", "routing_latency.counts.json"))

# Tier → roster model-id; order/ids identical to train_routing_from_matrix.jl.
const TIERS = ["haiku", "sonnet", "opus"]
const TIER_ID = Dict("haiku" => "claude-haiku-4-5", "sonnet" => "claude-sonnet-4-6",
                     "opus" => "claude-opus-4-8")
const BUCKETS = ["short", "medium", "long"]
# Gamma prior on turns — identical to tb_dominance.jl's TURNS_A0/TURNS_B0 (the reconstruction
# in RoutingBrain.reconstruct_latency reads this from the JSON, so it stays in lockstep).
const TURNS_A0, TURNS_B0 = 1.0, 0.1

length_bucket(n) = n <= 400 ? "short" : (n <= 1000 ? "medium" : "long")

function main()
    rows = [JSON3.read(ln) for ln in eachline(MATRIX) if !isempty(strip(ln))]
    # (tier, bucket) -> [sum_turns, n_obs];  tier -> [Σduration_s, Σturns] for the rate
    cells = Dict{Tuple{String, String}, Vector{Float64}}()
    rate_acc = Dict{String, Vector{Float64}}()
    for r in rows
        tier = String(r.tier)
        haskey(TIER_ID, tier) || continue
        r.num_turns === nothing && continue               # need a turn count for the Poisson belief
        turns = Float64(r.num_turns)
        c = get!(cells, (tier, length_bucket(Int(r.instr_len))), [0.0, 0.0])
        c[1] += turns; c[2] += 1.0
        if r.duration_s !== nothing                        # rate needs both duration and turns
            ra = get!(rate_acc, tier, [0.0, 0.0]); ra[1] += Float64(r.duration_s); ra[2] += turns
        end
    end

    per_model = Any[]
    for tier in TIERS
        ctxs = Any[]
        for b in BUCKETS
            c = get(cells, (tier, b), [0.0, 0.0])
            c[2] == 0.0 && continue                           # no data ⇒ leave the cell at prior
            push!(ctxs, Dict("ctx" => [b], "sum_turns" => c[1], "n_obs" => Int(c[2])))
        end
        ra = get(rate_acc, tier, [0.0, 0.0])
        rate_s = ra[2] > 0.0 ? ra[1] / ra[2] : 0.0            # measured Σduration / Σturns
        push!(per_model, Dict("model_id" => TIER_ID[tier], "rate_s" => rate_s, "contexts" => ctxs))
    end

    out = Dict(
        "artifact" => "credence-pi routing LATENCY belief E[time|model,prompt-length] — Poisson-Gamma turns × measured s/turn",
        "source" => "tb_matrix_rep3.jsonl (agentic Terminal-Bench), bucketed by instruction length (short ≤400 / medium ≤1000 / long)",
        "note" => "Gamma sufficient statistic (sum_turns,n_obs) per (model,bucket) + measured rate_s (s/turn) per model. E[time]=expect(Gamma(α0+Σt,β0+n),Identity)·rate_s. Buckets match routing-features.ts.",
        "turns_prior" => [TURNS_A0, TURNS_B0],
        "per_model" => per_model,
    )
    open(OUT, "w") do io; JSON3.pretty(io, out); end
    println("wrote ", OUT)
    for pm in per_model
        println("  ", rpad(pm["model_id"], 22), "rate_s=", round(pm["rate_s"], digits = 2), " s/turn")
        for ctx in pm["contexts"]
            mean_turns = (TURNS_A0 + ctx["sum_turns"]) / (TURNS_B0 + ctx["n_obs"])
            println("    ", rpad(ctx["ctx"][1], 8), "E[turns]=", round(mean_turns, digits = 2),
                    "  E[time]=", round(mean_turns * pm["rate_s"], digits = 1), "s",
                    "  (n=", ctx["n_obs"], ")")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
