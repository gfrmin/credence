# Role: eval
#
# train_routing_from_matrix.jl — warm the routing belief from the AGENTIC Terminal-Bench
# matrix (replacing the short-MCQ oracle grid that train_routing_brain.jl used).
#
# WHY (calibration-motivated): eval/calibration.jl showed the MCQ-warmed belief MIS-RANKS
# tiers on agentic tasks — it orders haiku<sonnet<opus, but tb reality is sonnet>opus
# (sonnet 36/51, opus 32/51). The router's job IS agentic (OpenClaw), so warm it from
# agentic data: per (tier, prompt-length bucket) resolved/not counts over the tb matrix. The
# length buckets match routing-features.ts EXACTLY (short ≤400, medium ≤1000, long >1000
# instruction chars) — the train/live synchronisation point — so finer prompt-length carries
# real, length-varying agentic signal from install.
#
# NON-CAUSAL (Role: eval): reads measured data, tallies counts, writes + verifies the
# routing_brain.counts.json fixture. No belief drives a decision here.
#   julia --project=. apps/credence-pi/eval/train_routing_from_matrix.jl

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using JSON3

const MATRIX = normpath(joinpath(@__DIR__, "live_ab", "results", "tb_matrix_rep3.jsonl"))
const OUT = normpath(joinpath(@__DIR__, "..", "brain", "routing_brain.counts.json"))

# Tier → roster model-id; order MUST match routing.bdsl routing-models (positional warm seed).
const TIERS = ["haiku", "sonnet", "opus"]
const TIER_ID = Dict("haiku" => "claude-haiku-4-5", "sonnet" => "claude-sonnet-4-6",
                     "opus" => "claude-opus-4-8")
const NAMES = ["cheap", "mid", "exp"]
const BUCKETS = ["short", "medium", "long"]

# prompt-length bucket — MUST match routing-features.ts (short ≤400, medium ≤1000, long >1000).
length_bucket(n) = n <= 400 ? "short" : (n <= 1000 ? "medium" : "long")

function main()
    rows = [JSON3.read(ln) for ln in eachline(MATRIX) if !isempty(strip(ln))]
    counts = Dict{Tuple{String, String}, Vector{Int}}()        # (tier, bucket) -> [n1, n0]
    for r in rows
        tier = String(r.tier)
        haskey(TIER_ID, tier) || continue
        c = get!(counts, (tier, length_bucket(Int(r.instr_len))), [0, 0])
        r.resolved ? (c[1] += 1) : (c[2] += 1)
    end

    per_model = Any[]
    for tier in TIERS
        ctxs = Any[]
        for b in BUCKETS
            c = get(counts, (tier, b), [0, 0])
            (c[1] == 0 && c[2] == 0) && continue              # no data ⇒ leave the cell at prior
            push!(ctxs, Dict("ctx" => [b], "n1" => c[1], "n0" => c[2]))
        end
        push!(per_model, Dict("contexts" => ctxs))
    end

    out = Dict(
        "artifact" => "credence-pi warm routing belief P(correct|prompt-length) — per-model counts",
        "source" => "tb_matrix_rep3.jsonl (agentic Terminal-Bench: 17 tasks × 3 tiers × 3 reps), bucketed by instruction length",
        "note" => "AGENTIC warm — replaces the short-MCQ oracle grid, which eval/calibration.jl showed mis-ranks tiers on agentic tasks (it ordered haiku<sonnet<opus; tb reality is sonnet>opus). Buckets match routing-features.ts: short ≤400, medium ≤1000, long >1000 chars. The daemon reconstructs K StructureBMA posteriors by replaying these counts via observe; empty cells stay at the prior and learn online.",
        "features" => ["prompt-length"],
        "feature_values" => [BUCKETS],
        "models" => NAMES,
        "model_ids" => [TIER_ID[t] for t in TIERS],
        "per_model" => per_model,
    )
    open(OUT, "w") do io; JSON3.pretty(io, out); end
    println("wrote ", OUT)
    for tier in TIERS, b in BUCKETS
        c = get(counts, (tier, b), [0, 0]); tot = c[1] + c[2]
        tot == 0 && continue
        println("  ", rpad(tier, 8), rpad(b, 8), "n1=", c[1], " n0=", c[2],
                "  rate=", round(c[1] / tot, digits = 3))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
