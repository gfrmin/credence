# Role: eval
#
# train_routing_brain.jl — distil the WARM routing belief from the measured oracle grid.
#
# SUPERSEDED (2026-06-17) by train_routing_from_matrix.jl: eval/calibration.jl showed this
# short-MCQ warm MIS-RANKS tiers on agentic tasks (it orders haiku<sonnet<opus, but tb
# reality is sonnet>opus). The shipped routing_brain.counts.json now comes from the AGENTIC
# Terminal-Bench matrix. Kept for provenance; do NOT run it to regenerate the shipped belief
# (it would re-introduce the mis-rank).
#
# Reads the real-model oracle grid (the de-risk fixture:
# apps/python/credence_router/experiments/routing_dominance/oracle_grid.json — per
# (model, question) measured MCQ correctness on the 50-item labelled bank) and emits
# per-model correct/incorrect COUNTS as apps/credence-pi/brain/routing_brain.counts.json.
# The daemon's wire_routing! reconstructs K StructureBMA posteriors from these counts by
# replaying `observe` (version-stable, like harm_brain.counts.json — NOT Serialization).
#
# Every benchmark item is a SHORT MCQ prompt, so each measured outcome seeds the "short"
# prompt-length cell; the "long" cell stays at the prior. This is the honest warm belief:
# we measured per-model accuracy on short prompts only, so the router starts knowing
# short-prompt accuracy and treats long prompts as partly-unknown (the marginal structure
# still lends them the overall signal; the length structure's long cell is at prior).
#
# This is the WARM SEED (a prior): the daemon then learns online from each routed turn's
# per-turn outcome, confound-aware (brain/routing_brain.jl `route_outcome!`). Run-level /
# multi-call credit assignment stays deferred (ROUTING_DOMINANCE.md).
#
# NON-CAUSAL (Role: eval): reads measured data, tallies counts, writes + verifies a
# fixture. No belief drives a decision here. Run from repo root:
#   julia --project=. apps/credence-pi/eval/train_routing_brain.jl

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using Credence
using JSON3
using Dates: now
include(joinpath(@__DIR__, "..", "brain", "feature_brain.jl"))
using .FeatureBrain: build_model
include(joinpath(@__DIR__, "..", "brain", "routing_brain.jl"))
using .RoutingBrain: posterior_accuracy

const GRID = normpath(joinpath(@__DIR__, "..", "..", "python", "credence_router",
    "experiments", "routing_dominance", "oracle_grid.json"))
const OUT = normpath(joinpath(@__DIR__, "..", "brain", "routing_brain.counts.json"))

# The routing feature schema — must equal routing.bdsl `routing-features`.
const FEATS = ["prompt-length"]
const VALS = [["short", "long"]]

function main()
    isfile(GRID) || error("oracle grid not found: $GRID (run the oracle first)")
    data = JSON3.read(read(GRID, String))
    models = collect(String.(data.models))
    model_ids = haskey(data, :model_ids) ? collect(String.(data.model_ids)) : copy(models)
    K = length(models)

    # Tally per-model correct (n1) / incorrect (n0) over every measured grid cell. All
    # items are short MCQ ⇒ everything lands in the "short" prompt-length cell.
    n1 = zeros(Int, K); n0 = zeros(Int, K)
    for (k, v) in data.grid
        mi = parse(Int, split(String(k), "|")[1])    # "mi|qid", mi is 0-based
        (v === true) ? (n1[mi + 1] += 1) : (n0[mi + 1] += 1)
    end

    per_model = [Dict("contexts" => [Dict("ctx" => ["short"], "n1" => n1[i], "n0" => n0[i])])
                 for i in 1:K]
    out = Dict(
        "artifact" => "credence-pi warm routing belief P(correct|prompt-length) — per-model counts",
        "source" => "oracle_grid.json (real-model de-risk: 3 frontier models × 50 labelled MCQ)",
        "models" => models, "model_ids" => model_ids,
        "features" => FEATS, "feature_values" => VALS,
        "note" => "All items are short MCQ ⇒ short cell only; long cell stays at prior. " *
                  "Daemon reconstructs K posteriors by replaying these counts via observe, " *
                  "then learns online per routed turn (confound-aware; see routing_brain.jl).",
        "trained_at" => string(now()), "warm_seed" => true,
        "per_model" => per_model)
    open(OUT, "w") do io; JSON3.pretty(io, out); end
    println("wrote per-model counts → $OUT")
    for i in 1:K
        acc = n1[i] / (n1[i] + n0[i])
        println("  ", rpad(models[i], 8), " (", model_ids[i], "): n1=$(n1[i]) n0=$(n0[i]) ",
                "measured acc=", round(acc; digits=3))
    end

    # Verify the shipped counts reconstruct into the warm belief, and report the
    # posterior-mean accuracy the router will actually use at a short-prompt context.
    model = build_model(FEATS, VALS; p_edge = 0.5)
    tops = RoutingBrain._reconstruct_routing_tops(model, K, OUT)
    println("warm belief P(correct|short) the router will use (Beta-shrunk posterior mean):")
    for i in 1:K
        θ = posterior_accuracy(model, tops[i], ["short"])
        println("  ", rpad(models[i], 8), " θ=", round(θ; digits=3))
    end
end

main()
