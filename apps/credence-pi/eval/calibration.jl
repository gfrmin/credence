# Role: eval
"""
    calibration.jl — is the routing belief's θ a TRUE probability?

Every routing decision rests on θ_a(X) = E[P(correct | model a, context X)] being calibrated:
the EU-max reward·θ − cost trades accuracy against cost in dollars, so a miscalibrated θ makes
the wrong trade even when it RANKS models correctly. The dominance eval measures realized
welfare; this measures whether the belief it routes on is honest.

NON-CAUSAL read-out (a diagnostic — nothing here feeds a routing decision or a belief): on
held-out (task, tier, rep) outcomes it compares the trained belief's predicted θ
(`posterior_accuracy`, the public diagnostic accessor) against the realized solve via proper
scores — ECE, Brier, log-score, AUROC — plus a reliability table. Reuses tb_dominance.jl's
matrix loader + per-tier StructureBMA training (the SAME belief the router uses); the 3-rep
matrix supplies the realized frequencies to calibrate against.

Run:
    julia --project=<repo-root> apps/credence-pi/eval/calibration.jl \\
        apps/credence-pi/eval/live_ab/results/tb_matrix_rep3.jsonl \\
        [--seeds 100] [--bins 10] [--train-frac 0.6] [--out .../calibration.summary.json]
"""

include(joinpath(@__DIR__, "tb_dominance.jl"))   # load_matrix, build_model, train_tops, …

using Printf
using JSON3
using Random: MersenneTwister, shuffle!

_clamp01(p) = min(1.0 - 1e-12, max(1e-12, p))

# Held-out (predicted θ, realized outcome) pairs: train the per-tier belief on the train split,
# then for each test (task, tier, rep) read the belief's predicted accuracy at that task's
# context against the rep's realized solve. `posterior_accuracy` is the public diagnostic
# accessor (E[θ|X] through `expect`); this read-out never feeds a decision.
function collect_pairs(tasks::Vector{TaskRow}, fvals, seeds::Int, train_frac::Float64)
    ps = Float64[]; ys = Float64[]
    ids = [t.id for t in tasks]
    for seed in 0:(seeds - 1)
        rng = MersenneTwister(seed)
        ord = shuffle!(rng, collect(eachindex(ids)))
        ntr = max(2, round(Int, train_frac * length(ids)))
        trainset = Set(ids[ord[1:ntr]])
        train = [t for t in tasks if t.id in trainset]
        test  = [t for t in tasks if !(t.id in trainset)]
        isempty(test) && continue
        model = build_model(FEATURES, fvals; alpha0 = ALPHA0, beta0 = BETA0, p_edge = PEDGE)
        tops = train_tops(model, train)
        for t in test, a in eachindex(TIERS)
            X = context_from_features(model, featuredict(t))
            p = posterior_accuracy(model, tops[a], X)         # predicted E[θ_a|X] (diagnostic)
            for r in 1:nreps(t)
                push!(ps, p); push!(ys, t.resolved[a][r] ? 1.0 : 0.0)
            end
        end
    end
    ps, ys
end

# ── proper scores over the (p, y) pairs — plain arithmetic on diagnostic readouts ──
_brier(ps, ys) = sum((ps .- ys) .^ 2) / length(ps)
_logscore(ps, ys) =
    -sum(ys .* log.(_clamp01.(ps)) .+ (1.0 .- ys) .* log.(1.0 .- _clamp01.(ps))) / length(ps)

_bin_idx(ps, lo, hi, last) =
    last ? findall(p -> lo <= p <= hi, ps) : findall(p -> lo <= p < hi, ps)

# Expected calibration error: equal-width bins, Σ (n_b/n)·|mean(y) − mean(p)| over bins.
function _ece(ps, ys, bins::Int)
    edges = range(0.0, 1.0; length = bins + 1)
    n = length(ps); ece = 0.0
    for b in 1:bins
        idx = _bin_idx(ps, edges[b], edges[b + 1], b == bins)
        isempty(idx) && continue
        ece += (length(idx) / n) * abs(sum(@view ys[idx]) - sum(@view ps[idx])) / length(idx)
    end
    ece
end

# AUROC via the Mann–Whitney U statistic (rank-based; ties take the average rank).
function _auroc(ps, ys)
    npos = count(==(1.0), ys); nneg = length(ys) - npos
    (npos == 0 || nneg == 0) && return NaN
    order = sortperm(ps)
    ranks = Vector{Float64}(undef, length(ps))
    i = 1
    while i <= length(ps)
        j = i
        while j < length(ps) && ps[order[j + 1]] == ps[order[i]]; j += 1; end
        r = (i + j) / 2.0
        for k in i:j; ranks[order[k]] = r; end
        i = j + 1
    end
    sum_pos = sum(ranks[k] for k in eachindex(ys) if ys[k] == 1.0; init = 0.0)
    (sum_pos - npos * (npos + 1) / 2) / (npos * nneg)
end

function reliability_table(ps, ys, bins::Int)
    edges = range(0.0, 1.0; length = bins + 1)
    println("  ", rpad("bin", 13), rpad("n", 8), rpad("mean pred", 12),
            rpad("mean real", 12), "gap")
    for b in 1:bins
        idx = _bin_idx(ps, edges[b], edges[b + 1], b == bins)
        isempty(idx) && continue
        mp = sum(@view ps[idx]) / length(idx); mr = sum(@view ys[idx]) / length(idx)
        println("  ", rpad(@sprintf("[%.1f,%.1f)", edges[b], edges[b + 1]), 13),
                rpad(string(length(idx)), 8), rpad(@sprintf("%.3f", mp), 12),
                rpad(@sprintf("%.3f", mr), 12), @sprintf("%+.3f", mr - mp))
    end
end

function parse_args(argv)
    a = Dict{String, Any}("path" => "", "seeds" => 100, "bins" => 10,
                          "train-frac" => 0.6, "out" => "")
    i = 1
    while i <= length(argv)
        t = argv[i]
        if     t == "--seeds";      a["seeds"] = parse(Int, argv[i + 1]);          i += 2
        elseif t == "--bins";       a["bins"] = parse(Int, argv[i + 1]);           i += 2
        elseif t == "--train-frac"; a["train-frac"] = parse(Float64, argv[i + 1]); i += 2
        elseif t == "--out";        a["out"] = argv[i + 1];                        i += 2
        elseif !startswith(t, "--") && isempty(a["path"]); a["path"] = t;          i += 1
        else; error("unknown arg $t"); end
    end
    isempty(a["path"]) && error("usage: calibration.jl <matrix.jsonl> [--seeds N] [--bins N] [--train-frac f] [--out f]")
    a
end

function run_calibration()
    args = parse_args(ARGS)
    tasks = load_matrix(args["path"])
    @assert length(tasks) >= 4 "need ≥4 fully-measured tasks, got $(length(tasks))"
    fvals = feature_values(tasks)
    ps, ys = collect_pairs(tasks, fvals, args["seeds"], args["train-frac"])

    ece = _ece(ps, ys, args["bins"]); brier = _brier(ps, ys)
    logs = _logscore(ps, ys); auroc = _auroc(ps, ys)

    bar = "="^72
    println("\n", bar)
    println("ROUTING-BELIEF CALIBRATION  (held-out predicted θ vs realized solve)")
    println(bar)
    println("tasks: ", length(tasks), "   tiers: ", join(TIERS, "/"),
            "   features: ", join(FEATURES, "/"),
            "   seeds: ", args["seeds"], "   pairs: ", length(ps))
    @printf("\n  ECE (%d-bin):  %.4f      Brier: %.4f      log-score: %.4f      AUROC: %.4f\n",
            args["bins"], ece, brier, logs, auroc)
    println("\nReliability (perfect calibration ⇒ gap ≈ 0 every bin):")
    reliability_table(ps, ys, args["bins"])
    println(bar, "\n")

    if !isempty(args["out"])
        out = Dict{String, Any}(
            "config" => Dict("tasks" => length(tasks), "tiers" => TIERS, "features" => FEATURES,
                             "seeds" => args["seeds"], "bins" => args["bins"],
                             "train_frac" => args["train-frac"], "pairs" => length(ps)),
            "ece" => ece, "brier" => brier, "log_score" => logs, "auroc" => auroc)
        mkpath(dirname(args["out"]))
        open(args["out"], "w") do io; JSON3.pretty(io, out); end
        println("summary → ", args["out"])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_calibration()
end
