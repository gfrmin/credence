#!/usr/bin/env julia
# Role: brain-side application
"""
    run.jl — Dominance benchmark entry point + the gate (dominance-design §7)

Running this file IS the check: it runs the full policy × seed grid, computes paired-seed
bootstrap CIs on the per-seed gaps, writes results/ artefacts, and ASSERTS the gate —
any failed gate is an error (halt-the-line: investigate, do not patch forward).

Manually run, out of the fast suite (like credence_router's test_live.py):

    julia apps/julia/dominance_benchmark/run.jl

Gate (§7):
  - CI on eu_max − random and eu_max − best-tuned fixed_schedule excludes 0 on BOTH
    realised value (AUC) and sample efficiency (steps-to-own-half, sign-flipped);
  - eu_max − never_explore reported as the HEADLINE (exploration's isolated value,
    the escape-mass heuristic held constant on both sides) and its CI excludes 0;
  - never_explore ≤ eu_max ≤ clairvoyant on mean AUC (the first is a hypothesis under
    test — on failure, diagnose the task's non-stationarity before blaming the policy;
    the second is a true sanity check);
  - minimax regret: the worst-seed AUC gap vs random and vs best fixed is ≥ 0;
  - behaviour-verified inversions extracted (concrete steps where eu_max grows and a
    baseline does not).

All statistics here are arithmetic on realised world outcomes (energy trajectories),
never on beliefs; the paired bootstrap resamples seed indices with a fixed RNG.
"""

include(joinpath(@__DIR__, "host.jl"))

# ── Paired-seed percentile bootstrap ──────────────────────────────────────────

"""
    bootstrap_ci(deltas; n_resamples, rng) → (mean, lo, hi)

Percentile bootstrap (2.5/97.5) of the mean of per-seed paired gaps: resample seed
indices uniformly with replacement, n_resamples times.
"""
function bootstrap_ci(deltas::Vector{Float64}; n_resamples::Int = 10_000,
                      rng = MersenneTwister(20260702))
    n = length(deltas)
    means = Vector{Float64}(undef, n_resamples)
    for r in 1:n_resamples
        s = 0.0
        for _ in 1:n
            s += deltas[rand(rng, 1:n)]
        end
        means[r] = s / n
    end
    sort!(means)
    lo = means[max(1, floor(Int, 0.025 * n_resamples))]
    hi = means[min(n_resamples, ceil(Int, 0.975 * n_resamples))]
    (sum(deltas) / n, lo, hi)
end

paired(a::Vector{RunSummary}, b::Vector{RunSummary}, f::Function) =
    Float64[f(a[i]) - f(b[i]) for i in eachindex(a)]

# ── Report writing ────────────────────────────────────────────────────────────

function write_results_tsv(path::String, results::Dict{String, Vector{RunSummary}})
    open(path, "w") do io
        println(io, "policy\tseed\tauc\tfinal_window_mean\tsteps_to_half\tn_meta\tn_grammars\tn_growth_ops")
        for name in sort(collect(keys(results))), r in results[name]
            println(io, "$(r.policy)\t$(r.seed)\t$(r.auc)\t$(r.final_window_mean)\t" *
                        "$(r.steps_to_half)\t$(r.n_meta)\t$(r.n_grammars)\t$(length(r.growth_steps))")
        end
    end
end

mean_of(rs::Vector{RunSummary}, f::Function) = sum(f(r) for r in rs) / length(rs)

# ── The gate ──────────────────────────────────────────────────────────────────

function main(; n_seeds::Int = DB_N_SEEDS)
    println("Dominance benchmark: $(length(policy_table())) policy configs × $n_seeds seeds, " *
            "task = $(DB_WORLD_RULES) with changes at $(DB_REGIME_STEPS), respawn on")
    results = run_benchmark(n_seeds = n_seeds)

    # Best-tuned baselines by mean AUC (anti-strawman: each family gets its best knob).
    fixed_names = ["fixed_k$(k)" for k in DB_K_SWEEP]
    best_fixed = fixed_names[argmax([mean_of(results[f], r -> r.auc) for f in fixed_names])]
    random_names = ["random_p$(replace(string(p), "." => ""))" for p in DB_P_SWEEP]
    best_random = random_names[argmax([mean_of(results[f], r -> r.auc) for f in random_names])]
    println("best-tuned fixed schedule: $best_fixed; best-tuned random: $best_random")

    eu = results["eu_max"]
    gaps = Dict{String, Dict{Symbol, Tuple{Float64, Float64, Float64}}}()
    for base in [best_random, best_fixed, "never_explore"]
        gaps[base] = Dict(
            :auc => bootstrap_ci(paired(eu, results[base], r -> r.auc)),
            # efficiency: fewer steps to own half-total is better ⇒ gap = baseline − eu_max.
            :efficiency => bootstrap_ci(paired(results[base], eu, r -> Float64(r.steps_to_half))),
        )
    end

    mean_auc = Dict(name => mean_of(rs, r -> r.auc) for (name, rs) in results)
    worst_gap = Dict(base => minimum(paired(eu, results[base], r -> r.auc))
                     for base in [best_random, best_fixed, "never_explore"])

    # Behaviour-verified inversions: concrete growth decisions eu_max made that the
    # de-confounded floor cannot make, on the same seeds.
    inversions = String[]
    for i in eachindex(eu)
        isempty(eu[i].growth_steps) && continue
        (step, op) = eu[i].growth_steps[1]
        push!(inversions,
              "seed $(eu[i].seed): eu_max takes $(op) at step $(step) " *
              "(never_explore: growth vetoed by construction; " *
              "auc gap $(round(eu[i].auc - results["never_explore"][i].auc, digits=2)))")
    end

    # ── Artefacts ──
    resdir = joinpath(@__DIR__, "results")
    mkpath(resdir)
    write_results_tsv(joinpath(resdir, "results.tsv"), results)
    open(joinpath(resdir, "summary.md"), "w") do io
        println(io, "# Dominance benchmark — results\n")
        println(io, "Task: `$(DB_WORLD_RULES)`, regime changes at `$(DB_REGIME_STEPS)`, " *
                    "$(DB_MAX_STEPS) steps, respawn on, $n_seeds seeds, " *
                    "paired-seed percentile bootstrap (10 000 resamples).\n")
        println(io, "| policy | mean AUC | mean final-window rate | mean steps-to-half | mean meta-actions |")
        println(io, "|---|---|---|---|---|")
        for name in sort(collect(keys(results)))
            rs = results[name]
            println(io, "| $name | $(round(mean_of(rs, r -> r.auc), digits=2)) | " *
                        "$(round(mean_of(rs, r -> r.final_window_mean), digits=3)) | " *
                        "$(round(mean_of(rs, r -> Float64(r.steps_to_half)), digits=1)) | " *
                        "$(round(mean_of(rs, r -> Float64(r.n_meta)), digits=1)) |")
        end
        println(io, "\n## Paired gaps (eu_max − baseline; efficiency sign-flipped so + favours eu_max)\n")
        println(io, "| baseline | AUC gap [95% CI] | efficiency gap [95% CI] | worst-seed AUC gap |")
        println(io, "|---|---|---|---|")
        for base in [best_random, best_fixed, "never_explore"]
            a = gaps[base][:auc]; e = gaps[base][:efficiency]
            println(io, "| $base | $(round(a[1], digits=2)) [$(round(a[2], digits=2)), $(round(a[3], digits=2))] " *
                        "| $(round(e[1], digits=1)) [$(round(e[2], digits=1)), $(round(e[3], digits=1))] " *
                        "| $(round(worst_gap[base], digits=2)) |")
        end
        println(io, "\n`eu_max − never_explore` is the headline: the escape-mass heuristic is " *
                    "identical on both sides, so this gap is exploration's isolated value.\n")
        println(io, "## Behaviour-verified inversions\n")
        for line in inversions[1:min(10, length(inversions))]
            println(io, "- $line")
        end
        println(io, "\nBracket: never_explore $(round(mean_auc["never_explore"], digits=2)) ≤ " *
                    "eu_max $(round(mean_auc["eu_max"], digits=2)) ≤ " *
                    "clairvoyant $(round(mean_auc["clairvoyant"], digits=2))")
    end
    println("wrote $(joinpath(resdir, "results.tsv")) and summary.md")

    # ── Assertions (running this file IS the gate) ──
    failures = String[]
    for base in [best_random, best_fixed]
        gaps[base][:auc][2] > 0.0 ||
            push!(failures, "CI(eu_max − $base) on AUC includes 0: $(gaps[base][:auc])")
        gaps[base][:efficiency][2] > 0.0 ||
            push!(failures, "CI(eu_max − $base) on efficiency includes 0: $(gaps[base][:efficiency])")
        worst_gap[base] >= 0.0 ||
            push!(failures, "minimax regret vs $base: worst-seed AUC gap $(worst_gap[base]) < 0")
    end
    gaps["never_explore"][:auc][2] > 0.0 ||
        push!(failures, "HEADLINE CI(eu_max − never_explore) on AUC includes 0: " *
                        "$(gaps["never_explore"][:auc]) — before blaming the policy, interrogate " *
                        "whether the task's non-stationarity is strong enough to reward " *
                        "exploration (dominance-design §6: the regime-shift magnitude is " *
                        "load-bearing for this gate meaning anything)")
    mean_auc["never_explore"] <= mean_auc["eu_max"] ||
        push!(failures, "bracket: never_explore mean AUC $(mean_auc["never_explore"]) > " *
                        "eu_max $(mean_auc["eu_max"]) (hypothesis under test — diagnose the task)")
    mean_auc["eu_max"] <= mean_auc["clairvoyant"] ||
        push!(failures, "SANITY: eu_max mean AUC $(mean_auc["eu_max"]) > clairvoyant " *
                        "$(mean_auc["clairvoyant"]) — must always hold, investigate")
    isempty(inversions) &&
        push!(failures, "no behaviour-verified inversions: eu_max never took a growth op")

    if isempty(failures)
        println("\nGATE PASSED — the deployed EU-max policy dominates random, the best-tuned " *
                "fixed schedule, and the never-explore floor, inside the clairvoyant ceiling.")
    else
        println("\nGATE FAILED:")
        for f in failures
            println("  ✗ $f")
        end
        error("dominance gate failed ($(length(failures)) assertion(s)) — halt the line")
    end
    results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
