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
  - CI on eu_max − random and eu_max − best-tuned fixed_schedule excludes 0 on ALL of
    realised value (AUC), the final-window rate (co-primary, §2c), and sample efficiency
    (steps to the shared per-seed level, sign-flipped);
  - eu_max − never_explore reported as the HEADLINE (exploration's isolated value,
    the learned-returns escape ops held constant on both sides) and its CI excludes 0;
  - never_explore ≤ eu_max ≤ clairvoyant on mean AUC (the first is a hypothesis under
    test — on failure, diagnose the task's non-stationarity before blaming the policy;
    the second is a true sanity check);
  - minimax regret: the worst-seed AUC gap vs random and vs best fixed is ≥ 0;
  - behaviour-verified inversions extracted (concrete steps where eu_max grows and a
    baseline does not).

Beyond the asserted gate, summary.md carries a REPORTED panel of alternative dominance
measures (win rate + exact sign test, median gap, 10th-percentile gap, final-regime rate)
so the dominance claim can be located precisely without moving the gate.

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

# ── Alternative dominance measures (REPORTED, not asserted) ──────────────────
# The gate's asserted measures (mean-gap bootstrap CIs + minimax) are kept for comparability;
# this panel reports complementary dominance notions on the same per-seed paired gaps so the
# §3.2 claim can be located precisely (author steer, 2026-07-03: "you can try different
# dominance measures"). All arithmetic on realised world outcomes.

median_of(v::Vector{Float64}) = begin
    s = sort(v); n = length(s)
    isodd(n) ? s[(n + 1) ÷ 2] : 0.5 * (s[n ÷ 2] + s[n ÷ 2 + 1])
end

# Nearest-rank lower quantile (q ∈ (0,1]) — the tail-regret measure softer than minimax.
quantile_of(v::Vector{Float64}, q::Float64) = sort(v)[clamp(ceil(Int, q * length(v)), 1, length(v))]

"""
    sign_test_p(deltas) → Float64

Exact two-sided sign-test p-value on the per-seed paired gaps (H₀: median gap = 0; ties
dropped). Distribution-free — the win-rate's significance without the bootstrap mean's
sensitivity to the winner's-curse outlier seeds.
"""
function sign_test_p(deltas::Vector{Float64})
    pos = count(>(0.0), deltas)
    neg = count(<(0.0), deltas)
    m = pos + neg
    m == 0 && return 1.0
    k = min(pos, neg)
    tail = sum(binomial(BigInt(m), BigInt(j)) for j in 0:k) / BigInt(2)^m
    min(1.0, 2.0 * Float64(tail))
end

# Realised per-step energy rate over the FINAL regime only: the converged-behaviour measure —
# an exploring policy pays early to earn late, so area over the whole run (AUC) charges
# exploration's tuition while this rate reads what it bought. The regime change applies at the
# TOP of the `step == DB_REGIME_STEPS[end]` iteration (grid_world host loop), so that step is
# the final regime's FIRST step (typically its worst): the window is ce[end] − ce[change − 1]
# over (n − change + 1) steps, not the off-by-one that would drop it (adversarial-review fix).
rate_final_regime(r::RunSummary) =
    (r.ce[end] - r.ce[DB_REGIME_STEPS[end] - 1]) / (length(r.ce) - DB_REGIME_STEPS[end] + 1)

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

    # Shared-reference efficiency (belief-derived-valuation §2c): one bar per seed — half the
    # per-seed BEST policy's final total — so collapsing early cannot look "efficient" (the
    # self-relative steps-to-own-half is retained in the tables for reporting only).
    all_names = collect(keys(results))
    level = Dict{Int, Float64}()
    for i in eachindex(eu)
        level[i] = 0.5 * maximum(results[nm][i].ce[end] for nm in all_names)
    end
    stl(rs::Vector{RunSummary}) = Float64[Float64(steps_to_level(rs[i], level[i])) for i in eachindex(rs)]

    gaps = Dict{String, Dict{Symbol, Tuple{Float64, Float64, Float64}}}()
    for base in [best_random, best_fixed, "never_explore"]
        gaps[base] = Dict(
            :auc => bootstrap_ci(paired(eu, results[base], r -> r.auc)),
            # co-primary realised-value gate (§2c): the end-state rate, not just the area.
            :final_window => bootstrap_ci(paired(eu, results[base], r -> r.final_window_mean)),
            # efficiency: fewer steps to the SHARED level is better ⇒ gap = baseline − eu_max.
            :efficiency => bootstrap_ci(stl(results[base]) .- stl(eu)),
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
        println(io, "\n## Paired gaps (eu_max − baseline; efficiency = steps to the shared per-seed")
        println(io, "level, sign-flipped so + favours eu_max — belief-derived-valuation §2c)\n")
        println(io, "| baseline | AUC gap [95% CI] | final-window gap [95% CI] | efficiency gap [95% CI] | worst-seed AUC gap |")
        println(io, "|---|---|---|---|---|")
        for base in [best_random, best_fixed, "never_explore"]
            a = gaps[base][:auc]; fw = gaps[base][:final_window]; e = gaps[base][:efficiency]
            println(io, "| $base | $(round(a[1], digits=2)) [$(round(a[2], digits=2)), $(round(a[3], digits=2))] " *
                        "| $(round(fw[1], digits=3)) [$(round(fw[2], digits=3)), $(round(fw[3], digits=3))] " *
                        "| $(round(e[1], digits=1)) [$(round(e[2], digits=1)), $(round(e[3], digits=1))] " *
                        "| $(round(worst_gap[base], digits=2)) |")
        end
        println(io, "\n`eu_max − never_explore` is the headline: the learned-returns escape ops are " *
                    "identical on both sides, so this gap is exploration's isolated value.\n")
        println(io, "## Alternative dominance measures (reported, not asserted)\n")
        println(io, "Complementary dominance notions on the same per-seed paired gaps: per-seed win")
        println(io, "rate with an exact two-sided sign test (distribution-free — robust to the")
        println(io, "winner's-curse outlier seeds the mean bootstrap is sensitive to), the median")
        println(io, "gap, the 10th-percentile gap (tail regret softer than minimax), and the")
        println(io, "final-regime rate (converged behaviour: AUC charges exploration's tuition over")
        println(io, "the whole run; this reads what it bought after the last change).\n")
        println(io, "| baseline | metric | mean gap | median gap | win rate | sign-test p | q10 gap |")
        println(io, "|---|---|---|---|---|---|---|")
        for base in [best_random, best_fixed, "never_explore"]
            for (label, f) in [("AUC", r -> r.auc),
                               ("final-window rate", r -> r.final_window_mean),
                               ("final-regime rate", rate_final_regime)]
                d = paired(eu, results[base], f)
                wins = count(>(0.0), d)
                losses = count(<(0.0), d)
                println(io, "| $base | $label | $(round(sum(d) / length(d), digits=3)) " *
                            "| $(round(median_of(d), digits=3)) " *
                            "| $(wins)–$(losses) of $(length(d)) " *
                            "| $(round(sign_test_p(d), digits=4)) " *
                            "| $(round(quantile_of(d, 0.1), digits=3)) |")
            end
        end
        println(io, "\n## Behaviour-verified inversions\n")
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
        gaps[base][:final_window][2] > 0.0 ||
            push!(failures, "CI(eu_max − $base) on final-window rate includes 0: $(gaps[base][:final_window])")
        gaps[base][:efficiency][2] > 0.0 ||
            push!(failures, "CI(eu_max − $base) on shared-level efficiency includes 0: $(gaps[base][:efficiency])")
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
