#!/usr/bin/env julia
# Role: brain-side application (experiment harness — NOT the gate)
"""
    experiment_priced.jl — Priced-exploration sweep for the dominance benchmark

EVIDENCE, not a fix. The first full gate run (results/summary.md, commit 3af9077) failed
honestly: eu_max at compute_cost = 0 over-fires the exact VOI tier (10.7 growth ops/run vs
the sparse optimum ~3-6), so best-tuned sparse baselines beat it on raw AUC even though the
headline (eu_max − never_explore) held. Diagnosis: the deployed VOI prices neither the
posterior-transition cost of a growth op nor its own compute — so any positive Δmll on a
noisy 30-obs window licenses a refinement.

This script measures whether a DECLARED positive exploration compute-cost restores
dominance, and at what price level. It sweeps eu_max over
exploration_cost ∈ {0.0, 0.25, 0.5, 1.0, 2.0} and compares each against the three fixed
reference baselines (random_p005, fixed_k50, never_explore) run at their own tuned
behaviour (cost 0 — the baselines are the fixed yardstick the gate compares against, so
their behaviour must not move with the lever).

The output table answers: is there a cost region where eu_max BOTH keeps the headline
(beats never_explore) AND beats the sparse baselines? And the mechanism: mean growth ops vs
cost.

Manually run, out of the fast suite:

    julia apps/julia/dominance_benchmark/experiment_priced.jl

Writes results/priced_experiment.md. NO gate assertions — this is a design-evidence run;
the tuned constant it surfaces is NOT baked anywhere.
"""

include(joinpath(@__DIR__, "run.jl"))   # brings host.jl + bootstrap_ci + paired + mean_of

const PRICE_SWEEP = Float64[0.0, 0.25, 0.5, 1.0, 2.0]

# eu_max variant names, keyed by cost. cost 0.0 keeps the plain name for continuity with the
# gate's summary.md; the priced variants get cNNN suffixes (0.25 → c025 etc.).
const EU_NAMES = Dict{Float64, String}(
    0.0  => "eu_max",
    0.25 => "eu_max_c025",
    0.5  => "eu_max_c05",
    1.0  => "eu_max_c1",
    2.0  => "eu_max_c2",
)

"""Mean number of recorded growth ops (explore / add_feature / perturb) per run."""
mean_growth(rs::Vector{RunSummary}) = sum(length(r.growth_steps) for r in rs) / length(rs)

function run_priced(; n_seeds::Int = DB_N_SEEDS)
    println("Priced-exploration sweep: eu_max at cost ∈ $(PRICE_SWEEP) plus baselines " *
            "random_p005, fixed_k50, never_explore, $(n_seeds) seeds each.")

    # ── Baselines (fixed reference; run at cost 0 = their own tuned behaviour) ──
    baseline_factories = Dict{String, Function}(
        "random_p005"   => (seed -> make_random(seed, 0.05)),
        "fixed_k50"     => (_seed -> make_fixed_schedule(50)),
        "never_explore" => (_seed -> make_never_explore()),
    )
    baselines = Dict{String, Vector{RunSummary}}()
    for (name, factory) in baseline_factories
        t0 = time()
        rs = RunSummary[run_cell(name, factory, seed; exploration_cost = 0.0)
                        for seed in 0:(n_seeds - 1)]
        baselines[name] = rs
        println("  $name: $(n_seeds) seeds in $(round(time() - t0, digits=1))s, " *
                "mean AUC $(round(mean_of(rs, r -> r.auc), digits=2)), " *
                "mean growth $(round(mean_growth(rs), digits=1))")
    end

    # ── eu_max at each price ──
    eu_by_cost = Dict{Float64, Vector{RunSummary}}()
    for c in PRICE_SWEEP
        name = EU_NAMES[c]
        t0 = time()
        rs = RunSummary[run_cell(name, _seed -> make_eu_max(), seed; exploration_cost = c)
                        for seed in 0:(n_seeds - 1)]
        eu_by_cost[c] = rs
        println("  $name (cost $c): $(n_seeds) seeds in $(round(time() - t0, digits=1))s, " *
                "mean AUC $(round(mean_of(rs, r -> r.auc), digits=2)), " *
                "mean growth $(round(mean_growth(rs), digits=1))")
    end

    # ── Table rows: for each cost, mean AUC, mean growth, paired gaps vs each baseline ──
    resdir = joinpath(@__DIR__, "results")
    mkpath(resdir)
    path = joinpath(resdir, "priced_experiment.md")
    open(path, "w") do io
        println(io, "# Priced-exploration sweep — dominance benchmark\n")
        println(io, "Task: `$(DB_WORLD_RULES)`, regime changes at `$(DB_REGIME_STEPS)`, " *
                    "$(DB_MAX_STEPS) steps, respawn on, $n_seeds seeds, " *
                    "paired-seed percentile bootstrap (10 000 resamples).\n")
        println(io, "eu_max swept over declared `exploration_cost` (Δ log-evidence nats, " *
                    "priced into BOTH the exact VOI lookahead and its matching execution). " *
                    "Baselines run at their own tuned behaviour (cost 0). A positive gap " *
                    "(eu_max − baseline) favours eu_max; efficiency sign-flipped so + favours eu_max.\n")

        # Baseline reference means.
        println(io, "## Baseline reference (fixed)\n")
        println(io, "| baseline | mean AUC | mean growth ops |")
        println(io, "|---|---|---|")
        for base in ["never_explore", "random_p005", "fixed_k50"]
            println(io, "| $base | $(round(mean_of(baselines[base], r -> r.auc), digits=2)) " *
                        "| $(round(mean_growth(baselines[base]), digits=1)) |")
        end

        # Main sweep table.
        println(io, "\n## eu_max priced sweep\n")
        println(io, "| cost | mean AUC | mean growth ops | AUC gap vs never_explore [CI] " *
                    "| AUC gap vs random_p005 [CI] | AUC gap vs fixed_k50 [CI] " *
                    "| worst-seed gap vs random | worst-seed gap vs fixed |")
        println(io, "|---|---|---|---|---|---|---|---|")
        for c in PRICE_SWEEP
            eu = eu_by_cost[c]
            g_ne = bootstrap_ci(paired(eu, baselines["never_explore"], r -> r.auc))
            g_rp = bootstrap_ci(paired(eu, baselines["random_p005"], r -> r.auc))
            g_fk = bootstrap_ci(paired(eu, baselines["fixed_k50"], r -> r.auc))
            w_rp = minimum(paired(eu, baselines["random_p005"], r -> r.auc))
            w_fk = minimum(paired(eu, baselines["fixed_k50"], r -> r.auc))
            fmtci(t) = "$(round(t[1], digits=2)) [$(round(t[2], digits=2)), $(round(t[3], digits=2))]"
            println(io, "| $c | $(round(mean_of(eu, r -> r.auc), digits=2)) " *
                        "| $(round(mean_growth(eu), digits=1)) " *
                        "| $(fmtci(g_ne)) | $(fmtci(g_rp)) | $(fmtci(g_fk)) " *
                        "| $(round(w_rp, digits=2)) | $(round(w_fk, digits=2)) |")
        end

        # Efficiency (steps-to-own-half, sign-flipped so + favours eu_max) vs sparse baselines.
        println(io, "\n## Efficiency gap (steps-to-own-half; baseline − eu_max, + favours eu_max)\n")
        println(io, "| cost | eff gap vs random_p005 [CI] | eff gap vs fixed_k50 [CI] |")
        println(io, "|---|---|---|")
        for c in PRICE_SWEEP
            eu = eu_by_cost[c]
            e_rp = bootstrap_ci(paired(baselines["random_p005"], eu, r -> Float64(r.steps_to_half)))
            e_fk = bootstrap_ci(paired(baselines["fixed_k50"], eu, r -> Float64(r.steps_to_half)))
            fmtci(t) = "$(round(t[1], digits=1)) [$(round(t[2], digits=1)), $(round(t[3], digits=1))]"
            println(io, "| $c | $(fmtci(e_rp)) | $(fmtci(e_fk)) |")
        end

        println(io, "\n## Reading the sweep\n")
        println(io, "- **Headline held** at a cost iff the `AUC gap vs never_explore [CI]` " *
                    "lower bound stays > 0 (exploration still pays for itself).")
        println(io, "- **Dominance restored** at a cost iff the `AUC gap vs random_p005` AND " *
                    "`AUC gap vs fixed_k50` CI lower bounds are both > 0 (or at minimum the " *
                    "point gaps turn non-negative and worst-seed gaps stop being deeply negative).")
        println(io, "- **Priced region exists** iff some cost row satisfies BOTH above.")
        println(io, "- **Mechanism**: mean growth ops should fall monotonically with cost; " *
                    "the failure mode is a cost so high that growth never fires and eu_max " *
                    "collapses into never_explore (headline gap → 0).")
    end
    println("\nwrote $path")

    (; eu_by_cost, baselines)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_priced()
end
