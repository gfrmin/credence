# Role: brain-side application
"""
    host.jl — Dominance-benchmark loop: policies × seeds over the non-stationary grid world

The task (fixed across all policies; the ONLY lever the policies control is meta-action
selection): three regimes forcing serial re-discovery — colour-typed (enemy = red) →
motion-typed (colour uninformative, enemy = fast) → territorial (colour and speed
uninformative, enemy = near walls) — with entity respawn on, so the encounter process is
recurrent and beliefs keep receiving evidence within each regime.

Per (policy, seed): one `run_agent` with that seed and policy, reduced by `summarise`.
"""

include(joinpath(@__DIR__, "..", "grid_world", "host.jl"))
include(joinpath(@__DIR__, "policies.jl"))
include(joinpath(@__DIR__, "metrics.jl"))

const DB_WORLD_RULES = Symbol[:colour_typed, :motion_typed, :territorial]
const DB_REGIME_STEPS = Int[70, 140]
const DB_MAX_STEPS = 210
const DB_N_SEEDS = 20
const DB_K_SWEEP = Int[5, 10, 25, 50]
const DB_P_SWEEP = Float64[0.05, 0.15, 0.4]

"""
The benchmark's declared starting vocabulary: colour features ONLY. Regime 2's predictor
(:speed) and regime 3's (:wall_dist) are absent from every seed grammar, so staying good
across regime changes REQUIRES feature discovery — with the stock 12-grammar pool the
needed programs pre-exist and mere reweighting suffices, which makes every policy tie
(measured; the §6 'regime-shift magnitude is load-bearing' diagnosis). The host still
extracts the full sensor superset each step, so discovery candidates are available.
"""
function colour_only_pool()::Vector{Grammar}
    reset_grammar_counter!()
    red_body = AndExpr(GTExpr(:red, 0.7), AndExpr(LTExpr(:green, 0.3), LTExpr(:blue, 0.3)))
    blue_body = AndExpr(LTExpr(:red, 0.3), AndExpr(LTExpr(:green, 0.3), GTExpr(:blue, 0.7)))
    Grammar[
        Grammar(Set([:red, :green, :blue]), ProductionRule[], next_grammar_id()),
        Grammar(Set([:red, :green, :blue]), [ProductionRule(:RED, red_body)], next_grammar_id()),
        Grammar(Set([:red, :green, :blue]), [ProductionRule(:BLUE, blue_body)], next_grammar_id()),
    ]
end

"""Run one (policy, seed) cell; `factory(seed)` builds a fresh policy closure per run."""
function run_cell(policy_name::String, factory::Function, seed::Int)::RunSummary
    growth_log = Tuple{Int, Symbol}[]
    policy = make_recording(factory(seed), growth_log)
    metrics, _state, _pool = run_agent(
        world_rules = DB_WORLD_RULES,
        regime_change_steps = DB_REGIME_STEPS,
        max_steps = DB_MAX_STEPS,
        program_max_depth = 2,
        max_meta_per_step = 3,
        verbose = false,
        rng_seed = seed,
        meta_policy = policy,
        respawn = true,
        observe_adjacent = true,
        seed_grammars = colour_only_pool(),
        explore_window = 30)
    summarise(policy_name, seed, metrics, growth_log)
end

"""All policy configurations of the benchmark, name → per-seed factory."""
function policy_table()
    entries = Tuple{String, Function}[
        ("eu_max", _seed -> make_eu_max()),
        ("never_explore", _seed -> make_never_explore()),
        ("clairvoyant", _seed -> make_clairvoyant(DB_REGIME_STEPS)),
    ]
    for p in DB_P_SWEEP
        push!(entries, ("random_p$(replace(string(p), "." => ""))", seed -> make_random(seed, p)))
    end
    for k in DB_K_SWEEP
        push!(entries, ("fixed_k$(k)", _seed -> make_fixed_schedule(k)))
    end
    entries
end

"""Run the full grid: every policy × seeds 0:DB_N_SEEDS-1."""
function run_benchmark(; n_seeds::Int = DB_N_SEEDS, verbose::Bool = true)
    results = Dict{String, Vector{RunSummary}}()
    for (name, factory) in policy_table()
        rs = RunSummary[]
        t0 = time()
        for seed in 0:(n_seeds - 1)
            push!(rs, run_cell(name, factory, seed))
        end
        results[name] = rs
        verbose && println("  $name: $(n_seeds) seeds in $(round(time() - t0, digits=1))s, " *
                           "mean AUC $(round(sum(r.auc for r in rs) / n_seeds, digits=2))")
    end
    results
end
