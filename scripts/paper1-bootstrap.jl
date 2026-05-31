#!/usr/bin/env julia
# scripts/paper1-bootstrap.jl — paired-seed bootstrap for Paper 1.
#
# Loads (agent, seed, total_score) from the benchmark SQLite DB by shelling
# out to the `sqlite3` CLI (no SQLite.jl dep needed in scripts/Project.toml),
# pairs scores by seed across every ordered agent pair, and bootstraps the
# mean Δ over seed-indices (the indices are the resampled units; within each
# resample every pair's score is taken from the same seed row, so seed-level
# variance cancels).
#
# Operates on observed total_score values from the DB — observed outcomes,
# not measure weights — so the single-reasoner invariant is not engaged.
# See precedent slug `baseline-comparison` (apps/julia/qa_benchmark/host.jl:66)
# for the Invariant-1 carve-out applied to diagnostic agent comparisons.
#
# Usage:
#   julia --project=scripts scripts/paper1-bootstrap.jl \
#     --db apps/julia/qa_benchmark/results/benchmark.db \
#     --output papers/paper1/bootstrap-results.md \
#     --resamples 10000 --rng-seed 42
#
# Runs two bootstraps internally: rng-seed (primary) and rng-seed+1
# (stability). Both are reported in a single output file.

using Random
using Printf
using Dates

# ─── CLI ─────────────────────────────────────────────────────────────────

function parse_args(args::Vector{String})
    opts = Dict{String,Any}(
        "db" => "apps/julia/qa_benchmark/results/benchmark.db",
        "output" => "papers/paper1/bootstrap-results.md",
        "resamples" => 10_000,
        "rng_seed" => 42,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--db"
            opts["db"] = args[i+1]; i += 2
        elseif a == "--output"
            opts["output"] = args[i+1]; i += 2
        elseif a == "--resamples"
            opts["resamples"] = parse(Int, args[i+1]); i += 2
        elseif a == "--rng-seed"
            opts["rng_seed"] = parse(Int, args[i+1]); i += 2
        else
            error("unknown arg: $a")
        end
    end
    opts
end

# ─── Data loading ────────────────────────────────────────────────────────

"""
Load (agent, seed, total_score) rows from the SQLite DB by shelling out to
the `sqlite3` CLI. Default `|` field separator (safe — no agent name uses
that character). Returns Dict{agent => Dict{seed => total_score}}.
"""
function load_scores(db_path::String)
    isfile(db_path) || error("DB not found: $db_path")
    cmd = `sqlite3 $db_path "SELECT agent, seed, total_score FROM runs ORDER BY agent, seed;"`
    raw = read(cmd, String)
    data = Dict{String, Dict{Int, Float64}}()
    for line in split(raw, '\n'; keepempty=false)
        parts = split(line, '|')
        length(parts) == 3 || error("unexpected row: $line")
        agent = String(parts[1])
        seed = parse(Int, parts[2])
        score = parse(Float64, parts[3])
        get!(data, agent, Dict{Int, Float64}())[seed] = score
    end
    data
end

# ─── Bootstrap ───────────────────────────────────────────────────────────

mean_(xs) = sum(xs) / length(xs)

function std_(xs)
    n = length(xs)
    n <= 1 && return 0.0
    m = mean_(xs)
    return sqrt(sum((x - m)^2 for x in xs) / (n - 1))
end

"""
Linear-interpolated quantile on a pre-sorted vector. q ∈ [0, 1].
"""
function quantile_sorted(v::Vector{Float64}, q::Float64)
    n = length(v)
    n == 0 && return NaN
    n == 1 && return v[1]
    pos = q * (n - 1) + 1               # 1-based position in [1, n]
    lo = clamp(floor(Int, pos), 1, n)
    hi = clamp(ceil(Int, pos), 1, n)
    lo == hi && return v[lo]
    f = pos - lo
    return v[lo] * (1 - f) + v[hi] * f
end

"""
Run the paired-seed bootstrap. Returns (results, common_seeds).

`results` is a sorted Vector of NamedTuples, one per ordered pair (A, B)
with A ≠ B; `mean_delta` is mean(score_A(s) − score_B(s)) over the
intersection of seeds present for both A and B.

Resample-of-seed-indices is drawn ONCE from `MersenneTwister(rng_seed)` and
reused across pairs, so the same bootstrap "world" applies to every pair.
"""
function bootstrap_pairs(data::Dict{String,Dict{Int,Float64}}, n_resamples::Int, rng_seed::Int)
    agents = sort(collect(keys(data)))
    seed_sets = [Set(keys(data[a])) for a in agents]
    common_seeds = sort(collect(reduce(intersect, seed_sets)))
    n = length(common_seeds)
    n >= 2 || error("need ≥2 common seeds across all agents (got $n)")

    rng = MersenneTwister(rng_seed)
    resample_idx = [rand(rng, 1:n, n) for _ in 1:n_resamples]

    results = NamedTuple[]
    for a in agents, b in agents
        a == b && continue
        deltas = [data[a][common_seeds[i]] - data[b][common_seeds[i]] for i in 1:n]
        observed = mean_(deltas)

        bs_means = Vector{Float64}(undef, n_resamples)
        for (r, idx) in enumerate(resample_idx)
            s = 0.0
            @inbounds for k in idx
                s += deltas[k]
            end
            bs_means[r] = s / n
        end
        sort!(bs_means)

        ci_lo = quantile_sorted(bs_means, 0.025)
        ci_hi = quantile_sorted(bs_means, 0.975)

        # Two-sided percentile-bootstrap p-value: count resamples with sign
        # opposite to observed mean Δ, double, divide by total. (Pre-spec'd
        # in Step A1.4.)
        opposite = if observed > 0
            count(<=(0.0), bs_means)
        elseif observed < 0
            count(>=(0.0), bs_means)
        else
            n_resamples ÷ 2
        end
        p_value = min(1.0, 2 * opposite / n_resamples)

        push!(results, (
            agent_a = a,
            agent_b = b,
            n = n,
            mean_delta = observed,
            ci_lo = ci_lo,
            ci_hi = ci_hi,
            p_value = p_value,
        ))
    end
    sort!(results, by = r -> -r.mean_delta)
    return (results = results, common_seeds = common_seeds)
end

# ─── Output ──────────────────────────────────────────────────────────────

function point_estimate_table(data::Dict{String,Dict{Int,Float64}})
    rows = []
    for (a, scores) in data
        xs = collect(values(scores))
        push!(rows, (
            agent = a,
            n = length(xs),
            mean_score = mean_(xs),
            std_score = std_(xs),
        ))
    end
    sort!(rows, by = r -> -r.mean_score)
    rows
end

function format_pair_row(r::NamedTuple)
    @sprintf("| `%s` | `%s` | %d | %+8.2f | [%+7.2f, %+7.2f] | %.4f |",
            r.agent_a, r.agent_b, r.n, r.mean_delta, r.ci_lo, r.ci_hi, r.p_value)
end

function write_report(opts::Dict{String,Any}, data, primary, stability)
    output_path = abspath(opts["output"])
    mkpath(dirname(output_path))

    seed_p = opts["rng_seed"]
    seed_s = seed_p + 1
    n_res = opts["resamples"]

    ts = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")

    open(output_path, "w") do io
        println(io, "# Paper 1 — paired-seed bootstrap")
        println(io)
        println(io, "Generated by `scripts/paper1-bootstrap.jl` on $ts")
        println(io)
        println(io, "- DB: `$(opts["db"])`")
        println(io, "- Resamples: $n_res")
        println(io, "- Primary RNG seed: $seed_p ; Stability RNG seed: $seed_s")
        println(io, "- Common seeds across all agents: $(length(primary.common_seeds))")
        println(io)

        println(io, "## Point-estimates (mean ± std over seeds)")
        println(io)
        println(io, "| Agent | N | Mean score | Std |")
        println(io, "|---|---|---|---|")
        for r in point_estimate_table(data)
            println(io, @sprintf("| `%s` | %d | %+8.2f | %7.2f |",
                                r.agent, r.n, r.mean_score, r.std_score))
        end
        println(io)

        # Headline
        headline_filter = filter(r -> r.agent_a == "ablation_greedy" && r.agent_b == "bayesian", primary.results)
        if !isempty(headline_filter)
            h = first(headline_filter)
            println(io, "## Headline: `ablation_greedy` − `bayesian`")
            println(io)
            println(io, @sprintf("Δ = %+0.2f, 95%% CI [%+0.2f, %+0.2f], p = %0.4f, paired N = %d.",
                                h.mean_delta, h.ci_lo, h.ci_hi, h.p_value, h.n))
            println(io)
        end

        # Full pairwise table (primary)
        println(io, "## Pairwise comparisons (primary, sorted by descending Δ)")
        println(io)
        println(io, "| Agent A | Agent B | N | Δ = mean(A − B) | 95% CI | p (two-sided) |")
        println(io, "|---|---|---|---|---|---|")
        for r in primary.results
            println(io, format_pair_row(r))
        end
        println(io)

        # Stability section
        println(io, "## Stability (rng-seed=$seed_p vs rng-seed=$seed_s)")
        println(io)
        println(io, "Compares CIs and p-values across two bootstrap RNG seeds. Δ values are")
        println(io, "deterministic (same observed paired differences) so only CI/p drift. A")
        println(io, "p-value crossing 0.05 between the two runs signals that `--resamples`")
        println(io, "(currently $n_res) is too low; bump and re-run both seeds.")
        println(io)
        println(io, "| Agent A | Agent B | Δ | p (seed $seed_p) | p (seed $seed_s) | crosses 0.05? |")
        println(io, "|---|---|---|---|---|---|")
        stab_lookup = Dict((r.agent_a, r.agent_b) => r for r in stability.results)
        cross_count = 0
        for r in primary.results
            s = stab_lookup[(r.agent_a, r.agent_b)]
            crosses = (r.p_value < 0.05) != (s.p_value < 0.05)
            crosses && (cross_count += 1)
            println(io, @sprintf("| `%s` | `%s` | %+0.2f | %.4f | %.4f | %s |",
                                r.agent_a, r.agent_b, r.mean_delta,
                                r.p_value, s.p_value, crosses ? "**YES**" : "no"))
        end
        println(io)
        println(io, "**Crossings: $cross_count of $(length(primary.results)) pairs.**")
        if cross_count > 0
            println(io, " Bump `--resamples` and re-run both seeds.")
        else
            println(io, " Resample count is sufficient at this CI level.")
        end
        println(io)

        # Sanity checks
        println(io, "## Sanity checks")
        println(io)

        # Self-consistency
        println(io, "### Self-consistency: bootstrap observed Δ matches direct DB mean(A) − mean(B)")
        println(io)
        full_ok = true
        max_dev = 0.0
        for r in primary.results
            direct = mean_(collect(values(data[r.agent_a]))) - mean_(collect(values(data[r.agent_b])))
            dev = abs(r.mean_delta - direct)
            dev > max_dev && (max_dev = dev)
            isapprox(r.mean_delta, direct; atol=1e-9) || (full_ok = false)
        end
        if full_ok
            println(io, @sprintf("All %d ordered pairs match direct computation (max |dev| = %.2e). ✓",
                                length(primary.results), max_dev))
        else
            println(io, @sprintf("**FAIL** — pairing logic divergence (max |dev| = %.2e).",
                                max_dev))
        end
        println(io)

        # Spot-check
        println(io, "### Spot-check: `random` − `single_best` (expected ≈ +11.2 from RESULTS.md)")
        println(io)
        rs_filter = filter(r -> r.agent_a == "random" && r.agent_b == "single_best", primary.results)
        if !isempty(rs_filter)
            rs = first(rs_filter)
            side = if rs.ci_lo > 0
                "above zero (random > single_best)"
            elseif rs.ci_hi < 0
                "below zero (random < single_best)"
            else
                "straddles zero"
            end
            println(io, @sprintf("Δ = %+0.2f, 95%% CI [%+0.2f, %+0.2f], p = %0.4f, N = %d. CI %s.",
                                rs.mean_delta, rs.ci_lo, rs.ci_hi, rs.p_value, rs.n, side))
        end
        println(io)
    end

    return output_path
end

# ─── Entry ───────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    opts = parse_args(ARGS)
    println(stderr, "Loading scores from $(opts["db"])...")
    data = load_scores(opts["db"])
    println(stderr, "Loaded $(length(data)) agents.")

    seed_p = opts["rng_seed"]
    seed_s = seed_p + 1
    n_res = opts["resamples"]

    println(stderr, "Running primary bootstrap (rng-seed=$seed_p, $n_res resamples)...")
    primary = bootstrap_pairs(data, n_res, seed_p)
    println(stderr, "Running stability bootstrap (rng-seed=$seed_s)...")
    stability = bootstrap_pairs(data, n_res, seed_s)

    out = write_report(opts, data, primary, stability)
    println(stderr, "Report written to $out")
end
