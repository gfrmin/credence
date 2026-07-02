# Role: brain-side application
"""
    metrics.jl — Thin per-run aggregation for the dominance benchmark

Reduces a grid_world MetricsTracker (per-step trajectories) to the per-(policy, seed)
quantities the gate compares (dominance-design §2, amended by belief-derived-valuation §2c):
realised value (AUC of the cumulative interaction-energy trajectory + final-window rate as a
CO-PRIMARY — no magic threshold), sample efficiency against a SHARED per-seed reference (steps
to half the best policy's final total — the self-relative variant rewarded early collapse and
is retained for reporting only), meta-action volume, and grammar growth. All arithmetic here is
on realised world outcomes, not on beliefs.
"""

struct RunSummary
    policy::String
    seed::Int
    auc::Float64                # mean of the cumulative-energy trajectory (area, normalised by steps)
    final_window_mean::Float64  # per-step energy rate over the last 20% of the run (co-primary, §2c)
    steps_to_half::Int          # first step reaching half the run's own final cumulative energy
                                # (self-relative; REPORTING ONLY — rewards early collapse, §2c)
    n_meta::Int                 # total meta-actions taken
    n_grammars::Int             # grammar-pool size at the end
    growth_steps::Vector{Tuple{Int, Symbol}}   # recorded growth ops (for the inversions report)
    ce::Vector{Float64}         # the cumulative-energy trajectory (for shared-reference efficiency)
end

function summarise(policy::String, seed::Int, m::MetricsTracker,
                   growth_steps::Vector{Tuple{Int, Symbol}})::RunSummary
    ce = m.cumulative_energy
    n = length(ce)
    auc = sum(ce) / n
    w = max(1, round(Int, 0.2 * n))
    fwm = (ce[end] - (n - w >= 1 ? ce[n - w] : 0.0)) / w
    sth = n
    if ce[end] > 0.0
        half = 0.5 * ce[end]
        for t in 1:n
            if ce[t] >= half
                sth = t
                break
            end
        end
    end
    RunSummary(policy, seed, auc, fwm, sth, sum(m.meta_actions_per_step),
               last(m.n_grammars), growth_steps, collect(ce))
end

"""
    steps_to_level(s, level) → Int

The fixed-reference efficiency metric (belief-derived-valuation §2c): first step at which the
run's cumulative energy reaches `level` (the gate uses half the per-seed BEST policy's final
total — one shared bar per seed, so collapsing early cannot look "efficient"). Runs that never
reach the level score the full run length.
"""
function steps_to_level(s::RunSummary, level::Float64)::Int
    for t in eachindex(s.ce)
        s.ce[t] >= level && return t
    end
    length(s.ce)
end
