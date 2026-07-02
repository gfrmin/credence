# Role: brain-side application
"""
    metrics.jl — Thin per-run aggregation for the dominance benchmark

Reduces a grid_world MetricsTracker (per-step trajectories) to the per-(policy, seed)
quantities the gate compares (dominance-design §2): realised value (AUC of the cumulative
interaction-energy trajectory + final-window rate — no magic threshold), sample efficiency
(steps to the run's OWN half-total, self-relative), meta-action volume, and grammar growth.
All arithmetic here is on realised world outcomes, not on beliefs.
"""

struct RunSummary
    policy::String
    seed::Int
    auc::Float64                # mean of the cumulative-energy trajectory (area, normalised by steps)
    final_window_mean::Float64  # per-step energy rate over the last 20% of the run
    steps_to_half::Int          # first step reaching half the run's own final cumulative energy
                                # (self-relative sample efficiency; n_steps when final ≤ 0)
    n_meta::Int                 # total meta-actions taken
    n_grammars::Int             # grammar-pool size at the end
    growth_steps::Vector{Tuple{Int, Symbol}}   # recorded growth ops (for the inversions report)
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
               last(m.n_grammars), growth_steps)
end
