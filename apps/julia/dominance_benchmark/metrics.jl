# Role: brain-side application
"""
    metrics.jl — Thin per-run aggregation for the dominance benchmark

Reduces a grid_world MetricsTracker (per-step trajectories) to the per-(policy, seed)
quantities the gate compares (dominance-design §2, amended by belief-derived-valuation
§2c): realised value (AUC of the cumulative interaction-energy trajectory + final-window
rate as CO-PRIMARY — no magic threshold), sample efficiency (fixed-reference: steps to a
level shared across policies, computed per seed in run.jl from the retained trajectory —
the self-relative steps-to-own-half rewards early collapse and is retained as a secondary
column only), meta-action volume, and grammar growth. All arithmetic here is on realised
world outcomes, not on beliefs.
"""

struct RunSummary
    policy::String
    seed::Int
    auc::Float64                # mean of the cumulative-energy trajectory (area, normalised by steps)
    final_window_mean::Float64  # per-step energy rate over the last 20% of the run (CO-PRIMARY, §2c)
    steps_to_half::Int          # first step reaching half the run's own final cumulative energy
                                # (self-relative; SECONDARY — rewards early collapse, §2c)
    n_meta::Int                 # total meta-actions taken
    n_grammars::Int             # grammar-pool size at the end
    growth_steps::Vector{Tuple{Int, Symbol}}   # recorded growth ops (for the inversions report)
    cumulative::Vector{Float64} # the retained trajectory — run.jl derives the fixed-reference
                                # efficiency (steps to half the best policy's per-seed total)
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
               last(m.n_grammars), growth_steps, copy(ce))
end

"""
    steps_to_level(r, level) → Int

Fixed-reference sample efficiency (belief-derived-valuation §2c): the first step at which
the run's cumulative energy reaches `level` (a reference SHARED across policies — run.jl
uses half the best policy's total on the same seed). Runs that never reach it score the
full run length — bounded, and comparable across policies unlike steps-to-own-half.
"""
function steps_to_level(r::RunSummary, level::Float64)::Int
    for t in eachindex(r.cumulative)
        r.cumulative[t] >= level && return t
    end
    length(r.cumulative)
end
