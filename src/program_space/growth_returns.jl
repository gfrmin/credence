"""
    growth_returns.jl — the learned returns-to-growth model (belief-derived-valuation §2b)

Replaces the entropy escape heuristic: the realised yield of every executed hypothesis-space op
is an OBSERVATION, a conjugate belief over per-op yields is conditioned through the one learning
mechanism, and the op's score is the posterior-predictive expected next yield minus a declared
compute price. Russell–Wefald metareasoning as inference — the constitution's "no separate
metareasoning layer" clause made mechanical.

The model (fixed at ratification, before the gate run — design §6's one-prior-choice discipline):
yields are observed in nats under an Exponential observation model with a Gamma(2, 1) prior on
the rate — conjugate (`GammaPrevision × Exponential`, the existing registry pair), expected next
yield `E[1/λ] = β/(α−1)` exact via the declared `ExponentialMean` TestFunction. Prior expected
yield = 1 nat of bounded initial optimism: an op fires once per fresh context at the default
price, then evidence takes over (one zero-yield observation halves the expectation — the decay
the entropy score never had).

Context is `(op, changed-since-last-fire)` (ratified Q3): four cells for the two escape ops,
independent conjugate beliefs. The regress terminates here (SPEC §1.3): the returns model's
hypothesis space and the compute price are declared data — one meta-level, no tower.
"""

using .Ontology

"""
    GrowthReturns(ops; prior_alpha = 2.0, prior_beta = 1.0)

The returns-to-growth belief: one `GammaPrevision` rate posterior per `(op, changed)` cell
(state-is-measure — the brain state a host carries alongside its explore buffer). `changed` is
host bookkeeping DATA: has the hypothesis space changed since this op last fired?
"""
struct GrowthReturns
    cells::Dict{Tuple{Symbol, Bool}, GammaPrevision}
end

function GrowthReturns(ops::Vector{Symbol}; prior_alpha::Float64 = 2.0, prior_beta::Float64 = 1.0)
    cells = Dict{Tuple{Symbol, Bool}, GammaPrevision}()
    for op in ops, changed in (false, true)
        cells[(op, changed)] = GammaPrevision(prior_alpha, prior_beta)
    end
    GrowthReturns(cells)
end

# The declared yield-observation kernel: y ~ Exponential(λ) given the cell's rate. Conjugate
# through the registry pair (GammaPrevision × Exponential); zero-valued yields are admitted
# (the dedup-no-op case — see the relaxed boundary in conjugate.jl).
_yield_kernel() = Kernel(PositiveReals(), PositiveReals(),
                         λ -> wrap_in_measure(GammaPrevision(1.0, λ)),
                         (λ, y) -> log(λ) - λ * y;
                         likelihood_family = Exponential())

"""
    observe_yield!(gr, op, changed, y) → GrowthReturns

Condition the `(op, changed)` cell on a realised yield `y` (nats, ≥ 0) through Tier-1
`condition` — the one learning mechanism; never decayed, never reset.
"""
function observe_yield!(gr::GrowthReturns, op::Symbol, changed::Bool, y::Float64)
    key = (op, changed)
    haskey(gr.cells, key) || error("GrowthReturns has no cell for $key")
    gr.cells[key] = condition(gr.cells[key], _yield_kernel(), y)
    gr
end

"""
    escape_score(gr, op, changed; compute_cost = 0.0) → Float64

The op's belief-derived score: the posterior-predictive expected next yield
(`expect(cell, ExponentialMean())`, exact `β/(α−1)`) net of the DECLARED compute price
(`net_value`; prices are utility data, never learned value substitutes — ratified Q6).
"""
function escape_score(gr::GrowthReturns, op::Symbol, changed::Bool; compute_cost::Float64 = 0.0)
    key = (op, changed)
    haskey(gr.cells, key) || error("GrowthReturns has no cell for $key")
    net_value(expect(gr.cells[key], ExponentialMean()), compute_cost)
end

"""
    injection_yield_nats(belief, n_added) → Float64

The realised yield of an injection, in nats: `−log(1 − mass)` where `mass` is the posterior
mass captured by the `n_added` most recently appended components (their tags are the trailing
positions — the `add_programs_to_state!` re-tag discipline). Exactly `0.0` for a dedup no-op.
With coherent injection this is exact and instantaneous: newcomers arrive at
evidence-conditioned weights, so the mass they hold IS how much the posterior credits what the
op admitted (zombie resurrections re-enter evidence-crushed and self-report as worthless).
Measure it BEFORE `sync_prune!`/`sync_truncate!` — they may drop the very components.

The ratified Q1 observable; the realised next-window predictive delta is the named finer
fidelity (design §5 Q1).
"""
function injection_yield_nats(belief::MixturePrevision, n_added::Int)
    n_added <= 0 && return 0.0
    n = length(belief.components)
    mass = probability(belief, TagSet(Interval(0.0, 1.0), Set((n - n_added + 1):n)))
    -log(1.0 - min(mass, 1.0 - 1e-12))
end
