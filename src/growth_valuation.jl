# growth_valuation.jl — belief-derived meta-action valuation (exploration-budget arc).
# Design: docs/exploration-budget/belief-derived-valuation-design.md (RATIFIED 2026-07-02).
# Included inside module Ontology by ontology.jl, AFTER conjugate.jl (Gamma×Exponential
# conjugacy) and events.jl (TagSet). Two halves, both replacing hand-written numbers at the
# selection seam with engine-computed expectations:
#
#   §2a growth_value    — the horizon-completed growth valuation (the net_value pattern:
#                         one canalised scalar reduction; hosts pass declared data and
#                         never multiply).
#   §2b GrowthReturns   — the learned returns-to-growth model for escape ops: realised
#                         yields are observations, a conjugate Gamma×Exponential belief
#                         per (op × changed-since-last-fire) cell is conditioned through
#                         the one learning mechanism, and the score is its posterior mean.
#
# Tested by test/test_growth_returns.jl (unit) and test/test_grid_world_meta.jl (seam).

"""
    growth_value(Δℓ, n_buf, plateau, H; prior_term = 0.0, compute_cost = 0.0) -> Float64

The horizon-completed value of a growth meta-action (design §2a):

    plateau · (Δℓ / n_buf) · H  +  prior_term  −  compute_cost

- `Δℓ` — the measured window-total predictive gain of the op's best candidate (the
  lookahead's fit axis, in nats; `exploration_fit` / `feature_discovery_fit`).
- `n_buf` — the window length the gain was measured on, so `Δℓ/n_buf` is the
  per-conditioning-event gain.
- `plateau` — P(the measured gain is a persistent plateau, not transient): *whether*
  the gain is real (Move-2 semantics, unchanged).
- `H` — expected remaining conditioning events: *how long* it pays. Declared task
  data + host bookkeeping counts. `nothing` declares an open-ended host (§5 Q2)
  ⇒ `H = n_buf`, recovering the window-total score — this defaulting rule lives
  HERE and only here, so every scoring seam and executor completes identically.
- `prior_term` — the one-time Occam charge (e.g. `complexity_logprior(Δcomplexity)`
  for a feature add). A prior over grammars is paid ONCE — never multiplied by the
  horizon, never discounted by `plateau` (it is not a measured gain).
- `compute_cost` — the declared price of the op's compute (utility data).

Computed as `plateau · Δℓ · (H / n_buf)` so the `H == n_buf` case reproduces the
window-total score `plateau·Δℓ + prior_term − compute_cost` EXACTLY (`H/n_buf == 1.0`
in floats) — the nested-special-case pin, test_growth_returns.jl §1. `n_buf ≤ 0`
(empty window) means no measured gain: the value is `prior_term − compute_cost`.

The subtraction routes through the one canalised scalar reduction `net_value`
(`src/net_value.jl` — pure linear value − cost, no clamp; this site is on that
invariant's audit surface, paired back-reference kept there). The no-op floor
(a lookahead with no clearing candidate scores 0) belongs to the callers
(`exploration_voi` / `feature_discovery_voi`), not to this functional.
"""
function growth_value(Δℓ::Float64, n_buf::Int, plateau::Float64,
                      H::Union{Nothing, Float64};
                      prior_term::Float64 = 0.0, compute_cost::Float64 = 0.0)::Float64
    n_buf <= 0 && return net_value(prior_term, compute_cost)
    h = H === nothing ? Float64(n_buf) : H
    net_value(plateau * Δℓ * (h / n_buf) + prior_term, compute_cost)
end

# ═══════════════════════════════════════
# The learned returns-to-growth model (design §2b)
# ═══════════════════════════════════════

# The declared observation model for realised yields: yield ~ Exponential(λ) with a
# Gamma(α, β) prior on the rate — conjugate (`GammaPrevision × Exponential`, the
# registry pair in conjugate.jl). Expected next yield E[y] = E[1/λ] = β/(α−1), exact
# via the declared `Reciprocal` test function. Built once; stateless.
# NOTE: apps/skin/server.jl builds the same declared shape for the wire's "exponential"
# kernel (with a numerics guard on log(λ) its non-conjugate paths need); lifting one
# shared constructor into kernels.jl is a follow-up — kept separate here to leave the
# skin bit-stable in this move.
const _GROWTH_YIELD_KERNEL = Kernel(PositiveReals(), PositiveReals(),
    _ -> error("generate not used in condition"),
    (λ, y) -> log(λ) - λ * y;
    likelihood_family = Exponential())

"""
    GrowthReturns

The returns-to-growth belief state (design §2b): one conjugate Gamma×Exponential cell
per `(op, changed-since-last-fire)` context (§5 Q3 — the minimal honest context), plus
the epoch bookkeeping that derives the context bit. Brain state (state-is-measure):
cells are Previsions, updated ONLY through Tier-1 `condition` (`observe_growth_yield!`),
read ONLY through `expect` (`expected_growth_yield`). Never decayed, never reset — a
context shift is a different cell, not a forgotten one.

Cold start: a fresh cell is the declared prior `Gamma(prior_alpha, prior_beta)` —
Gamma(2, 1) by default, i.e. prior expected yield `β/(α−1) = 1` nat of bounded initial
optimism that decays under evidence (after one zero-yield observation the expectation
halves to 1/2; after k, `1/(1+k)`). Fixed at ratification, before the gate run (§6's
one-prior-choice discipline).
"""
mutable struct GrowthReturns
    cells::Dict{Tuple{Symbol, Bool}, GammaPrevision}
    space_epoch::Int                      # bumped by note_space_change! on any hypothesis-space change
    last_fire_epoch::Dict{Symbol, Int}    # per op: space_epoch at its last firing
    prior_alpha::Float64
    prior_beta::Float64
end

"""
    initial_growth_returns(; alpha = 2.0, beta = 1.0) → GrowthReturns

The fresh returns state. `alpha`/`beta` are the declared per-cell Gamma prior
(shape/rate on the yield rate λ); `alpha > 1` is required for the expected yield
`β/(α−1)` to exist (`Reciprocal` guards it at read time).
"""
initial_growth_returns(; alpha::Float64 = 2.0, beta::Float64 = 1.0) =
    GrowthReturns(Dict{Tuple{Symbol, Bool}, GammaPrevision}(), 0, Dict{Symbol, Int}(),
                  alpha, beta)

# The context bit: has the hypothesis space changed since this op last fired? An op that
# has never fired is in the fresh (`changed = true`) context. Self-effects count — an op
# whose own execution changed the space (resurrections, depth raise) reads the changed
# cell next time; both cells accumulate their own evidence forever.
_growth_context(gr::GrowthReturns, op::Symbol)::Tuple{Symbol, Bool} =
    (op, !haskey(gr.last_fire_epoch, op) || gr.space_epoch > gr.last_fire_epoch[op])

_growth_cell(gr::GrowthReturns, ctx::Tuple{Symbol, Bool})::GammaPrevision =
    get(gr.cells, ctx) do
        GammaPrevision(gr.prior_alpha, gr.prior_beta)
    end

"""
    expected_growth_yield(gr, op) → Float64

The posterior-predictive expected next yield (nats) of `op` in its current context:
`expect(cell, Reciprocal())` = `β/(α−1)` on the cell's Gamma posterior — a Tier-1 read.
The escape-op score is `net_value(expected_growth_yield(gr, op), declared_price)`.
"""
expected_growth_yield(gr::GrowthReturns, op::Symbol)::Float64 =
    expect(_growth_cell(gr, _growth_context(gr, op)), Reciprocal())

"""
    observe_growth_yield!(gr, op, yield) → GrowthReturns

Condition `op`'s current-context cell on its realised `yield` (nats, ≥ 0) through
Tier-1 `condition` with the declared Exponential kernel — the one learning mechanism —
and record the firing epoch (the next context bit is measured from here). Call once
per executed op, with the yield measured by `growth_yield` immediately after the
op's injection (before any prune/truncate re-tags).
"""
function observe_growth_yield!(gr::GrowthReturns, op::Symbol, yield::Float64)
    ctx = _growth_context(gr, op)
    gr.cells[ctx] = condition(_growth_cell(gr, ctx), _GROWTH_YIELD_KERNEL, yield)
    gr.last_fire_epoch[op] = gr.space_epoch
    gr
end

"""
    note_space_change!(gr) → GrowthReturns

Declare that the hypothesis space changed (components injected, depth raised, grammar
added). Ops whose last firing precedes this earn fresh consideration — their context
bit flips to `changed = true`, reading the changed-context cell (§5 Q3). Data
bookkeeping, not a belief operation: nothing is reweighted or reset.
"""
function note_space_change!(gr::GrowthReturns)
    gr.space_epoch += 1
    gr
end

"""
    growth_yield(belief, tags) → Float64

The realised yield observable of an executed op (design §5 Q1, ratified: injected
posterior mass): the posterior mass captured by the op's coherently-injected
components, converted to nats as `−log(1 − mass)` (the log-evidence the op claimed).
Exactly `0.0` when the op injected nothing (dedup no-op: empty `tags` ⇒ mass 0),
near-zero when zombies re-enter at evidence-crushed weight (the churn self-reports
as worthless). Read immediately after injection, BEFORE `sync_prune!`/`sync_truncate!`
re-tag.

Computed from the INCUMBENT side — `1 − mass = probability(belief, TagSet(incumbent
tags))`, a Tier-1 read — so a high-yield injection never suffers the `1 − mass`
cancellation: when the injected block holds all but ~1e-16 of the mass, the incumbent
sum is that small number exactly, not a rounded 0. The mixture's components carry the
contiguous tags `1..N` (the sync re-tag discipline), so the incumbent set is the
complement of `tags`. An incumbent mass that underflows to exactly `0.0`
(incumbents > ~745 nats under the newcomers) saturates at the measurement's
representable bound `−log(floatmin(Float64))` — a type fact, not a tuned constant:
the instrument's resolution, not a decision.

The belief is assumed to be a Beta-carrier tagged mixture (`TaggedBetaPrevision`
components — the program-space shape); components without tags contribute no mass.
"""
function growth_yield(belief::MixturePrevision, tags)::Float64
    tagset = Set{Int}(tags)
    isempty(tagset) && return 0.0
    # Declared structure, fail loud (Invariant 2): the TagSet mass read silently skips
    # untagged components, which here would fabricate zero yields and silently mis-train
    # the returns cells — so the unsupported shape errors instead.
    all(c isa TaggedBetaPrevision for c in belief.components) ||
        error("growth_yield: the belief must be a tagged Beta mixture " *
              "(TaggedBetaPrevision components); an untagged component would silently " *
              "contribute zero mass and corrupt the yield observable")
    incumbents = setdiff(Set{Int}(1:length(belief.components)), tagset)
    isempty(incumbents) &&
        error("growth_yield: no incumbent components — the yield observable requires " *
              "incumbents (injection into an empty belief is not an escape-op shape)")
    inc_mass = probability(belief, TagSet(Interval(0.0, 1.0), incumbents))
    inc_mass > 0.0 ? -log(inc_mass) : -log(floatmin(Float64))
end
