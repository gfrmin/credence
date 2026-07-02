# net_value.jl — the single net-expected-value shape. collapse-towers Phase 3.
# Tested by test/test_net_value.jl.

"""
    net_value(delta_value, cost) -> Float64

The net expected value of an action: the expected improvement it buys, minus its cost — the **scalar
reduction** of `E[value] − cost` (SPEC's meta-action passage: "think more or act now is one argmax EU").

Instances: `net_voi` (action = observe; `delta_value = voi`, in utility currency, `src/stdlib.jl`) and
`net_voc` (action = a grammar perturbation; `delta_value = Δcomplexity_logprior`, in **log-prior nats**,
`src/program_space/perturbation.jl`, collapse-towers Phase 5). `net_voc` is the same functional FORM in
a different currency: depth-one the metalevel can only afford the change in the program-space complexity
prior, not achievable EU (the value-proxy the complexity-prior axiom exists to supply — SPEC §1.3). The
form unifies across instances; the currency is whatever each decision context can afford.

The same semantics also lives in a **general (Functional-offset) representation** —
`eu(joint, LinearCombination([(reward, Projection(a))], -(cost + time_cost)))` — used by the routing
EU (`_eu_functional`, `src/routing.jl`) and the `decide_with_voi` block payoff, where the value is
integrated over the joint by `expect` and the cost rides in the offset. The scalar `net_value` here is
that representation's reduction for when value is already a scalar.

INVARIANT (load-bearing for the arc's "one argmax EU subsumes every lever" thesis): this is **pure
linear `value − cost` — no clamp, no nonlinearity.** Both representations must stay semantically
identical; if a nonlinearity is added to either site, the unification breaks and must be revisited.
A paired back-reference comment is kept at `_eu_functional` (`src/routing.jl`) so an edit at either
site is visible against this invariant.
"""
net_value(delta_value::Real, cost::Real) = delta_value - cost

"""
    growth_value(fit, n_buf, plateau, horizon; prior_term = 0.0, compute_cost = 0.0) -> Float64

The horizon-completed Δ log-evidence value of a hypothesis-space growth op
(belief-derived-valuation design §2a):

    plateau · fit · (horizon / n_buf) + prior_term − compute_cost

`fit` is the lookahead's window-total Δℓ measured over `n_buf` conditioning events; `horizon` is
the expected number of remaining conditioning events (DECLARED task data — the episode length is
the host's to declare; open-ended hosts pass `horizon = n_buf` and recover the window-total
valuation); `plateau` is P(the measured gain is a persistent plateau, Move 2) — *whether* the
gain is real vs *how long* it pays, no double count. `prior_term` (the one-time Occam charge,
e.g. −log 2 per feature symbol) is never horizon-multiplied — it is a prior over grammars, paid
once. With `horizon == n_buf` and `plateau == 1` this reduces BIT-EXACTLY to
`net_value(fit + prior_term, compute_cost)`, the pre-move valuation (the multiplier is exactly
1.0; pinned by test_growth_returns.jl §5). `n_buf ≤ 0` (empty window) ⇒ the fit term is 0.

Same linear-value−cost INVARIANT as `net_value` above: no clamp, no nonlinearity.
"""
growth_value(fit::Real, n_buf::Integer, plateau::Real, horizon::Real;
             prior_term::Real = 0.0, compute_cost::Real = 0.0) =
    (n_buf <= 0 ? 0.0 : plateau * fit * (horizon / n_buf)) + prior_term - compute_cost
