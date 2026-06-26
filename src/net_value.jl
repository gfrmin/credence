# net_value.jl — the single net-expected-value shape. collapse-towers Phase 3.
# Tested by test/test_net_value.jl.

"""
    net_value(delta_value, cost) -> Float64

The net expected value of an action: the expected improvement it buys, minus its cost — the **scalar
reduction** of `E[value] − cost` (SPEC's meta-action passage: "think more or act now is one argmax EU").

Instances: `net_voi` (action = observe; `delta_value = voi`, `src/stdlib.jl`) and `net_voc` (action = a
meta-action; `delta_value = E[value after the computation] − value(now)`; collapse-towers Phase 5).

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
