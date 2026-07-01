# gather_voi.jl — the gather VOI: pricing + which-gather argmax for grow (recall/discovery)
# actuators. `net_voi` prices observations over a *closed* hypothesis set; a grow actuator
# instead proposes to ENLARGE the set so a missing truth can enter, so it is priced by a
# separate recovery belief `g = P(recover | sensors)` — a structure-BMA readout
# (`belief_at_context`) — against the consumer's terminal EU. Connection-generic: any body
# with actuators + sensors uses this exactly as it uses `optimise`/`net_voi`.
# Lifted from apps/answer-brain (the conferred gather half — life-agent
# docs/ask-as-connection.md §4/§7); pure functions over the above, no new frozen type.

"""
    grow_value(g, u_correct, eu, cost) -> Float64

The gather VOI of one grow (recall/discovery) actuator. A grow move may enlarge the
consumer's hypothesis set to admit a missing truth; `g = P(recover | sensors)` is a
structure-BMA belief at the sensor context. The value is the gain from converting the
current terminal outcome (EU `eu`) into the correct-terminal value (`u_correct`), realised
with probability `g`, net of `cost`. The missing mass **self-gates through `eu`**: a
confident terminal has `eu ≈ u_correct` ⇒ prices `≈ −cost` (don't grow a solved decision);
a withhold has low `eu` ⇒ a large gain. There is no missing-mass branch — `p_none`/entropy
are features of the sensor context that shape `g`. Optimistic in the post-recovery value
(a bound; `g` carries the realism).
"""
grow_value(g::Float64, u_correct::Float64, eu::Float64, cost::Float64)::Float64 =
    g * (u_correct - eu) - cost

"""
    best_grow(actuators, u_correct, eu) -> (probe::Union{String,Nothing}, value::Float64)

The which-gather argmax over grow actuators (e.g. re-extract vs retrieve-wider). Each
actuator is `(probe::String, g::Float64, cost::Float64)`; the winner is the `grow_value`
argmax, returned iff it strictly clears 0 (mirrors the `:voi` `net_voi > 0` gate). None
clears ⇒ `(nothing, 0.0)` — no grow, the terminal decision stands.
"""
function best_grow(actuators, u_correct::Float64, eu::Float64)
    best = nothing
    best_v = 0.0
    for (probe, g, cost) in actuators
        v = grow_value(Float64(g), u_correct, eu, Float64(cost))
        v > best_v && (best_v = v; best = String(probe))
    end
    (best, best_v)
end
