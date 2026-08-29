#!/usr/bin/env julia
# Role: tests
"""
    test_extra_actions.jl — body-priced terminal rows (life-agent r30b).

The body may declare terminal actions of its own: `{name, act, values}` rows over the same K+1
atoms, ranked beside the built-in ones by the SAME `optimise` call. This is the generic form of
what `report_scoped_j` has always been on the skin lane — a row whose VALUE the body computed
and whose RANKING the engine owns.

The daemon must do NO arithmetic on them. Their loss is declared once on the body's side, where
it is also graded; a second spelling here would let the body be graded on a loss it did not
decide under. Invariant 1 is unchanged: the effector still comes from `optimise`.

`act` names the SPEECH ACT the row belongs to, so the registry's eligibility predicates — the
owner-scoped attribution guard, the §2-A rescue gate — keep asking "did this commit?" rather
than matching a wire name.

Synthetic scenarios; no PII.

Run from the credence repo root:
    julia --project=. apps/answer-brain/tests/julia/test_extra_actions.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "src"))
using Credence
using JSON3

include(joinpath(@__DIR__, "..", "..", "brain", "answer_brain.jl"))
using .AnswerBrain
include(joinpath(@__DIR__, "..", "..", "daemon", "server.jl"))
using .Server

const PASSED = String[]
function check(name::AbstractString, cond::Bool; detail::AbstractString = "")
    if cond
        push!(PASSED, name); println("PASSED: ", name)
    else
        println("FAILED: ", name, " — ", detail); error("assertion failed: $name")
    end
end

const UBAR = Dict("u_correct" => 1.0, "u_wrong" => -5.0, "u_hedged" => 0.4,
                  "u_abstain" => 0.0, "lambda_int" => 1.0)

# A dispersed two-candidate posterior: every crisp report is below the bar, so the built-in
# action set withholds. This is exactly the population r30b's lever addresses.
const DISPERSED = candidate_posterior(2, Obs[Obs(0, 0, 0.9, 1.0, 1.0), Obs(1, 1, 0.9, 1.0, 1.0)], 0.7)

println("="^64)
println("answer-brain — body-priced terminal rows (extra_actions)")
println("="^64)

# ── 1. Absent extras are byte-identical to before ────────────────────────────────────────
let (o1, f1) = decision_fpa(2, UBAR), (o2, f2) = decision_fpa(2, UBAR; extra = [])
    check("absent extras ⇒ the same order vector", o1 == o2)
    check("absent extras ⇒ the same utility rows",
          all(f1[k].values == f2[k].values for k in o1))
end
let a = decide_full(DISPERSED, 2, UBAR), b = decide_full(DISPERSED, 2, UBAR; extra = [])
    check("absent extras ⇒ the same decision", a == b)
end

# ── 2. A row that beats every built-in action WINS, and answers by its own name ───────────
let extra = [("interval_0_1", "report", [0.9, 0.9, -5.0])]
    (action, report_index, eu) = decide_full(DISPERSED, 2, UBAR; extra = extra)
    check("a dominating body row wins the argmax", action == "interval_0_1")
    check("a body row is not a candidate report", report_index === nothing)
    check("its EU is the engine's, over the posterior", eu > 0.0)
end

# ── 3. A row that loses does NOT displace the built-in decision ───────────────────────────
let extra = [("interval_0_1", "report", [-9.0, -9.0, -9.0])]
    (with, _, _) = decide_full(DISPERSED, 2, UBAR; extra = extra)
    (without, _, _) = decide_full(DISPERSED, 2, UBAR)
    check("a dominated body row changes nothing", with == without)
end

# ── 4. The engine prices the row it was GIVEN — no arithmetic of its own ──────────────────
let vals = [0.9, 0.1, -5.0], extra = [("row", "report", vals)]
    (order, fpa) = decision_fpa(2, UBAR; extra = extra)
    check("the body's row is placed verbatim", fpa[order[end]].values == vals)
end

# ── 5. A malformed row fails loud, never silently ─────────────────────────────────────────
let bad = [("row", "report", [0.9, 0.1])]        # 2 values for a 3-atom space
    threw = false
    try; decision_fpa(2, UBAR; extra = bad); catch; threw = true; end
    check("a row that does not span the atoms is an error", threw)
end

# ── 6. Eligibility is keyed on the SPEECH ACT, not the wire name ──────────────────────────
let extra = [("interval_0_1", "report", [0.9, 0.9, -5.0])]
    check("terminal_class maps a body row to its declared act",
          terminal_class("interval_0_1", extra) == "report")
    check("terminal_class passes a built-in action through",
          terminal_class("abstain", extra) == "abstain")
    # the owner-scoped attribution guard defends an OUT-OF-MODEL risk on any commit
    (eff, _, probe, _, _) = gather_decide(DISPERSED, 2, UBAR; owner_scoped = true,
                                          extra = extra)
    check("an owner-scoped body-row commit still corroborates first",
          eff == "gather" && probe == "corroborate")
end

# ── 7. The row survives the wire ──────────────────────────────────────────────────────────
let req = Dict{String, Any}(
        "candidates" => ["a", "b"], "rho" => 0.7,
        "observations" => [Dict("reports" => 0, "group" => 0, "authority" => 0.9,
                                "subject_factor" => 1.0, "time_factor" => 1.0),
                           Dict("reports" => 1, "group" => 1, "authority" => 0.9,
                                "subject_factor" => 1.0, "time_factor" => 1.0)],
        "u_bar" => UBAR,
        "extra_actions" => [Dict("name" => "interval_0_1", "act" => "report",
                                 "values" => [0.9, 0.9, -5.0])])
    resp = Server.decide_response(req)
    check("the wire returns the winning row's own name", resp["effector"] == "interval_0_1")
    check("a body row carries no report_index", resp["report_index"] === nothing)
    check("a body row carries no candidate value", resp["value"] === nothing)
    # and the same request WITHOUT the block is unchanged
    resp0 = Server.decide_response(Dict(k => v for (k, v) in req if k != "extra_actions"))
    check("absent extra_actions ⇒ the pre-r30b reply", resp0["effector"] == "abstain")
end

println()
println("-"^64)
println("extra_actions: $(length(PASSED)) checks passed")
