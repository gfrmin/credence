#!/usr/bin/env julia
# Role: tests
"""
    test_grow.jl — app-side integration checks for the gather VOI (the ruling's "B half").

The pricing (`grow_value`) and the which-gather argmax (`best_grow`) are ENGINE stdlib
(`src/gather_voi.jl`, behaviour pinned by `test/test_gather_voi.jl`); the app imports and
re-exports them. These checks pin only the integration: the app surface IS the engine
function (no shadowing fork), and one behavioural smoke each through the app name.

Run from the credence repo root:
    julia --project=. apps/answer-brain/tests/julia/test_grow.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "src"))
using Credence

include(joinpath(@__DIR__, "..", "..", "brain", "answer_brain.jl"))
using .AnswerBrain

const PASSED = String[]
function check(name::AbstractString, cond::Bool; detail::AbstractString = "")
    if cond
        push!(PASSED, name); println("PASSED: ", name)
    else
        println("FAILED: ", name, " — ", detail); error("assertion failed: $name")
    end
end
approx(a, b; atol = 1e-9) = abs(a - b) <= atol

println("="^64)
println("answer-brain grow — engine gather-VOI integration")
println("="^64)

# The app surface is the engine function, not a fork.
check("AnswerBrain.grow_value === Credence.grow_value",
      AnswerBrain.grow_value === Credence.grow_value)
check("AnswerBrain.best_grow === Credence.best_grow",
      AnswerBrain.best_grow === Credence.best_grow)

# One behavioural smoke each through the app name (self-gating + which-gather).
let u_c = 1.0, cost = 0.1
    check("grow_value self-gates at a confident report (≈ −cost)",
          approx(AnswerBrain.grow_value(0.9, u_c, u_c, cost), -cost))
end
let u_c = 1.0, eu = -0.2
    best, bv = AnswerBrain.best_grow([("re-extract", 0.7, 0.1), ("retrieve-wider", 0.4, 0.1)], u_c, eu)
    check("best_grow discriminates which-gather through the app surface",
          best == "re-extract" && approx(bv, AnswerBrain.grow_value(0.7, u_c, eu, 0.1)))
end

# ── Slice 4: the scheduler's third stanza — grow actuators on the priced menu ───────────
# `schedule_decide`/`gather_decide` gain `grows` (per-actuator `(probe, g, cost)`, g already
# read from the gather structure-BMA): guards fire first; then the grow argmax competes with
# the :voi argmax in one EU comparison; `applied_probes` retires grow probes (termination).
# Parity: `grows` empty ⇒ byte-identical to today.
const UBAR = Dict("u_correct" => 1.0, "u_wrong" => -4.0, "u_hedged" => -0.5,
                  "u_abstain" => 0.0, "lambda_int" => 1.0)

# A dispersed (withholding) and a heavily-corroborated (confident-report) posterior.
dispersed = candidate_posterior(2, Obs[Obs(0, 0, 0.9, 1.0, 1.0), Obs(1, 1, 0.9, 1.0, 1.0)], 0.7)
confident = candidate_posterior(2, Obs[Obs(0, i, 0.95, 1.0, 1.0) for i in 0:4], 0.7)

let (a_d, _, eu_d) = decide_full(dispersed, 2, UBAR),
    (a_c, _, eu_c) = decide_full(confident, 2, UBAR)
    check("precondition: dispersed withholds, confident reports above eu 0.6",
          a_d != "report" && a_c == "report" && eu_c > 0.6 && eu_d < 0.4;
          detail = "got a_d=$a_d a_c=$a_c eu_c=$eu_c eu_d=$eu_d")

    grows = [("re-extract", 0.5, 0.2)]

    # Parity: no grows ⇒ the terminal tuple exactly (dispersed, no registry flags).
    let (eff, idx, probe, tgt, eu) = gather_decide(dispersed, 2, UBAR)
        (eff2, idx2, probe2, tgt2, eu2) = gather_decide(dispersed, 2, UBAR; grows = Tuple{String,Float64,Float64}[])
        check("grows=[] is parity-exact with today's call",
              (eff, idx, probe, tgt, eu) == (eff2, idx2, probe2, tgt2, eu2))
    end

    # A withhold with a clearing grow ⇒ gather(probe), targeting the provisional leader.
    let (eff, idx, probe, tgt, _) = gather_decide(dispersed, 2, UBAR; grows = grows)
        check("withhold + clearing grow ⇒ gather(re-extract) at the leader",
              eff == "gather" && probe == "re-extract" && idx === nothing &&
              tgt == provisional_leader(dispersed, 2, UBAR);
              detail = "got eff=$eff probe=$probe tgt=$tgt")
    end

    # Self-gating: the same grow at a confident report prices negative ⇒ the report stands.
    let (eff, idx, _, _, _) = gather_decide(confident, 2, UBAR; grows = grows)
        check("confident report + same grow ⇒ terminal stands (self-gating, no p_none branch)",
              eff == "report" && idx == 0; detail = "got eff=$eff idx=$idx")
    end

    # Which-gather: the higher-value actuator's probe is returned.
    let two = [("re-extract", 0.3, 0.05), ("retrieve-wider", 0.7, 0.05)]
        (eff, _, probe, _, _) = gather_decide(dispersed, 2, UBAR; grows = two)
        check("which-gather argmax through the scheduler", eff == "gather" && probe == "retrieve-wider";
              detail = "got eff=$eff probe=$probe")
    end

    # applied_probes retires a grow probe (termination); the remaining one still prices.
    let two = [("re-extract", 0.7, 0.05), ("retrieve-wider", 0.5, 0.05)]
        (eff, _, probe, _, _) = gather_decide(dispersed, 2, UBAR; grows = two,
                                              applied_probes = ["re-extract"])
        check("an applied grow probe is retired; the next prices",
              eff == "gather" && probe == "retrieve-wider"; detail = "got eff=$eff probe=$probe")
        (eff2, _, _, _, _) = gather_decide(dispersed, 2, UBAR; grows = two,
                                           applied_probes = ["re-extract", "retrieve-wider"])
        check("all grow probes applied ⇒ terminal stands (terminates)", eff2 != "gather";
              detail = "got $eff2")
    end

    # Guards keep precedence: era_split fires recency BEFORE any grow.
    let (eff, _, probe, _, _) = gather_decide(dispersed, 2, UBAR; era_split = true, grows = grows)
        check("guards precede grow (recency first on era_split)",
              eff == "gather" && probe == "recency"; detail = "got eff=$eff probe=$probe")
    end
end

println("-"^64)
println("grow integration: ", length(PASSED), " checks passed")
