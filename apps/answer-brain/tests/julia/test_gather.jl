#!/usr/bin/env julia
# Role: tests
"""
    test_gather.jl — the daemon's gather branch (Move 4: the feature-policy).

The forward gather steer as an operator-set feature-policy (move-4-design §2C, §5 Q2): when the
terminal decision is NOT a confident report and a class-valid discriminating probe is
available-and-unapplied, the brain emits `gather(probe, target)` instead of withholding; else the
terminal decision stands. v0's class-valid probe is `recency` on an `era_split` (the validated
lever, `master-plan.md` §"probe library"). `applied_probes` (body-held, resent) guarantees
termination. `provisional_leader` names the candidate the decision mechanism would report — via
`optimise` over the report actions, NOT `argmax(weights)` (Invariant 1). A correctness check, not a
parity check; the scenarios are synthetic (no PII).

Run from the credence repo root:
    julia --project=. apps/answer-brain/tests/julia/test_gather.jl
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

# The owner's Ū with a harsh wrong-answer cost — reporting a below-bar leader is EU-negative,
# so a led-but-dispersed posterior withholds and the gather policy can fire.
const UBAR = Dict("u_correct" => 1.0, "u_wrong" => -4.0, "u_hedged" => -0.5,
                  "u_abstain" => 0.0, "lambda_int" => 1.0)

println("="^64)
println("answer-brain Move 4 — the gather feature-policy")
println("="^64)

# ── 0. Preconditions: a sharp posterior reports; a dispersed one withholds ───────────────
sharp     = candidate_posterior(2, Obs[Obs(0, 0, 0.95, 1.0, 1.0), Obs(0, 1, 0.95, 1.0, 1.0)], 0.7)
dispersed = candidate_posterior(2, Obs[Obs(0, 0, 0.9, 1.0, 1.0),  Obs(1, 1, 0.9, 1.0, 1.0)],  0.7)
let (a, _, _) = decide_full(sharp, 2, UBAR)
    check("precondition: sharp posterior → report", a == "report"; detail = "got $a")
end
let (a, _, _) = decide_full(dispersed, 2, UBAR)
    check("precondition: dispersed posterior → withhold (not report)", a != "report"; detail = "got $a")
end

# ── 1. provisional_leader = the EU-leader (optimise over reports, not argmax weights) ────
check("provisional_leader picks the dominant candidate",
      provisional_leader(sharp, 2, UBAR) == 0;
      detail = "got $(provisional_leader(sharp, 2, UBAR))")

# ── 2. a confident report is never deflected to gather (even with era_split) ──────────────
let (eff, idx, probe, _, _) = gather_decide(sharp, 2, UBAR; era_split = true)
    check("report is not deflected to gather",
          eff == "report" && idx == 0 && probe === nothing;
          detail = "got eff=$eff idx=$idx probe=$probe")
end

# ── 3. below-bar + era-split + recency unapplied ⇒ gather(recency, leader) ────────────────
let (eff, idx, probe, tgt, _) = gather_decide(dispersed, 2, UBAR; era_split = true)
    check("below-bar era-split ⇒ gather(recency)", eff == "gather" && probe == "recency";
          detail = "got eff=$eff probe=$probe")
    check("gather names the provisional leader as target",
          tgt == provisional_leader(dispersed, 2, UBAR); detail = "got tgt=$tgt")
    check("a gather carries no report index", idx === nothing)
end

# ── 4. no era-split ⇒ no gather (the discriminating probe's precondition is absent) ───────
let (eff, _, _, _, _) = gather_decide(dispersed, 2, UBAR; era_split = false)
    check("no era-split ⇒ terminal stands (no gather)", eff != "gather"; detail = "got $eff")
end

# ── 5. recency already applied ⇒ no re-gather (termination guard) ─────────────────────────
let (eff, _, _, _, _) = gather_decide(dispersed, 2, UBAR; era_split = true,
                                      applied_probes = ["recency"])
    check("recency applied ⇒ terminal stands (terminates)", eff != "gather"; detail = "got $eff")
end

# ── 6. the loop closes: stale-led ⇒ gather(recency); recency-decayed ⇒ report(current) ───
# candidate 0 = current (2 obs); candidate 1 = stale (2 obs, slightly higher source authority,
# so it leads modestly — below the report bar). The mobile-number class in miniature.
let stale_obs = Obs[Obs(0, 0, 0.9, 1.0, 1.0),  Obs(0, 1, 0.9, 1.0, 1.0),
                    Obs(1, 2, 0.95, 1.0, 1.0), Obs(1, 3, 0.95, 1.0, 1.0)]
    stale_led = candidate_posterior(2, stale_obs, 0.7)
    let (a, _, _) = decide_full(stale_led, 2, UBAR)
        check("precondition: stale-led posterior is below bar (withholds)", a != "report";
              detail = "got $a — stale leader cleared the bar; make it less corroborated")
    end
    let (eff, _, probe, tgt, _) = gather_decide(stale_led, 2, UBAR; era_split = true)
        check("stale-led, below bar ⇒ gather(recency) on the stale leader",
              eff == "gather" && probe == "recency" && tgt == 1;
              detail = "got eff=$eff probe=$probe tgt=$tgt")
    end

    # recency applied: the stale candidate's (old) observations decay (time_factor 0.02);
    # the current candidate's observations keep time_factor 1.0, and now clear the bar.
    recency_obs = Obs[Obs(0, 0, 0.9, 1.0, 1.0),  Obs(0, 1, 0.9, 1.0, 1.0),
                      Obs(1, 2, 0.95, 1.0, 0.02), Obs(1, 3, 0.95, 1.0, 0.02)]
    after = candidate_posterior(2, recency_obs, 0.7)
    let (eff, idx, _, _, _) = gather_decide(after, 2, UBAR; era_split = true,
                                            applied_probes = ["recency"])
        check("after recency ⇒ report the current candidate (loop closed)",
              eff == "report" && idx == 0; detail = "got eff=$eff idx=$idx")
    end
end

println("\n", "="^64)
println("answer-brain Move 4 gather-policy: $(length(PASSED)) checks PASSED")
println("="^64)
