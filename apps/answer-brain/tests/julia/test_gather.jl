#!/usr/bin/env julia
# Role: tests
"""
    test_gather.jl — the daemon's gather branch (Move 4: the feature-policy).

The forward gather steer as an operator-set feature-policy (move-4-design §2C, §5 Q2). v0's
class-valid probe is `recency` on an `era_split` — a **cheap re-weighting** probe, so it is applied
BEFORE the terminal report, not gated behind it: a count-led STALE value can sit ABOVE the EU bar,
and a confident report must rule out that staleness first (`gather.py` applies recency pre-decision;
the daemon ports that). `applied_probes` (body-held, resent) makes it fire at most once →
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

# The owner's Ū with a harsh wrong-answer cost.
const UBAR = Dict("u_correct" => 1.0, "u_wrong" => -4.0, "u_hedged" => -0.5,
                  "u_abstain" => 0.0, "lambda_int" => 1.0)

println("="^64)
println("answer-brain Move 4 — the gather feature-policy")
println("="^64)

# ── 0. Preconditions: a sharp posterior reports; a dispersed one withholds (no era_split) ─
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

# ── 2. era_split forces a recency-check BEFORE the report (even a sharp/confident posterior) ─
let (eff, _, probe, _, _) = gather_decide(sharp, 2, UBAR; era_split = true)
    check("era_split ⇒ recency-check precedes the report (even when sharp)",
          eff == "gather" && probe == "recency"; detail = "got eff=$eff probe=$probe")
end
let (eff, idx, _, _, _) = gather_decide(sharp, 2, UBAR; era_split = true, applied_probes = ["recency"])
    check("recency checked ⇒ the sharp report stands", eff == "report" && idx == 0;
          detail = "got eff=$eff idx=$idx")
end
let (eff, idx, _, _, _) = gather_decide(sharp, 2, UBAR; era_split = false)
    check("no era_split ⇒ report directly (no confound to rule out)", eff == "report" && idx == 0;
          detail = "got eff=$eff idx=$idx")
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

# ── 6. confident-stale safety: a STALE value above the bar is recency-checked, not reported ─
# candidate 0 = current (2 obs); candidate 1 = stale (3 obs — out-documents the current, so it
# confidently leads ABOVE the bar at baseline). The exact trap the below-bar gate would have sprung.
let stale_obs = Obs[Obs(0, 0, 0.9, 1.0, 1.0),  Obs(0, 1, 0.9, 1.0, 1.0),
                    Obs(1, 2, 0.9, 1.0, 1.0),  Obs(1, 3, 0.9, 1.0, 1.0), Obs(1, 4, 0.9, 1.0, 1.0)]
    stale_led = candidate_posterior(2, stale_obs, 0.7)
    let (a, idx, _) = decide_full(stale_led, 2, UBAR)
        check("precondition: stale value leads CONFIDENTLY (reports the stale at baseline)",
              a == "report" && idx == 1;
              detail = "got act=$a idx=$idx — make the stale leader more corroborated")
    end
    let (eff, _, probe, tgt, _) = gather_decide(stale_led, 2, UBAR; era_split = true)
        check("confident-stale + era_split ⇒ gather(recency), NOT a confident-wrong report",
              eff == "gather" && probe == "recency" && tgt == 1;
              detail = "got eff=$eff probe=$probe tgt=$tgt")
    end

    # recency applied: the stale candidate's (old) observations decay (time_factor 0.02);
    # the current candidate's observations keep time_factor 1.0, and now lead.
    recency_obs = Obs[Obs(0, 0, 0.9, 1.0, 1.0),  Obs(0, 1, 0.9, 1.0, 1.0),
                      Obs(1, 2, 0.9, 1.0, 0.02), Obs(1, 3, 0.9, 1.0, 0.02), Obs(1, 4, 0.9, 1.0, 0.02)]
    after = candidate_posterior(2, recency_obs, 0.7)
    let (eff, idx, _, _, _) = gather_decide(after, 2, UBAR; era_split = true,
                                            applied_probes = ["recency"])
        check("after recency ⇒ report the current candidate (loop closed, correct value)",
              eff == "report" && idx == 0; detail = "got eff=$eff idx=$idx")
    end
end

println("\n", "="^64)
println("answer-brain Move 4 gather-policy: $(length(PASSED)) checks PASSED")
println("="^64)
