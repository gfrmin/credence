#!/usr/bin/env julia
# Role: tests
"""
    test_answer_brain.jl — Stage-1 parity + the net_voi forward capability.

Parity: on each fixture case the native brain reproduces the Stage-0 lookup answerer's
posterior weights, chosen effector, and EU (move-1-design §3). The reference is
`apps/answer-brain/tests/fixtures/stage0_parity.json` (life-agent `c1a781f`). Both sides call
the same Credence `condition`/`optimise`, so a mismatch is a porting bug in the density /
utility construction — exactly what these cases pin.

net_voi: a correctness (not parity) check of the forward gather/ask gate — a perfect probe
prices above a useless one; cost subtracts (move-1-design §5 Open-Q3).

Run from the credence repo root:
    julia --project=. apps/answer-brain/tests/julia/test_answer_brain.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "src"))
using Credence
using Credence: weights, Kernel, Finite, PushOnly
using JSON3

include(joinpath(@__DIR__, "..", "..", "brain", "answer_brain.jl"))
using .AnswerBrain
include(joinpath(@__DIR__, "..", "..", "daemon", "observation_log.jl"))
using .ObservationLog

const FIXTURE = joinpath(@__DIR__, "..", "fixtures", "stage0_parity.json")
const ATOL = 1e-9

const PASSED = String[]
function check(name::AbstractString, cond::Bool; detail::AbstractString = "")
    if cond
        push!(PASSED, name)
        println("PASSED: ", name)
    else
        println("FAILED: ", name, " — ", detail)
        error("assertion failed: $name")
    end
end
approx(a, b; atol = ATOL) = abs(a - b) <= atol

println("="^64)
println("answer-brain Stage-1 — parity vs Stage 0 + net_voi")
println("="^64)

data = JSON3.read(read(FIXTURE, String))

# ── 0. Channel-param drift guard: the fixture's params == the brain's constants ──────────
let ch = data.channel_params
    cp = ChannelParams(Float64(ch.A_alternatives), Float64(ch.beta_ancestry),
                       Float64(ch.beta_model), Float64(ch.p_none_prior),
                       Float64(ch.oracle_p), Float64(ch.prob_eps))
    check("channel params match brain CANONICAL_CHANNEL",
          cp.a_alternatives == CANONICAL_CHANNEL.a_alternatives &&
          cp.beta_ancestry  == CANONICAL_CHANNEL.beta_ancestry &&
          cp.beta_model     == CANONICAL_CHANNEL.beta_model &&
          cp.p_none_prior   == CANONICAL_CHANNEL.p_none_prior &&
          cp.oracle_p       == CANONICAL_CHANNEL.oracle_p &&
          cp.prob_eps       == CANONICAL_CHANNEL.prob_eps;
          detail = "fixture $(cp) vs brain $(CANONICAL_CHANNEL)")
end

# ── 1. Per-case parity: posterior weights, chosen effector, EU ──────────────────────────
for case in data.cases
    name = String(case.name)
    k = Int(case.k)
    rho = Float64(case.rho)
    ubar = Dict{String, Float64}(String(kk) => Float64(vv) for (kk, vv) in case.u_bar)
    obs = Obs[Obs(Int(o.reports), Int(o.group), Float64(o.authority),
                  Float64(o.subject_factor), Float64(o.time_factor))
              for o in case.observations]

    post = candidate_posterior(k, obs, rho)
    w = weights(post)
    exp_w = [Float64(x) for x in case.expected.weights]
    check("[$name] weight-vector length", length(w) == length(exp_w);
          detail = "got $(length(w)) want $(length(exp_w))")
    check("[$name] posterior weights match Stage-0",
          all(approx(w[i], exp_w[i]) for i in eachindex(w));
          detail = "got $w want $exp_w")
    check("[$name] p_none matches", approx(w[end], Float64(case.expected.p_none)))

    act, eu = terminal_decide(post, k, ubar)
    check("[$name] chosen effector matches Stage-0", act == String(case.expected.action);
          detail = "got $act want $(case.expected.action)")
    check("[$name] EU matches Stage-0", approx(eu, Float64(case.expected.eu));
          detail = "got $eu want $(case.expected.eu)")
end

# ── 2. net_voi forward gate (correctness, not parity) ───────────────────────────────────
let k = 2,
    ubar = Dict("u_correct" => 1.0, "u_wrong" => -4.0, "u_hedged" => -0.5,
                "u_abstain" => 0.0, "lambda_int" => 1.0),
    obs = Obs[Obs(0, 0, 0.9, 1.0, 1.0), Obs(1, 1, 0.9, 1.0, 1.0)]   # dispersed two-way

    post = candidate_posterior(k, obs, 0.7)
    atoms = collect(Float64, 0:k)
    reveal = Finite(collect(Float64, 0:(k - 1)))
    # perfect probe: the observation reveals which candidate is true
    perfect = Kernel(Finite(atoms), reveal, _ -> error("gen"),
                     (h, o) -> Int(round(h)) == Int(round(o)) ? 0.0 : -Inf;
                     likelihood_family = PushOnly())
    # useless probe: flat — every observation equally likely under every hypothesis
    useless = Kernel(Finite(atoms), reveal, _ -> error("gen"),
                     (_h, _o) -> 0.0; likelihood_family = PushOnly())
    possible = [0.0, 1.0]

    vp = voi_gather(post, k, ubar, perfect, possible, 0.0)
    vu = voi_gather(post, k, ubar, useless, possible, 0.0)
    check("net_voi: perfect probe is valued above a useless one", vp > vu + 1e-6;
          detail = "perfect=$vp useless=$vu")
    check("net_voi: a useless probe carries ~zero gross VOI", approx(vu, 0.0);
          detail = "got $vu")
    check("net_voi: cost subtracts pointwise",
          approx(voi_gather(post, k, ubar, useless, possible, 0.05), -0.05))
    check("net_voi: a perfect probe earns positive net value at small cost",
          voi_gather(post, k, ubar, perfect, possible, 0.01) > 0.0)
end

# ── 2b. owner_scoped corroborate guard (correctness; additive ⇒ parity-preserving) ──────
let k = 2,
    ubar = Dict("u_correct" => 1.0, "u_wrong" => -4.0, "u_hedged" => -0.5,
                "u_abstain" => 0.0, "lambda_int" => 1.0),
    # two independent groups both report candidate 0 ⇒ a confident in-set leader that REPORTS
    obs = Obs[Obs(0, 0, 1.0, 1.0, 1.0), Obs(0, 1, 1.0, 1.0, 1.0)]

    post = candidate_posterior(k, obs, 0.95)
    a0, _, _, _, _ = gather_decide(post, k, ubar)
    check("gather_decide: a confident leader reports when not owner-scoped", a0 == "report";
          detail = "got $a0")
    # owner_scoped=false is byte-identical to the plain terminal decision (the parity guarantee)
    check("gather_decide: owner_scoped=false == the unchanged terminal decision",
          a0 == terminal_decide(post, k, ubar)[1])

    eff, _, probe, target, _ = gather_decide(post, k, ubar; owner_scoped = true)
    check("gather_decide: an owner-scoped report corroborates first (attribution guard)",
          eff == "gather" && probe == "corroborate"; detail = "got $eff/$probe")
    check("gather_decide: the corroborate target is the provisional leader", target == 0;
          detail = "got $target")

    # once corroborated (the body re-extracted + re-posted), the terminal report stands ⇒ the loop
    # terminates; a disagreeing re-read would instead have moved mass to NONE and withheld.
    a2, _, _, _, _ = gather_decide(post, k, ubar; owner_scoped = true,
                                   applied_probes = ["corroborate"])
    check("gather_decide: after corroborate the owner-scoped report stands (termination)",
          a2 == "report"; detail = "got $a2")
end

# ── 3. Observation-log replay reconstructs the posterior exactly ────────────────────────
let case = first(data.cases),
    k = Int(case.k), rho = Float64(case.rho)

    obs = Obs[Obs(Int(o.reports), Int(o.group), Float64(o.authority),
                  Float64(o.subject_factor), Float64(o.time_factor))
              for o in case.observations]
    direct = weights(candidate_posterior(k, obs, rho))

    path = tempname() * ".jsonl"
    for o in obs
        ObservationLog.append_observation!(path; question_id = "q", reports = o.reports,
            group = o.group, authority = o.authority,
            subject_factor = o.subject_factor, time_factor = o.time_factor)
    end
    replayed = [Obs(r.reports, r.group, r.authority, r.subject_factor, r.time_factor)
                for r in ObservationLog.replay_observations(path)]
    rm(path; force = true)

    check("replay: same observation count", length(replayed) == length(obs);
          detail = "got $(length(replayed)) want $(length(obs))")
    rebuilt = weights(candidate_posterior(k, replayed, rho))
    check("replay reconstructs the posterior exactly",
          length(rebuilt) == length(direct) &&
          all(approx(direct[i], rebuilt[i]) for i in eachindex(direct));
          detail = "got $rebuilt want $direct")
end

println("\n", "="^64)
println("answer-brain Stage-1: $(length(PASSED)) checks PASSED · $(length(data.cases)) parity cases")
println("="^64)
