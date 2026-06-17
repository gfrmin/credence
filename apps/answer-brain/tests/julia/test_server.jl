#!/usr/bin/env julia
# Role: tests
"""
    test_server.jl — Stage-2a wire parity: the daemon's `/decide` ≡ the Stage-1 brain.

The decision must survive the wire. On each Stage-0 parity fixture, `Server.decide_response` (the
request→response handler) must reproduce the chosen effector + EU + posterior the native brain does —
same oracle as `test_answer_brain.jl` (`tests/fixtures/stage0_parity.json`, life-agent `c1a781f`),
now exercised through the JSON request shape. Also checks: the report index `optimise` returned agrees
with the weight leader (Invariant-1 consistency), the handler is stateless (no cross-question bleed),
and a live HTTP round-trip serves `/decide` + `/ready`.

Run from the credence repo root:
    julia --project=. apps/answer-brain/tests/julia/test_server.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "src"))
using Credence
using JSON3
using HTTP

include(joinpath(@__DIR__, "..", "..", "brain", "answer_brain.jl"))
using .AnswerBrain
include(joinpath(@__DIR__, "..", "..", "daemon", "server.jl"))
using .Server

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
println("answer-brain Stage-2a — daemon /decide wire parity")
println("="^64)

data = JSON3.read(read(FIXTURE, String))

# Build the wire request for a fixture case. The fixtures carry abstract observations + `k` but no
# candidate display strings (the brain is string-blind), so synthesise k labels — parity is over
# effector/EU/weights, which don't depend on the labels.
function request_for(case)
    k = Int(case.k)
    Dict{String, Any}(
        "question_id" => "wire-$(String(case.name))",
        "candidates"  => ["c$(j)" for j in 0:(k - 1)],
        "rho"         => Float64(case.rho),
        "u_bar"       => Dict{String, Any}(String(kk) => Float64(vv) for (kk, vv) in case.u_bar),
        "observations" => [Dict{String, Any}(
            "reports" => Int(o.reports), "group" => Int(o.group),
            "authority" => Float64(o.authority),
            "subject_factor" => Float64(o.subject_factor),
            "time_factor" => Float64(o.time_factor)) for o in case.observations],
    )
end

# A synthetic below-bar two-candidate request (dispersed, harsh u_wrong) — the Move-4 gather inputs
# ride on the same wire. `current`/`stale` are display labels only; era_split drives the gather branch.
function gather_request(; era_split::Bool, applied::Vector{String} = String[])
    Dict{String, Any}(
        "question_id"    => "wire-gather",
        "candidates"     => ["current", "stale"],
        "rho"            => 0.7,
        "u_bar"          => Dict{String, Any}("u_correct" => 1.0, "u_wrong" => -4.0,
                            "u_hedged" => -0.5, "u_abstain" => 0.0, "lambda_int" => 1.0),
        "observations"   => [
            Dict{String, Any}("reports" => 0, "group" => 0, "authority" => 0.9,
                              "subject_factor" => 1.0, "time_factor" => 1.0),
            Dict{String, Any}("reports" => 1, "group" => 1, "authority" => 0.9,
                              "subject_factor" => 1.0, "time_factor" => 1.0)],
        "era_split"      => era_split,
        "applied_probes" => applied,
    )
end

# ── 0. Drift guard: the handler's default channel == the fixture's params (move-1 §0) ────
let ch = data.channel_params
    cp = CANONICAL_CHANNEL
    check("default channel == fixture channel_params",
          cp.a_alternatives == Float64(ch.A_alternatives) &&
          cp.beta_ancestry  == Float64(ch.beta_ancestry) &&
          cp.beta_model     == Float64(ch.beta_model) &&
          cp.p_none_prior   == Float64(ch.p_none_prior) &&
          cp.oracle_p       == Float64(ch.oracle_p) &&
          cp.prob_eps       == Float64(ch.prob_eps);
          detail = "brain $(cp) vs fixture $(ch)")
end

# ── 1. Per-case wire parity: effector, EU, posterior ─────────────────────────────────────
for case in data.cases
    name = String(case.name)
    k = Int(case.k)
    resp = Server.decide_response(request_for(case))

    check("[$name] wire effector == Stage-0", resp["effector"] == String(case.expected.action);
          detail = "got $(resp["effector"]) want $(case.expected.action)")
    check("[$name] wire EU == Stage-0", approx(Float64(resp["eu"]), Float64(case.expected.eu));
          detail = "got $(resp["eu"]) want $(case.expected.eu)")

    exp_w = [Float64(x) for x in case.expected.weights]          # candidates then NONE
    got_w = vcat([Float64(x) for x in resp["credences"]], [Float64(resp["p_none"])])
    check("[$name] wire posterior == Stage-0 (credences ⧺ p_none)",
          length(got_w) == length(exp_w) &&
          all(approx(got_w[i], exp_w[i]) for i in eachindex(exp_w));
          detail = "got $got_w want $exp_w")

    # report_index: present iff a report, and it is the weight leader (optimise ≡ argmax wₖ for the
    # symmetric report utility — the property that lets the body display the leader without deciding).
    if resp["effector"] == "report"
        idx = resp["report_index"]
        leader = argmax([Float64(x) for x in resp["credences"]]) - 1   # 0-based
        check("[$name] report_index is the weight leader", idx == leader;
              detail = "report_index=$idx leader=$leader")
        check("[$name] value == candidates[report_index]",
              resp["value"] == request_for(case)["candidates"][idx + 1])
    else
        check("[$name] non-report ⇒ report_index/value are null",
              resp["report_index"] === nothing && resp["value"] === nothing)
    end
end

# ── 2. Stateless: deciding another question between two decides of the same one changes nothing ──
let a = first(data.cases), b = last(data.cases)
    r1 = Server.decide_response(request_for(a))
    _ = Server.decide_response(request_for(b))            # interleave a different question
    r2 = Server.decide_response(request_for(a))
    check("stateless: same request decides identically across an interleaved one",
          r1["effector"] == r2["effector"] && approx(Float64(r1["eu"]), Float64(r2["eu"])) &&
          all(approx(Float64(r1["credences"][i]), Float64(r2["credences"][i]))
              for i in eachindex(r1["credences"]));
          detail = "r1=$(r1["effector"])/$(r1["eu"]) r2=$(r2["effector"])/$(r2["eu"])")
end

# ── 3. Live HTTP round-trip: the plumbing serves /decide + /ready ────────────────────────
let port = 8791
    server = start_daemon(; port = port, host = "127.0.0.1")
    try
        ready = HTTP.get("http://127.0.0.1:$port/ready"; retry = false, readtimeout = 5)
        check("HTTP /ready returns 200 ok",
              ready.status == 200 && strip(String(ready.body)) == "ok")

        case = first(data.cases)
        body = JSON3.write(request_for(case))
        http = HTTP.post("http://127.0.0.1:$port/decide", ["Content-Type" => "application/json"],
                         body; retry = false, readtimeout = 5)
        resp = JSON3.read(String(http.body))
        check("HTTP /decide returns 200 + Stage-0 effector",
              http.status == 200 && String(resp.effector) == String(case.expected.action);
              detail = "status=$(http.status) effector=$(get(resp, :effector, "∅"))")
        check("HTTP /decide EU matches", approx(Float64(resp.eu), Float64(case.expected.eu)))

        bad = HTTP.post("http://127.0.0.1:$port/decide", ["Content-Type" => "application/json"],
                        "{not json"; status_exception = false, retry = false, readtimeout = 5)
        check("HTTP /decide rejects malformed JSON with 400", bad.status == 400;
              detail = "status=$(bad.status)")

        # Move 4: a gather request over live HTTP returns the forward steer.
        ghttp = HTTP.post("http://127.0.0.1:$port/decide", ["Content-Type" => "application/json"],
                          JSON3.write(gather_request(; era_split = true)); retry = false, readtimeout = 5)
        gresp = JSON3.read(String(ghttp.body))
        check("HTTP /decide gather: era_split ⇒ gather(recency)",
              ghttp.status == 200 && String(gresp.effector) == "gather" &&
              String(gresp.probe) == "recency";
              detail = "status=$(ghttp.status) effector=$(get(gresp, :effector, "∅"))")
    finally
        stop_daemon(server)
    end
end

# ── 4. The gather branch over the wire (Move 4): the forward steer + backward compatibility ──
let r = Server.decide_response(gather_request(; era_split = true))
    check("wire gather: below-bar + era_split ⇒ gather(recency)",
          r["effector"] == "gather" && r["probe"] == "recency";
          detail = "got effector=$(r["effector"]) probe=$(get(r, "probe", "∅"))")
    check("wire gather: target is a candidate string, report_index null",
          r["target"] in ("current", "stale") && r["report_index"] === nothing;
          detail = "target=$(r["target"]) report_index=$(r["report_index"])")
end
let r = Server.decide_response(gather_request(; era_split = false))
    check("wire gather: era_split absent ⇒ no gather (Stage-2a terminal, additive fields null)",
          r["effector"] != "gather" && r["probe"] === nothing && r["target"] === nothing;
          detail = "got $(r["effector"])")
end
let r = Server.decide_response(gather_request(; era_split = true, applied = ["recency"]))
    check("wire gather: recency applied ⇒ no re-gather (terminates)",
          r["effector"] != "gather"; detail = "got $(r["effector"])")
end

println("\n", "="^64)
println("answer-brain Stage-2a: $(length(PASSED)) checks PASSED · $(length(data.cases)) wire-parity cases")
println("="^64)
