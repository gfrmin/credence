#!/usr/bin/env julia
"""
    test_observation_log.jl — Step 3 of credence-pi.

Round-trip, schema rejection, malformed-line tolerance, and the
end-to-end replay assertion: a fresh BDSL environment that ingests a
log produced by an earlier session must recover a posterior whose
expectations match the in-memory posterior of the producing session.

State-equivalence is asserted through `expect`, not struct-field
access. The "no struct internals in tests" rule is constitutional
(Invariant 3, single-responsibility representations); it applies here
even though the comparison happens between two BetaMeasures whose
.alpha/.beta we could trivially read.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "src"))
using Credence
using Credence.Previsions: Identity

include(joinpath(@__DIR__, "..", "..", "daemon", "observation_log.jl"))
using .ObservationLog

const BDSL_DIR = joinpath(@__DIR__, "..", "..", "bdsl")

function load_pass1_env()
    env = Credence.Eval.default_env()
    env[:__toplevel__] = true
    stdlib_path = joinpath(@__DIR__, "..", "..", "..", "..", "src", "stdlib.bdsl")
    for expr in Credence.Parse.parse_all(read(stdlib_path, String))
        Credence.Eval.eval_dsl(expr, env)
    end
    for fname in ("capabilities.bdsl", "features.bdsl",
                  "prior.bdsl", "kernel.bdsl", "decide.bdsl")
        for expr in Credence.Parse.parse_all(read(joinpath(BDSL_DIR, fname), String))
            Credence.Eval.eval_dsl(expr, env)
        end
    end
    env
end

const PASSED = String[]
function ok(name)
    push!(PASSED, name)
    println("PASSED: ", name)
end

const TOL = 1e-12

# ── 1. Round trip: append → read ────────────────────────────────────────

let path = tempname() * ".jsonl"
    e1 = Dict("event_type" => "tool-proposed", "event_id" => "evt_1")
    e2 = Dict("event_type" => "user-responded", "in_response_to" => "evt_1",
              "response" => "yes")
    e3 = Dict("event_type" => "tool-completed", "event_id" => "evt_3",
              "in_response_to" => "evt_1")

    r1 = append_event!(path, e1)
    r2 = append_event!(path, e2)
    r3 = append_event!(path, e3)

    @assert r1.schema == "credence-pi/v1"
    @assert r1.event["event_type"] == "tool-proposed"
    @assert r2.event["response"] == "yes"

    records = read_log(path)
    @assert length(records) == 3
    @assert all(r -> r.schema == "credence-pi/v1", records)
    @assert [r.event["event_type"] for r in records] ==
            ["tool-proposed", "user-responded", "tool-completed"]
    ok("append_event! → read_log round-trip preserves order, schema, and event payload")

    rm(path)
end

# ── 2. Schema enforcement ───────────────────────────────────────────────
#
# The log file may contain lines from a future schema, hand-written
# debugging entries, or partial writes from a crashed process. These
# must not crash the daemon at startup.

let path = tempname() * ".jsonl"
    # One good line via the public API.
    append_event!(path, Dict("event_type" => "user-responded",
                             "response" => "yes"))

    # Hand-write three bad lines: missing schema, wrong schema, malformed JSON.
    open(path, "a") do io
        write(io, """{"received_at":"x","event":{"event_type":"user-responded","response":"yes"}}\n""")
        write(io, """{"schema":"credence-pi/v999","received_at":"x","event":{"event_type":"user-responded","response":"no"}}\n""")
        write(io, "this is not json at all\n")
    end

    # And one more good line so we know reading continued past the bad ones.
    append_event!(path, Dict("event_type" => "user-responded",
                             "response" => "no"))

    records = redirect_stderr(devnull) do
        read_log(path)
    end
    @assert length(records) == 2
    @assert records[1].event["response"] == "yes"
    @assert records[2].event["response"] == "no"
    ok("read_log skips lines with missing schema, wrong schema, and JSON parse errors")

    rm(path)
end

# ── 3. replay_user_responses extracts the right subset ─────────────────

let path = tempname() * ".jsonl"
    append_event!(path, Dict("event_type" => "tool-proposed", "event_id" => "p1"))
    append_event!(path, Dict("event_type" => "user-responded",
                             "in_response_to" => "p1", "response" => "yes"))
    append_event!(path, Dict("event_type" => "tool-completed",
                             "in_response_to" => "p1"))
    append_event!(path, Dict("event_type" => "user-responded",
                             "response" => "no"))
    append_event!(path, Dict("event_type" => "user-responded",
                             "response" => "timeout"))
    append_event!(path, Dict("event_type" => "user-responded",
                             "response" => "yes"))

    obs = replay_user_responses(path)
    @assert obs == [1, 0, 1]
    ok("replay_user_responses ignores non-user-responded and timeout, maps yes→1 / no→0")

    rm(path)
end

# ── 4. End-to-end replay: log-replayed posterior == in-memory posterior ─
#
# State-equivalence asserted through `expect`, not struct access. We
# compare both the first moment (Identity functional, dispatches to the
# closed-form α/(α+β)) and the second moment (a polynomial closure that
# routes through the post-step-2 Gauss-Jacobi quadrature path). Both
# matching to 1e-12 demonstrates the full BetaMeasure state agrees,
# without reading α/β directly.

function replay_to_posterior(env, path)
    posterior = env[Symbol("make-prior")]()
    obs_fn    = env[Symbol("observe-response")]
    for o in replay_user_responses(path)
        posterior = obs_fn(posterior, o)
    end
    posterior
end

let path = tempname() * ".jsonl"
    # Session A: produce a sequence of observations, build the posterior
    # in memory, and append each user-responded event to the log.
    env_a = load_pass1_env()
    obs_fn_a = env_a[Symbol("observe-response")]
    posterior_a = env_a[Symbol("make-prior")]()
    for (response, obs) in (("yes", 1), ("yes", 1), ("no", 0),
                            ("yes", 1), ("no", 0))
        append_event!(path, Dict("event_type" => "user-responded",
                                 "response" => response))
        posterior_a = obs_fn_a(posterior_a, obs)
    end

    # Mix in a tool-proposed and a tool-completed event that the BDSL
    # never conditions on. The replay must ignore them.
    append_event!(path, Dict("event_type" => "tool-proposed", "event_id" => "x"))
    append_event!(path, Dict("event_type" => "tool-completed",
                             "in_response_to" => "x"))

    # Session B: fresh BDSL environment, fresh prior, replay from log.
    env_b = load_pass1_env()
    posterior_b = replay_to_posterior(env_b, path)

    # First moment via the Identity fast path (closed-form on Beta).
    mean_a = expect(posterior_a, Identity())
    mean_b = expect(posterior_b, Identity())
    @assert isapprox(mean_a, mean_b; atol=TOL)
    ok("end-to-end replay: E[θ] of replayed posterior matches in-memory within 1e-12")

    # Second moment via a polynomial closure — exercises the Gauss-
    # Jacobi quadrature path. If the replay had reconstructed a
    # different posterior, the closure-evaluated second moments would
    # diverge well above 1e-12.
    sq_a = expect(posterior_a, θ -> θ^2)
    sq_b = expect(posterior_b, θ -> θ^2)
    @assert isapprox(sq_a, sq_b; atol=TOL)
    ok("end-to-end replay: E[θ²] of replayed posterior matches in-memory within 1e-12")

    rm(path)
end

# ── 5. Replay on missing log file is a no-op (returns prior) ───────────

let path = tempname() * ".jsonl"   # never created
    @assert !isfile(path)
    @assert read_log(path) == ObservationLog.LogRecord[]
    @assert replay_user_responses(path) == Int[]

    env = load_pass1_env()
    prior      = env[Symbol("make-prior")]()
    replayed   = replay_to_posterior(env, path)
    @assert isapprox(expect(prior, Identity()),
                     expect(replayed, Identity());
                     atol=TOL)
    ok("replay against a missing log file is a no-op (returns prior unchanged)")
end

# ── 6. fsync: hard-power-cut survival is the property we want, but we
# can't simulate that in-process. Instead, we assert the OS-level effect
# we rely on — every line written via append_event! is durable on disk
# at the moment append_event! returns, observable by re-reading the
# file from a different process-style file handle.

let path = tempname() * ".jsonl"
    for i in 1:5
        append_event!(path, Dict("event_type" => "user-responded",
                                 "response" => i % 2 == 0 ? "yes" : "no"))
        # Open a fresh handle each iteration; eachline on it must see
        # exactly i lines.
        n = 0
        open(path, "r") do io
            for line in eachline(io)
                isempty(strip(line)) || (n += 1)
            end
        end
        @assert n == i
    end
    ok("each append_event! is durably visible to a fresh reader before returning")

    rm(path)
end

println()
println("=" ^ 60)
println("ALL ", length(PASSED), " ASSERTIONS PASSED")
println("=" ^ 60)
