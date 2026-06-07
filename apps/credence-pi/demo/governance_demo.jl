#!/usr/bin/env julia
# Role: demo (display-only; exercises the real Pass-2 feature brain end-to-end)
"""
    governance_demo.jl — credence-pi governance, end to end, on a synthetic
    session (no live OpenClaw, no real data required).

THE SURGICAL WIN (Pass 2 / Route B). An agent is stuck in a wasteful loop,
re-proposing the same `bash` call (high repetition). Interleaved, it makes a
legitimate NOVEL call (`read`, first time). At cold start credence-pi ASKS
both (the VOI gate). The user DENIES the loop and APPROVES the novel call.
The feature-conditioned brain learns PER CONTEXT: it comes to AUTO-BLOCK the
repeated loop while STILL allowing the novel call — by expected-utility
maximisation, no hard-coded rule. A single global posterior (Pass 1) could not
do this: once it had learned to refuse, it would block everything.

The contrast is the whole point — the same brain, at the same moment, blocks
one context and allows another. Finally the dollars-saved surface reports the
spend governance flagged.

Run:  julia --project=. apps/credence-pi/demo/governance_demo.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "src"))
using Credence
using Credence: Identity

include(joinpath(@__DIR__, "..", "daemon", "server.jl"))
using .Server: init_state, handle_sensor_event, snapshot, drain!
using .Server.FeatureBrain: build_model_from_env, belief_at_context, context_from_features

include(joinpath(@__DIR__, "..", "savings.jl"))
using .Savings: savings_report, format_report

const BDSL_DIR = joinpath(@__DIR__, "..", "bdsl")

feats(; tool, wd="project-root", parent="none", rep, ident="ident-0", since="lt-30s") =
    Dict{String, Any}("tool-name" => tool, "working-directory-relative" => wd,
                      "parent-tool-call-name" => parent, "recent-repetition-count" => rep,
                      "recent-identical-call-count" => ident,
                      "time-since-last-user-message" => since)

# The two contexts: a wasteful loop (the SAME call repeated → ident-3plus), and a
# legitimate novel call (first time → ident-0).
const LOOP  = feats(tool = "bash", parent = "bash", rep = "rep-3plus", ident = "ident-3plus")
const NOVEL = feats(tool = "read", parent = "edit", rep = "rep-0", since = "lt-2m")

# Post one event through the real daemon dispatcher; return the emitted effector.
function step!(state, event)::Union{String, Nothing}
    drain!(state.signal_queue)
    handle_sensor_event(state, event)
    sigs = snapshot(state.signal_queue)
    isempty(sigs) ? nothing : string(sigs[1]["effector"])
end

# P(approve|X) off the live posterior, via the brain's per-context view (display).
function predict(state, model, features)
    expect(belief_at_context(model, state.posterior[], context_from_features(model, features)), Identity())
end

# Propose `features`; if the brain asks, the user answers `response`. Returns the
# brain's decision string.
function turn!(state, eid, features, response)
    eff = step!(state, Dict{String, Any}(
        "event_type" => "tool-proposed", "event_id" => eid, "session_id" => "demo",
        "features" => features,
        "proposed_call" => Dict("tool_name" => features["tool-name"],
                                "input" => Dict("command" => "npm run build"))))
    if eff == "ask"
        step!(state, Dict{String, Any}(
            "event_type" => "user-responded", "event_id" => "r_$eid",
            "in_response_to" => eid, "response" => response))
    end
    # The turn burned tokens regardless — a moderately-priced agent turn.
    step!(state, Dict{String, Any}(
        "event_type" => "turn-cost", "session_id" => "demo",
        "usd" => 0.50, "total_tokens" => 1500, "model" => "claude-opus-4-8"))
    eff
end

function run_demo(io::IO = stdout)
    path = tempname() * ".jsonl"
    state = init_state(; bdsl_dir = BDSL_DIR, log_path = path)
    model = build_model_from_env(state.env)

    println(io, "credence-pi governance demo (Pass 2) — the agent loops on a wasteful")
    println(io, "`bash` call while also making a legitimate novel `read` call. The brain")
    println(io, "learns PER CONTEXT: deny→block the loop, approve→allow the novel call.\n")
    println(io, "round |   context   | P(approve|X) | brain decision | user")
    println(io, "------+-------------+--------------+----------------+------")

    first_block = 0
    for k in 1:5
        # The wasteful loop: user denies when asked.
        d = turn!(state, "loop_$k", LOOP, "no")
        p = predict(state, model, LOOP)
        println(io, lpad(k, 5), " | ", rpad("bash/loop", 11), " |    ",
                rpad(round(p; digits = 3), 7), "  | ", rpad(d === nothing ? "(none)" : d, 14),
                " | ", d == "ask" ? "deny" : "—")
        d == "block" && first_block == 0 && (first_block = k)

        # The legitimate novel call: user approves when asked.
        d2 = turn!(state, "novel_$k", NOVEL, "yes")
        p2 = predict(state, model, NOVEL)
        println(io, lpad(k, 5), " | ", rpad("read/novel", 11), " |    ",
                rpad(round(p2; digits = 3), 7), "  | ", rpad(d2 === nothing ? "(none)" : d2, 14),
                " | ", d2 == "ask" ? "appr" : "—")
    end

    p_loop  = predict(state, model, LOOP)
    p_novel = predict(state, model, NOVEL)
    println(io)
    println(io, ">> THE SURGICAL WIN — same brain, same moment, opposite calls:")
    println(io, "   P(approve | bash/loop)  = ", round(p_loop;  digits = 3),
                "  → decision: ", turn_decision(state, model, LOOP))
    println(io, "   P(approve | read/novel) = ", round(p_novel; digits = 3),
                "  → decision: ", turn_decision(state, model, NOVEL))
    if first_block > 0
        println(io, "   The loop AUTO-BLOCKED from round $first_block (EU-max, no hard-coded rule);")
        println(io, "   the novel call was never blocked. One global posterior cannot do this.\n")
    else
        println(io, "   (the loop did not reach the auto-block threshold in these rounds)\n")
    end

    format_report(io, savings_report(path))
    rm(path; force = true)
    return first_block
end

# A dry-run decision on a context (no logging) for the final contrast line.
function turn_decision(state, model, features)
    decide = state.env[Symbol("decide-action")]
    string(decide(state.posterior[], features, state.last_cost[]))
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
