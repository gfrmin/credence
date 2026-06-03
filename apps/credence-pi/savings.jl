# Role: brain (reporting — display-only, non-causal)
"""
    savings.jl — the dollars-saved surface for credence-pi.

Reads the append-only observation log and produces a human-facing
governance + spend report: total LLM spend, how many tool calls the
agent proposed, how many credence-pi asked about, how many the user then
denied (stopping that work), and an explicitly-bounded estimate of the
spend those denials avoided.

DISPLAY-ONLY / NON-CAUSAL. This module summarises what already happened;
its arithmetic never feeds a decision or a belief update, so it sits
outside Invariant 1's scope (the constitution's "display formatting,
diagnostic telemetry … out of scope" carve-out). It calls neither
`condition` nor `expect`; it is reporting/IO over already-logged values.

Honesty notes baked into the report:
  - Cost is per-TURN, not per-tool-call (pi exposes no per-tool usage), so
    "prevented spend" is an ESTIMATE: denied-calls x average-turn-cost. It
    is labelled as such, never presented as a guaranteed figure.
  - The log records inbound sensor events, not the brain's effector
    signals, so the unambiguous governance signal is "asked -> user
    responded yes/no". proceed/block without an ask are not distinguished
    here (a Phase-2 enhancement would log decisions).
"""
module Savings

include(joinpath(@__DIR__, "daemon", "observation_log.jl"))
using .ObservationLog: read_log

export savings_report, format_report

"""
    savings_report(path) -> NamedTuple

Compute the governance + spend tallies from the observation log at
`path`. Pure over the file contents; returns a NamedTuple of counts and
(clearly-labelled) estimates. Empty/missing log yields an all-zero report.
"""
function savings_report(path::AbstractString)
    records = read_log(path)

    sessions = Set{String}()
    n_by_type = Dict{String, Int}()
    first_at = ""
    last_at = ""

    # spend
    total_usd = 0.0
    priced_turns = 0
    total_tokens = 0
    n_turns = 0

    # governance correlation
    proposed_tool = Dict{String, String}()   # tool-proposed event_id -> tool name
    responses = Dict{String, String}()        # in_response_to -> response
    decisions = Dict{String, Int}()           # brain decision action -> count
    n_completed = 0

    for r in records
        ev = r.event
        et = string(get(ev, "event_type", ""))
        n_by_type[et] = get(n_by_type, et, 0) + 1
        sid = get(ev, "session_id", nothing)
        sid === nothing || push!(sessions, string(sid))
        ra = r.received_at
        if !isempty(ra)
            isempty(first_at) && (first_at = ra)
            last_at = ra
        end

        if et == "turn-cost"
            n_turns += 1
            usd = get(ev, "usd", nothing)
            if usd isa Number
                total_usd += float(usd)
                priced_turns += 1
            end
            tok = get(ev, "total_tokens", nothing)
            tok isa Number && (total_tokens += Int(round(tok)))
        elseif et == "tool-proposed"
            eid = string(get(ev, "event_id", ""))
            pc = get(ev, "proposed_call", nothing)
            tool = pc === nothing ? "(unknown)" : string(get(pc, "tool_name", "(unknown)"))
            proposed_tool[eid] = tool
        elseif et == "user-responded"
            irt = string(get(ev, "in_response_to", ""))
            responses[irt] = string(get(ev, "response", ""))
        elseif et == "tool-completed"
            n_completed += 1
        elseif et == "decision"
            act = string(get(ev, "action", ""))
            decisions[act] = get(decisions, act, 0) + 1
        end
    end

    n_proposed = length(proposed_tool)
    asked_ids = [eid for eid in keys(proposed_tool) if haskey(responses, eid)]
    n_asked = length(asked_ids)
    n_approved = count(eid -> responses[eid] == "yes", asked_ids)
    n_denied = count(eid -> responses[eid] == "no", asked_ids)
    n_timeout = count(eid -> responses[eid] == "timeout", asked_ids)
    denied_tools = sort([proposed_tool[eid] for eid in asked_ids if responses[eid] == "no"])

    avg_usd_per_turn = priced_turns == 0 ? 0.0 : total_usd / priced_turns

    # Brain decisions (one "decision" record per emitted effector signal).
    # These count the silent auto-blocks / auto-proceeds the user-response
    # view misses — every `block` prevented a tool call from running.
    n_block = get(decisions, "block", 0)
    n_proceed = get(decisions, "proceed", 0)
    n_ask = get(decisions, "ask", 0)
    prevented_calls = n_block

    # ESTIMATE only — see module docstring. Per-turn cost (pi exposes no
    # per-tool cost), so avoided spend ~ prevented calls x average turn cost.
    estimated_prevented_usd = prevented_calls * avg_usd_per_turn

    return (
        total_events = length(records),
        sessions = length(sessions),
        first_at = first_at,
        last_at = last_at,
        events_by_type = n_by_type,
        total_usd = total_usd,
        priced_turns = priced_turns,
        turns = n_turns,
        total_tokens = total_tokens,
        avg_usd_per_turn = avg_usd_per_turn,
        tool_calls_proposed = n_proposed,
        decided_block = n_block,
        decided_proceed = n_proceed,
        decided_ask = n_ask,
        prevented_calls = prevented_calls,
        asked = n_asked,
        approved = n_approved,
        denied = n_denied,
        timed_out = n_timeout,
        completed = n_completed,
        denied_tools = denied_tools,
        estimated_prevented_usd = estimated_prevented_usd,
    )
end

_usd(x) = string("\$", string(round(x; digits=4)))

"""
    format_report(io, report)

Pretty-print a `savings_report` NamedTuple. Human-facing; the estimate is
labelled as an estimate.
"""
function format_report(io::IO, r)
    println(io, "=" ^ 60)
    println(io, "credence-pi — governance & spend report")
    println(io, "=" ^ 60)
    if r.total_events == 0
        println(io, "No observations yet. Deploy the plugin and run some sessions.")
        return
    end
    println(io, "Window:        $(r.first_at)  ->  $(r.last_at)")
    println(io, "Sessions:      $(r.sessions)")
    println(io, "Events:        $(r.total_events)  $(r.events_by_type)")
    println(io)
    println(io, "Spend observed (per-turn LLM cost):")
    println(io, "  turns priced:  $(r.priced_turns) / $(r.turns)")
    println(io, "  total:         $(_usd(r.total_usd))   ($(r.total_tokens) tokens)")
    println(io, "  avg / turn:    $(_usd(r.avg_usd_per_turn))")
    println(io)
    println(io, "Governance (brain decisions):")
    println(io, "  tool calls proposed: $(r.tool_calls_proposed)")
    println(io, "  proceed:             $(r.decided_proceed)")
    println(io, "  blocked:             $(r.decided_block)")
    println(io, "  asked:               $(r.decided_ask)   (user approved $(r.approved), denied $(r.denied), timed-out $(r.timed_out))")
    println(io, "  completed (ran):     $(r.completed)")
    if !isempty(r.denied_tools)
        println(io, "  user-denied tools:   ", join(r.denied_tools, ", "))
    end
    println(io)
    println(io, "Estimated prevented spend (blocked calls x avg turn cost):")
    println(io, "  ~ $(_usd(r.estimated_prevented_usd))   [ESTIMATE — cost is per-turn, not per-tool; see savings.jl]")
    println(io, "=" ^ 60)
end

# CLI: julia apps/credence-pi/savings.jl [log-path]
if abspath(PROGRAM_FILE) == @__FILE__
    path = length(ARGS) >= 1 ? ARGS[1] : joinpath(homedir(), ".credence-pi", "observations.jsonl")
    format_report(stdout, savings_report(path))
end

end # module Savings
