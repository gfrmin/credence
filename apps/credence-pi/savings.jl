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
  - "Prevented" counts only ENFORCED blocks. The daemon logs a `decision`
    record per brain decision regardless of mode; in shadow mode the body
    proceeds anyway and logs a `counterfactual-decision`. Every shadowed block
    is therefore BOTH a decision(block) and a counterfactual(block), so we
    subtract the latter (enforced = decided - would) — otherwise /report would
    claim prevention that never happened in an observe-only deployment.
  - "Would have blocked" (shadow) is reported separately, with a bounded
    false-block PROXY: of the exact-repeat would-blocks, how many had an
    intervening workspace mutation (a plausible legitimate re-run) vs none
    (almost-certainly waste). write/edit = hard mutation; bash/exec = ambiguous
    but counted (the safe direction — it over-counts "legitimate", so it never
    under-reports false blocks). The gold rate still needs opt-in user labels.
"""
module Savings

include(joinpath(@__DIR__, "daemon", "observation_log.jl"))
using .ObservationLog: read_log
using JSON3   # canonical fingerprint for the exact-repeat / false-block proxy

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
    tps = NamedTuple[]                         # tool-proposed records in log order (false-block proxy)
    counterfactuals = Tuple{String, String}[] # shadow-mode (in_response_to, effector) — would-have-enforced
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
            # Keep the input + session + order for the exact-repeat proxy below.
            input = pc === nothing ? nothing : get(pc, "input", nothing)
            push!(tps, (eid = eid, sid = sid === nothing ? "" : string(sid), tool = tool, input = input))
        elseif et == "counterfactual-decision"
            push!(counterfactuals,
                  (string(get(ev, "in_response_to", "")), string(get(ev, "effector", ""))))
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

    # Shadow mode: the brain still emits a `decision`, but the body proceeds and
    # logs a `counterfactual-decision`. Every shadowed block is BOTH a
    # decision(block) and a counterfactual(block); enforced blocks are only the
    # former. So enforced = decided - would, and "prevented" counts only enforced
    # ones (else /report claims prevention that never happened in shadow mode).
    would_block = count(c -> c[2] == "block", counterfactuals)
    would_ask   = count(c -> c[2] == "ask", counterfactuals)
    prevented_calls = max(0, n_block - would_block)

    # ESTIMATE only — see module docstring. Per-turn cost (pi exposes no
    # per-tool cost), so avoided spend ~ prevented calls x average turn cost.
    estimated_prevented_usd = prevented_calls * avg_usd_per_turn
    # Same per-turn estimate on the shadow would-blocks: what governance WOULD
    # have saved on this traffic if enforcing. Labelled counterfactual.
    estimated_counterfactual_usd = would_block * avg_usd_per_turn

    # False-block PROXY (display-only): of the shadow would-blocks on an
    # exact-repeat call, how many are plausibly LEGITIMATE re-runs (the workspace
    # changed in between) vs almost-certainly waste (nothing changed). The brain
    # blocks on (tool, args) alone and cannot see a change; this bounded post-hoc
    # check is the honest estimate of that false-block rate. Asymmetry is
    # deliberate: write/edit are a HARD mutation signal; bash/exec are AMBIGUOUS
    # but counted as possible mutations — over-counting `candidate_legitimate`,
    # which errs toward NOT under-reporting false blocks (the safe direction).
    # Limitation: with redactToolInputs the input is null, so identical-call
    # matching degrades (null inputs in a session collapse together) — noted, not
    # fatal. Read-only / unknown intervening tools are treated as non-mutating.
    tp_by_eid = Dict{String, Int}(t.eid => i for (i, t) in enumerate(tps))
    _fp(input) = input === nothing ? "∅" : JSON3.write(input)
    _is_mut(tool) = lowercase(tool) in ("write", "edit", "bash", "exec")   # KNOWN_TOOLS minus read
    wb_candidate = 0
    wb_waste = 0
    for (irt, eff) in counterfactuals
        eff == "block" || continue
        i = get(tp_by_eid, irt, 0)
        if i == 0
            wb_candidate += 1   # cannot locate the call ⇒ don't call it waste (safe direction)
            continue
        end
        cur = tps[i]
        jprior = 0                              # nearest prior identical (tool, args) in this session
        for j in (i - 1):-1:1
            t = tps[j]
            (t.sid == cur.sid && lowercase(t.tool) == lowercase(cur.tool) &&
                _fp(t.input) == _fp(cur.input)) || continue
            jprior = j
            break
        end
        if jprior == 0
            wb_candidate += 1   # no prior identical ⇒ not an exact-repeat we can call pure waste
            continue
        end
        mutated = any(j -> tps[j].sid == cur.sid && _is_mut(tps[j].tool), (jprior + 1):(i - 1))
        mutated ? (wb_candidate += 1) : (wb_waste += 1)
    end

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
        would_block_calls = would_block,
        would_ask_calls = would_ask,
        would_block_candidate_legitimate = wb_candidate,
        would_block_likely_waste = wb_waste,
        asked = n_asked,
        approved = n_approved,
        denied = n_denied,
        timed_out = n_timeout,
        completed = n_completed,
        denied_tools = denied_tools,
        estimated_prevented_usd = estimated_prevented_usd,
        estimated_counterfactual_usd = estimated_counterfactual_usd,
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
    println(io, "Estimated prevented spend (enforced blocks x avg turn cost):")
    println(io, "  ~ $(_usd(r.estimated_prevented_usd))   [ESTIMATE — cost is per-turn, not per-tool; see savings.jl]")
    if r.would_block_calls > 0 || r.would_ask_calls > 0
        println(io)
        println(io, "Shadow / counterfactual (observed only — NOT enforced):")
        println(io, "  would-block:   $(r.would_block_calls)   (~$(_usd(r.estimated_counterfactual_usd)) would-be-saved [ESTIMATE])")
        println(io, "  would-ask:     $(r.would_ask_calls)")
        println(io, "  would-blocks split: likely-waste $(r.would_block_likely_waste), candidate legitimate re-run $(r.would_block_candidate_legitimate)")
        println(io, "                      [PROXY — write/edit=mutation, bash/exec=ambiguous; gold rate needs user labels]")
    end
    println(io, "=" ^ 60)
end

# CLI: julia apps/credence-pi/savings.jl [log-path]
if abspath(PROGRAM_FILE) == @__FILE__
    path = length(ARGS) >= 1 ? ARGS[1] : joinpath(homedir(), ".credence-pi", "observations.jsonl")
    format_report(stdout, savings_report(path))
end

end # module Savings
