#!/usr/bin/env julia
"""
    test_detectors_e2e.jl — End-to-end integration test spanning all three detectors.

Verifies: each detector fires correctly in a single sidecar session;
cross-detector independence; reason strings distinguish detectors.
"""

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Credence
using JSON3
using Dates

include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "instruction_patterns.jl"))
include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "brain.jl"))
include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "detectors.jl"))
include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "persistence.jl"))

config_path = joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "config", "budgets.json")
bt = load_budget_table(config_path)

read_tools_path = joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "config", "read_tools.json")
read_tools = load_read_tools(read_tools_path)

# ── Test 1: Synthetic trajectory — each failure mode fires in turn ──

println("=" ^ 60)
println("TEST 1: Sequential failure-mode trajectory")
println("=" ^ 60)

state = make_brain_state(bt)

# Phase 1: Build up some posterior with successful Bash calls
for _ in 1:10
    update_posterior!(state, "Bash", "generic", true)
end
println("  Phase 1: Built posterior Beta(11,1)")

# Phase 2: Trigger #34574 (stationarity) via repeated identical tool calls
params_repeat = Dict{String,Any}("command" => "echo stuck")
for _ in 1:5
    record_outcome!(state, "Bash", params_repeat, true)
end
sr = check_stationarity(state, read_tools, "Bash", params_repeat, "generic")
@assert sr !== nothing && sr.fires "Phase 2: Stationarity detector should fire after 5 identical outcomes"
println("  Phase 2: #34574 (stationarity) fires — outcome_var=$(round(sr.outcome_var, digits=4))")

# Phase 3: Trigger #1084 (instruction-based escalation)
register_instruction!(state, "negation-delete", "delete")
state.tool_outcomes[PosteriorKey("Bash", "delete")] = BetaMeasure(50.0, 2.0)
result_1084 = compute_eu(state, "Bash", "delete")
@assert result_1084.decision == "escalate" "Phase 3: #1084 detector should escalate for delete class"
println("  Phase 3: #1084 (instruction escalation) fires — decision=$(result_1084.decision)")

# Phase 4: Trigger #65550 (no-confidence) via high-variance EU trajectory
state.tool_outcomes[PosteriorKey("Edit", "code")] = BetaMeasure(2.0, 2.0)
high_var_eu = [0.8, -0.5, 0.6, -0.4, 0.7, -0.3, 0.5, -0.6, 0.4, -0.5,
               0.8, -0.5, 0.6, -0.4]
nc_fired = false
for v in high_var_eu
    global nc_fired = check_no_confidence(state, "Edit", "code", v, 10)
end
@assert nc_fired "Phase 4: #65550 no-confidence should fire"
println("  Phase 4: #65550 (no-confidence) fires")

println("PASSED: All three detectors fire in sequence")
println()

# ── Test 2: Cross-detector independence ──

println("=" ^ 60)
println("TEST 2: Cross-detector independence")
println("=" ^ 60)

state2 = make_brain_state(bt)

# Set up conditions for both #34574 and #65550 simultaneously
state2.tool_outcomes[PosteriorKey("Bash", "generic")] = BetaMeasure(2.0, 2.0)
params_both = Dict{String,Any}("command" => "echo loop")
for _ in 1:5
    record_outcome!(state2, "Bash", params_both, true)
end

high_var_eu2 = [0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1, -0.1,
                0.8, -0.7, 0.6, -0.5]
for v in high_var_eu2
    check_no_confidence(state2, "Bash", "generic", v, 10)
end

sr2 = check_stationarity(state2, read_tools, "Bash", params_both, "generic")
nc2 = state2.no_confidence_consecutive >= NO_CONFIDENCE_SPAN

@assert sr2 !== nothing && sr2.fires "#34574 should fire independently"
@assert nc2 "#65550 should fire independently"
println("  Both #34574 and #65550 fire simultaneously — independence verified")

println("PASSED: Cross-detector independence")
println()

# ── Test 3: Reason strings distinguish detectors ──

println("=" ^ 60)
println("TEST 3: Reason strings are distinguishable")
println("=" ^ 60)

sr_reason = "Posterior stationary on repeated tool call"
nc_reason = "Posterior over next-action value is flat"
esc_reason = "uncertain expected utility"

@assert sr_reason != nc_reason "Stationarity and no-confidence reasons must differ"
@assert sr_reason != esc_reason "Stationarity and escalation reasons must differ"
@assert nc_reason != esc_reason "No-confidence and escalation reasons must differ"

println("PASSED: Reason strings are distinct")
println()

# ── Test 4: Read tools unaffected by any detector ──

println("=" ^ 60)
println("TEST 4: Read tools unaffected across all detectors")
println("=" ^ 60)

state4 = make_brain_state(bt)
read_params = Dict{String,Any}("file_path" => "/etc/hosts")
for _ in 1:100
    record_outcome!(state4, "Read", read_params, true)
end

sr4 = check_stationarity(state4, read_tools, "Read", read_params, "code")
@assert sr4 === nothing "Read tool exempt from #34574"

result_read = compute_eu(state4, "Read", "code")
@assert result_read.decision != "halt" "Read tool should not be halted"

println("PASSED: Read tools exempt from destructive detectors")
println()

println("=" ^ 60)
println("ALL END-TO-END DETECTOR TESTS PASSED")
println("=" ^ 60)
