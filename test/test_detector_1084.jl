#!/usr/bin/env julia
"""
    test_detector_1084.jl — Tests for compaction-wipes-confirm-instruction detector.

Verifies: instruction-based escalation elevation fires when a registered
instruction matches the candidate's action class, magnitude tracks decay
state, and the detector integrates with the full lifecycle.
"""

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Credence
using JSON3
using Dates

include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "instruction_patterns.jl"))
include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "brain.jl"))

config_path = joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "config", "budgets.json")
bt = load_budget_table(config_path)

# ── Test 1: Registered instruction → escalate fires ──

println("=" ^ 60)
println("TEST 1: Registered instruction for matching class → escalate")
println("=" ^ 60)

state1 = make_brain_state(bt)
state1.tool_outcomes[PosteriorKey("Bash", "delete")] = BetaMeasure(50.0, 2.0)

result_before = compute_eu(state1, "Bash", "delete")
@assert result_before.decision == "proceed" "Concentrated posterior without instruction should proceed, got $(result_before.decision)"

register_instruction!(state1, "negation-delete", "delete")
result_after = compute_eu(state1, "Bash", "delete")
@assert result_after.decision == "escalate" "With registered instruction, should escalate, got $(result_after.decision)"

println("PASSED: Instruction registration triggers escalation")
println()

# ── Test 2: Instruction for non-matching class → no escalation ──

println("=" ^ 60)
println("TEST 2: Instruction for different class → no effect")
println("=" ^ 60)

state2 = make_brain_state(bt)
state2.tool_outcomes[PosteriorKey("Bash", "version-control")] = BetaMeasure(50.0, 2.0)

register_instruction!(state2, "negation-delete", "delete")
result_vc = compute_eu(state2, "Bash", "version-control")
@assert result_vc.decision == "proceed" "Instruction for 'delete' should not affect 'version-control', got $(result_vc.decision)"

println("PASSED: Non-matching instruction class has no effect")
println()

# ── Test 3: No registered instruction → no escalation from detector ──

println("=" ^ 60)
println("TEST 3: No instruction → no escalation from this detector")
println("=" ^ 60)

state3 = make_brain_state(bt)
state3.tool_outcomes[PosteriorKey("Bash", "delete")] = BetaMeasure(50.0, 2.0)
@assert isempty(state3.registered_instructions)

result_no_inst = compute_eu(state3, "Bash", "delete")
@assert result_no_inst.decision == "proceed" "No instruction, concentrated posterior → proceed, got $(result_no_inst.decision)"

println("PASSED: No instruction means no escalation from #1084 detector")
println()

# ── Test 4: Fresh instruction → strong elevation ──

println("=" ^ 60)
println("TEST 4: Fresh instruction has strong elevation magnitude")
println("=" ^ 60)

state4 = make_brain_state(bt)
state4.tool_outcomes[PosteriorKey("Bash", "delete")] = BetaMeasure(50.0, 2.0)
register_instruction!(state4, "negation-delete", "delete")

result_fresh = compute_eu(state4, "Bash", "delete")
@assert result_fresh.decision == "escalate" "Fresh instruction should escalate"
fresh_eu_escalate = result_fresh.eu_escalate
println("  Fresh instruction: eu_escalate = $(round(fresh_eu_escalate, digits=4))")

println("PASSED: Fresh instruction produces strong escalation")
println()

# ── Test 5: Decayed instruction → weak elevation ──

println("=" ^ 60)
println("TEST 5: Decayed instruction has weak elevation magnitude")
println("=" ^ 60)

state5 = make_brain_state(bt)
state5.tool_outcomes[PosteriorKey("Bash", "delete")] = BetaMeasure(50.0, 2.0)
register_instruction!(state5, "negation-delete", "delete")

state5.registered_instructions[1]["approvals"] = 40

result_decayed = compute_eu(state5, "Bash", "delete")
decayed_eu_escalate = result_decayed.eu_escalate
println("  Decayed instruction (40 approvals): eu_escalate = $(round(decayed_eu_escalate, digits=4))")

@assert decayed_eu_escalate < fresh_eu_escalate "Decayed instruction should have weaker elevation than fresh"
@assert result_decayed.decision != "escalate" "After 40 approvals, boost too weak against Beta(50,2), got $(result_decayed.decision)"

println("PASSED: Instruction boost decays with approvals")
println()

# ── Test 6: Denial-heavy instruction → strong elevation persists ──

println("=" ^ 60)
println("TEST 6: Denial-heavy instruction maintains strong elevation")
println("=" ^ 60)

state6 = make_brain_state(bt)
state6.tool_outcomes[PosteriorKey("Bash", "delete")] = BetaMeasure(50.0, 2.0)
register_instruction!(state6, "negation-delete", "delete")
state6.registered_instructions[1]["denials"] = 40

result_denied = compute_eu(state6, "Bash", "delete")
@assert result_denied.decision == "escalate" "40 denials should keep escalation strong, got $(result_denied.decision)"

println("PASSED: Denials maintain escalation strength")
println()

# ── Test 7: Full lifecycle — register → escalate → approve × N → retire → proceed ──

println("=" ^ 60)
println("TEST 7: Full #1084 detector lifecycle")
println("=" ^ 60)

state7 = make_brain_state(bt)
for _ in 1:30
    update_posterior!(state7, "Bash", "delete", true)
end

register_instruction!(state7, "confirm-before-delete", "delete")
r_escalate = compute_eu(state7, "Bash", "delete")
@assert r_escalate.decision == "escalate" "After registration, should escalate"

for _ in 1:49
    update_instruction_decay!(state7, "delete", true)
end
@assert isempty(state7.registered_instructions) "Instruction should retire after 49 approvals"

r_proceed = compute_eu(state7, "Bash", "delete")
@assert r_proceed.decision == "proceed" "After retirement, should proceed, got $(r_proceed.decision)"

println("PASSED: Full lifecycle: register → escalate → retire → proceed")
println()

# ── Test 8: any-destructive covers multiple categories ──

println("=" ^ 60)
println("TEST 8: any-destructive instruction covers delete, deploy, privileged-exec")
println("=" ^ 60)

state8 = make_brain_state(bt)
state8.tool_outcomes[PosteriorKey("Bash", "delete")] = BetaMeasure(50.0, 2.0)
state8.tool_outcomes[PosteriorKey("Bash", "deploy")] = BetaMeasure(50.0, 2.0)
state8.tool_outcomes[PosteriorKey("Bash", "privileged-exec")] = BetaMeasure(50.0, 2.0)
state8.tool_outcomes[PosteriorKey("Edit", "code")] = BetaMeasure(50.0, 2.0)

register_instruction!(state8, "ask-before-any", "any-destructive")

@assert compute_eu(state8, "Bash", "delete").decision == "escalate"
@assert compute_eu(state8, "Bash", "deploy").decision == "escalate"
@assert compute_eu(state8, "Bash", "privileged-exec").decision == "escalate"
@assert compute_eu(state8, "Edit", "code").decision != "escalate" "code should NOT match any-destructive"

println("PASSED: any-destructive covers destructive categories, not code")
println()

println("=" ^ 60)
println("ALL #1084 DETECTOR TESTS PASSED")
println("=" ^ 60)
