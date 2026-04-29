#!/usr/bin/env julia
"""
    test_instruction_decay.jl — Tests for instruction decay and retirement.

Verifies: approval-driven retirement, denial-driven persistence,
retirement threshold (10× prior_strength), and full lifecycle
trajectories.
"""

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Credence
using JSON3
using Dates

include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "instruction_patterns.jl"))
include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "brain.jl"))
include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "persistence.jl"))

function with_temp_state_dir(f)
    dir = mktempdir()
    old = get(ENV, "CREDENCE_STATE_DIR", nothing)
    ENV["CREDENCE_STATE_DIR"] = dir
    try
        f(dir)
    finally
        if old === nothing
            delete!(ENV, "CREDENCE_STATE_DIR")
        else
            ENV["CREDENCE_STATE_DIR"] = old
        end
        rm(dir; recursive=true, force=true)
    end
end

config_path = joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "config", "budgets.json")
bt = load_budget_table(config_path)

# ── Test 1: Approval increments ──

println("=" ^ 60)
println("TEST 1: Approval increments approval count")
println("=" ^ 60)

brain1 = make_brain_state(bt)
register_instruction!(brain1, "negation-delete", "delete")
@assert brain1.registered_instructions[1]["approvals"] == 0

update_instruction_decay!(brain1, "delete", true)
@assert brain1.registered_instructions[1]["approvals"] == 1 "Expected 1 approval, got $(brain1.registered_instructions[1]["approvals"])"
@assert brain1.registered_instructions[1]["denials"] == 0

update_instruction_decay!(brain1, "delete", true)
@assert brain1.registered_instructions[1]["approvals"] == 2

println("PASSED: Approvals increment correctly")
println()

# ── Test 2: Denial increments ──

println("=" ^ 60)
println("TEST 2: Denial increments denial count")
println("=" ^ 60)

brain2 = make_brain_state(bt)
register_instruction!(brain2, "negation-delete", "delete")

update_instruction_decay!(brain2, "delete", false)
@assert brain2.registered_instructions[1]["denials"] == 1
@assert brain2.registered_instructions[1]["approvals"] == 0

update_instruction_decay!(brain2, "delete", false)
@assert brain2.registered_instructions[1]["denials"] == 2

println("PASSED: Denials increment correctly")
println()

# ── Test 3: Retirement after sufficient approvals ──

println("=" ^ 60)
println("TEST 3: Instruction retires after α+β > 10 × prior_strength")
println("=" ^ 60)

brain3 = make_brain_state(bt)
register_instruction!(brain3, "negation-delete", "delete")
@assert length(brain3.registered_instructions) == 1

for _ in 1:47
    update_instruction_decay!(brain3, "delete", true)
    @assert length(brain3.registered_instructions) == 1 "Should not retire yet"
end

@assert brain3.registered_instructions[1]["approvals"] == 47
# precision = 2 + 47 + 0 = 49, threshold = 50 → not retired
@assert length(brain3.registered_instructions) == 1 "49 < 50, should not retire"

update_instruction_decay!(brain3, "delete", true)
# precision = 2 + 48 + 0 = 50, threshold = 50, approvals(48) > denials(0) → NOT retired (need > not >=)
@assert brain3.registered_instructions[1]["approvals"] == 48

# One more to get precision = 51 > 50
update_instruction_decay!(brain3, "delete", true)
@assert isempty(brain3.registered_instructions) "precision=51 > 50, approvals(49) > denials(0) → should retire"

println("PASSED: Retirement fires at correct threshold")
println()

# ── Test 4: Denials prevent retirement ──

println("=" ^ 60)
println("TEST 4: Denials prevent retirement even with high precision")
println("=" ^ 60)

brain4 = make_brain_state(bt)
register_instruction!(brain4, "negation-delete", "delete")

for _ in 1:30
    update_instruction_decay!(brain4, "delete", false)
end
for _ in 1:25
    update_instruction_decay!(brain4, "delete", true)
end

# precision = 2 + 25 + 30 = 57 > 50, but approvals(25) < denials(30) → no retirement
@assert length(brain4.registered_instructions) == 1 "Denials should prevent retirement"
@assert brain4.registered_instructions[1]["approvals"] == 25
@assert brain4.registered_instructions[1]["denials"] == 30

println("PASSED: Denials prevent retirement")
println()

# ── Test 5: Mixed trajectory — eventual retirement when approvals dominate ──

println("=" ^ 60)
println("TEST 5: Mixed trajectory — approvals eventually dominate → retirement")
println("=" ^ 60)

brain5 = make_brain_state(bt)
register_instruction!(brain5, "negation-delete", "delete")

for _ in 1:5
    update_instruction_decay!(brain5, "delete", false)
end
for _ in 1:50
    update_instruction_decay!(brain5, "delete", true)
end

# precision = 2 + 50 + 5 = 57 > 50, approvals(50) > denials(5) → retired
@assert isempty(brain5.registered_instructions) "Should retire when approvals dominate"

println("PASSED: Mixed trajectory retires correctly")
println()

# ── Test 6: Only matching category affected ──

println("=" ^ 60)
println("TEST 6: Decay only affects matching action class")
println("=" ^ 60)

brain6 = make_brain_state(bt)
register_instruction!(brain6, "negation-delete", "delete")
register_instruction!(brain6, "negation-deploy-to-protected", "deploy")

update_instruction_decay!(brain6, "delete", true)
delete_inst = first(i for i in brain6.registered_instructions if i["pattern"] == "negation-delete")
deploy_inst = first(i for i in brain6.registered_instructions if i["pattern"] == "negation-deploy-to-protected")

@assert delete_inst["approvals"] == 1 "delete instruction should have 1 approval"
@assert deploy_inst["approvals"] == 0 "deploy instruction should be unaffected"

println("PASSED: Only matching class affected")
println()

# ── Test 7: any-destructive decays on multiple categories ──

println("=" ^ 60)
println("TEST 7: any-destructive instruction decays on any destructive class")
println("=" ^ 60)

brain7 = make_brain_state(bt)
register_instruction!(brain7, "ask-before-any", "any-destructive")

update_instruction_decay!(brain7, "delete", true)
update_instruction_decay!(brain7, "deploy", true)
update_instruction_decay!(brain7, "privileged-exec", false)

@assert brain7.registered_instructions[1]["approvals"] == 2
@assert brain7.registered_instructions[1]["denials"] == 1

update_instruction_decay!(brain7, "code", true)
@assert brain7.registered_instructions[1]["approvals"] == 2 "code should not match any-destructive"

println("PASSED: any-destructive decays on destructive categories")
println()

# ── Test 8: Retirement removes from persistence ──

println("=" ^ 60)
println("TEST 8: Retired instructions removed from persisted state")
println("=" ^ 60)

with_temp_state_dir() do dir
    brain8 = make_brain_state(bt)
    (; user_id, created_at) = load_sidecar_state!(brain8)

    register_instruction!(brain8, "negation-delete", "delete")
    register_instruction!(brain8, "negation-deploy-to-protected", "deploy")
    save_sidecar_state(brain8, user_id, created_at)

    for _ in 1:49
        update_instruction_decay!(brain8, "delete", true)
    end
    @assert length(brain8.registered_instructions) == 1 "delete instruction should be retired"
    @assert brain8.registered_instructions[1]["pattern"] == "negation-deploy-to-protected" "deploy instruction should remain"

    save_sidecar_state(brain8, user_id, created_at)

    brain8b = make_brain_state(bt)
    load_sidecar_state!(brain8b)
    @assert length(brain8b.registered_instructions) == 1 "Only deploy instruction should survive reload"
    @assert brain8b.registered_instructions[1]["pattern"] == "negation-deploy-to-protected"
end
println("PASSED: Retirement persists correctly")
println()

# ── Test 9: Full self-tuning lifecycle ──

println("=" ^ 60)
println("TEST 9: Full self-tuning lifecycle — register → escalate → approve × N → retire")
println("=" ^ 60)

brain9 = make_brain_state(bt)
for _ in 1:30
    update_posterior!(brain9, "Bash", "delete", true)
end

register_instruction!(brain9, "confirm-before-delete", "delete")

r_before = compute_eu(brain9, "Bash", "delete")
@assert r_before.decision == "escalate" "Should escalate after registration"

for i in 1:49
    update_instruction_decay!(brain9, "delete", true)
end

@assert isempty(brain9.registered_instructions) "Instruction should retire after 49 approvals"

r_after = compute_eu(brain9, "Bash", "delete")
@assert r_after.decision == "proceed" "After retirement, concentrated posterior should proceed, got $(r_after.decision)"

println("PASSED: Full lifecycle: register → escalate → approve × 49 → retire → proceed")
println()

# ── Test 10: Denial-heavy lifecycle — instruction persists ──

println("=" ^ 60)
println("TEST 10: Denial-heavy lifecycle — instruction never retires")
println("=" ^ 60)

brain10 = make_brain_state(bt)
for _ in 1:30
    update_posterior!(brain10, "Bash", "delete", true)
end
register_instruction!(brain10, "confirm-before-delete", "delete")

for _ in 1:60
    update_instruction_decay!(brain10, "delete", false)
end

@assert length(brain10.registered_instructions) == 1 "Instruction should persist after 60 denials"

r_denied = compute_eu(brain10, "Bash", "delete")
@assert r_denied.decision == "escalate" "Should still escalate after 60 denials, got $(r_denied.decision)"

println("PASSED: Denial-heavy lifecycle — instruction persists and keeps escalating")
println()

println("=" ^ 60)
println("ALL INSTRUCTION DECAY TESTS PASSED")
println("=" ^ 60)
