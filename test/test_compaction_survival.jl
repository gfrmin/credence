#!/usr/bin/env julia
"""
    test_compaction_survival.jl — Integration tests for compaction-survival.

Verifies: instruction registration, persistence across restarts,
escalation elevation, deduplication, and the full Yue/Meta lifecycle.
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

# ── Test 1: Instruction registration ──

println("=" ^ 60)
println("TEST 1: register_instruction! creates entry")
println("=" ^ 60)

brain1 = make_brain_state(bt)
@assert isempty(brain1.registered_instructions)

added = register_instruction!(brain1, "negation-delete", "delete")
@assert added == true "First registration should return true"
@assert length(brain1.registered_instructions) == 1
inst = brain1.registered_instructions[1]
@assert inst["pattern"] == "negation-delete"
@assert inst["action_class"] == "delete"
@assert inst["prior_strength"] == 5.0
@assert inst["approvals"] == 0
@assert inst["denials"] == 0
@assert haskey(inst, "registered_at")
@assert haskey(inst, "last_seen")
println("PASSED: Instruction registered correctly")
println()

# ── Test 2: Deduplication ──

println("=" ^ 60)
println("TEST 2: Deduplication — same pattern not re-registered")
println("=" ^ 60)

added2 = register_instruction!(brain1, "negation-delete", "delete")
@assert added2 == false "Duplicate registration should return false"
@assert length(brain1.registered_instructions) == 1 "Should still be 1 instruction"

added3 = register_instruction!(brain1, "confirm-before-delete", "delete")
@assert added3 == true "Different pattern should register"
@assert length(brain1.registered_instructions) == 2
println("PASSED: Deduplication works")
println()

# ── Test 3: Persistence across restart ──

println("=" ^ 60)
println("TEST 3: Instructions persist across sidecar restart")
println("=" ^ 60)

with_temp_state_dir() do dir
    brain_a = make_brain_state(bt)
    (; user_id, created_at) = load_sidecar_state!(brain_a)

    register_instruction!(brain_a, "negation-delete", "delete")
    register_instruction!(brain_a, "negation-deploy-to-protected", "deploy")
    save_sidecar_state(brain_a, user_id, created_at)

    brain_b = make_brain_state(bt)
    load_sidecar_state!(brain_b)

    @assert length(brain_b.registered_instructions) == 2 "Expected 2 instructions after reload, got $(length(brain_b.registered_instructions))"
    patterns = Set(inst["pattern"] for inst in brain_b.registered_instructions)
    @assert "negation-delete" in patterns
    @assert "negation-deploy-to-protected" in patterns
end
println("PASSED: Instructions survive restart")
println()

# ── Test 4: Escalation elevation — post-registration ──

println("=" ^ 60)
println("TEST 4: Post-registration escalation fires for guarded class")
println("=" ^ 60)

brain4 = make_brain_state(bt)
brain4.tool_outcomes[PosteriorKey("Bash", "delete")] = BetaMeasure(50.0, 2.0)

result_before = compute_eu(brain4, "Bash", "delete")
@assert result_before.decision == "proceed" "Before registration, concentrated Beta(50,2) should proceed, got $(result_before.decision)"

register_instruction!(brain4, "negation-delete", "delete")

result_after = compute_eu(brain4, "Bash", "delete")
@assert result_after.decision == "escalate" "After registration, delete-class tool should escalate, got $(result_after.decision)"

println("PASSED: Escalation fires after instruction registration")
println()

# ── Test 5: any-destructive matches multiple categories ──

println("=" ^ 60)
println("TEST 5: any-destructive instruction escalates multiple classes")
println("=" ^ 60)

brain5 = make_brain_state(bt)
brain5.tool_outcomes[PosteriorKey("Bash", "delete")] = BetaMeasure(50.0, 2.0)
brain5.tool_outcomes[PosteriorKey("Bash", "deploy")] = BetaMeasure(50.0, 2.0)
brain5.tool_outcomes[PosteriorKey("Bash", "privileged-exec")] = BetaMeasure(50.0, 2.0)
brain5.tool_outcomes[PosteriorKey("Edit", "code")] = BetaMeasure(50.0, 2.0)

register_instruction!(brain5, "ask-before-any", "any-destructive")

@assert compute_eu(brain5, "Bash", "delete").decision == "escalate" "delete should escalate"
@assert compute_eu(brain5, "Bash", "deploy").decision == "escalate" "deploy should escalate"
@assert compute_eu(brain5, "Bash", "privileged-exec").decision == "escalate" "privileged-exec should escalate"
@assert compute_eu(brain5, "Edit", "code").decision != "escalate" "code should NOT escalate via any-destructive"

println("PASSED: any-destructive covers delete, deploy, privileged-exec but not code")
println()

# ── Test 6: Instruction boost decays with approvals ──

println("=" ^ 60)
println("TEST 6: Instruction boost weakens with approvals")
println("=" ^ 60)

brain6 = make_brain_state(bt)
brain6.tool_outcomes[PosteriorKey("Bash", "delete")] = BetaMeasure(50.0, 2.0)
register_instruction!(brain6, "negation-delete", "delete")

result_fresh = compute_eu(brain6, "Bash", "delete")
@assert result_fresh.decision == "escalate" "Fresh instruction should escalate"

brain6.registered_instructions[1]["approvals"] = 40
result_decayed = compute_eu(brain6, "Bash", "delete")
@assert result_decayed.decision != "escalate" "After 40 approvals, instruction boost should be too weak to escalate against Beta(50,2), got $(result_decayed.decision)"

println("PASSED: Instruction boost decays with approvals")
println()

# ── Test 7: Denials keep boost strong ──

println("=" ^ 60)
println("TEST 7: Denials maintain instruction strength")
println("=" ^ 60)

brain7 = make_brain_state(bt)
brain7.tool_outcomes[PosteriorKey("Bash", "delete")] = BetaMeasure(50.0, 2.0)
register_instruction!(brain7, "negation-delete", "delete")
brain7.registered_instructions[1]["denials"] = 40

result_denied = compute_eu(brain7, "Bash", "delete")
@assert result_denied.decision == "escalate" "After 40 denials, instruction should still escalate, got $(result_denied.decision)"

println("PASSED: Denials maintain escalation")
println()

# ── Test 8: Unrelated categories not affected ──

println("=" ^ 60)
println("TEST 8: Registration doesn't affect unrelated categories")
println("=" ^ 60)

brain8 = make_brain_state(bt)
brain8.tool_outcomes[PosteriorKey("Bash", "version-control")] = BetaMeasure(50.0, 2.0)
register_instruction!(brain8, "negation-delete", "delete")

result_vc = compute_eu(brain8, "Bash", "version-control")
@assert result_vc.decision == "proceed" "version-control should not be affected by delete instruction, got $(result_vc.decision)"

println("PASSED: Unrelated categories unaffected")
println()

# ── Test 9: Full Yue/Meta inbox-deletion lifecycle ──

println("=" ^ 60)
println("TEST 9: Yue/Meta lifecycle — register via compaction → escalate")
println("=" ^ 60)

with_temp_state_dir() do dir
    brain9 = make_brain_state(bt)
    (; user_id, created_at) = load_sidecar_state!(brain9)

    for _ in 1:30
        update_posterior!(brain9, "Bash", "delete", true)
    end
    r_before = compute_eu(brain9, "Bash", "delete")
    @assert r_before.decision == "proceed" "Before instruction, concentrated posterior should proceed"

    messages = [
        Dict{String,Any}("role" => "user", "content" => "please confirm before deleting any files"),
        Dict{String,Any}("role" => "assistant", "content" => "Understood, I'll ask before deleting."),
    ]
    matches = match_instructions(messages)
    @assert length(matches) >= 1 "Should find at least one instruction"
    for (_, action_class) in matches
        for p in INSTRUCTION_PATTERNS
            if p.action_class == action_class
                register_instruction!(brain9, p.id, action_class)
                break
            end
        end
    end
    save_sidecar_state(brain9, user_id, created_at)

    r_after = compute_eu(brain9, "Bash", "delete")
    @assert r_after.decision == "escalate" "After instruction, delete-class should escalate"

    brain9b = make_brain_state(bt)
    load_sidecar_state!(brain9b)
    r_reloaded = compute_eu(brain9b, "Bash", "delete")
    @assert r_reloaded.decision == "escalate" "After restart, instruction should still escalate"
end
println("PASSED: Yue/Meta lifecycle — compaction → registration → escalation → persistence")
println()

println("=" ^ 60)
println("ALL COMPACTION-SURVIVAL TESTS PASSED")
println("=" ^ 60)
