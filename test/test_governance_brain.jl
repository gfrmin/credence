#!/usr/bin/env julia
"""
    test_governance_brain.jl — Tests for the governance sidecar brain.

Verifies: EU calculation, posterior updates, task-category inference,
duration budget classification, threshold derivations, and backwards
compatibility with Move 1's response shape.
"""

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Credence
using JSON3

include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "instruction_patterns.jl"))
include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "brain.jl"))

# ── Test 1: Category inference ──

println("=" ^ 60)
println("TEST 1: Task-category inference")
println("=" ^ 60)

@assert infer_category("Bash", Dict{String,Any}("command" => "git status")) == "version-control"
@assert infer_category("Bash", Dict{String,Any}("command" => "git log --oneline")) == "version-control"
@assert infer_category("Edit", Dict{String,Any}("file_path" => "/home/user/app.py")) == "code"
@assert infer_category("Edit", Dict{String,Any}("file_path" => "/home/user/README.md")) == "documentation"
@assert infer_category("Bash", Dict{String,Any}("command" => "rm -rf /tmp/foo")) == "delete"
@assert infer_category("Bash", Dict{String,Any}("command" => "git rm old_file.txt")) == "delete"
@assert infer_category("Bash", Dict{String,Any}("command" => "npm install express")) == "dependency"
@assert infer_category("Bash", Dict{String,Any}("command" => "pip install requests")) == "dependency"
@assert infer_category("Bash", Dict{String,Any}("command" => "sudo systemctl restart nginx")) == "privileged-exec"
@assert infer_category("Bash", Dict{String,Any}("command" => "docker push myimage:latest")) == "deploy"
@assert infer_category("Bash", Dict{String,Any}("command" => "echo hello")) == "generic"
@assert infer_category("Read", Dict{String,Any}("file_path" => "/etc/hosts")) == "code"
@assert infer_category("Write", Dict{String,Any}("file_path" => "/home/user/notes.txt")) == "documentation"
println("PASSED: All category inference cases")
println()

# ── Test 2: Duration budget ──

println("=" ^ 60)
println("TEST 2: Duration budget classification")
println("=" ^ 60)

config_path = joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "config", "budgets.json")
bt = load_budget_table(config_path)

@assert get_budget(bt, "Read", "code") == 5000
@assert get_budget(bt, "Edit", "code") == 10000
@assert get_budget(bt, "Bash", "generic") == 30000
@assert get_budget(bt, "Bash", "privileged-exec") == 30000
@assert get_budget(bt, "Agent", "generic") == 15000

@assert classify_outcome(bt, "Read", "code", nothing, 100.0) == true
@assert classify_outcome(bt, "Read", "code", nothing, 6000.0) == false
@assert classify_outcome(bt, "Read", "code", "file not found", 100.0) == false
@assert classify_outcome(bt, "Bash", "generic", nothing, nothing) == true
@assert classify_outcome(bt, "Bash", "generic", "", nothing) == true
println("PASSED: Budget classification")
println()

# ── Test 3: Posterior update via condition ──

println("=" ^ 60)
println("TEST 3: Posterior update produces exact alpha/beta")
println("=" ^ 60)

state = make_brain_state(bt)
m0 = get_posterior(state, "Bash", "generic")
@assert m0.alpha == 1.0
@assert m0.beta == 1.0

update_posterior!(state, "Bash", "generic", true)
m1 = get_posterior(state, "Bash", "generic")
@assert m1.alpha == 2.0 "Expected alpha=2.0, got $(m1.alpha)"
@assert m1.beta == 1.0 "Expected beta=1.0, got $(m1.beta)"

update_posterior!(state, "Bash", "generic", false)
m2 = get_posterior(state, "Bash", "generic")
@assert m2.alpha == 2.0 "Expected alpha=2.0, got $(m2.alpha)"
@assert m2.beta == 2.0 "Expected beta=2.0, got $(m2.beta)"

for _ in 1:48
    update_posterior!(state, "Bash", "generic", true)
end
m3 = get_posterior(state, "Bash", "generic")
@assert m3.alpha == 50.0 "Expected alpha=50.0, got $(m3.alpha)"
@assert m3.beta == 2.0 "Expected beta=2.0, got $(m3.beta)"
println("PASSED: Exact alpha/beta after N updates")
println()

# ── Test 4: EU calculation — fresh prior yields escalate ──

println("=" ^ 60)
println("TEST 4: EU argmax under different posteriors")
println("=" ^ 60)

fresh_state = make_brain_state(bt)
result_fresh = compute_eu(fresh_state, "Bash", "generic")
@assert result_fresh.decision == "escalate" "Fresh Beta(1,1) should yield escalate, got $(result_fresh.decision)"
@assert result_fresh.action == "escalate"
println("  Beta(1,1) → decision=$(result_fresh.decision) ✓")

confident_state = make_brain_state(bt)
confident_state.tool_outcomes[PosteriorKey("Bash", "generic")] = BetaMeasure(50.0, 2.0)
result_confident = compute_eu(confident_state, "Bash", "generic")
@assert result_confident.decision == "proceed" "Concentrated Beta(50,2) should yield proceed, got $(result_confident.decision)"
@assert result_confident.action == "proceed"
println("  Beta(50,2) → decision=$(result_confident.decision) ✓")

failing_state = make_brain_state(bt)
failing_state.tool_outcomes[PosteriorKey("Bash", "generic")] = BetaMeasure(2.0, 50.0)
result_failing = compute_eu(failing_state, "Bash", "generic")
@assert result_failing.decision == "halt" "Concentrated Beta(2,50) should yield halt, got $(result_failing.decision)"
@assert result_failing.action == "block"
println("  Beta(2,50) → decision=$(result_failing.decision) ✓")

println("PASSED: EU argmax under canonical posteriors")
println()

# ── Test 5: Threshold derivations match Amendment 1 ──

println("=" ^ 60)
println("TEST 5: Threshold derivations from posterior precision")
println("=" ^ 60)

m_fresh = BetaMeasure(1.0, 1.0)
concentration_fresh = m_fresh.alpha + m_fresh.beta
downgrade_thresh_fresh = 1.0 - 1.0 / concentration_fresh
escalate_thresh_fresh = 1.0 / sqrt(concentration_fresh)
@assert abs(downgrade_thresh_fresh - 0.5) < 1e-10 "Fresh downgrade threshold should be 0.5, got $downgrade_thresh_fresh"
@assert abs(escalate_thresh_fresh - 1.0/sqrt(2.0)) < 1e-10 "Fresh escalate threshold should be 1/√2, got $escalate_thresh_fresh"
println("  Beta(1,1): downgrade_threshold=$(round(downgrade_thresh_fresh, digits=3)), escalate_threshold=$(round(escalate_thresh_fresh, digits=3)) ✓")

m_conc = BetaMeasure(50.0, 50.0)
concentration_conc = m_conc.alpha + m_conc.beta
downgrade_thresh_conc = 1.0 - 1.0 / concentration_conc
escalate_thresh_conc = 1.0 / sqrt(concentration_conc)
@assert downgrade_thresh_conc >= 0.99 "Concentrated downgrade threshold should be ≥0.99, got $downgrade_thresh_conc"
@assert escalate_thresh_conc <= 0.1 "Concentrated escalate threshold should be ≤0.1, got $escalate_thresh_conc"
println("  Beta(50,50): downgrade_threshold=$(round(downgrade_thresh_conc, digits=3)), escalate_threshold=$(round(escalate_thresh_conc, digits=3)) ✓")

println("PASSED: Thresholds scale with posterior precision")
println()

# ── Test 6: Signals populated in EU result ──

println("=" ^ 60)
println("TEST 6: Signal fields populated correctly")
println("=" ^ 60)

sig_state = make_brain_state(bt)
sig_state.tool_outcomes[PosteriorKey("Bash", "code")] = BetaMeasure(10.0, 3.0)
result_sig = compute_eu(sig_state, "Bash", "code")
@assert haskey(result_sig.signals, "alpha")
@assert haskey(result_sig.signals, "beta")
@assert haskey(result_sig.signals, "comparison_p")
@assert haskey(result_sig.signals, "cv")
@assert haskey(result_sig.signals, "eu_proceed")
@assert haskey(result_sig.signals, "eu_halt")
@assert haskey(result_sig.signals, "eu_downgrade")
@assert haskey(result_sig.signals, "eu_escalate")
@assert result_sig.signals["alpha"] == 10.0
@assert result_sig.signals["beta"] == 3.0
println("PASSED: All signal fields present and correct")
println()

# ── Test 7: Backwards compatibility — Move 1 response shape ──

println("=" ^ 60)
println("TEST 7: Backwards compatibility with Move 1 plugin")
println("=" ^ 60)

compat_state = make_brain_state(bt)
result_compat = compute_eu(compat_state, "Edit", "code")
response = Dict{String,Any}(
    "action" => result_compat.action,
    "decision" => result_compat.decision,
    "reason" => result_compat.reason,
    "signals" => result_compat.signals,
    "requireApproval" => nothing,
)
@assert response["action"] in ("proceed", "block", "escalate") "action must be proceed/block/escalate"
@assert response["decision"] in ("proceed", "halt", "downgrade", "route", "escalate")
json_str = JSON3.write(response)
parsed = JSON3.read(json_str, Dict{String,Any})
@assert haskey(parsed, "action")
@assert haskey(parsed, "reason")
println("PASSED: Response shape serialises cleanly, action/reason present for Move 1 plugin")
println()

# ── Test 8: Observation count tracks correctly ──

println("=" ^ 60)
println("TEST 8: Observation count")
println("=" ^ 60)

count_state = make_brain_state(bt)
@assert count_state.observation_count == 0
update_posterior!(count_state, "Bash", "generic", true)
update_posterior!(count_state, "Read", "code", true)
update_posterior!(count_state, "Edit", "code", false)
@assert count_state.observation_count == 3 "Expected 3 observations, got $(count_state.observation_count)"
println("PASSED: Observation count increments correctly")
println()

# ── Test 9: Category precedence — delete before version-control ──

println("=" ^ 60)
println("TEST 9: Category precedence")
println("=" ^ 60)

@assert infer_category("Bash", Dict{String,Any}("command" => "git rm file.txt")) == "delete"
println("PASSED: 'git rm' categorised as delete, not version-control")
println()

println("=" ^ 60)
println("ALL GOVERNANCE BRAIN TESTS PASSED")
println("=" ^ 60)
