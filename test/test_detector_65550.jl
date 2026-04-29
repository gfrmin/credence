#!/usr/bin/env julia
"""
    test_detector_65550.jl — Tests for no-confidence dreaming-loop detector.

Verifies: CV threshold fires after 5+ consecutive high-variance evaluations,
does not fire at 4, threshold scales with posterior concentration, and
stable trajectories do not trigger.
"""

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Credence
using JSON3

include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "instruction_patterns.jl"))
include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "brain.jl"))
include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "detectors.jl"))

config_path = joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "config", "budgets.json")
bt = load_budget_table(config_path)

# ── Test 1: Stable EU(proceed) → no-confidence does NOT fire ──

println("=" ^ 60)
println("TEST 1: Stable EU trajectory → no fire")
println("=" ^ 60)

state1 = make_brain_state(bt)
state1.tool_outcomes[PosteriorKey("Bash", "generic")] = BetaMeasure(50.0, 2.0)
window = 10

result1 = false
for _ in 1:15
    global result1 = check_no_confidence(state1, "Bash", "generic", 0.85, window)
end
@assert !result1 "Stable EU(proceed)=0.85 should not fire no-confidence"

println("PASSED: Stable trajectory does not trigger")
println()

# ── Test 2: High-variance EU for 5+ consecutive → fires ──

println("=" ^ 60)
println("TEST 2: High-variance EU for 5+ consecutive → fires")
println("=" ^ 60)

state2 = make_brain_state(bt)
state2.tool_outcomes[PosteriorKey("Bash", "generic")] = BetaMeasure(2.0, 2.0)
window2 = 10

high_var_values = [0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1, -0.1,
                   0.8, -0.7, 0.6, -0.5]
result2 = false
for v in high_var_values
    global result2 = check_no_confidence(state2, "Bash", "generic", v, window2)
end

@assert result2 "High-variance trajectory with fresh posterior should fire after 5+ consecutive above threshold"

println("PASSED: High-variance trajectory fires no-confidence")
println()

# ── Test 3: Consecutive span < 5 → does NOT fire ──

println("=" ^ 60)
println("TEST 3: 4 consecutive high-CV evaluations → does NOT fire")
println("=" ^ 60)

state3 = make_brain_state(bt)
state3.tool_outcomes[PosteriorKey("Bash", "generic")] = BetaMeasure(2.0, 2.0)
window3 = 10

# Fill window with high-variance values to establish CV
for v in [0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1, -0.1]
    check_no_confidence(state3, "Bash", "generic", v, window3)
end
# Flush entire window with stable values to reset counter
for _ in 1:10
    check_no_confidence(state3, "Bash", "generic", 0.2, window3)
end
@assert state3.no_confidence_consecutive == 0 "Stable values should reset counter"

# Now add exactly 4 high-variance values (high CV after mixing into window)
for v in [0.9, -0.8, 0.7, -0.6]
    global result3 = check_no_confidence(state3, "Bash", "generic", v, window3)
end
@assert state3.no_confidence_consecutive <= 4 "Should have at most 4 consecutive"
@assert !result3 "4 consecutive should not fire (need 5)"

println("PASSED: 4 consecutive high-CV does not fire")
println()

# ── Test 4: Concentrated posterior → demanding threshold ──

println("=" ^ 60)
println("TEST 4: Concentrated posterior needs very high CV to fire")
println("=" ^ 60)

state4 = make_brain_state(bt)
state4.tool_outcomes[PosteriorKey("Bash", "generic")] = BetaMeasure(100.0, 100.0)
window4 = 10

# Low variance values — CV should be below 1/√200 ≈ 0.0707
low_var = [0.200, 0.201, 0.199, 0.200, 0.201, 0.199, 0.200, 0.201, 0.199, 0.200,
           0.201, 0.199, 0.200, 0.201]
result4 = false
for v in low_var
    global result4 = check_no_confidence(state4, "Bash", "generic", v, window4)
end

# credence-lint: allow — precedent:display-arithmetic — test oracle CV computation
μ4 = sum(low_var[end-9:end]) / 10
σ4 = sqrt(sum((x - μ4)^2 for x in low_var[end-9:end]) / 10)
cv4 = σ4 / abs(μ4)
# credence-lint: allow — precedent:display-arithmetic — test oracle threshold
threshold4 = 1.0 / sqrt(200.0)
println("  CV = $(round(cv4, digits=6)), threshold = $(round(threshold4, digits=4))")
@assert !result4 "Low-variance trajectory below concentrated threshold should not fire"
println("PASSED: Low variance below concentrated-posterior threshold")
println()

# ── Test 5: Fresh posterior → permissive threshold, fires ──

println("=" ^ 60)
println("TEST 5: Fresh posterior fires at moderate CV")
println("=" ^ 60)

state5 = make_brain_state(bt)
window5 = 10

moderate_var_2 = [0.5, -0.1, 0.4, -0.2, 0.3, -0.1, 0.4, -0.2, 0.3, -0.1,
                  0.5, -0.1, 0.4, -0.2]
result5 = false
for v in moderate_var_2
    global result5 = check_no_confidence(state5, "Bash", "generic", v, window5)
end

# credence-lint: allow — precedent:expect-through-accessor — test oracle threshold
m5 = get_posterior(state5, "Bash", "generic")
threshold5 = 1.0 / sqrt(m5.alpha + m5.beta)  # credence-lint: allow — precedent:display-arithmetic — test oracle
println("  Fresh posterior threshold = $(round(threshold5, digits=4))")
println("  fired = $result5")

@assert result5 "Fresh Beta(1,1) with high-variance trajectory should fire (threshold=1/√2≈0.707)"

println("PASSED: Fresh posterior fires at moderate CV")
println()

# ── Test 6: Consecutive span resets on stable evaluation ──

println("=" ^ 60)
println("TEST 6: Consecutive counter resets on stable evaluation")
println("=" ^ 60)

state6 = make_brain_state(bt)
state6.tool_outcomes[PosteriorKey("Bash", "generic")] = BetaMeasure(2.0, 2.0)
window6 = 10

for v in [0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1, -0.1]
    check_no_confidence(state6, "Bash", "generic", v, window6)
end
@assert state6.no_confidence_consecutive >= 1 "Should have some consecutive count"

for _ in 1:10
    check_no_confidence(state6, "Bash", "generic", 0.5, window6)
end
@assert state6.no_confidence_consecutive == 0 "Stable values should reset consecutive counter, got $(state6.no_confidence_consecutive)"

println("PASSED: Consecutive counter resets correctly")
println()

# ── Test 7: Window size insufficient → no fire ──

println("=" ^ 60)
println("TEST 7: Fewer values than window size → no fire")
println("=" ^ 60)

state7 = make_brain_state(bt)
window7 = 10

result7 = false
for v in [0.9, -0.8, 0.7]
    global result7 = check_no_confidence(state7, "Bash", "generic", v, window7)
    @assert !result7 "Fewer than window_size values should not fire"
end

println("PASSED: Insufficient data prevents firing")
println()

println("=" ^ 60)
println("ALL #65550 DETECTOR TESTS PASSED")
println("=" ^ 60)
