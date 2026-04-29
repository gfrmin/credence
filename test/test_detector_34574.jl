#!/usr/bin/env julia
"""
    test_detector_34574.jl — Tests for exec-repetition stationarity detector.

Verifies: stationarity fires after K identical observations, does not fire
with fewer or mixed outcomes, read-tool exemption, and window-size scaling
with posterior concentration.
"""

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Credence
using JSON3

include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "instruction_patterns.jl"))
include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "brain.jl"))
include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "detectors.jl"))

config_path = joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "config", "budgets.json")
bt = load_budget_table(config_path)

read_tools_path = joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "config", "read_tools.json")
read_tools = load_read_tools(read_tools_path)

# ── Test 1: Window size scales with posterior concentration ──

println("=" ^ 60)
println("TEST 1: Stationarity window size")
println("=" ^ 60)

@assert stationarity_window(BetaMeasure(1.0, 1.0)) == 2 "Beta(1,1): K should be 2, got $(stationarity_window(BetaMeasure(1.0, 1.0)))"
@assert stationarity_window(BetaMeasure(50.0, 50.0)) == 10 "Beta(50,50): K should be 10, got $(stationarity_window(BetaMeasure(50.0, 50.0)))"
@assert stationarity_window(BetaMeasure(5.0, 5.0)) >= 4 "Beta(5,5): K should be >= 4"
@assert stationarity_window(BetaMeasure(100.0, 100.0)) >= 14 "Beta(100,100): K should be >= 14"

println("PASSED: Window size scales with sqrt(concentration)")
println()

# ── Test 2: 10 identical observations → stationarity fires ──

println("=" ^ 60)
println("TEST 2: 10 identical observations → stationarity fires")
println("=" ^ 60)

state2 = make_brain_state(bt)
state2.tool_outcomes[PosteriorKey("Bash", "generic")] = BetaMeasure(5.0, 5.0)
params2 = Dict{String,Any}("command" => "ls -la")

for _ in 1:10
    record_outcome!(state2, "Bash", params2, true)
end

sr2 = check_stationarity(state2, read_tools, "Bash", params2, "generic")
@assert sr2 !== nothing "Stationarity check should return a result"
@assert sr2.fires "Stationarity should fire after 10 identical observations"

println("PASSED: Stationarity fires after 10 identical observations")
println()

# ── Test 3: 9 identical + 1 differing → does NOT fire ──

println("=" ^ 60)
println("TEST 3: 9 identical + 1 differing → does NOT fire")
println("=" ^ 60)

state3 = make_brain_state(bt)
state3.tool_outcomes[PosteriorKey("Bash", "generic")] = BetaMeasure(5.0, 5.0)
params3 = Dict{String,Any}("command" => "echo test")

for _ in 1:9
    record_outcome!(state3, "Bash", params3, true)
end
record_outcome!(state3, "Bash", params3, false)

sr3 = check_stationarity(state3, read_tools, "Bash", params3, "generic")
@assert sr3 !== nothing "Stationarity check should return a result"
@assert !sr3.fires "Stationarity should NOT fire with mixed outcomes"

println("PASSED: Mixed outcomes prevent stationarity")
println()

# ── Test 4: Read tool exemption ──

println("=" ^ 60)
println("TEST 4: Read tool with 100 repetitions → exempt")
println("=" ^ 60)

state4 = make_brain_state(bt)
params4 = Dict{String,Any}("file_path" => "/etc/hosts")

for _ in 1:100
    record_outcome!(state4, "Read", params4, true)
end

sr4 = check_stationarity(state4, read_tools, "Read", params4, "code")
@assert sr4 === nothing "Read tool should be exempt from stationarity detection"

sr4_grep = check_stationarity(state4, read_tools, "Grep", Dict{String,Any}(), "code")
@assert sr4_grep === nothing "Grep tool should be exempt"

sr4_ls = check_stationarity(state4, read_tools, "LS", Dict{String,Any}(), "code")
@assert sr4_ls === nothing "LS tool should be exempt"

sr4_glob = check_stationarity(state4, read_tools, "Glob", Dict{String,Any}(), "code")
@assert sr4_glob === nothing "Glob tool should be exempt"

println("PASSED: Read-like tools exempt from stationarity detection")
println()

# ── Test 5: Beta(1,1) with K=2 identical → fires ──

println("=" ^ 60)
println("TEST 5: Beta(1,1) prior with K=2 same outcome → fires")
println("=" ^ 60)

state5 = make_brain_state(bt)
params5 = Dict{String,Any}("command" => "date")

record_outcome!(state5, "Bash", params5, true)
record_outcome!(state5, "Bash", params5, true)

sr5 = check_stationarity(state5, read_tools, "Bash", params5, "generic")
@assert sr5 !== nothing "Should get stationarity result"
@assert sr5.fires "Beta(1,1) with K=2 identical outcomes should fire"
@assert sr5.window_size == 2 "Window should be 2 for Beta(1,1)"

println("PASSED: Low-concentration posterior fires with minimal data")
println()

# ── Test 6: Beta(50,50) with K=2 identical → does NOT fire ──

println("=" ^ 60)
println("TEST 6: Beta(50,50) prior with K=2 same outcome → does NOT fire")
println("=" ^ 60)

state6 = make_brain_state(bt)
state6.tool_outcomes[PosteriorKey("Bash", "generic")] = BetaMeasure(50.0, 50.0)
params6 = Dict{String,Any}("command" => "whoami")

record_outcome!(state6, "Bash", params6, true)
record_outcome!(state6, "Bash", params6, true)

sr6 = check_stationarity(state6, read_tools, "Bash", params6, "generic")
@assert sr6 === nothing "Beta(50,50) with only 2 observations should not have enough data (K=10)"

println("PASSED: High-concentration posterior requires more observations")
println()

# ── Test 7: Prototype fallback removed ──

println("=" ^ 60)
println("TEST 7: Prototype fallback fields removed from BrainState")
println("=" ^ 60)

state7 = make_brain_state(bt)
@assert !hasproperty(state7, :prototype_fallback_enabled) "prototype_fallback_enabled should be removed"
@assert !hasproperty(state7, :max_repetitions) "max_repetitions should be removed"
@assert hasproperty(state7, :tool_args_outcomes) "tool_args_outcomes should exist"

println("PASSED: Prototype fallback replaced by stationarity detector")
println()

# ── Test 8: canonical_key still works ──

println("=" ^ 60)
println("TEST 8: canonical_key produces stable keys")
println("=" ^ 60)

k1 = canonical_key("Bash", Dict{String,Any}("command" => "ls", "timeout" => 5000))
k2 = canonical_key("Bash", Dict{String,Any}("timeout" => 5000, "command" => "ls"))
@assert k1 == k2 "Same params in different order should produce same key"

k3 = canonical_key("Bash", Dict{String,Any}("command" => "ls"))
@assert k1 != k3 "Different params should produce different keys"

println("PASSED: canonical_key is order-independent")
println()

# ── Test 9: Insufficient observations → no result ──

println("=" ^ 60)
println("TEST 9: Insufficient observations returns nothing")
println("=" ^ 60)

state9 = make_brain_state(bt)
state9.tool_outcomes[PosteriorKey("Bash", "generic")] = BetaMeasure(10.0, 10.0)
params9 = Dict{String,Any}("command" => "pwd")

record_outcome!(state9, "Bash", params9, true)

sr9 = check_stationarity(state9, read_tools, "Bash", params9, "generic")
@assert sr9 === nothing "1 observation when K>1 should return nothing"

println("PASSED: Insufficient observations correctly handled")
println()

println("=" ^ 60)
println("ALL STATIONARITY DETECTOR TESTS PASSED")
println("=" ^ 60)
