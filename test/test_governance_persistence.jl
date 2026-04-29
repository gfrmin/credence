#!/usr/bin/env julia
"""
    test_governance_persistence.jl — Tests for sidecar state persistence.

Verifies: bootstrap, round-trip, schema validation, atomic save,
concurrent writes, file permissions, UUID format, observation count
reconstruction.
"""

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Credence
using JSON3

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

# ── Test 1: Bootstrap — no state file ──

println("=" ^ 60)
println("TEST 1: Bootstrap with no existing state file")
println("=" ^ 60)

with_temp_state_dir() do dir
    brain = make_brain_state(bt)
    (; user_id, created_at) = load_sidecar_state!(brain)

    @assert !isempty(user_id) "user_id should not be empty"
    @assert occursin(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", user_id) "user_id should be UUID v4, got: $user_id"
    @assert !isempty(created_at) "created_at should not be empty"

    state_path = joinpath(dir, "posterior.json")
    @assert isfile(state_path) "State file should exist after bootstrap"

    dir_mode = filemode(dir) & 0o777
    @assert dir_mode == 0o700 "State directory mode should be 0700, got $(string(dir_mode, base=8))"

    file_mode = filemode(state_path) & 0o777
    @assert file_mode == 0o600 "State file mode should be 0600, got $(string(file_mode, base=8))"

    data = JSON3.read(read(state_path, String), Dict{String,Any})
    @assert data["schema_version"] == 1 "schema_version should be 1"
    @assert data["user_id"] == user_id
    @assert data["created_at"] == created_at
    @assert haskey(data, "updated_at")
    @assert data["tool_posteriors"] isa Dict
    @assert data["model_posteriors"] isa Dict
    @assert data["registered_instructions"] isa AbstractVector
    @assert isempty(data["registered_instructions"])
end
println("PASSED: Bootstrap creates valid state file")
println()

# ── Test 2: Round-trip — save, reload, verify exact match ──

println("=" ^ 60)
println("TEST 2: Round-trip save and reload")
println("=" ^ 60)

with_temp_state_dir() do dir
    brain = make_brain_state(bt)
    (; user_id, created_at) = load_sidecar_state!(brain)

    update_posterior!(brain, "Bash", "generic", true)
    update_posterior!(brain, "Bash", "generic", true)
    update_posterior!(brain, "Bash", "generic", false)
    update_posterior!(brain, "Read", "code", true)
    save_sidecar_state(brain, user_id, created_at)

    brain2 = make_brain_state(bt)
    result2 = load_sidecar_state!(brain2)

    @assert result2.user_id == user_id "user_id should survive round-trip"
    @assert result2.created_at == created_at "created_at should survive round-trip"

    m_bash = get_posterior(brain2, "Bash", "generic")
    @assert m_bash.alpha == 3.0 "Expected Bash:generic alpha=3.0, got $(m_bash.alpha)"
    @assert m_bash.beta == 2.0 "Expected Bash:generic beta=2.0, got $(m_bash.beta)"

    m_read = get_posterior(brain2, "Read", "code")
    @assert m_read.alpha == 2.0 "Expected Read:code alpha=2.0, got $(m_read.alpha)"
    @assert m_read.beta == 1.0 "Expected Read:code beta=1.0, got $(m_read.beta)"

    @assert brain2.observation_count == 4 "Expected observation_count=4, got $(brain2.observation_count)"
end
println("PASSED: Round-trip preserves exact alpha/beta and observation count")
println()

# ── Test 3: Schema version validation — reject newer ──

println("=" ^ 60)
println("TEST 3: Reject schema_version > 1")
println("=" ^ 60)

with_temp_state_dir() do dir
    state_path = joinpath(dir, "posterior.json")
    data = Dict{String,Any}(
        "schema_version" => 2,
        "user_id" => "00000000-0000-4000-8000-000000000000",
        "created_at" => "2026-01-01T00:00:00Z",
        "updated_at" => "2026-01-01T00:00:00Z",
        "tool_posteriors" => Dict{String,Any}(),
        "model_posteriors" => Dict{String,Any}(),
        "registered_instructions" => Any[],
    )
    open(state_path, "w") do io
        JSON3.pretty(io, data)
    end

    brain = make_brain_state(bt)
    rejected = false
    msg = ""
    try
        load_sidecar_state!(brain)
    catch e
        rejected = true
        msg = sprint(showerror, e)
    end
    @assert rejected "Should reject schema_version 2"
    @assert occursin("newer than this sidecar supports", msg) "Error message should mention upgrade, got: $msg"
end
println("PASSED: Schema version 2 rejected with clear message")
println()

# ── Test 4: Missing required fields ──

println("=" ^ 60)
println("TEST 4: Reject state files with missing required fields")
println("=" ^ 60)

required_fields = ["user_id", "tool_posteriors", "model_posteriors"]

with_temp_state_dir() do dir
    for field in required_fields
        state_path = joinpath(dir, "posterior.json")
        data = Dict{String,Any}(
            "schema_version" => 1,
            "user_id" => "00000000-0000-4000-8000-000000000000",
            "created_at" => "2026-01-01T00:00:00Z",
            "updated_at" => "2026-01-01T00:00:00Z",
            "tool_posteriors" => Dict{String,Any}(),
            "model_posteriors" => Dict{String,Any}(),
            "registered_instructions" => Any[],
        )
        delete!(data, field)
        open(state_path, "w") do io
            JSON3.pretty(io, data)
        end

        brain = make_brain_state(bt)
        rejected = false
        msg = ""
        try
            load_sidecar_state!(brain)
        catch e
            rejected = true
            msg = sprint(showerror, e)
        end
        @assert rejected "Should reject state file missing '$field'"
        @assert occursin(field, msg) "Error should name missing field '$field', got: $msg"
    end
end
println("PASSED: Missing required fields detected")
println()

# ── Test 5: Malformed JSON ──

println("=" ^ 60)
println("TEST 5: Reject malformed JSON")
println("=" ^ 60)

with_temp_state_dir() do dir
    state_path = joinpath(dir, "posterior.json")
    write(state_path, "{ not valid json !!!")

    brain = make_brain_state(bt)
    rejected = false
    try
        load_sidecar_state!(brain)
    catch e
        rejected = true
    end
    @assert rejected "Should reject malformed JSON"
end
println("PASSED: Malformed JSON rejected")
println()

# ── Test 6: Non-positive Beta parameters ──

println("=" ^ 60)
println("TEST 6: Reject non-positive alpha/beta")
println("=" ^ 60)

with_temp_state_dir() do dir
    state_path = joinpath(dir, "posterior.json")
    data = Dict{String,Any}(
        "schema_version" => 1,
        "user_id" => "00000000-0000-4000-8000-000000000000",
        "created_at" => "2026-01-01T00:00:00Z",
        "updated_at" => "2026-01-01T00:00:00Z",
        "tool_posteriors" => Dict{String,Any}(
            "Bash:generic" => Dict{String,Any}("alpha" => 0.0, "beta" => 1.0)
        ),
        "model_posteriors" => Dict{String,Any}(),
        "registered_instructions" => Any[],
    )
    open(state_path, "w") do io
        JSON3.pretty(io, data)
    end

    brain = make_brain_state(bt)
    rejected = false
    msg = ""
    try
        load_sidecar_state!(brain)
    catch e
        rejected = true
        msg = sprint(showerror, e)
    end
    @assert rejected "Should reject alpha=0.0"
    @assert occursin("positive reals", msg) "Error should mention positive reals, got: $msg"

    data["tool_posteriors"] = Dict{String,Any}(
        "Bash:generic" => Dict{String,Any}("alpha" => 1.0, "beta" => -0.5)
    )
    open(state_path, "w") do io
        JSON3.pretty(io, data)
    end

    brain2 = make_brain_state(bt)
    rejected2 = false
    try
        load_sidecar_state!(brain2)
    catch
        rejected2 = true
    end
    @assert rejected2 "Should reject beta=-0.5"
end
println("PASSED: Non-positive parameters rejected")
println()

# ── Test 7: File permissions enforced on every save ──

println("=" ^ 60)
println("TEST 7: File permissions re-enforced on save")
println("=" ^ 60)

with_temp_state_dir() do dir
    brain = make_brain_state(bt)
    (; user_id, created_at) = load_sidecar_state!(brain)
    state_path = joinpath(dir, "posterior.json")

    chmod(state_path, 0o644)
    @assert (filemode(state_path) & 0o777) == 0o644 "Sanity: mode should be 644 after chmod"

    save_sidecar_state(brain, user_id, created_at)
    restored_mode = filemode(state_path) & 0o777
    @assert restored_mode == 0o600 "Mode should be restored to 0600 after save, got $(string(restored_mode, base=8))"
end
println("PASSED: Permissions restored on save")
println()

# ── Test 8: Concurrent saves produce well-formed state ──

println("=" ^ 60)
println("TEST 8: Concurrent saves don't corrupt state")
println("=" ^ 60)

with_temp_state_dir() do dir
    brain = make_brain_state(bt)
    (; user_id, created_at) = load_sidecar_state!(brain)

    state_lock = ReentrantLock()
    tasks = map(1:100) do i
        Threads.@spawn begin
            lock(state_lock) do
                update_posterior!(brain, "Bash", "generic", rand(Bool))
                save_sidecar_state(brain, user_id, created_at)
            end
        end
    end
    foreach(wait, tasks)

    state_path = joinpath(dir, "posterior.json")
    raw = read(state_path, String)
    data = JSON3.read(raw, Dict{String,Any})
    @assert data["schema_version"] == 1 "State file should be well-formed after concurrent writes"
    @assert haskey(data["tool_posteriors"], "Bash:generic")

    brain2 = make_brain_state(bt)
    result2 = load_sidecar_state!(brain2)
    @assert result2.user_id == user_id "user_id should survive concurrent writes"

    m = get_posterior(brain2, "Bash", "generic")
    # credence-lint: allow — precedent:expect-through-accessor — verifying persisted alpha/beta after concurrent writes
    total_obs = m.alpha + m.beta - 2.0
    @assert total_obs == 100.0 "Expected 100 observations, got $total_obs"
end
println("PASSED: 100 concurrent saves produce well-formed state")
println()

# ── Test 9: UUID v4 format ──

println("=" ^ 60)
println("TEST 9: UUID v4 generation format")
println("=" ^ 60)

for _ in 1:20
    uuid = generate_uuid4()
    @assert occursin(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", uuid) "UUID should match v4 format, got: $uuid"
end
println("PASSED: 20 UUIDs all match v4 format")
println()

# ── Test 10: Observation count reconstruction ──

println("=" ^ 60)
println("TEST 10: Observation count reconstructed from posteriors")
println("=" ^ 60)

with_temp_state_dir() do dir
    brain = make_brain_state(bt)
    load_sidecar_state!(brain)

    for i in 1:15
        update_posterior!(brain, "Edit", "code", true)
    end
    for i in 1:7
        update_posterior!(brain, "Bash", "deploy", false)
    end
    for i in 1:3
        update_posterior!(brain, "Read", "documentation", true)
    end
    @assert brain.observation_count == 25 "Expected 25 observations in-memory"

    save_sidecar_state(brain, "test-uid", "2026-01-01T00:00:00Z")

    brain2 = make_brain_state(bt)
    load_sidecar_state!(brain2)
    @assert brain2.observation_count == 25 "Expected 25 observations after reload, got $(brain2.observation_count)"
end
println("PASSED: Observation count matches after reload")
println()

# ── Test 11: Missing schema_version field ──

println("=" ^ 60)
println("TEST 11: Reject state file with no schema_version")
println("=" ^ 60)

with_temp_state_dir() do dir
    state_path = joinpath(dir, "posterior.json")
    data = Dict{String,Any}(
        "user_id" => "00000000-0000-4000-8000-000000000000",
        "tool_posteriors" => Dict{String,Any}(),
        "model_posteriors" => Dict{String,Any}(),
        "registered_instructions" => Any[],
    )
    open(state_path, "w") do io
        JSON3.pretty(io, data)
    end

    brain = make_brain_state(bt)
    rejected = false
    msg = ""
    try
        load_sidecar_state!(brain)
    catch e
        rejected = true
        msg = sprint(showerror, e)
    end
    @assert rejected "Should reject state file without schema_version"
    @assert occursin("schema_version", msg) "Error should mention schema_version, got: $msg"
end
println("PASSED: Missing schema_version rejected")
println()

# ── Test 12: CREDENCE_STATE_DIR override ──

println("=" ^ 60)
println("TEST 12: CREDENCE_STATE_DIR override")
println("=" ^ 60)

custom_dir = mktempdir()
old_env = get(ENV, "CREDENCE_STATE_DIR", nothing)
ENV["CREDENCE_STATE_DIR"] = custom_dir
try
    @assert get_state_dir() == custom_dir "get_state_dir should return custom dir"
    @assert get_state_path() == joinpath(custom_dir, "posterior.json")

    brain = make_brain_state(bt)
    load_sidecar_state!(brain)
    @assert isfile(joinpath(custom_dir, "posterior.json")) "State file should exist in custom dir"
finally
    if old_env === nothing
        delete!(ENV, "CREDENCE_STATE_DIR")
    else
        ENV["CREDENCE_STATE_DIR"] = old_env
    end
    rm(custom_dir; recursive=true, force=true)
end
println("PASSED: CREDENCE_STATE_DIR override works")
println()

# ── Test 13: Model posteriors round-trip ──

println("=" ^ 60)
println("TEST 13: Model posteriors serialisation round-trip")
println("=" ^ 60)

with_temp_state_dir() do dir
    brain = make_brain_state(bt)
    (; user_id, created_at) = load_sidecar_state!(brain)

    brain.model_quality["gpt-4"] = Dict{String,BetaMeasure}(
        "code" => BetaMeasure(10.0, 3.0),
        "documentation" => BetaMeasure(5.0, 2.0),
    )
    save_sidecar_state(brain, user_id, created_at)

    brain2 = make_brain_state(bt)
    load_sidecar_state!(brain2)

    @assert haskey(brain2.model_quality, "gpt-4") "Model quality should survive round-trip"
    @assert brain2.model_quality["gpt-4"]["code"].alpha == 10.0
    @assert brain2.model_quality["gpt-4"]["code"].beta == 3.0
    @assert brain2.model_quality["gpt-4"]["documentation"].alpha == 5.0
    @assert brain2.model_quality["gpt-4"]["documentation"].beta == 2.0
end
println("PASSED: Model posteriors survive round-trip")
println()

println("=" ^ 60)
println("ALL GOVERNANCE PERSISTENCE TESTS PASSED")
println("=" ^ 60)
