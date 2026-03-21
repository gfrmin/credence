#!/usr/bin/env julia
"""
    test_grid_world.jl — Tier 3 tests: grid-world domain

Tests that need the grid-world simulation: world creation, sensor
projection, full agent runs, regime change, meta-learning.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: expect, condition, weights, mean
using Credence: BetaMeasure, TaggedBetaMeasure, MixtureMeasure, Finite, Interval, Kernel, Measure
using Credence: prune, truncate

# Load grid-world domain
include(joinpath(@__DIR__, "..", "domains", "grid_world", "host.jl"))

using Random

# ═══════════════════════════════════════
# TEST 1: Grid world basic functionality
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 1: Grid world creates and simulates correctly")
println("=" ^ 60)

let
    Random.seed!(42)
    world = create_world(:colour_typed)
    @assert length(world.entities) >= 4 "Expected ≥ 4 entities"
    @assert world.agent_energy == 100.0 "Initial energy should be 100"

    for e in world.entities
        if e.kind == ENEMY
            @assert e.rgb[1] > 0.8 "Enemy should be red under colour rule"
            @assert e.energy < 0 "Enemy should have negative energy"
        elseif e.kind == FOOD
            @assert e.rgb[3] > 0.8 "Food should be blue under colour rule"
            @assert e.energy > 0 "Food should have positive energy"
        end
    end
    println("PASSED: World created with correct entity properties")

    set_rule!(world, :motion_typed)
    for e in world.entities
        if e.kind == ENEMY
            @assert e.speed > 0.5 "Enemy should be fast under motion rule"
        elseif e.kind == FOOD
            @assert e.speed == 0.0 "Food should be stationary under motion rule"
        end
    end
    println("PASSED: Regime change updates entity properties")

    initial_energy = world.agent_energy
    world_step!(world, MOVE_N)
    @assert world.agent_energy < initial_energy "Moving should cost energy"
    println("PASSED: World step executes correctly")
end
println()

# ═══════════════════════════════════════
# TEST 2: Sensor projection
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 2: Sensor projection produces correct-dimensionality output")
println("=" ^ 60)

let
    Random.seed!(42)
    true_state = [0.9, 0.1, 0.1, 0.5, 0.5, 0.8, 0.3, 0.4]

    sv1 = project(true_state, minimal_sensor_config())
    @assert length(sv1) == 2 "Minimal config should produce 2 readings"

    sv2 = project(true_state, full_sensor_config())
    @assert length(sv2) == 8 "Full config should produce 8 readings"

    for r in [sv1..., sv2...]
        @assert 0.0 <= r <= 1.0 "Sensor reading out of [0,1]: $r"
    end

    println("PASSED: Sensor projection produces correct dimensionality and range")
end
println()

# ═══════════════════════════════════════
# TEST 3: AgentState sync_prune! keeps arrays aligned
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 3: AgentState sync_prune! keeps parallel arrays aligned")
println("=" ^ 60)

let
    Random.seed!(42)
    grammars = generate_seed_grammars()

    g1 = grammars[2]
    g2 = grammars[3]
    p1 = enumerate_programs(g1, 2)
    p2 = enumerate_programs(g2, 2)

    components = Measure[]
    log_prior = Float64[]
    meta = Tuple{Int, Int}[]
    ck = CompiledKernel[]
    progs = Program[]

    idx = 0
    for (pi, p) in enumerate(p1)
        idx += 1
        push!(components, TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaMeasure(1.0, 1.0)))
        push!(log_prior, -g1.complexity * log(2) - p.complexity * log(2))
        push!(meta, (g1.id, pi))
        push!(ck, compile_kernel(p, g1, pi))
        push!(progs, p)
    end
    for (pi, p) in enumerate(p2)
        idx += 1
        push!(components, TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaMeasure(1.0, 1.0)))
        push!(log_prior, -g2.complexity * log(2) - p.complexity * log(2))
        push!(meta, (g2.id, pi))
        push!(ck, compile_kernel(p, g2, pi))
        push!(progs, p)
    end

    belief = MixtureMeasure(Interval(0.0, 1.0), components, log_prior)
    state = AgentState(belief, meta, ck, progs)

    n_before = length(state.belief.components)

    sync_prune!(state; threshold=-5.0)
    n_after = length(state.belief.components)

    @assert length(state.metadata) == n_after "metadata length mismatch"
    @assert length(state.compiled_kernels) == n_after "compiled_kernels length mismatch"
    @assert length(state.all_programs) == n_after "all_programs length mismatch"

    for (i, comp) in enumerate(state.belief.components)
        @assert comp isa TaggedBetaMeasure
        @assert comp.tag == i "Tag reindex failed: expected $i, got $(comp.tag)"
    end

    println("PASSED: Before=$n_before, after=$n_after, all arrays aligned, tags reindexed")
end
println()

# ═══════════════════════════════════════
# TEST 4: AgentState sync_truncate! keeps arrays aligned
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 4: AgentState sync_truncate! keeps parallel arrays aligned")
println("=" ^ 60)

let
    Random.seed!(42)
    grammars = generate_seed_grammars()

    g = grammars[2]
    programs = enumerate_programs(g, 2)

    components = Measure[]
    log_prior = Float64[]
    meta = Tuple{Int, Int}[]
    ck = CompiledKernel[]
    progs = Program[]

    for (pi, p) in enumerate(programs)
        push!(components, TaggedBetaMeasure(Interval(0.0, 1.0), pi, BetaMeasure(1.0, 1.0)))
        push!(log_prior, -g.complexity * log(2) - p.complexity * log(2))
        push!(meta, (g.id, pi))
        push!(ck, compile_kernel(p, g, pi))
        push!(progs, p)
    end

    belief = MixtureMeasure(Interval(0.0, 1.0), components, log_prior)
    state = AgentState(belief, meta, ck, progs)

    n_before = length(state.belief.components)
    max_c = 10
    sync_truncate!(state; max_components=max_c)
    n_after = length(state.belief.components)

    @assert n_after <= max_c "Truncate should limit to $max_c, got $n_after"
    @assert length(state.metadata) == n_after
    @assert length(state.compiled_kernels) == n_after
    @assert length(state.all_programs) == n_after

    for (i, comp) in enumerate(state.belief.components)
        @assert comp.tag == i "Tag reindex after truncate: expected $i, got $(comp.tag)"
    end

    println("PASSED: Before=$n_before, truncated to $n_after, all aligned")
end
println()

# ═══════════════════════════════════════
# TEST 5: Per-grammar sensor projection produces different vectors
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 5: Per-grammar sensor projection produces different vectors")
println("=" ^ 60)

let
    Random.seed!(42)
    true_state = [0.9, 0.1, 0.1, 0.5, 0.5, 0.8, 0.3, 0.4]

    grammars = generate_seed_grammars()

    gsvs = project_per_grammar(true_state, grammars)

    @assert length(gsvs) == length(grammars) "Should have one vector per grammar"

    colour_g = grammars[2]
    motion_g = grammars[3]

    sv_colour = gsvs[colour_g.id]
    sv_motion = gsvs[motion_g.id]

    @assert length(sv_colour) == 3 "Colour grammar should produce 3-dim vector"
    @assert length(sv_motion) == 2 "Motion grammar should produce 2-dim vector"

    @assert sv_colour[1] > 0.7 "Colour should see high red (true state r=0.9)"
    @assert sv_motion[1] > 0.5 "Motion should see high speed (true state speed=0.8)"

    println("PASSED: Colour vector=$(round.(sv_colour, digits=3)), " *
            "Motion vector=$(round.(sv_motion, digits=3))")
end
println()

# ═══════════════════════════════════════
# TEST 6: TaggedBetaMeasure kernel dispatch — firing vs non-firing diverge
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 6: TaggedBetaMeasure kernel dispatch with compiled predicates")
println("=" ^ 60)

let
    Random.seed!(42)
    grammars = generate_seed_grammars()

    g = grammars[2]
    programs = enumerate_programs(g, 2)

    red_prog = nothing
    blue_prog = nothing
    red_sv = [0.95, 0.1, 0.1]
    blue_sv = [0.1, 0.1, 0.95]
    ts = Dict{Symbol, Any}()

    for (i, p) in enumerate(programs)
        ck_test = compile_kernel(p, g, i)
        if red_prog === nothing && ck_test.evaluate(red_sv, ts)
            red_prog = (i, p, ck_test)
        end
        if blue_prog === nothing && !ck_test.evaluate(red_sv, ts) && ck_test.evaluate(blue_sv, ts)
            blue_prog = (i, p, ck_test)
        end
        red_prog !== nothing && blue_prog !== nothing && break
    end

    @assert red_prog !== nothing "Should find a program that fires on red"
    @assert blue_prog !== nothing "Should find a program that fires on blue (not red)"

    comp1 = TaggedBetaMeasure(Interval(0.0, 1.0), 1, BetaMeasure(1.0, 1.0))
    comp2 = TaggedBetaMeasure(Interval(0.0, 1.0), 2, BetaMeasure(1.0, 1.0))
    belief = MixtureMeasure(Interval(0.0, 1.0), Measure[comp1, comp2], [0.0, 0.0])

    ck_vec = [red_prog[3], blue_prog[3]]

    grammar_sensor_vectors = Dict{Int, Vector{Float64}}(g.id => red_sv)

    posterior = belief
    for _ in 1:5
        k = build_observation_kernel(ck_vec, grammar_sensor_vectors, ts)
        posterior = condition(posterior, k, 1.0)
    end

    c1 = posterior.components[1]
    c2 = posterior.components[2]
    @assert c1 isa TaggedBetaMeasure
    @assert c2 isa TaggedBetaMeasure

    # Red-firing program (tag 1) should have updated beta: Beta(6, 1)
    @assert c1.beta.alpha ≈ 6.0 "Red-firing program should have α=6 after 5 enemy obs, got $(c1.beta.alpha)"

    # Blue-firing program (tag 2) on red entity doesn't fire — but non-firing programs
    # now predict base rate (50/50) and get conjugate updates too: Beta(6,1) after 5 enemy obs
    @assert c2.beta.alpha ≈ 6.0 "Non-firing program should also get conjugate update, got α=$(c2.beta.alpha)"
    @assert c2.beta.beta ≈ 1.0 "Non-firing program β should be 1, got $(c2.beta.beta)"

    # Firing program gains weight because its ll is better than log(0.5) base rate
    w = weights(posterior)
    @assert w[1] > w[2] "Firing program should gain weight over non-firing after multiple obs"

    println("PASSED: Red-fires α=$(c1.beta.alpha), Blue-non-firing α=$(c2.beta.alpha), " *
            "weights=[$(round(w[1], digits=4)), $(round(w[2], digits=4))]")
end
println()

# ═══════════════════════════════════════
# TEST 7: Regime change causes prediction disruption (emergent)
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 7: Regime change — surprise increases after change (disruption)")
println("=" ^ 60)

let
    metrics, _, _ = run_agent(
        world_rules=[:colour_typed, :motion_typed],
        regime_change_steps=[50],
        max_steps=100,
        grammar_perturbation_interval=typemax(Int),
        verbose=false,
        rng_seed=42)

    function mean_surprise_in_range(m, start_step, end_step)
        vals = Float64[]
        for (idx, step) in enumerate(m.steps)
            start_step <= step <= end_step || continue
            m.surprise[idx] > 0.0 || continue
            push!(vals, m.surprise[idx])
        end
        vals
    end

    pre = mean_surprise_in_range(metrics, 20, 50)
    post = mean_surprise_in_range(metrics, 51, 70)

    @assert length(pre) >= 3 "Need ≥3 interactions in steps 20-50, got $(length(pre))"
    @assert length(post) >= 2 "Need ≥2 interactions in steps 51-70, got $(length(post))"

    mean_pre = sum(pre) / length(pre)
    mean_post = sum(post) / length(post)

    @assert mean_post > mean_pre (
        "Surprise should increase after regime change (disruption): " *
        "pre=$(round(mean_pre, digits=3)), post=$(round(mean_post, digits=3))")

    println("PASSED: Mean surprise pre=$(round(mean_pre, digits=3)) ($( length(pre)) obs), " *
            "post=$(round(mean_post, digits=3)) ($(length(post)) obs)")
end
println()

# ═══════════════════════════════════════
# TEST 8: Regime change re-learning occurs (emergent)
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 8: Regime change — surprise decreases during re-learning")
println("=" ^ 60)

let
    metrics, _, _ = run_agent(
        world_rules=[:colour_typed, :motion_typed],
        regime_change_steps=[50],
        max_steps=100,
        grammar_perturbation_interval=typemax(Int),
        verbose=false,
        rng_seed=42)

    surprise_early = Float64[]
    surprise_late = Float64[]

    for (idx, step) in enumerate(metrics.steps)
        s = metrics.surprise[idx]
        s == 0.0 && continue
        if 51 <= step <= 70
            push!(surprise_early, s)
        elseif 75 <= step <= 100
            push!(surprise_late, s)
        end
    end

    @assert length(surprise_early) >= 2 "Need ≥2 interactions in steps 51-70, got $(length(surprise_early))"
    @assert length(surprise_late) >= 2 "Need ≥2 interactions in steps 75-100, got $(length(surprise_late))"

    mean_early = sum(surprise_early) / length(surprise_early)
    mean_late = sum(surprise_late) / length(surprise_late)

    @assert mean_late < mean_early (
        "Surprise should decrease during re-learning: " *
        "early=$(round(mean_early, digits=3)), late=$(round(mean_late, digits=3))")

    println("PASSED: Mean surprise early=$(round(mean_early, digits=3)) ($(length(surprise_early)) obs), " *
            "late=$(round(mean_late, digits=3)) ($(length(surprise_late)) obs)")
end
println()

# ═══════════════════════════════════════
# TEST 9: Meta-learning controlled comparison (emergent)
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 9: Meta-learning — perturbation vs no perturbation")
println("=" ^ 60)

let
    seed = 42

    metrics_a, _, _ = run_agent(
        world_rules=[:colour_typed, :motion_typed, :colour_typed],
        regime_change_steps=[75, 150],
        max_steps=225,
        grammar_perturbation_interval=25,
        verbose=false,
        rng_seed=seed)

    metrics_b, _, _ = run_agent(
        world_rules=[:colour_typed, :motion_typed, :colour_typed],
        regime_change_steps=[75, 150],
        max_steps=225,
        grammar_perturbation_interval=typemax(Int),
        verbose=false,
        rng_seed=seed)

    function regime3_accuracy(m::MetricsTracker)
        hits = 0
        total = 0
        for (idx, step) in enumerate(m.steps)
            150 <= step <= 225 || continue
            m.surprise[idx] > 0.0 || continue
            total += 1
            m.prediction_correct[idx] && (hits += 1)
        end
        total == 0 ? 0.0 : hits / total
    end

    acc_a = regime3_accuracy(metrics_a)
    acc_b = regime3_accuracy(metrics_b)

    @assert acc_a > acc_b (
        "Perturbation agent should have higher regime-3 accuracy: " *
        "A=$(round(acc_a, digits=3)) vs B=$(round(acc_b, digits=3))")

    println("PASSED: Agent A accuracy=$(round(acc_a, digits=3)), " *
            "Agent B accuracy=$(round(acc_b, digits=3))")
end
println()

# ═══════════════════════════════════════
# TEST 10: Grammar pool evolution is real (emergent)
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 10: Grammar pool evolution — perturbed grammars are useful")
println("=" ^ 60)

let
    metrics, state, pool = run_agent(
        world_rules=[:colour_typed, :motion_typed, :colour_typed],
        regime_change_steps=[75, 150],
        max_steps=225,
        grammar_perturbation_interval=25,
        verbose=false,
        rng_seed=42)

    perturbed = filter(g -> g.id > 12, pool)
    @assert !isempty(perturbed) "Should have perturbed grammars in pool"

    w = weights(state.belief)
    gw = aggregate_grammar_weights(w, state.metadata)

    perturbed_with_weight = filter(g -> get(gw, g.id, 0.0) > 0.0, perturbed)
    @assert !isempty(perturbed_with_weight) "Some perturbed grammars should have positive posterior weight"

    seed_channel_counts = Set(length(g.sensor_config.channels) for g in pool if g.id <= 12)
    novel = filter(g -> length(g.sensor_config.channels) ∉ seed_channel_counts || !isempty(g.rules),
                   perturbed)

    println("  Perturbed grammars: $(length(perturbed))")
    println("  With positive weight: $(length(perturbed_with_weight))")
    println("  With novel structure: $(length(novel))")
    for g in perturbed_with_weight[1:min(5, length(perturbed_with_weight))]
        println("  Grammar $(g.id): weight=$(round(get(gw, g.id, 0.0), digits=6)), " *
                "channels=$(length(g.sensor_config.channels)), rules=$(length(g.rules))")
    end

    println("PASSED: Grammar pool evolution — $(length(perturbed)) perturbed, " *
            "$(length(perturbed_with_weight)) with positive weight")
end
println()

println("=" ^ 60)
println("ALL GRID WORLD TESTS PASSED")
println("=" ^ 60)
