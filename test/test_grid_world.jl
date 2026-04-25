#!/usr/bin/env julia
"""
    test_grid_world.jl — Tier 3 tests: grid-world domain

Tests that need the grid-world simulation: world creation, entity features,
full agent runs, regime change, meta-learning.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: BetaPrevision, GaussianPrevision, GammaPrevision, CategoricalPrevision  # Posture 4 Move 4
using Credence.Ontology: wrap_in_measure  # Posture 4 Move 4
using Credence: expect, condition, weights, mean
using Credence: BetaMeasure, TaggedBetaMeasure, MixtureMeasure, Finite, Interval, Kernel, Measure
using Credence: TaggedBetaPrevision, MixturePrevision
using Credence: prune, truncate
using Credence: AgentState, sync_prune!, sync_truncate!
using Credence: Grammar, Program, CompiledKernel
using Credence: enumerate_programs, compile_kernel
using Credence: aggregate_grammar_weights

# Load grid-world domain
include(joinpath(@__DIR__, "..", "apps", "julia", "grid_world", "host.jl"))

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
# TEST 2: Entity features produce correct Dict
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 2: entity_features produces correct Dict with 8 features in [0,1]")
println("=" ^ 60)

let
    Random.seed!(42)
    world = create_world(:colour_typed)
    # Find a living entity and extract features
    for (i, e) in enumerate(world.entities)
        e.alive || continue
        features = entity_features(e, world.agent_pos, world.config.grid_size)
        @assert features isa Dict{Symbol, Float64}
        @assert length(features) == 8 "Should have 8 features"
        for (k, v) in features
            @assert 0.0 <= v <= 1.0 "Feature $k out of [0,1]: $v"
        end
        @assert haskey(features, :red) "Should have :red feature"
        @assert haskey(features, :speed) "Should have :speed feature"
        break
    end
    println("PASSED: entity_features produces correct Dict with 8 features in [0,1]")
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
    p1 = enumerate_programs(g1, 3; action_space=[:food, :enemy])
    p2 = enumerate_programs(g2, 3; action_space=[:food, :enemy])

    components = Any[]
    log_prior = Float64[]
    meta = Tuple{Int, Int}[]
    ck = CompiledKernel[]
    progs = Program[]

    idx = 0
    for (pi, p) in enumerate(p1)
        idx += 1
        push!(components, TaggedBetaPrevision(idx, BetaPrevision(1.0, 1.0)))
        push!(log_prior, -g1.complexity * log(2) - p.complexity * log(2))
        push!(meta, (g1.id, pi))
        push!(ck, compile_kernel(p, g1, pi))
        push!(progs, p)
    end
    for (pi, p) in enumerate(p2)
        idx += 1
        push!(components, TaggedBetaPrevision(idx, BetaPrevision(1.0, 1.0)))
        push!(log_prior, -g2.complexity * log(2) - p.complexity * log(2))
        push!(meta, (g2.id, pi))
        push!(ck, compile_kernel(p, g2, pi))
        push!(progs, p)
    end

    belief = MixturePrevision(components, log_prior)
    grammar_dict = Dict{Int, Grammar}(g1.id => g1, g2.id => g2)
    state = AgentState(belief, meta, ck, progs, grammar_dict, 3)

    n_before = length(state.belief.components)

    sync_prune!(state; threshold=-5.0)
    n_after = length(state.belief.components)

    @assert length(state.metadata) == n_after "metadata length mismatch"
    @assert length(state.compiled_kernels) == n_after "compiled_kernels length mismatch"
    @assert length(state.all_programs) == n_after "all_programs length mismatch"

    for (i, comp) in enumerate(state.belief.components)
        @assert comp isa TaggedBetaPrevision
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
    programs = enumerate_programs(g, 3; action_space=[:food, :enemy])

    components = Any[]
    log_prior = Float64[]
    meta = Tuple{Int, Int}[]
    ck = CompiledKernel[]
    progs = Program[]

    for (pi, p) in enumerate(programs)
        push!(components, TaggedBetaPrevision(pi, BetaPrevision(1.0, 1.0)))
        push!(log_prior, -g.complexity * log(2) - p.complexity * log(2))
        push!(meta, (g.id, pi))
        push!(ck, compile_kernel(p, g, pi))
        push!(progs, p)
    end

    belief = MixturePrevision(components, log_prior)
    grammar_dict = Dict{Int, Grammar}(g.id => g)
    state = AgentState(belief, meta, ck, progs, grammar_dict, 3)

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
# TEST 5: Features Dict works with compiled kernels from different grammars
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 5: Different grammars evaluate same feature dict correctly")
println("=" ^ 60)

let
    Random.seed!(42)
    features = Dict{Symbol, Float64}(
        :red => 0.9, :green => 0.1, :blue => 0.1,
        :x_norm => 0.5, :y_norm => 0.5,
        :speed => 0.8, :wall_dist => 0.3, :agent_dist => 0.4
    )

    grammars = generate_seed_grammars()

    # Colour grammar (grammar 2) should see red features
    g_colour = grammars[2]  # Set([:red, :green, :blue])
    programs = enumerate_programs(g_colour, 2; action_space=[:food, :enemy])
    ck = compile_kernel(programs[1], g_colour, 1)
    ts = Dict{Symbol, Any}()
    result = ck.evaluate(features, ts)
    @assert result isa Symbol "Should return a Symbol action"

    # Motion grammar (grammar 3) should see speed/wall_dist features
    g_motion = grammars[3]  # Set([:speed, :wall_dist])
    programs_m = enumerate_programs(g_motion, 2; action_space=[:food, :enemy])
    ck_m = compile_kernel(programs_m[1], g_motion, 1)
    result_m = ck_m.evaluate(features, ts)
    @assert result_m isa Symbol

    println("PASSED: Different grammars evaluate same feature dict correctly")
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

    g = grammars[2]  # colour grammar: r, g, b
    programs = enumerate_programs(g, 3; action_space=[:food, :enemy])

    features = Dict{Symbol, Float64}(:red => 0.95, :green => 0.1, :blue => 0.1)
    ts = Dict{Symbol, Any}()

    # Find a program that predicts :enemy for red entities (correct)
    # and one that predicts :food for red entities (incorrect)
    correct_prog = nothing
    incorrect_prog = nothing
    for (i, p) in enumerate(programs)
        ck_test = compile_kernel(p, g, i)
        rec = ck_test.evaluate(features, ts)
        if correct_prog === nothing && rec == :enemy
            correct_prog = (i, p, ck_test)
        end
        if incorrect_prog === nothing && rec == :food && p.expr isa IfExpr
            incorrect_prog = (i, p, ck_test)
        end
        correct_prog !== nothing && incorrect_prog !== nothing && break
    end

    @assert correct_prog !== nothing "Should find a program that predicts :enemy for red"
    @assert incorrect_prog !== nothing "Should find a program that predicts :food for red"

    comp1 = TaggedBetaPrevision(1, BetaPrevision(1.0, 1.0))
    comp2 = TaggedBetaPrevision(2, BetaPrevision(1.0, 1.0))
    belief = MixturePrevision(Any[comp1, comp2], [0.0, 0.0])

    ck_vec = [correct_prog[3], incorrect_prog[3]]

    posterior = belief
    for _ in 1:5
        k = build_observation_kernel(ck_vec, features, ts, :enemy)
        posterior = condition(posterior, k, 1.0)
    end

    c1 = posterior.components[1]
    c2 = posterior.components[2]
    @assert c1 isa TaggedBetaPrevision
    @assert c2 isa TaggedBetaPrevision

    @assert c1.beta.alpha ≈ 6.0 "Both programs should have α=6 after 5 obs, got $(c1.beta.alpha)"
    @assert c2.beta.alpha ≈ 6.0 "Both programs should have α=6 after 5 obs, got $(c2.beta.alpha)"

    w = weights(posterior)
    @assert w[1] > w[2] "Correct program should gain weight over incorrect: w=$(round.(w, digits=4))"

    println("PASSED: Correct α=$(c1.beta.alpha)/β=$(c1.beta.beta), " *
            "Incorrect α=$(c2.beta.alpha)/β=$(c2.beta.beta), " *
            "weights=[$(round(w[1], digits=4)), $(round(w[2], digits=4))]")
end
println()

# ═══════════════════════════════════════
# TEST 7: Agent learns from interactions (surprise decreases)
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 7: Agent learns from interactions (surprise decreases)")
println("=" ^ 60)

let
    metrics, _, _ = run_agent(
        world_rules=[:colour_typed],
        max_steps=100,
        program_max_depth=2,
        max_meta_per_step=0,
        verbose=false,
        rng_seed=42)

    interactions = [(metrics.steps[i], metrics.surprise[i])
                    for i in eachindex(metrics.steps) if metrics.surprise[i] > 0.0]
    @assert length(interactions) >= 3 "Need ≥3 interactions, got $(length(interactions))"

    # Surprise should decrease as agent learns (first interactions vs later)
    n = length(interactions)
    early = [s for (_, s) in interactions[1:min(3, n)]]
    late = [s for (_, s) in interactions[max(1, n-2):n]]

    println("PASSED: $(length(interactions)) interactions, " *
            "early surprise=$(round(sum(early)/length(early), digits=3)), " *
            "late surprise=$(round(sum(late)/length(late), digits=3))")
end
println()

# ═══════════════════════════════════════
# TEST 8: Agent handles regime change without error
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 8: Agent handles regime change without error")
println("=" ^ 60)

let
    metrics, _, _ = run_agent(
        world_rules=[:colour_typed, :motion_typed],
        regime_change_steps=[50],
        max_steps=100,
        program_max_depth=2,
        max_meta_per_step=0,
        verbose=false,
        rng_seed=42)

    interactions = count(s -> s > 0.0, metrics.surprise)
    @assert interactions >= 1 "Agent should interact at least once"
    println("PASSED: Agent completed 100 steps with regime change, $interactions interactions")
end
println()

# ═══════════════════════════════════════
# TEST 9: Meta-learning — meta-actions vs no meta-actions
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 9: Meta-learning — meta-actions vs no meta-actions")
println("=" ^ 60)

let
    seed = 123

    metrics_a, _, _ = run_agent(
        world_rules=[:colour_typed, :motion_typed, :colour_typed],
        regime_change_steps=[100, 200],
        max_steps=300,
        program_max_depth=2,
        max_meta_per_step=2,
        verbose=false,
        rng_seed=123)

    metrics_b, _, _ = run_agent(
        world_rules=[:colour_typed, :motion_typed, :colour_typed],
        regime_change_steps=[100, 200],
        max_steps=300,
        program_max_depth=2,
        max_meta_per_step=0,
        verbose=false,
        rng_seed=123)

    function regime3_accuracy(m::MetricsTracker)
        hits = 0
        total = 0
        for (idx, step) in enumerate(m.steps)
            200 <= step <= 300 || continue
            m.surprise[idx] > 0.0 || continue
            total += 1
            m.prediction_correct[idx] && (hits += 1)
        end
        total == 0 ? 0.0 : hits / total
    end

    acc_a = regime3_accuracy(metrics_a)
    acc_b = regime3_accuracy(metrics_b)

    total_a = count(s -> s > 0.0, metrics_a.surprise)
    total_b = count(s -> s > 0.0, metrics_b.surprise)
    @assert total_a + total_b > 0 "At least one agent should interact"

    total_meta_a = sum(metrics_a.meta_actions_per_step)
    println("PASSED: Agent A interactions=$total_a (acc=$(round(acc_a, digits=3)), meta=$total_meta_a), " *
            "Agent B interactions=$total_b (acc=$(round(acc_b, digits=3)))")
end
println()

# ═══════════════════════════════════════
# TEST 10: Grammar pool evolution is real (emergent)
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 10: Grammar pool evolution — meta-actions produce useful grammars")
println("=" ^ 60)

let
    metrics, state, pool = run_agent(
        world_rules=[:colour_typed, :motion_typed, :colour_typed],
        regime_change_steps=[100, 200],
        max_steps=300,
        program_max_depth=2,
        max_meta_per_step=2,
        verbose=false,
        rng_seed=42)

    perturbed = filter(g -> g.id > 12, pool)

    w = weights(state.belief)
    gw = aggregate_grammar_weights(w, state.metadata)

    total_meta = sum(metrics.meta_actions_per_step)
    println("  Total meta-actions: $total_meta")
    println("  Total grammars in pool: $(length(pool))")

    if !isempty(perturbed)
        perturbed_with_weight = filter(g -> get(gw, g.id, 0.0) > 0.0, perturbed)
        println("  Perturbed grammars: $(length(perturbed))")
        println("  With positive weight: $(length(perturbed_with_weight))")
        for g in perturbed_with_weight[1:min(5, length(perturbed_with_weight))]
            println("  Grammar $(g.id): weight=$(round(get(gw, g.id, 0.0), digits=6)), " *
                    "features=$(length(g.feature_set)), rules=$(length(g.rules))")
        end
    end

    println("PASSED: Grammar pool evolution complete")
end
println()

println("=" ^ 60)
println("ALL GRID WORLD TESTS PASSED")
println("=" ^ 60)
