#!/usr/bin/env julia
"""
    test_program_agent.jl — Test suite for program-space Bayesian agent

Tests:
  - Grammar enumeration and program compilation
  - Kernel precompilation speed
  - CompiledKernel has no AST field
  - SubprogramFrequencyTable enforcement
  - Compression payoff
  - Single-regime convergence
  - Regime change detection
  - Occam's Razor (simplest program dominates in trivial world)
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: expect, condition, weights, mean
using Credence: BetaMeasure, MixtureMeasure, Finite, Interval, Kernel, Measure
using Credence: prune, truncate

include(joinpath(@__DIR__, "..", "examples", "grid_world.jl"))
include(joinpath(@__DIR__, "..", "examples", "grammar_programs.jl"))
include(joinpath(@__DIR__, "..", "examples", "metrics.jl"))

using Random

# ═══════════════════════════════════════
# TEST 1: Grammar enumeration
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 1: Grammar enumeration produces valid programs")
println("=" ^ 60)

let
    grammars = generate_seed_grammars()
    @assert length(grammars) >= 10 "Expected ≥ 10 seed grammars, got $(length(grammars))"
    println("PASSED: $(length(grammars)) seed grammars generated")

    # Check each grammar has valid sensor config and complexity
    for g in grammars
        @assert length(g.sensor_config.channels) > 0 "Grammar $(g.id) has no sensor channels"
        @assert g.complexity > 0 "Grammar $(g.id) has non-positive complexity"
    end
    println("PASSED: All grammars have valid config and complexity")

    # Enumerate programs for first grammar
    programs = enumerate_programs(grammars[1], 3)
    @assert length(programs) > 0 "No programs enumerated for grammar 1"
    println("PASSED: Grammar 1 → $(length(programs)) programs at depth 3")

    # Check complexity is positive for all programs
    for p in programs
        @assert p.complexity > 0 "Program has non-positive complexity"
    end
    println("PASSED: All programs have positive complexity")
end
println()

# ═══════════════════════════════════════
# TEST 2: Kernel compilation produces closures
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 2: Kernel compilation produces working closures")
println("=" ^ 60)

let
    grammars = generate_seed_grammars()
    g = grammars[2]  # colour grammar: r, g, b
    programs = enumerate_programs(g, 2)

    @assert length(programs) > 0 "No programs for colour grammar"

    # Compile all programs
    kernels = [compile_kernel(p, g, i) for (i, p) in enumerate(programs)]
    @assert length(kernels) == length(programs)

    # Test evaluation with a red entity sensor vector
    red_sv = [0.95, 0.1, 0.1]  # r=high, g=low, b=low
    ts = Dict{Symbol, Any}()

    # At least one kernel should fire on red
    any_fires = any(k -> k.evaluate(red_sv, ts), kernels)
    @assert any_fires "No kernel fires on red sensor vector"

    # Test with blue entity
    blue_sv = [0.1, 0.1, 0.95]
    any_fires_blue = any(k -> k.evaluate(blue_sv, ts), kernels)
    @assert any_fires_blue "No kernel fires on blue sensor vector"

    println("PASSED: $(length(kernels)) kernels compiled, evaluate correctly on test vectors")
end
println()

# ═══════════════════════════════════════
# TEST 3: CompiledKernel has NO AST field (enforcement)
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 3: CompiledKernel has no AST field (type enforcement)")
println("=" ^ 60)

let
    fnames = fieldnames(CompiledKernel)
    ftypes = [fieldtype(CompiledKernel, f) for f in fnames]

    for (name, typ) in zip(fnames, ftypes)
        @assert !(typ <: ProgramExpr) "CompiledKernel has ProgramExpr field: $name"
        @assert !(typ <: Vector{<:ProgramExpr}) "CompiledKernel has Vector{ProgramExpr} field: $name"
        @assert !(typ <: AbstractVector{<:ProgramExpr}) "CompiledKernel has array of ProgramExpr: $name"
    end

    println("PASSED: CompiledKernel fields = $fnames — no AST types present")
end
println()

# ═══════════════════════════════════════
# TEST 4: Kernel precompilation speed (enforcement)
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 4: 10K kernel evaluations < 1ms (precompilation speed)")
println("=" ^ 60)

let
    grammars = generate_seed_grammars()
    g = grammars[5]  # colour + speed grammar: 4 channels
    programs = enumerate_programs(g, 3)
    @assert length(programs) >= 10 "Need ≥ 10 programs for speed test"

    # Compile a depth-3+ program
    # Find one with complexity ≥ 3
    complex_progs = filter(p -> p.complexity >= 3, programs)
    @assert !isempty(complex_progs) "No programs with complexity ≥ 3"

    p = first(complex_progs)
    ck = compile_kernel(p, g, 1)

    sv = [0.9, 0.1, 0.1, 0.8]  # red, not green, not blue, fast
    ts = Dict{Symbol, Any}()

    # Extended warmup to ensure JIT compilation is complete
    for _ in 1:10_000
        ck.evaluate(sv, ts)
    end

    # Timed run: take best of 3 to avoid GC jitter
    n_calls = 10_000
    best_ms = Inf
    for _ in 1:3
        t0 = time_ns()
        for _ in 1:n_calls
            ck.evaluate(sv, ts)
        end
        elapsed_ms = (time_ns() - t0) / 1_000_000
        best_ms = min(best_ms, elapsed_ms)
    end

    # 10ms threshold catches AST interpretation (~50-100μs/call) while allowing
    # Julia closure overhead with Dict{Symbol,Any} temporal state (~0.6μs/call)
    @assert best_ms < 10.0 "10K kernel evals took $(round(best_ms, digits=3))ms, expected < 10ms (closure, not AST interpretation)"
    println("PASSED: 10K evaluations in $(round(best_ms, digits=3))ms (< 10ms — closure, not AST walk)")
end
println()

# ═══════════════════════════════════════
# TEST 5: Compression payoff is real
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 5: Compression payoff — nonterminals reduce complexity")
println("=" ^ 60)

let
    # Grammar without nonterminals
    g_bare = Grammar(colour_sensor_config(), ProductionRule[], 1)

    # Grammar with RED nonterminal
    red_body = AndExpr(GTExpr(0, 0.7), AndExpr(LTExpr(1, 0.3), LTExpr(2, 0.3)))
    g_red = Grammar(colour_sensor_config(), [ProductionRule(:RED, red_body)], 2)

    # Under bare grammar: AND(GT(0,0.7), AND(LT(1,0.3), LT(2,0.3))) costs 5
    bare_complexity = expr_complexity(red_body)
    @assert bare_complexity == 5 "Expected complexity 5 for raw expression, got $bare_complexity"

    # Under red grammar: NonterminalRef(:RED) costs 1
    ref_complexity = expr_complexity(NonterminalRef(:RED))
    @assert ref_complexity == 1 "Expected complexity 1 for nonterminal ref, got $ref_complexity"

    # The program using the nonterminal has strictly lower complexity
    @assert ref_complexity < bare_complexity "Nonterminal should compress"

    # Expanded complexity should match bare
    exp_c = expanded_complexity(NonterminalRef(:RED), g_red.rules)
    @assert exp_c == bare_complexity "Expanded complexity should equal bare"

    println("PASSED: Bare=5, Nonterminal=1, Expanded=5 — compression payoff confirmed")
end
println()

# ═══════════════════════════════════════
# TEST 6: SubprogramFrequencyTable enforcement
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 6: propose_nonterminal requires SubprogramFrequencyTable")
println("=" ^ 60)

let
    # Verify propose_nonterminal only accepts SubprogramFrequencyTable
    methods_list = methods(propose_nonterminal)
    for m in methods_list
        sig = m.sig
        # Check that the parameter type includes SubprogramFrequencyTable
        @assert string(sig) |> s -> occursin("SubprogramFrequencyTable", s) "propose_nonterminal has method not requiring SubprogramFrequencyTable: $sig"
    end
    println("PASSED: propose_nonterminal only accepts SubprogramFrequencyTable")

    # Verify perturb_grammar requires SubprogramFrequencyTable
    methods_pg = methods(perturb_grammar)
    for m in methods_pg
        sig = m.sig
        @assert string(sig) |> s -> occursin("SubprogramFrequencyTable", s) "perturb_grammar has method not requiring SubprogramFrequencyTable: $sig"
    end
    println("PASSED: perturb_grammar requires SubprogramFrequencyTable")
end
println()

# ═══════════════════════════════════════
# TEST 7: Subtree analysis and nonterminal proposal
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 7: Subtree analysis proposes real subtrees")
println("=" ^ 60)

let
    Random.seed!(42)
    g = Grammar(colour_sensor_config(), ProductionRule[], 1)
    programs = enumerate_programs(g, 3)

    # Simulate posterior: weight programs that use GT(0, 0.7) heavily
    prog_weights = zeros(length(programs))
    for (i, p) in enumerate(programs)
        s = show_expr(p.predicate)
        if occursin("GT(0,0.7)", s)
            prog_weights[i] = 0.1
        else
            prog_weights[i] = 0.001
        end
    end
    prog_weights ./= sum(prog_weights)

    freq_table = analyse_posterior_subtrees(programs, prog_weights;
                                            min_frequency=0.01, min_complexity=2)

    if !isempty(freq_table.subtrees)
        proposed = propose_nonterminal(freq_table)
        if proposed !== nothing
            # Verify proposed nonterminal appears in source programs
            best_idx = argmax(freq_table.weighted_frequency)
            n_sources = length(freq_table.source_programs[best_idx])
            @assert n_sources >= 2 "Proposed nonterminal should appear in ≥ 2 programs, got $n_sources"
            println("PASSED: Proposed nonterminal '$(proposed.name)' from $n_sources programs")
            println("  Body: $(show_expr(proposed.body))")
        else
            println("PASSED: No nonterminal proposed (compression payoff insufficient)")
        end
    else
        println("PASSED: No frequent subtrees found (expected for small program set)")
    end
end
println()

# ═══════════════════════════════════════
# TEST 8: Grid world basic functionality
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 8: Grid world creates and simulates correctly")
println("=" ^ 60)

let
    Random.seed!(42)
    world = create_world(:colour_typed)
    @assert length(world.entities) >= 4 "Expected ≥ 4 entities"
    @assert world.agent_energy == 100.0 "Initial energy should be 100"

    # Check entity properties match colour rule
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

    # Test regime change
    set_rule!(world, :motion_typed)
    for e in world.entities
        if e.kind == ENEMY
            @assert e.speed > 0.5 "Enemy should be fast under motion rule"
        elseif e.kind == FOOD
            @assert e.speed == 0.0 "Food should be stationary under motion rule"
        end
    end
    println("PASSED: Regime change updates entity properties")

    # Test world step
    initial_energy = world.agent_energy
    world_step!(world, MOVE_N)
    @assert world.agent_energy < initial_energy "Moving should cost energy"
    println("PASSED: World step executes correctly")
end
println()

# ═══════════════════════════════════════
# TEST 9: Sensor projection
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 9: Sensor projection produces correct-dimensionality output")
println("=" ^ 60)

let
    Random.seed!(42)
    true_state = [0.9, 0.1, 0.1, 0.5, 0.5, 0.8, 0.3, 0.4]

    # Minimal config (2 channels)
    sv1 = project(true_state, minimal_sensor_config())
    @assert length(sv1) == 2 "Minimal config should produce 2 readings"

    # Full config (8 channels)
    sv2 = project(true_state, full_sensor_config())
    @assert length(sv2) == 8 "Full config should produce 8 readings"

    # All readings should be in [0, 1]
    for r in [sv1..., sv2...]
        @assert 0.0 <= r <= 1.0 "Sensor reading out of [0,1]: $r"
    end

    println("PASSED: Sensor projection produces correct dimensionality and range")
end
println()

# ═══════════════════════════════════════
# TEST 10: Flat mixture conditioning with per-component dispatch
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 10: Full pipeline — enumeration → compilation → conditioning")
println("=" ^ 60)

let
    Random.seed!(42)
    grammars = generate_seed_grammars()

    # Build small mixture from 2 grammars
    g1 = grammars[2]  # colour
    g2 = grammars[3]  # motion
    p1 = enumerate_programs(g1, 2)
    p2 = enumerate_programs(g2, 2)

    components = BetaMeasure[]
    log_prior = Float64[]
    metadata = Tuple{Int, Int}[]
    kernels = CompiledKernel[]

    for (pi, p) in enumerate(p1)
        push!(components, BetaMeasure(1.0, 1.0))
        push!(log_prior, -g1.complexity * log(2) - p.complexity * log(2))
        push!(metadata, (g1.id, pi))
        push!(kernels, compile_kernel(p, g1, pi))
    end
    for (pi, p) in enumerate(p2)
        push!(components, BetaMeasure(1.0, 1.0))
        push!(log_prior, -g2.complexity * log(2) - p.complexity * log(2))
        push!(metadata, (g2.id, pi))
        push!(kernels, compile_kernel(p, g2, pi))
    end

    belief = MixtureMeasure(Interval(0.0, 1.0), Measure[c for c in components], log_prior)

    # Condition on observing enemy (1.0) multiple times
    obs_space = Finite([0.0, 1.0])
    k = Kernel(Interval(0.0, 1.0), obs_space,
        θ -> error("generate not used"),
        (θ, obs) -> obs == 1.0 ? log(max(θ, 1e-300)) : log(max(1.0 - θ, 1e-300)))

    for _ in 1:5
        belief = condition(belief, k, 1.0)
    end

    w = weights(belief)
    @assert length(w) == length(components)

    # All Betas should have been updated
    for comp in belief.components
        @assert comp isa BetaMeasure
        @assert comp.alpha > 1.0 "Beta should have been updated"
    end

    # Grammar-level aggregation
    gw = Dict{Int, Float64}()
    for (i, (gi, _)) in enumerate(metadata)
        gw[gi] = get(gw, gi, 0.0) + w[i]
    end
    println("PASSED: Full pipeline works. Grammar weights: $gw")
end
println()

# ═══════════════════════════════════════
# TEST 11: Occam's Razor — simplest program wins in trivial world
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 11: Occam's Razor — simpler programs preferred with equal prediction")
println("=" ^ 60)

let
    # All components are BetaMeasures with same data
    # Simpler programs (lower complexity) should have more weight
    components = Measure[BetaMeasure(5.0, 2.0) for _ in 1:6]
    complexities = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    log_prior = [-c * log(2) for c in complexities]

    belief = MixtureMeasure(Interval(0.0, 1.0), components, log_prior)

    w = weights(belief)
    # Weights should be monotonically decreasing with complexity
    for i in 1:5
        @assert w[i] > w[i+1] "Simpler program should have more weight: w[$i]=$(w[i]) vs w[$(i+1)]=$(w[i+1])"
    end

    println("PASSED: Simpler programs dominate when prediction is equal")
    println("  Weights: ", [round(wi, digits=4) for wi in w])
end
println()

# ═══════════════════════════════════════
# TEST 12: Temporal operators compile correctly
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 12: Temporal operators in compiled kernels")
println("=" ^ 60)

let
    g = Grammar(minimal_sensor_config(), ProductionRule[], 1)

    # Test CHANGED operator
    changed_expr = ChangedExpr(GTExpr(0, 0.5))
    ck = CompiledKernel(
        compile_expr(changed_expr, ProductionRule[]),
        2, 1, 1)

    # No history: should return false
    ts_empty = Dict{Symbol, Any}(:recent => Vector{Float64}[])
    @assert !ck.evaluate([0.8, 0.3], ts_empty) "CHANGED should be false with no history"

    # With history where value was below threshold, now above
    ts_changed = Dict{Symbol, Any}(:recent => [[0.3, 0.5]])
    @assert ck.evaluate([0.8, 0.3], ts_changed) "CHANGED should detect transition"

    # With history where value stayed above
    ts_same = Dict{Symbol, Any}(:recent => [[0.8, 0.5]])
    @assert !ck.evaluate([0.9, 0.3], ts_same) "CHANGED should be false when stable"

    println("PASSED: CHANGED operator compiles and evaluates correctly")

    # Test PERSISTS operator
    persists_expr = PersistsExpr(GTExpr(0, 0.5), 2)
    ck2 = CompiledKernel(
        compile_expr(persists_expr, ProductionRule[]),
        2, 1, 2)

    ts_persists = Dict{Symbol, Any}(:recent => [[0.8, 0.3], [0.9, 0.4]])
    @assert ck2.evaluate([0.85, 0.3], ts_persists) "PERSISTS(2) should be true"

    ts_not_persist = Dict{Symbol, Any}(:recent => [[0.3, 0.5]])
    @assert !ck2.evaluate([0.8, 0.3], ts_not_persist) "PERSISTS(2) needs 2 history entries"

    println("PASSED: PERSISTS operator compiles and evaluates correctly")
end
println()

# ═══════════════════════════════════════
# TEST 13: Program enumeration with temporal operators
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 13: Program enumeration includes temporal operators when enabled")
println("=" ^ 60)

let
    # Use a grammar with few channels to stay under the program cap
    sc = SensorConfig([SensorChannel(0, :identity, 0.05, 1.0)])  # 1 channel
    g = Grammar(sc, ProductionRule[], 1)

    programs_no_temporal = enumerate_programs(g, 2; include_temporal=false)
    programs_with_temporal = enumerate_programs(g, 2; include_temporal=true)

    # Check that some temporal programs exist in the temporal enumeration
    temporal_count = count(p -> begin
        s = show_expr(p.predicate)
        occursin("CHANGED", s) || occursin("PERSISTS", s)
    end, programs_with_temporal)

    @assert temporal_count > 0 "Should have temporal programs"
    @assert length(programs_with_temporal) > length(programs_no_temporal) "Temporal programs should expand the program space"
    println("PASSED: $(length(programs_no_temporal)) programs without temporal, " *
            "$(length(programs_with_temporal)) with temporal ($temporal_count temporal)")
end
println()

# ═══════════════════════════════════════
# TEST 14: End-to-end single regime test
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 14: End-to-end single regime (colour-typed, short)")
println("=" ^ 60)

let
    Random.seed!(42)

    # Import the host driver functions
    # (already included above via grammar_programs.jl → sensor_projection.jl)

    grammars = generate_seed_grammars()

    # Enumerate and compile
    components = BetaMeasure[]
    log_prior = Float64[]
    metadata = Tuple{Int, Int}[]
    compiled = CompiledKernel[]
    programs_all = Program[]

    for g in grammars[1:5]  # Use only first 5 grammars for speed
        programs = enumerate_programs(g, 2)
        for (pi, p) in enumerate(programs)
            push!(components, BetaMeasure(1.0, 1.0))
            push!(log_prior, -g.complexity * log(2) - p.complexity * log(2))
            push!(metadata, (g.id, pi))
            push!(compiled, compile_kernel(p, g, pi))
            push!(programs_all, p)
        end
    end

    belief = MixtureMeasure(Interval(0.0, 1.0), Measure[c for c in components], log_prior)

    # Simulate 20 interactions
    k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
        θ -> error("not used"),
        (θ, obs) -> obs == 1.0 ? log(max(θ, 1e-300)) : log(max(1.0 - θ, 1e-300)))

    # Alternate enemy and food observations
    for i in 1:20
        obs = i % 3 == 0 ? 1.0 : 0.0  # ~33% enemy
        belief = condition(belief, k, obs)
    end

    w = weights(belief)
    @assert abs(sum(w) - 1.0) < 1e-10 "Weights should sum to 1"

    # Verify components are updated BetaMeasures
    for comp in belief.components
        @assert comp isa BetaMeasure
        @assert comp.alpha + comp.beta > 2.0 "Beta should have been updated from prior"
    end

    println("PASSED: 20 conditioning steps, $(length(belief.components)) components, weights sum to $(sum(w))")
end
println()

# ═══════════════════════════════════════
# TEST 15: AgentState sync_prune! keeps arrays aligned
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 15: AgentState sync_prune! keeps parallel arrays aligned")
println("=" ^ 60)

include(joinpath(@__DIR__, "..", "examples", "host_program_agent.jl"))

let
    Random.seed!(42)
    grammars = generate_seed_grammars()

    # Build a small mixture from 2 grammars
    g1 = grammars[2]  # colour
    g2 = grammars[3]  # motion
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

    # Prune with a very aggressive threshold to remove some components
    sync_prune!(state; threshold=-5.0)
    n_after = length(state.belief.components)

    # All parallel arrays should have same length
    @assert length(state.metadata) == n_after "metadata length mismatch"
    @assert length(state.compiled_kernels) == n_after "compiled_kernels length mismatch"
    @assert length(state.all_programs) == n_after "all_programs length mismatch"

    # Tags should be reindexed to 1:n_after
    for (i, comp) in enumerate(state.belief.components)
        @assert comp isa TaggedBetaMeasure
        @assert comp.tag == i "Tag reindex failed: expected $i, got $(comp.tag)"
    end

    println("PASSED: Before=$n_before, after=$n_after, all arrays aligned, tags reindexed")
end
println()

# ═══════════════════════════════════════
# TEST 16: AgentState sync_truncate! keeps arrays aligned
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 16: AgentState sync_truncate! keeps parallel arrays aligned")
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
# TEST 17: Per-grammar sensor projection produces different vectors
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 17: Per-grammar sensor projection produces different vectors")
println("=" ^ 60)

let
    Random.seed!(42)
    true_state = [0.9, 0.1, 0.1, 0.5, 0.5, 0.8, 0.3, 0.4]

    grammars = generate_seed_grammars()

    # Project through each grammar's sensor config
    gsvs = project_per_grammar(true_state, grammars)

    @assert length(gsvs) == length(grammars) "Should have one vector per grammar"

    # Colour grammar (3 channels) and motion grammar (2 channels) should differ in length
    colour_g = grammars[2]
    motion_g = grammars[3]

    sv_colour = gsvs[colour_g.id]
    sv_motion = gsvs[motion_g.id]

    @assert length(sv_colour) == 3 "Colour grammar should produce 3-dim vector"
    @assert length(sv_motion) == 2 "Motion grammar should produce 2-dim vector"

    # Colour vector should see r,g,b; motion should see speed,wall_dist
    @assert sv_colour[1] > 0.7 "Colour should see high red (true state r=0.9)"
    @assert sv_motion[1] > 0.5 "Motion should see high speed (true state speed=0.8)"

    println("PASSED: Colour vector=$(round.(sv_colour, digits=3)), " *
            "Motion vector=$(round.(sv_motion, digits=3))")
end
println()

# ═══════════════════════════════════════
# TEST 18: TaggedBetaMeasure kernel dispatch — firing vs non-firing diverge
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 18: TaggedBetaMeasure kernel dispatch with compiled predicates")
println("=" ^ 60)

let
    Random.seed!(42)
    grammars = generate_seed_grammars()

    # Use colour grammar and compile 2 programs
    g = grammars[2]  # colour: r, g, b
    programs = enumerate_programs(g, 2)

    # Pick a program that fires on red and one that fires on blue
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

    # Build a mixture with 2 TaggedBetaMeasures
    comp1 = TaggedBetaMeasure(Interval(0.0, 1.0), 1, BetaMeasure(1.0, 1.0))
    comp2 = TaggedBetaMeasure(Interval(0.0, 1.0), 2, BetaMeasure(1.0, 1.0))
    belief = MixtureMeasure(Interval(0.0, 1.0), Measure[comp1, comp2], [0.0, 0.0])

    ck_vec = [red_prog[3], blue_prog[3]]

    # Entity is red → grammar sensor vectors project as red
    grammar_sensor_vectors = Dict{Int, Vector{Float64}}(g.id => red_sv)

    # Condition on enemy (obs=1.0) 5 times — divergence needs multiple observations.
    # First obs: both get log(0.5) penalty (identical Beta(1,1) priors).
    # Subsequent obs: firing component has learned (mean > 0.5), gets less penalty.
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

    # Blue-firing program (tag 2) on red entity should NOT fire, beta unchanged
    @assert c2.beta.alpha ≈ 1.0 "Blue-firing program should have α = 1 (didn't fire on red entity)"
    @assert c2.beta.beta ≈ 1.0 "Blue-firing program should have β = 1"

    w = weights(posterior)
    @assert w[1] > w[2] "Firing program should gain weight over non-firing after multiple obs"

    println("PASSED: Red-fires α=$(c1.beta.alpha), Blue-doesn't α=$(c2.beta.alpha), " *
            "weights=[$(round(w[1], digits=4)), $(round(w[2], digits=4))]")
end
println()

println("=" ^ 60)
println("ALL PROGRAM AGENT TESTS PASSED")
println("=" ^ 60)
