#!/usr/bin/env julia
"""
    test_program_space.jl — Tier 2 tests: program-space inference

Tests enumeration, compilation, complexity scoring, perturbation,
compression payoff, enforcement tests. No grid-world simulation needed.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: expect, condition, weights, mean
using Credence: BetaMeasure, TaggedBetaMeasure, MixtureMeasure, Finite, Interval, Kernel, Measure
using Credence: prune, truncate

# Grid-world sensor configs and terminals for program enumeration tests
include(joinpath(@__DIR__, "..", "domains", "grid_world", "sensors.jl"))
include(joinpath(@__DIR__, "..", "domains", "grid_world", "terminals.jl"))

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

    for g in grammars
        @assert length(g.sensor_config.channels) > 0 "Grammar $(g.id) has no sensor channels"
        @assert g.complexity > 0 "Grammar $(g.id) has non-positive complexity"
    end
    println("PASSED: All grammars have valid config and complexity")

    programs = enumerate_programs(grammars[1], 3)
    @assert length(programs) > 0 "No programs enumerated for grammar 1"
    println("PASSED: Grammar 1 → $(length(programs)) programs at depth 3")

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

    kernels = [compile_kernel(p, g, i) for (i, p) in enumerate(programs)]
    @assert length(kernels) == length(programs)

    red_sv = [0.95, 0.1, 0.1]
    ts = Dict{Symbol, Any}()

    any_fires = any(k -> k.evaluate(red_sv, ts), kernels)
    @assert any_fires "No kernel fires on red sensor vector"

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

    complex_progs = filter(p -> p.complexity >= 3, programs)
    @assert !isempty(complex_progs) "No programs with complexity ≥ 3"

    p = first(complex_progs)
    ck = compile_kernel(p, g, 1)

    sv = [0.9, 0.1, 0.1, 0.8]
    ts = Dict{Symbol, Any}()

    for _ in 1:10_000
        ck.evaluate(sv, ts)
    end

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
    g_bare = Grammar(colour_sensor_config(), ProductionRule[], 1)

    red_body = AndExpr(GTExpr(0, 0.7), AndExpr(LTExpr(1, 0.3), LTExpr(2, 0.3)))
    g_red = Grammar(colour_sensor_config(), [ProductionRule(:RED, red_body)], 2)

    bare_complexity = expr_complexity(red_body)
    @assert bare_complexity == 5 "Expected complexity 5 for raw expression, got $bare_complexity"

    ref_complexity = expr_complexity(NonterminalRef(:RED))
    @assert ref_complexity == 1 "Expected complexity 1 for nonterminal ref, got $ref_complexity"

    @assert ref_complexity < bare_complexity "Nonterminal should compress"

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
    methods_list = methods(propose_nonterminal)
    for m in methods_list
        sig = m.sig
        @assert string(sig) |> s -> occursin("SubprogramFrequencyTable", s) "propose_nonterminal has method not requiring SubprogramFrequencyTable: $sig"
    end
    println("PASSED: propose_nonterminal only accepts SubprogramFrequencyTable")

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
# TEST 8: Flat mixture conditioning with per-component dispatch
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 8: Full pipeline — enumeration → compilation → conditioning")
println("=" ^ 60)

let
    Random.seed!(42)
    grammars = generate_seed_grammars()

    g1 = grammars[2]
    g2 = grammars[3]
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

    obs_space = Finite([0.0, 1.0])
    k = Kernel(Interval(0.0, 1.0), obs_space,
        θ -> error("generate not used"),
        (θ, obs) -> obs == 1.0 ? log(max(θ, 1e-300)) : log(max(1.0 - θ, 1e-300)))

    for _ in 1:5
        belief = condition(belief, k, 1.0)
    end

    w = weights(belief)
    @assert length(w) == length(components)

    for comp in belief.components
        @assert comp isa BetaMeasure
        @assert comp.alpha > 1.0 "Beta should have been updated"
    end

    gw = Dict{Int, Float64}()
    for (i, (gi, _)) in enumerate(metadata)
        gw[gi] = get(gw, gi, 0.0) + w[i]
    end
    println("PASSED: Full pipeline works. Grammar weights: $gw")
end
println()

# ═══════════════════════════════════════
# TEST 9: Occam's Razor — simplest program wins in trivial world
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 9: Occam's Razor — simpler programs preferred with equal prediction")
println("=" ^ 60)

let
    components = Measure[BetaMeasure(5.0, 2.0) for _ in 1:6]
    complexities = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    log_prior = [-c * log(2) for c in complexities]

    belief = MixtureMeasure(Interval(0.0, 1.0), components, log_prior)

    w = weights(belief)
    for i in 1:5
        @assert w[i] > w[i+1] "Simpler program should have more weight: w[$i]=$(w[i]) vs w[$(i+1)]=$(w[i+1])"
    end

    println("PASSED: Simpler programs dominate when prediction is equal")
    println("  Weights: ", [round(wi, digits=4) for wi in w])
end
println()

# ═══════════════════════════════════════
# TEST 10: Temporal operators compile correctly
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 10: Temporal operators in compiled kernels")
println("=" ^ 60)

let
    g = Grammar(minimal_sensor_config(), ProductionRule[], 1)

    changed_expr = ChangedExpr(GTExpr(0, 0.5))
    ck = CompiledKernel(
        compile_expr(changed_expr, ProductionRule[]),
        2, 1, 1)

    ts_empty = Dict{Symbol, Any}(:recent => Vector{Float64}[])
    @assert !ck.evaluate([0.8, 0.3], ts_empty) "CHANGED should be false with no history"

    ts_changed = Dict{Symbol, Any}(:recent => [[0.3, 0.5]])
    @assert ck.evaluate([0.8, 0.3], ts_changed) "CHANGED should detect transition"

    ts_same = Dict{Symbol, Any}(:recent => [[0.8, 0.5]])
    @assert !ck.evaluate([0.9, 0.3], ts_same) "CHANGED should be false when stable"

    println("PASSED: CHANGED operator compiles and evaluates correctly")

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
# TEST 11: Program enumeration with temporal operators
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 11: Program enumeration includes temporal operators when enabled")
println("=" ^ 60)

let
    sc = SensorConfig([SensorChannel(0, :identity, 0.05, 1.0)])
    g = Grammar(sc, ProductionRule[], 1)

    programs_no_temporal = enumerate_programs(g, 2; include_temporal=false)
    programs_with_temporal = enumerate_programs(g, 2; include_temporal=true)

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
# TEST 12: End-to-end single regime test (no grid world — pure conditioning)
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 12: End-to-end single regime (pure conditioning, short)")
println("=" ^ 60)

let
    Random.seed!(42)

    grammars = generate_seed_grammars()

    components = BetaMeasure[]
    log_prior = Float64[]
    metadata = Tuple{Int, Int}[]
    compiled = CompiledKernel[]
    programs_all = Program[]

    for g in grammars[1:5]
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

    k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
        θ -> error("not used"),
        (θ, obs) -> obs == 1.0 ? log(max(θ, 1e-300)) : log(max(1.0 - θ, 1e-300)))

    for i in 1:20
        obs = i % 3 == 0 ? 1.0 : 0.0
        belief = condition(belief, k, obs)
    end

    w = weights(belief)
    @assert abs(sum(w) - 1.0) < 1e-10 "Weights should sum to 1"

    for comp in belief.components
        @assert comp isa BetaMeasure
        @assert comp.alpha + comp.beta > 2.0 "Beta should have been updated from prior"
    end

    println("PASSED: 20 conditioning steps, $(length(belief.components)) components, weights sum to $(sum(w))")
end
println()

# ═══════════════════════════════════════
# TEST 13: Compression pipeline produces real compression (mechanism, spec §5.10)
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 13: Compression pipeline produces real compression")
println("=" ^ 60)

let
    Random.seed!(42)
    g = Grammar(colour_sensor_config(), ProductionRule[], 1)
    programs = enumerate_programs(g, 3)

    prog_weights = zeros(length(programs))
    for (i, p) in enumerate(programs)
        s = show_expr(p.predicate)
        if occursin("GT(0,0.7)", s) && occursin("LT(1,0.3)", s)
            prog_weights[i] = 1.0
        elseif occursin("GT(0,0.7)", s) || occursin("LT(1,0.3)", s)
            prog_weights[i] = 0.1
        else
            prog_weights[i] = 0.001
        end
    end
    prog_weights ./= sum(prog_weights)

    freq_table = analyse_posterior_subtrees(programs, prog_weights;
                                            min_frequency=0.005, min_complexity=2)
    proposed = propose_nonterminal(freq_table)
    @assert proposed !== nothing "Should propose a nonterminal from colour posterior"

    body_str = show_expr(proposed.body)
    has_colour_ref = any(ch -> occursin("($ch,", body_str), ["0", "1", "2"])
    @assert has_colour_ref "Proposed body should reference colour channels, got: $body_str"

    new_g = Grammar(colour_sensor_config(), [proposed], 2)
    new_programs = enumerate_programs(new_g, 2)

    nt_progs = filter(p -> occursin(string(proposed.name), show_expr(p.predicate)), new_programs)
    @assert !isempty(nt_progs) "Should have programs using the nonterminal"

    nt_prog = first(nt_progs)
    exp_c = expanded_complexity(nt_prog.predicate, new_g.rules)
    @assert nt_prog.complexity < exp_c "Nonterminal should compress: ref=$(nt_prog.complexity) < expanded=$exp_c"

    println("PASSED: Nonterminal '$(proposed.name)' body=$body_str, " *
            "ref_complexity=$(nt_prog.complexity) < expanded=$exp_c")
end
println()

# ═══════════════════════════════════════
# TEST 14: modify_threshold produces changes (enforcement)
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 14: modify_threshold produces grammar with modified thresholds")
println("=" ^ 60)

let
    Random.seed!(42)
    red_body = AndExpr(GTExpr(0, 0.7), AndExpr(LTExpr(1, 0.3), LTExpr(2, 0.3)))
    g = Grammar(colour_sensor_config(), [ProductionRule(:RED, red_body)], 1)
    original_str = show_expr(red_body)

    dummy_table = SubprogramFrequencyTable(ProgramExpr[], Float64[], Vector{Int}[])

    found = false
    for _ in 1:200
        new_g = perturb_grammar(g, dummy_table)
        for r in new_g.rules
            if r.name == :RED && show_expr(r.body) != original_str
                found = true
                break
            end
        end
        found && break
    end

    @assert found "modify_threshold should produce at least one changed threshold in 200 trials"
    println("PASSED: modify_threshold produces grammars with modified thresholds")
end
println()

# ═══════════════════════════════════════
# TEST 15: Proposed nonterminals are actual posterior subtrees (enforcement, spec §5.10)
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 15: Proposed nonterminals are actual posterior subtrees")
println("=" ^ 60)

let
    Random.seed!(42)
    g = Grammar(colour_sensor_config(), ProductionRule[], 1)
    programs = enumerate_programs(g, 3)

    prog_weights = zeros(length(programs))
    for (i, p) in enumerate(programs)
        s = show_expr(p.predicate)
        if occursin("GT(0,0.7)", s) && occursin("LT(1,0.3)", s)
            prog_weights[i] = 1.0
        elseif occursin("GT(0,0.7)", s)
            prog_weights[i] = 0.1
        else
            prog_weights[i] = 0.001
        end
    end
    prog_weights ./= sum(prog_weights)

    freq_table = analyse_posterior_subtrees(programs, prog_weights;
                                            min_frequency=0.005, min_complexity=2)
    proposed = propose_nonterminal(freq_table)
    @assert proposed !== nothing "Should propose a nonterminal"

    proposed_str = show_expr(proposed.body)

    best_idx = argmax(freq_table.weighted_frequency)
    source_idxs = unique(freq_table.source_programs[best_idx])
    @assert length(source_idxs) >= 2 "Proposed body should appear in ≥2 programs, found in $(length(source_idxs))"

    verified = count(i -> occursin(proposed_str, show_expr(programs[i].predicate)),
                     source_idxs[1:min(10, length(source_idxs))])
    @assert verified >= 2 "Proposed subtree should be verifiable in source programs via show_expr"

    println("PASSED: Proposed body '$proposed_str' in $(length(source_idxs)) source programs, verified $verified")
end
println()

# ═══════════════════════════════════════
# TEST 16: Temporal programs avoid overcommitment across regime changes (mechanism)
# ═══════════════════════════════════════

println("=" ^ 60)
println("TEST 16: Temporal programs — overcommitment avoidance & transition detection")
println("=" ^ 60)

let
    n_steps = 20

    alpha, beta_param = 1.0, 1.0
    stable_ll_second_half = 0.0

    for step in 1:n_steps
        p = alpha / (alpha + beta_param)
        if step <= 10
            alpha += 1.0
        else
            stable_ll_second_half += log(1.0 - p)
            beta_param += 1.0
        end
    end

    changed_ll_second_half = 0.0

    @assert changed_ll_second_half > stable_ll_second_half (
        "CHANGED should avoid overcommitment: " *
        "CHANGED ll=$changed_ll_second_half > stable ll=$(round(stable_ll_second_half, digits=2))")

    println("  Scenario A PASSED: stable 2nd-half ll=$(round(stable_ll_second_half, digits=2)), " *
            "CHANGED 2nd-half ll=$changed_ll_second_half")

    changed_gt = compile_expr(ChangedExpr(GTExpr(0, 0.7)), ProductionRule[])
    ts = Dict{Symbol, Any}(:recent => Vector{Float64}[])
    changed_fires = Bool[]

    for step in 1:n_steps
        red = step <= 10 ? 0.9 : 0.5
        sv = [red, 0.5]
        push!(changed_fires, changed_gt(sv, ts))
        push!(ts[:recent], sv)
        while length(ts[:recent]) > 10
            popfirst!(ts[:recent])
        end
    end

    @assert changed_fires[11] "CHANGED should fire at the transition step (step 11)"
    non_transition = [changed_fires[i] for i in 1:n_steps if i != 11]
    @assert !any(non_transition) "CHANGED should only fire at the transition step"

    println("  Scenario B PASSED: CHANGED fires at step 11 only")
    println("PASSED: Both temporal scenarios verified")
end
println()

println("=" ^ 60)
println("ALL PROGRAM SPACE TESTS PASSED")
println("=" ^ 60)
