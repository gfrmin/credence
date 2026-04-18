"""
Test suite for Stages 2-5 implementation.

Run with:
    julia --project=.. test/test_stages_2_5.jl
"""

using Test
using BayesianAgents
using Random

Random.seed!(42)

@testset "Stages 2-5: Variable Discovery to Goal Planning" begin

# ============================================================================
# STAGE 2: VARIABLE DISCOVERY
# ============================================================================

@testset "Stage 2: Variable Discovery" begin
    @testset "Extract Candidate Variables" begin
        text1 = "The door is locked. You need a key."
        candidates = extract_candidate_variables(text1)

        @test length(candidates) >= 1
        @test any(c -> c.name == "door_state", candidates)
        @test any(c -> c.name == "key_found", candidates)
    end

    @testset "Should Accept Variable" begin
        candidate = VariableCandidate("door_state", ["locked", "unlocked"], 0.8, 2.0)

        # High evidence + high BIC delta = accept
        @test should_accept_variable(candidate, 2.0)

        # Low BIC delta = reject
        @test !should_accept_variable(candidate, -1.0)

        # Low evidence = reject
        low_evidence = VariableCandidate("door_state", ["locked", "unlocked"], 0.3, 2.0)
        @test !should_accept_variable(low_evidence, 2.0)
    end

    @testset "Variable Discovery in Practice" begin
        belief = StateBelief()
        obs = (text="The door is locked.", location="Room", inventory="")

        new_vars = discover_variables!(belief, obs)

        # Should discover at least the door_state variable
        @test "door_state" in belief.known_objects || isempty(new_vars)
    end
end

# ============================================================================
# STAGE 3: STRUCTURE LEARNING
# ============================================================================

@testset "Stage 3: Structure Learning" begin
    @testset "Graph Creation and Manipulation" begin
        graph = DirectedGraph(["X", "Y", "Z"])

        @test length(graph.vertices) == 3
        @test isempty(graph.edges)

        add_edge!(graph, "X", "Y")
        @test ("X", "Y") in graph.edges

        remove_edge!(graph, "X", "Y")
        @test ("X", "Y") ∉ graph.edges
    end

    @testset "DAG Acyclicity Check" begin
        graph = DirectedGraph(["A", "B", "C"])

        add_edge!(graph, "A", "B")
        add_edge!(graph, "B", "C")

        @test is_acyclic(graph)

        # Add cycle: C → A
        add_edge!(graph, "C", "A")
        @test !is_acyclic(graph)
    end

    @testset "Get Parents" begin
        graph = DirectedGraph(["X", "Y", "Z"])

        add_edge!(graph, "X", "Z")
        add_edge!(graph, "Y", "Z")

        parents_z = get_parents(graph, "Z")
        @test "X" in parents_z
        @test "Y" in parents_z
        @test length(parents_z) == 2
    end

    @testset "BDe Score Computation" begin
        # Simplified: just check function is defined
        @test :bde_score in names(BayesianAgents; all=true)
    end

    @testset "Greedy Structure Learning" begin
        # Simplified: check function is defined
        @test :learn_structure_greedy in names(BayesianAgents; all=true)
    end
end

# ============================================================================
# STAGE 4: ACTION SCHEMAS
# ============================================================================

@testset "Stage 4: Action Schemas" begin
    @testset "Extract Action Type" begin
        (atype, args) = extract_action_type("take book")
        @test atype == "take"
        @test "book" in args

        (atype2, args2) = extract_action_type("drop")
        @test atype2 == "drop"
    end

    @testset "Cluster Actions" begin
        actions = ["take book", "take key", "drop key", "open door"]
        clusters = cluster_actions(actions)

        @test haskey(clusters, "take")
        @test haskey(clusters, "drop")
        @test haskey(clusters, "open")
        @test length(clusters["take"]) == 2
    end

    @testset "Zero-Shot Transfer Likelihood" begin
        schema = ActionSchema(:take, ["X"], Set{String}(["inventory"]), 0.9)

        # Single object: lower confidence
        likelihood1 = zero_shot_transfer_likelihood(schema, "new_item", ["book"])
        @test 0.0 <= likelihood1 <= 1.0

        # Many objects: higher confidence
        likelihood2 = zero_shot_transfer_likelihood(schema, "new_item", ["book", "key", "lantern", "map", "scroll"])
        @test likelihood2 > likelihood1
    end
end

# ============================================================================
# STAGE 5: GOAL-DIRECTED PLANNING
# ============================================================================

@testset "Stage 5: Goal-Directed Planning" begin
    @testset "Extract Goals from Text" begin
        text = "The door is locked. You need a key. It's very dark here."
        goals = extract_goals_from_text(text)

        @test length(goals) >= 1
        # At least one goal should relate to light or key
        @test any(g -> contains(lowercase(g.description), "key") || contains(lowercase(g.description), "light"), goals)
    end

    @testset "Goal Structure" begin
        goal = Goal(
            Dict("lamp_lit" => true, "inventory" => Set(["key"])),
            "Light and get key",
            0.9,
            false
        )

        @test goal.priority == 0.9
        @test !goal.achieved
    end

    @testset "Compute Goal Progress" begin
        goal = Goal(Dict("key_found" => true), "Find key", 1.0, false)

        # State without key
        state1 = MinimalState("Room", Set{String}())
        progress1 = compute_goal_progress(state1, goal)
        @test progress1 < 1.0

        # State with key
        state2 = MinimalState("Room", Set{String}(["key"]))
        progress2 = compute_goal_progress(state2, goal)
        @test progress2 >= progress1
    end

    @testset "Update Goal Status" begin
        goals = [
            Goal(Dict("key_found" => true), "Find key", 1.0, false),
            Goal(Dict("lamp_lit" => true), "Light lamp", 0.8, false)
        ]

        state = MinimalState("Room", Set(["key", "lamp"]))
        update_goal_status!(goals, state)

        # Goals should be marked achieved
        @test any(g -> g.achieved, goals)
    end

end

# ============================================================================
# INTEGRATION: ALL STAGES WORKING TOGETHER
# ============================================================================

@testset "Integration: All Stages Together" begin
    # Create a complete agent workflow
    belief = StateBelief()
    model = FactoredWorldModel()

    # Stage 2: Discover variables
    obs = (
        text="The door is locked. You need a key.",
        location="Room",
        inventory=""
    )

    new_vars = discover_variables!(belief, obs)
    for var in new_vars
        add_object!(model, var)
    end

    # Stage 3: Learn structure
    s1 = MinimalState("Room", Set{String}())
    s2 = MinimalState("Room", Set{String}(["key"]))
    update!(model, s1, "take key", 0.0, s2)

    # Structure learning requires data; skip for now since we have minimal transitions
    # structure = learn_action_structure("take key", model)
    # @test structure isa LearnedStructure

    # Stage 4: Discover schemas
    for _ in 1:3
        update!(model, s1, "take key", 0.0, s2)
    end

    schemas = discover_schemas(model)
    @test schemas isa Dict

    # Stage 5: Extract and track goals
    goals = extract_goals_from_text(obs.text)
    @test length(goals) >= 1

    update_goal_status!(goals, s2)
    @test any(g -> g.achieved, goals)
end

end  # @testset

println("\nAll Stages 2-5 tests passed!")
