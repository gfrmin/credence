# Role: brain-side application
"""
Test suite for BayesianAgents Stage 1 (MVBN).

Run with:
    julia --project=.. -e 'using Pkg; Pkg.test()'

or

    julia --project=.. test/runtests.jl
"""

using Test
using BayesianAgents
using Distributions
using Random
import BayesianAgents: entropy, mode, predict  # Resolve naming conflicts

Random.seed!(42)  # Reproducibility

@testset "BayesianAgents Stage 1: MVBN" begin

# ============================================================================
# DIRICHLET-CATEGORICAL CONJUGACY TESTS
# ============================================================================

@testset "DirichletCategorical Conjugacy" begin
    # Prior: Dirichlet([1.0, 1.0, 1.0])
    cpd = DirichletCategorical(["a", "b", "c"], 1.0)

    # Check initial state
    @test length(cpd.domain) == 3
    @test length(cpd.alpha) == 3
    @test all(cpd.counts .== 0)
    @test length(cpd.domain) == length(cpd.alpha) == length(cpd.counts)

    # Update with observations
    update!(cpd, "a")
    update!(cpd, "a")
    update!(cpd, "b")

    # Check counts
    @test cpd.counts == [2, 1, 0]

    # Posterior should be Dirichlet([1+2, 1+1, 1+0]) = Dirichlet([3, 2, 1])
    # Predictive: P(V=v|data) = (α_v + count_v) / Σ
    pred = predict(cpd)
    expected = [3.0, 2.0, 1.0] ./ 6.0
    @test pred ≈ expected

    @test sum(pred) ≈ 1.0
end

@testset "DirichletCategorical Mode and Entropy" begin
    cpd = DirichletCategorical(["x", "y", "z"], 0.1)

    # Prior: uniform (maximum entropy)
    prior_entropy = entropy(cpd)
    @test prior_entropy > 0.5  # Log scale, 3 categories

    # After 10 observations of "x"
    for _ in 1:10
        update!(cpd, "x")
    end

    # Entropy should decrease (but only with enough data)
    posterior_entropy = entropy(cpd)
    @test posterior_entropy <= prior_entropy

    # Mode should be "x" after 10 observations
    m = mode(cpd)
    @test m == "x"
end

@testset "DirichletCategorical Sampling" begin
    cpd = DirichletCategorical(["red", "green", "blue"], 1.0)

    # After 100 observations strongly favor "red"
    for _ in 1:100
        update!(cpd, "red")
    end
    for _ in 1:1
        update!(cpd, "blue")
    end

    # Sample many times
    samples = [rand(cpd) for _ in 1:1000]

    # Most should be "red"
    red_count = count(s -> s == "red", samples)
    @test red_count > 800  # High probability
end

@testset "DirichletCategorical Copy and Reset" begin
    cpd1 = DirichletCategorical(["a", "b"], 0.5)
    update!(cpd1, "a")
    update!(cpd1, "a")

    # Copy
    cpd2 = copy(cpd1)
    @test cpd2.counts == cpd1.counts
    @test cpd2.alpha == cpd1.alpha

    # Modify copy without affecting original
    update!(cpd2, "b")
    @test cpd1.counts == [2, 0]
    @test cpd2.counts == [2, 1]

    # Reset
    reset!(cpd1)
    @test all(cpd1.counts .== 0)
end

# ============================================================================
# MINIMAL STATE TESTS
# ============================================================================

@testset "MinimalState Equality and Hashing" begin
    s1 = MinimalState("Kitchen", Set(["book", "lantern"]))
    s2 = MinimalState("Kitchen", Set(["lantern", "book"]))  # Different order, same set
    s3 = MinimalState("Forest", Set(["book", "lantern"]))

    @test s1 == s2  # Sets are unordered
    @test s1 != s3  # Different location
    @test hash(s1) == hash(s2)  # Same hash for equal states
end

@testset "MinimalState Construction" begin
    s1 = MinimalState("Room", Set{String}())
    @test s1.location == "Room"
    @test isempty(s1.inventory)

    s2 = MinimalState("Room", "book,key,lantern")
    @test s2.location == "Room"
    # Trim whitespace since split might create strings
    expected_items = Set(strip.(["book", "key", "lantern"]))
    @test s2.inventory == expected_items
end

# ============================================================================
# STATE BELIEF TESTS
# ============================================================================

@testset "StateBelief Initialization" begin
    belief = StateBelief()
    @test length(belief.location_belief.domain) >= 3  # Common locations
    @test isempty(belief.inventory_beliefs)
    @test isempty(belief.history)
end

@testset "StateBelief Object Registration" begin
    belief = StateBelief()

    add_object!(belief, "book")
    @test haskey(belief.inventory_beliefs, "book")
    @test "book" ∈ belief.known_objects

    # Adding again doesn't create duplicate
    add_object!(belief, "book")
    @test length(belief.known_objects) == 1
end

@testset "StateBelief State Update" begin
    belief = StateBelief()
    state = MinimalState("Kitchen", Set(["book"]))

    update_from_state!(belief, state)

    @test length(belief.history) == 1
    @test "book" ∈ belief.known_objects

    # Location belief should be updated
    pred_loc = predict(belief.location_belief)
    # "Kitchen" should have higher probability now
    kitchen_idx = findfirst(l -> l == "Kitchen", belief.location_belief.domain)
    @test pred_loc[kitchen_idx] > 1.0 / length(belief.location_belief.domain)
end

@testset "StateBelief Sampling" begin
    belief = StateBelief()

    # Add some observations to shape beliefs
    for _ in 1:10
        update_from_state!(belief, MinimalState("Kitchen", Set(["book"])))
    end
    for _ in 1:5
        update_from_state!(belief, MinimalState("Forest", Set{String}()))
    end

    # Kitchen and Forest should have highest probability
    loc_pred = predict(belief.location_belief)
    kitchen_idx = findfirst(l -> l == "Kitchen", belief.location_belief.domain)
    forest_idx = findfirst(l -> l == "Forest", belief.location_belief.domain)

    # Kitchen seen 10 times, Forest 5 times, both should be above average
    avg_prob = 1.0 / length(belief.location_belief.domain)
    @test loc_pred[kitchen_idx] > avg_prob
    @test loc_pred[forest_idx] > avg_prob
end

@testset "StateBelief Entropy" begin
    belief = StateBelief()

    # Before any data: high entropy
    h_prior = entropy(belief)

    # Add observations to strongly favor Kitchen
    for _ in 1:100
        update_from_state!(belief, MinimalState("Kitchen", Set(["book"])))
    end

    # After strong data: lower entropy
    h_posterior = entropy(belief)
    @test h_posterior <= h_prior
    # Should be noticeably lower due to strong observations
    @test (h_prior - h_posterior) > 0.1
end

# ============================================================================
# FACTORED WORLD MODEL TESTS
# ============================================================================

@testset "FactoredWorldModel Initialization" begin
    model = FactoredWorldModel(0.1)
    @test isempty(model.cpds)
    @test isempty(model.transitions)
    @test isempty(model.known_locations)
    @test isempty(model.known_objects)
end

@testset "FactoredWorldModel Update" begin
    model = FactoredWorldModel(0.5)

    s = MinimalState("Kitchen", Set(["book"]))
    s_next = MinimalState("Forest", Set(["book", "key"]))

    # Update with transition
    update!(model, s, "north", 0.0, s_next)

    # Check registration
    @test "Kitchen" ∈ model.known_locations
    @test "Forest" ∈ model.known_locations
    @test "book" ∈ model.known_objects
    @test "key" ∈ model.known_objects

    # Check CPDs created
    @test haskey(model.cpds, "north")
    @test haskey(model.cpds["north"], "location")
    @test haskey(model.cpds["north"], "inventory_book")
    @test haskey(model.cpds["north"], "inventory_key")
end

@testset "FactoredWorldModel Self-loop Tracking" begin
    model = FactoredWorldModel()

    s = MinimalState("Room", Set{String}())
    a = "look around"

    # Mark as self-loop
    mark_selfloop!(model, s, a)
    @test is_selfloop(model, s, a)

    # Different state should not be marked
    s_other = MinimalState("Room", Set(["book"]))
    @test !is_selfloop(model, s_other, a)
end

@testset "FactoredWorldModel Thompson Sampling" begin
    model = FactoredWorldModel()

    # Add some observations (both with same target location to avoid domain expansion)
    s = MinimalState("Kitchen", Set{String}())
    s_north = MinimalState("Forest", Set{String}())
    s_south = MinimalState("Forest", Set(["book"]))  # Same location, different inventory

    for _ in 1:5
        update!(model, s, "north", 0.0, s_north)
    end
    for _ in 1:2
        update!(model, s, "north", 0.0, s_south)
    end

    # Sample dynamics
    sampled = sample_dynamics(model)
    @test sampled isa SampledFactoredDynamics
    @test haskey(sampled.sampled_cpds, "north")
end

# ============================================================================
# STATE BELIEF + WORLD MODEL INTEGRATION
# ============================================================================

@testset "Full MVBN Workflow" begin
    # Create belief and model
    belief = StateBelief()
    model = FactoredWorldModel()

    # Simulate a few transitions
    states = [
        MinimalState("Kitchen", Set{String}()),
        MinimalState("Kitchen", Set(["book"])),
        MinimalState("Forest", Set(["book"])),
    ]

    for i in 1:length(states)-1
        state = states[i]
        next_state = states[i+1]
        action = if i == 1 "take book" else "go north" end
        reward = 0.0

        # Update belief
        update_from_state!(belief, state)

        # Update model
        update!(model, state, action, reward, next_state)
    end

    # Check results
    @test belief.location_belief.counts[1] >= 1  # Saw Kitchen
    @test "book" ∈ belief.known_objects
    @test "Forest" ∈ model.known_locations
end

end  # @testset

println("\nAll Stage 1 tests passed!")
