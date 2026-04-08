"""
Example: Running a Bayesian agent on GridWorld

This demonstrates:
1. Creating a world
2. Setting up the Bayesian world model
3. Using Thompson MCTS for planning
4. Running episodes and observing learning
"""

# Add the package to path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using BayesianAgents
using Random

# Include world implementations
include(joinpath(@__DIR__, "..", "worlds", "grid_world.jl"))
using .GridWorldEnv

function main()
    println("=== Bayesian Agent on GridWorld ===\n")
    
    # Create world
    world = GridWorld(width=8, height=8, n_food_types=4, seed=42)
    
    # Create world model (Bayesian)
    model = TabularWorldModel(
        transition_prior=0.1,
        reward_prior_mean=0.0,
        reward_prior_variance=2.0
    )
    
    # Create planner (Thompson MCTS)
    planner = ThompsonMCTS(
        iterations=50,
        depth=5,
        discount=0.99,
        ucb_c=2.0
    )
    
    # Create state abstractor
    # Start with identity (no abstraction) for simplicity
    abstractor = IdentityAbstractor()
    
    # Configuration
    config = AgentConfig(
        planning_depth=5,
        mcts_iterations=50,
        discount=0.99,
        use_intrinsic_reward=true,
        intrinsic_scale=0.1
    )
    
    # Create agent
    agent = BayesianAgent(world, model, planner, abstractor; config=config)
    
    # Run episodes
    n_episodes = 20
    rewards = Float64[]
    
    println("Running $n_episodes episodes...\n")
    
    for ep in 1:n_episodes
        total_reward = run_episode!(agent; max_steps=100)
        push!(rewards, total_reward)
        
        if ep % 5 == 0
            mean_recent = sum(rewards[max(1,ep-4):ep]) / min(ep, 5)
            println("Episode $ep: reward = $(round(total_reward, digits=2)), " *
                    "mean(last 5) = $(round(mean_recent, digits=2)), " *
                    "model entropy = $(round(entropy(model), digits=2))")
        end
    end
    
    # Summary
    println("\n=== Summary ===")
    println("Total episodes: $n_episodes")
    println("Mean reward (first 5): $(round(sum(rewards[1:5])/5, digits=2))")
    println("Mean reward (last 5): $(round(sum(rewards[end-4:end])/5, digits=2))")
    println("Final model entropy: $(round(entropy(model), digits=2))")
    
    # Show learned model statistics
    println("\n=== Learned Dynamics ===")
    println("Observed state-action pairs: $(length(model.transition_counts))")
    println("Known states: $(length(model.known_states))")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
