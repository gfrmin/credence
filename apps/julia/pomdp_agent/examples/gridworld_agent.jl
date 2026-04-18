"""
    Bayesian Agent Example: Grid World

This example demonstrates the complete Bayesian agent framework:
1. Learning world dynamics via Bayesian inference
2. Planning via Thompson Sampling MCTS
3. State abstraction via bisimulation
4. Optional: LLM sensor for guidance

Run with:
    julia --project=. examples/gridworld_agent.jl
"""

# Ensure we're using the local package
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Random
using Printf

# Import the framework
include("../src/BayesianAgents.jl")
using .BayesianAgents

# ============================================================================
# SIMPLE ORACLE SENSOR (for testing without LLM)
# ============================================================================

"""
A simple oracle that knows the true food energies.
Used for testing the sensor learning mechanism.
"""
struct OracleClient
    world::GridWorld
    noise::Float64  # Probability of giving wrong answer
end

function (oracle::OracleClient)(prompt::String)
    # Parse the action from the prompt
    action_match = match(r"action \"([^\"]+)\"", prompt)
    if isnothing(action_match)
        return rand() < 0.5 ? "yes" : "no"
    end
    
    action = Symbol(action_match.captures[1])
    
    # Determine if action would actually help
    helps = false
    
    if action == :eat && haskey(oracle.world.food, oracle.world.agent_pos)
        energy = oracle.world.food[oracle.world.agent_pos].energy
        helps = energy > 0  # Eating positive energy food helps
    elseif action in [:north, :south, :east, :west]
        # Movement helps if it gets closer to goal or food
        new_pos = move(oracle.world.agent_pos, action)
        if is_valid_pos(oracle.world, new_pos)
            # Check if closer to any positive food
            for (pos, food) in oracle.world.food
                if food.energy > 0
                    old_dist = sum(abs.(oracle.world.agent_pos .- pos))
                    new_dist = sum(abs.(new_pos .- pos))
                    if new_dist < old_dist
                        helps = true
                        break
                    end
                end
            end
        end
    end
    
    # Add noise
    if rand() < oracle.noise
        helps = !helps
    end
    
    return helps ? "yes" : "no"
end

# ============================================================================
# STATE PREPROCESSING
# ============================================================================

"""
Strip agent_energy and steps from the observation (they change every tick,
making every state unique). Keep agent_pos and food layout — these are
strategically relevant (food disappears when eaten).
"""
gridworld_preprocess(obs) = (agent_pos = obs.agent_pos, food = obs.food)

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

function run_experiment(;
    n_episodes::Int = 50,
    grid_size::Int = 8,
    use_sensor::Bool = true,
    sensor_noise::Float64 = 0.2,
    verbose::Bool = true
)
    println("=" ^ 60)
    println("BAYESIAN AGENT EXPERIMENT")
    println("=" ^ 60)
    println()
    
    # Create world
    world = GridWorld(grid_size, grid_size; max_steps=50)
    
    # Create world model
    model = TabularWorldModel(
        transition_prior = 0.1,
        reward_prior_mean = 0.0,
        reward_prior_variance = 1.0
    )
    
    # Create planner
    planner = ThompsonMCTS(
        iterations = 80,
        depth = 8,
        discount = 0.99,
        ucb_c = 2.0
    )
    
    # Create state abstractor with preprocessing to collapse the state space
    abstractor = BisimulationAbstractor(
        similarity_threshold = 0.95,
        reward_threshold = 0.1,
        preprocess_fn = gridworld_preprocess
    )
    
    # Create sensor (optional)
    sensors = Sensor[]
    if use_sensor
        oracle = OracleClient(world, sensor_noise)
        sensor = BinarySensor("oracle", (state, q) -> oracle(q) == "yes")
        push!(sensors, sensor)
    end
    
    # Create agent
    agent = BayesianAgent(
        world, model, planner, abstractor;
        sensors = sensors,
        config = AgentConfig(
            planning_depth = 8,
            mcts_iterations = 80,
            sensor_cost = 0.01
        )
    )
    
    # Run episodes
    episode_rewards = Float64[]
    episode_lengths = Int[]
    
    for episode in 1:n_episodes
        reward = run_episode!(agent; max_steps=50)
        push!(episode_rewards, reward)
        push!(episode_lengths, agent.step_count)
        
        if verbose && episode % 10 == 0
            recent_avg = mean(episode_rewards[max(1, episode-9):episode])
            println(@sprintf("Episode %3d: reward = %6.2f | recent_avg = %6.2f | states = %d",
                episode, reward, recent_avg, length(model.known_states)))
        end
    end
    
    # Summary
    println()
    println("=" ^ 60)
    println("RESULTS")
    println("=" ^ 60)
    println(@sprintf("Total episodes: %d", n_episodes))
    println(@sprintf("Average reward: %.2f", mean(episode_rewards)))
    println(@sprintf("Final 10 avg:   %.2f", mean(episode_rewards[end-9:end])))
    println(@sprintf("States learned: %d", length(model.known_states)))
    println(@sprintf("Transitions:    %d", length(model.transition_counts)))
    
    if use_sensor && !isempty(sensors)
        sensor = sensors[1]
        println()
        println("Sensor reliability:")
        println(@sprintf("  TPR: %.3f (queries: %d)", tpr(sensor), sensor.n_queries))
        println(@sprintf("  FPR: %.3f", fpr(sensor)))
    end

    println()
    println(abstraction_summary(abstractor))
    
    return (
        rewards = episode_rewards,
        lengths = episode_lengths,
        agent = agent
    )
end

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function mean(x)
    return sum(x) / length(x)
end

# ============================================================================
# RUN
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Run experiment
    results = run_experiment(
        n_episodes = 100,
        grid_size = 6,
        use_sensor = true,
        sensor_noise = 0.15,
        verbose = true
    )
    
    println()
    println("Learning curve:")
    for (i, chunk) in enumerate(Iterators.partition(results.rewards, 10))
        avg = mean(collect(chunk))
        bar = "█" ^ max(0, round(Int, avg + 5))
        println(@sprintf("Episodes %2d-%2d: %s %.2f", 
            (i-1)*10+1, i*10, bar, avg))
    end
end
