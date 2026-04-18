"""
    GridWorld

A simple grid world environment for testing the Bayesian agent framework.

Features:
- Configurable size, obstacles, and goals
- Food items with hidden energy values (to be learned)
- Stochastic transitions (optional)
- Deterministic or stochastic rewards
"""

using Random

"""
    GridWorld

A grid-based environment with food items to collect.
"""
mutable struct GridWorld <: World
    # Grid dimensions
    width::Int
    height::Int
    
    # Agent state
    agent_pos::Tuple{Int, Int}
    agent_energy::Float64
    
    # Goal and obstacles
    goal_pos::Union{Nothing, Tuple{Int, Int}}
    obstacles::Set{Tuple{Int, Int}}
    
    # Food items: position → (true energy, appearance)
    # The agent sees the appearance, not the true energy
    food::Dict{Tuple{Int, Int}, NamedTuple{(:energy, :appearance), Tuple{Float64, Symbol}}}
    
    # Episode tracking
    steps::Int
    max_steps::Int
    total_reward::Float64
    done::Bool
    
    # Stochasticity
    transition_noise::Float64  # Probability of random movement
    
    # Random seed
    rng::AbstractRNG
end

"""
    GridWorld(width, height; kwargs...)

Create a new grid world.

# Arguments
- `width`, `height`: Grid dimensions
- `goal_pos`: Optional goal position
- `obstacles`: Set of obstacle positions
- `transition_noise`: Probability of random movement (0 = deterministic)
- `max_steps`: Maximum steps per episode
- `seed`: Random seed
"""
function GridWorld(
    width::Int,
    height::Int;
    goal_pos::Union{Nothing, Tuple{Int, Int}} = nothing,
    obstacles::Set{Tuple{Int, Int}} = Set{Tuple{Int, Int}}(),
    transition_noise::Float64 = 0.0,
    max_steps::Int = 100,
    seed::Int = 42
)
    rng = MersenneTwister(seed)
    
    # Find valid starting position
    start_pos = (1, 1)
    while start_pos in obstacles || start_pos == goal_pos
        start_pos = (rand(rng, 1:width), rand(rng, 1:height))
    end
    
    return GridWorld(
        width, height,
        start_pos, 10.0,  # Start with 10 energy
        goal_pos, obstacles,
        Dict{Tuple{Int, Int}, NamedTuple{(:energy, :appearance), Tuple{Float64, Symbol}}}(),
        0, max_steps, 0.0, false,
        transition_noise,
        rng
    )
end

"""
    reset!(world::GridWorld) → observation

Reset the world to initial state.
"""
function reset!(world::GridWorld)
    # Reset agent
    world.agent_pos = (1, 1)
    world.agent_energy = 10.0
    world.steps = 0
    world.total_reward = 0.0
    world.done = false
    
    # Respawn food
    spawn_food!(world)
    
    return observe(world)
end

"""
    spawn_food!(world::GridWorld)

Spawn food items in the world with random positions and hidden energy values.
"""
function spawn_food!(world::GridWorld; n_food::Int = 5)
    empty!(world.food)
    
    # Food appearances
    appearances = [:circle, :square, :triangle]
    
    # Hidden energy distributions for each appearance
    # The agent must learn these!
    energy_dists = Dict(
        :circle => Normal(2.0, 0.5),   # Good food
        :square => Normal(-1.0, 0.5),  # Bad food
        :triangle => Normal(0.0, 1.0)  # Uncertain food
    )
    
    for _ in 1:n_food
        # Random position not on obstacle, goal, or existing food
        pos = (rand(world.rng, 1:world.width), rand(world.rng, 1:world.height))
        attempts = 0
        while (pos in world.obstacles || 
               pos == world.goal_pos || 
               pos == world.agent_pos ||
               haskey(world.food, pos)) && attempts < 100
            pos = (rand(world.rng, 1:world.width), rand(world.rng, 1:world.height))
            attempts += 1
        end
        
        if attempts < 100
            appearance = rand(world.rng, appearances)
            energy = rand(world.rng, energy_dists[appearance])
            world.food[pos] = (energy=energy, appearance=appearance)
        end
    end
end

"""
    observe(world::GridWorld) → observation

Return the current observation.
"""
function observe(world::GridWorld)
    # The agent sees: its position, energy, visible food (with appearance only), goal
    visible_food = Dict{Tuple{Int, Int}, Symbol}()
    for (pos, food) in world.food
        visible_food[pos] = food.appearance
    end
    
    return (
        agent_pos = world.agent_pos,
        agent_energy = world.agent_energy,
        food = visible_food,
        goal = world.goal_pos,
        steps = world.steps
    )
end

"""
    actions(world::GridWorld, observation) → Vector{Symbol}

Return available actions.
"""
function actions(world::GridWorld, observation = nothing)
    return [:north, :south, :east, :west, :stay, :eat]
end

"""
    step!(world::GridWorld, action::Symbol) → (observation, reward, done, info)

Execute an action and return the result.
"""
function step!(world::GridWorld, action::Symbol)
    if world.done
        return observe(world), 0.0, true, Dict()
    end
    
    world.steps += 1
    reward = 0.0
    
    # Movement cost
    world.agent_energy -= 0.1
    
    # Handle stochastic transitions
    if rand(world.rng) < world.transition_noise
        action = rand(world.rng, [:north, :south, :east, :west])
    end
    
    # Execute action
    if action in [:north, :south, :east, :west]
        new_pos = move(world.agent_pos, action)
        
        # Check validity
        if is_valid_pos(world, new_pos)
            world.agent_pos = new_pos
        end
        
    elseif action == :eat
        # Try to eat food at current position
        if haskey(world.food, world.agent_pos)
            food = world.food[world.agent_pos]
            energy_gained = food.energy
            world.agent_energy += energy_gained
            reward = energy_gained  # Reward is the energy gained
            delete!(world.food, world.agent_pos)
        end
    end
    # :stay does nothing
    
    # Check terminal conditions
    if world.agent_energy <= 0
        world.done = true
        reward -= 10.0  # Death penalty
    elseif world.agent_pos == world.goal_pos
        world.done = true
        reward += 10.0  # Goal reward
    elseif world.steps >= world.max_steps
        world.done = true
    end
    
    world.total_reward += reward
    
    info = Dict(
        :energy_gained => action == :eat ? reward : 0.0,
        :moved => action in [:north, :south, :east, :west]
    )
    
    return observe(world), reward, world.done, info
end

"""
    move(pos, direction) → new_pos

Compute new position after moving in a direction.
"""
function move(pos::Tuple{Int, Int}, direction::Symbol)
    dx, dy = Dict(
        :north => (0, 1),
        :south => (0, -1),
        :east => (1, 0),
        :west => (-1, 0)
    )[direction]
    
    return (pos[1] + dx, pos[2] + dy)
end

"""
    is_valid_pos(world, pos) → Bool

Check if a position is valid (in bounds and not an obstacle).
"""
function is_valid_pos(world::GridWorld, pos::Tuple{Int, Int})
    x, y = pos
    return (1 <= x <= world.width && 
            1 <= y <= world.height && 
            pos ∉ world.obstacles)
end

"""
    render(world::GridWorld) → String

Render the world as a string.
"""
function render(world::GridWorld)
    lines = String[]
    
    push!(lines, "Energy: $(round(world.agent_energy, digits=1)) | Steps: $(world.steps) | Reward: $(round(world.total_reward, digits=1))")
    push!(lines, "+" * "-"^world.width * "+")
    
    for y in world.height:-1:1
        row = "|"
        for x in 1:world.width
            pos = (x, y)
            if pos == world.agent_pos
                row *= "@"
            elseif pos == world.goal_pos
                row *= "G"
            elseif pos in world.obstacles
                row *= "#"
            elseif haskey(world.food, pos)
                appearance = world.food[pos].appearance
                char = Dict(:circle => "●", :square => "■", :triangle => "▲")[appearance]
                row *= char
            else
                row *= " "
            end
        end
        row *= "|"
        push!(lines, row)
    end
    
    push!(lines, "+" * "-"^world.width * "+")
    
    return join(lines, "\n")
end

"""
    seed!(world::GridWorld, seed::Int)

Set the random seed.
"""
function seed!(world::GridWorld, seed::Int)
    world.rng = MersenneTwister(seed)
end

# Export
export GridWorld, spawn_food!, move, is_valid_pos
