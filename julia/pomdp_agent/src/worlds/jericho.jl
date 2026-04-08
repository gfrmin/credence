"""
    JerichoWorld

Adapter for Interactive Fiction games via the Jericho library.

Jericho is a Python library, so we use PyCall to interface with it.
The adapter presents IF games through the standard World interface.

# Requirements

Install Jericho in your Python environment:
```
pip install jericho
```

Download game files (.z5, .z8) from the IF Archive:
https://www.ifarchive.org/
"""

using PyCall

# Lazy import of Jericho
const jericho = PyNULL()

function __init_jericho__()
    if ispynull(jericho)
        copy!(jericho, pyimport("jericho"))
    end
end

"""
    JerichoWorld

An Interactive Fiction game world via Jericho.
"""
mutable struct JerichoWorld <: World
    # Jericho environment
    env::PyObject
    game_path::String
    
    # Current state
    current_obs::String
    current_score::Int
    max_score::Int
    done::Bool
    
    # Episode tracking
    steps::Int
    max_steps::Int
    
    # Cached valid actions
    cached_actions::Vector{String}
    actions_cached_at_hash::UInt64
end

"""
    JerichoWorld(game_path; max_steps=100)

Create a Jericho world from a game file.

# Arguments
- `game_path`: Path to the .z5 or .z8 game file
- `max_steps`: Maximum steps per episode
"""
function JerichoWorld(game_path::String; max_steps::Int = 100)
    __init_jericho__()
    
    env = jericho.FrotzEnv(game_path)
    obs, info = env.reset()
    
    return JerichoWorld(
        env, game_path,
        string(obs), 0, env.get_max_score(),
        false,
        0, max_steps,
        String[], UInt64(0)
    )
end

"""
    reset!(world::JerichoWorld) → observation

Reset the game to its initial state.
"""
function reset!(world::JerichoWorld)
    obs, info = world.env.reset()
    
    world.current_obs = string(obs)
    world.current_score = 0
    world.done = false
    world.steps = 0
    world.cached_actions = String[]
    world.actions_cached_at_hash = UInt64(0)
    
    return observe(world)
end

"""
    observe(world::JerichoWorld) → NamedTuple

Return the current observation as a structured object.
"""
function observe(world::JerichoWorld)
    return (
        text = world.current_obs,
        score = world.current_score,
        max_score = world.max_score,
        steps = world.steps,
        location = safe_get_location(world),
        inventory = safe_get_inventory(world),
        state_hash = safe_get_hash(world)
    )
end

"""
    actions(world::JerichoWorld, observation=nothing) → Vector{String}

Return valid actions from the current state.

Uses Jericho's action recognition, which combines:
- Parser-valid actions (syntactically correct)
- Object-based actions (interact with visible objects)
"""
function actions(world::JerichoWorld, observation = nothing)
    current_hash = safe_get_hash(world)
    
    # Return cached if still valid
    if current_hash == world.actions_cached_at_hash && !isempty(world.cached_actions)
        return world.cached_actions
    end
    
    # Get valid actions from Jericho
    try
        valid = world.env.get_valid_actions()
        world.cached_actions = [string(a) for a in valid]
        world.actions_cached_at_hash = current_hash
    catch e
        # Fallback to basic actions if Jericho fails
        world.cached_actions = ["look", "inventory", "north", "south", "east", "west", "up", "down"]
    end
    
    return world.cached_actions
end

"""
    step!(world::JerichoWorld, action::String) → (observation, reward, done, info)

Execute an action in the game.
"""
function step!(world::JerichoWorld, action::String)
    if world.done
        return observe(world), 0.0, true, Dict()
    end
    
    world.steps += 1
    
    # Execute action
    obs, reward, done, info = world.env.step(action)
    
    world.current_obs = string(obs)
    world.current_score += Int(reward)
    world.done = done || world.steps >= world.max_steps
    
    # Invalidate action cache
    world.cached_actions = String[]
    
    return observe(world), Float64(reward), world.done, Dict(
        :info => info,
        :action => action
    )
end

"""
    render(world::JerichoWorld) → String

Return the current game text.
"""
function render(world::JerichoWorld)
    return """
    ┌─────────────────────────────────────────────────────────────────┐
    │ Score: $(world.current_score)/$(world.max_score) │ Steps: $(world.steps) │ Done: $(world.done)
    ├─────────────────────────────────────────────────────────────────┤
    $(world.current_obs)
    └─────────────────────────────────────────────────────────────────┘
    """
end

"""
    seed!(world::JerichoWorld, seed::Int)

Set the random seed (if the game supports it).
"""
function seed!(world::JerichoWorld, seed::Int)
    try
        world.env.seed(seed)
    catch
        # Not all games support seeding
    end
end

# ============================================================================
# SAFE ACCESSORS (handle Jericho API errors gracefully)
# ============================================================================

function safe_get_location(world::JerichoWorld)
    try
        loc = world.env.get_player_location()
        return isnothing(loc) ? "unknown" : string(loc.name)
    catch
        return "unknown"
    end
end

function safe_get_inventory(world::JerichoWorld)
    try
        inv = world.env.get_inventory()
        return string(inv)
    catch
        return ""
    end
end

function safe_get_hash(world::JerichoWorld)
    try
        return UInt64(hash(world.env.get_state()))
    catch
        return UInt64(hash(world.current_obs))
    end
end

# ============================================================================
# STATE SERIALIZATION (for model learning)
# ============================================================================

"""
    state_signature(world::JerichoWorld) → String

Return a compact state signature for the current game state.
This is used as the state key in the world model.
"""
function state_signature(world::JerichoWorld)
    loc = safe_get_location(world)
    inv = safe_get_inventory(world)
    
    # Hash inventory to keep signature compact
    inv_hash = string(hash(inv), base=16)[1:8]
    
    return "$(loc)|$(inv_hash)"
end

"""
    detailed_state(world::JerichoWorld) → Dict

Return a detailed state representation for debugging.
"""
function detailed_state(world::JerichoWorld)
    return Dict(
        :location => safe_get_location(world),
        :inventory => safe_get_inventory(world),
        :score => world.current_score,
        :text => world.current_obs,
        :hash => safe_get_hash(world)
    )
end

# ============================================================================
# LLM INTEGRATION HELPERS
# ============================================================================

"""
    observation_for_llm(world::JerichoWorld) → String

Format the current observation for LLM queries.
"""
function observation_for_llm(world::JerichoWorld)
    return """
    Location: $(safe_get_location(world))
    Score: $(world.current_score)/$(world.max_score)
    
    $(world.current_obs)
    
    Inventory: $(safe_get_inventory(world))
    """
end

"""
    action_question(action::String, world::JerichoWorld) → String

Generate a yes/no question about whether an action will help.
"""
function action_question(action::String, world::JerichoWorld)
    context = observation_for_llm(world)
    return """
    Context:
    $context
    
    Question: Will the action "$action" help make progress toward winning the game?
    Answer only 'yes' or 'no'.
    """
end

# Export
export JerichoWorld, state_signature, detailed_state, observation_for_llm, action_question
