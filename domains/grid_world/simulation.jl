"""
    grid_world.jl — 5×5 grid world simulation for program-space agent

Entities have hidden types (FOOD, ENEMY, NEUTRAL) and observable properties.
The environment maintains a true state vector per entity, projected through
the agent's SensorConfig to produce sensor readings.

World rules determine classification logic. The active rule can change
mid-run (regime change) without notification to the agent.
"""

include("sensors.jl")

# ── Enums ──

@enum EntityKind FOOD ENEMY NEUTRAL
@enum TerrainType GROUND WALL MUD
@enum Action MOVE_N MOVE_S MOVE_E MOVE_W INTERACT OBSERVE WAIT
@enum MovementPattern STATIONARY PATROL CHASE WANDER

# ── Types ──

struct Pos
    x::Int
    y::Int
end

mutable struct Entity
    kind::EntityKind
    pos::Pos
    rgb::Tuple{Float64, Float64, Float64}
    speed::Float64
    energy::Float64           # energy delta on interaction
    movement::MovementPattern
    patrol_path::Vector{Pos}  # for PATROL
    patrol_idx::Int
    alive::Bool
end

struct WorldConfig
    grid_size::Int
    terrain::Matrix{TerrainType}
    rule_name::Symbol
end

mutable struct WorldState
    config::WorldConfig
    entities::Vector{Entity}
    agent_pos::Pos
    agent_energy::Float64
    step::Int
    last_feedback::Union{Nothing, Float64}
    last_interaction_entity::Union{Nothing, Int}
    rng_seed::Int
end

# ── Temporal state for sensor transforms ──

mutable struct TemporalWindow
    history::Dict{Int, Vector{Vector{Float64}}}  # entity_id → past true states
    max_history::Int
end

TemporalWindow(; max_history::Int=10) =
    TemporalWindow(Dict{Int, Vector{Vector{Float64}}}(), max_history)

function update!(tw::TemporalWindow, entity_states::Vector{Tuple{Int, Vector{Float64}}})
    for (eid, state) in entity_states
        hist = get!(tw.history, eid, Vector{Float64}[])
        push!(hist, state)
        while length(hist) > tw.max_history
            popfirst!(hist)
        end
    end
end

# ── World creation ──

function default_terrain(n::Int)
    t = fill(GROUND, n, n)
    # Add some walls along edges and a mud patch
    for i in 1:n
        t[1, i] = WALL
        t[n, i] = WALL
        t[i, 1] = WALL
        t[i, n] = WALL
    end
    t[3, 3] = MUD
    t
end

function create_entity(kind::EntityKind, pos::Pos, rule::Symbol)
    rgb, speed, energy, movement = entity_properties(kind, rule)
    Entity(kind, pos, rgb, speed, energy, movement, Pos[], 1, true)
end

function entity_properties(kind::EntityKind, rule::Symbol)
    if rule == :all_food
        rgb = (0.3 + 0.4 * rand(), 0.3 + 0.4 * rand(), 0.3 + 0.4 * rand())
        return (rgb, 0.0, 5.0, STATIONARY)

    elseif rule == :colour_typed
        if kind == ENEMY
            return ((0.9, 0.1, 0.1), 0.0, -5.0, STATIONARY)
        elseif kind == FOOD
            return ((0.1, 0.1, 0.9), 0.0, 5.0, STATIONARY)
        else
            return ((0.1, 0.9, 0.1), 0.0, 0.0, STATIONARY)
        end

    elseif rule == :motion_typed
        if kind == ENEMY
            return ((0.5, 0.5, 0.5), 0.8, -5.0, CHASE)
        elseif kind == FOOD
            return ((0.5, 0.5, 0.5), 0.0, 5.0, STATIONARY)
        else
            return ((0.5, 0.5, 0.5), 0.3, 0.0, WANDER)
        end

    elseif rule == :territorial
        if kind == ENEMY
            # Enemies near walls
            return ((0.5, 0.5, 0.5), 0.0, -5.0, STATIONARY)
        elseif kind == FOOD
            # Food in centre
            return ((0.5, 0.5, 0.5), 0.0, 5.0, STATIONARY)
        else
            return ((0.5, 0.5, 0.5), 0.0, 0.0, STATIONARY)
        end

    elseif rule == :mixed
        if kind == ENEMY
            return ((0.9, 0.1, 0.1), 0.8, -5.0, CHASE)
        elseif kind == FOOD
            return ((0.1, 0.1, 0.9), 0.0, 5.0, STATIONARY)
        else
            return ((0.1, 0.9, 0.1), 0.3, 0.0, WANDER)
        end

    else
        error("Unknown rule: $rule")
    end
end

function spawn_entities(rule::Symbol, grid_size::Int; n_entities::Int=6)
    entities = Entity[]
    kinds = [FOOD, FOOD, ENEMY, ENEMY, NEUTRAL, FOOD]
    for (i, kind) in enumerate(kinds[1:min(n_entities, length(kinds))])
        # Place in interior (avoid walls)
        x = rand(2:grid_size-1)
        y = rand(2:grid_size-1)
        e = create_entity(kind, Pos(x, y), rule)

        # For territorial rule, place enemies near walls, food in centre
        if rule == :territorial
            if kind == ENEMY
                # Near edge
                if rand() < 0.5
                    e.pos = Pos(2, rand(2:grid_size-1))
                else
                    e.pos = Pos(grid_size-1, rand(2:grid_size-1))
                end
            elseif kind == FOOD
                cx, cy = div(grid_size, 2) + 1, div(grid_size, 2) + 1
                e.pos = Pos(cx + rand(-1:1), cy + rand(-1:1))
            end
        end

        push!(entities, e)
    end
    entities
end

function create_world(rule::Symbol; grid_size::Int=5, n_entities::Int=6)
    terrain = default_terrain(grid_size)
    config = WorldConfig(grid_size, terrain, rule)
    entities = spawn_entities(rule, grid_size; n_entities)
    cx = div(grid_size, 2) + 1
    WorldState(config, entities, Pos(cx, cx), 100.0, 0, nothing, nothing, 42)
end

# ── Regime change ──

function set_rule!(state::WorldState, rule::Symbol)
    new_config = WorldConfig(state.config.grid_size, state.config.terrain, rule)
    state.config = new_config
    # Re-assign entity properties under new rule
    for e in state.entities
        e.alive || continue
        rgb, speed, energy, movement = entity_properties(e.kind, rule)
        e.rgb = rgb
        e.speed = speed
        e.energy = energy
        e.movement = movement
    end
end

# ── Entity true state vector ──

function entity_true_state(e::Entity, agent_pos::Pos, grid_size::Int)::Vector{Float64}
    wall_dist = min(e.pos.x - 1, e.pos.y - 1,
                    grid_size - e.pos.x, grid_size - e.pos.y) / (grid_size / 2)
    agent_dist = sqrt((e.pos.x - agent_pos.x)^2 + (e.pos.y - agent_pos.y)^2) /
                 (sqrt(2) * grid_size)
    Float64[
        e.rgb[1],           # 0: red
        e.rgb[2],           # 1: green
        e.rgb[3],           # 2: blue
        e.pos.x / grid_size, # 3: x_norm
        e.pos.y / grid_size, # 4: y_norm
        e.speed,             # 5: speed
        clamp(wall_dist, 0.0, 1.0),  # 6: wall_dist
        clamp(agent_dist, 0.0, 1.0), # 7: agent_dist
    ]
end

# ── Entity movement ──

function move_entity!(e::Entity, agent_pos::Pos, grid_size::Int)
    e.alive || return
    e.speed == 0.0 && return

    if e.movement == STATIONARY
        return
    elseif e.movement == CHASE
        # Step toward agent
        dx = sign(agent_pos.x - e.pos.x)
        dy = sign(agent_pos.y - e.pos.y)
        new_pos = Pos(clamp(e.pos.x + dx, 2, grid_size - 1),
                      clamp(e.pos.y + dy, 2, grid_size - 1))
        e.pos = new_pos
    elseif e.movement == PATROL
        if !isempty(e.patrol_path)
            e.patrol_idx = mod1(e.patrol_idx + 1, length(e.patrol_path))
            e.pos = e.patrol_path[e.patrol_idx]
        end
    elseif e.movement == WANDER
        dx = rand(-1:1)
        dy = rand(-1:1)
        new_pos = Pos(clamp(e.pos.x + dx, 2, grid_size - 1),
                      clamp(e.pos.y + dy, 2, grid_size - 1))
        e.pos = new_pos
    end
end

# ── World step ──

function world_step!(state::WorldState, action::Action)
    state.step += 1
    state.last_feedback = nothing
    state.last_interaction_entity = nothing
    gs = state.config.grid_size

    # Move entities
    for e in state.entities
        move_entity!(e, state.agent_pos, gs)
    end

    # Execute agent action
    if action == MOVE_N
        new_y = max(2, state.agent_pos.y - 1)
        state.agent_pos = Pos(state.agent_pos.x, new_y)
        state.agent_energy -= 1.0
    elseif action == MOVE_S
        new_y = min(gs - 1, state.agent_pos.y + 1)
        state.agent_pos = Pos(state.agent_pos.x, new_y)
        state.agent_energy -= 1.0
    elseif action == MOVE_E
        new_x = min(gs - 1, state.agent_pos.x + 1)
        state.agent_pos = Pos(new_x, state.agent_pos.y)
        state.agent_energy -= 1.0
    elseif action == MOVE_W
        new_x = max(2, state.agent_pos.x - 1)
        state.agent_pos = Pos(new_x, state.agent_pos.y)
        state.agent_energy -= 1.0
    elseif action == INTERACT
        # Interact with nearest adjacent entity
        best_dist = Inf
        best_idx = 0
        for (i, e) in enumerate(state.entities)
            e.alive || continue
            d = abs(e.pos.x - state.agent_pos.x) + abs(e.pos.y - state.agent_pos.y)
            d <= 1 || continue
            if d < best_dist
                best_dist = d
                best_idx = i
            end
        end
        if best_idx > 0
            e = state.entities[best_idx]
            state.last_feedback = e.energy
            state.last_interaction_entity = best_idx
            state.agent_energy += e.energy
            e.alive = false
        end
    elseif action == OBSERVE
        state.agent_energy -= 1.0
    # WAIT: do nothing
    end

    state.last_feedback
end

# ── Queries ──

has_feedback(state::WorldState) = state.last_feedback !== nothing
get_feedback(state::WorldState) = state.last_feedback

function get_visible_entities(state::WorldState)::Vector{Tuple{Int, Entity}}
    [(i, e) for (i, e) in enumerate(state.entities) if e.alive]
end

function get_entity_states(state::WorldState)::Vector{Tuple{Int, Vector{Float64}}}
    [(i, entity_true_state(e, state.agent_pos, state.config.grid_size))
     for (i, e) in enumerate(state.entities) if e.alive]
end

"""Get nearest entity within interaction range."""
function nearest_entity(state::WorldState)::Union{Nothing, Tuple{Int, Entity}}
    best_dist = Inf
    best = nothing
    for (i, e) in enumerate(state.entities)
        e.alive || continue
        d = abs(e.pos.x - state.agent_pos.x) + abs(e.pos.y - state.agent_pos.y)
        if d < best_dist
            best_dist = d
            best = (i, e)
        end
    end
    best
end
