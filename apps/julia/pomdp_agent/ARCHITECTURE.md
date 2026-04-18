# Bayesian Agent Framework: Architecture Design

## Vision

A general-purpose framework for building agents that:

1. **Maintain principled uncertainty** over world states and dynamics
2. **Maximise expected utility** under that uncertainty  
3. **Decide rationally** when to act vs. gather information
4. **Adapt to non-stationarity** as the world changes
5. **Connect to arbitrary environments** via a clean interface

The framework derives all behaviour from first principles—no ad-hoc exploration bonuses, no hardcoded heuristics, no loop detection hacks.

---

## Core Philosophy

### What Is Fixed (The Physics)

- **Bayesian inference** as the belief update rule
- **Expected utility maximisation** as the decision rule
- **Value of information** as the criterion for information-gathering
- **Coherence**: beliefs must form valid probability distributions

### What Is Learned

- **World dynamics**: P(s' | s, a)
- **Reward structure**: P(r | s, a)  
- **Sensor reliability**: P(observation | true_state, sensor)
- **State abstractions**: which distinctions matter for decision-making

### What Is Specified Per Domain

- **State space** (or how to discover it)
- **Action space** (or how to enumerate it)
- **Observation model** (how the world presents itself)
- **Utility function** (what the agent cares about)
- **Available sensors** (including LLMs, APIs, databases)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              WORLD INTERFACE                            │
│                                                                         │
│   Jericho (IF)  │  GridWorld  │  RealDataFeed  │  Gymnasium  │  ...    │
│                                                                         │
│   Provides: observations, valid_actions, rewards, terminal conditions   │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            BELIEF SYSTEM                                │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │  State Belief   │  │ Dynamics Belief │  │   Sensor Reliability    │ │
│  │                 │  │                 │  │                         │ │
│  │  P(s | history) │  │ P(s'|s,a)       │  │  P(obs | truth, sensor) │ │
│  │                 │  │ learned from    │  │  learned from outcomes  │ │
│  │  Updated via    │  │ experience      │  │                         │ │
│  │  Bayes' rule    │  │                 │  │  TPR, FPR per sensor    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │
│                                                                         │
│  State Abstraction: φ(concrete_state) → abstract_state                 │
│  Learns equivalence classes via bisimulation                           │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           SENSOR BANK                                   │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │  LLM Oracle  │  │  Database    │  │  API Query   │  │  Direct Obs │ │
│  │              │  │              │  │              │  │             │ │
│  │  Binary Q&A  │  │  Structured  │  │  External    │  │  From world │ │
│  │  TPR/FPR     │  │  lookup      │  │  data        │  │  interface  │ │
│  │  learned     │  │              │  │              │  │             │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
│                                                                         │
│  Each sensor: query(state, question) → observation                      │
│  Each has learned reliability parameters                                │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DECISION ENGINE                                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Value of Information                         │   │
│  │                                                                 │   │
│  │  VOI(query) = E[max_a EU(a) | after_query] - max_a EU(a) | now │   │
│  │                                                                 │   │
│  │  Decide: gather info iff VOI > cost                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Planner                                    │   │
│  │                                                                 │   │
│  │  Thompson Sampling over trajectories:                          │   │
│  │    1. Sample world model θ ~ P(θ | history)                    │   │
│  │    2. Plan optimal trajectory under θ (MCTS / tree search)     │   │
│  │    3. Execute first action                                     │   │
│  │    4. Observe outcome, update beliefs                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Planning horizon: configurable (1-step to full rollout)               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Type System (Julia)

### Abstract Types

```julia
# The fundamental abstractions

abstract type AbstractState end
abstract type AbstractAction end  
abstract type AbstractObservation end
abstract type AbstractWorld end
abstract type AbstractSensor end
abstract type AbstractBelief end
abstract type AbstractPlanner end

# A world provides the interface to an environment
abstract type World{S<:AbstractState, A<:AbstractAction, O<:AbstractObservation} end

# A sensor provides information about the world
abstract type Sensor{Q, R} end  # Q = query type, R = response type
```

### Core Interfaces

```julia
# World interface - what any environment must provide
function reset!(world::World) end
function step!(world::World, action) end  # returns (obs, reward, done, info)
function valid_actions(world::World) end
function observe(world::World) end

# Sensor interface - what any information source must provide  
function query(sensor::Sensor, state, question) end
function update_reliability!(sensor::Sensor, prediction, ground_truth) end
function reliability(sensor::Sensor) end  # returns (TPR, FPR) or equivalent

# Belief interface - what any belief system must provide
function update!(belief::AbstractBelief, observation) end
function sample(belief::AbstractBelief) end  # for Thompson sampling
function entropy(belief::AbstractBelief) end
function expected_value(belief::AbstractBelief, f::Function) end
```

---

## Module Structure

```
BayesianAgents/
├── src/
│   ├── BayesianAgents.jl          # Main module
│   │
│   ├── core/
│   │   ├── types.jl               # Abstract types and interfaces
│   │   ├── belief.jl              # Belief representations
│   │   ├── inference.jl           # Bayesian update machinery
│   │   └── utilities.jl           # Expected utility calculations
│   │
│   ├── dynamics/
│   │   ├── tabular.jl             # Discrete state/action dynamics
│   │   ├── continuous.jl          # Continuous dynamics (Gaussian processes)
│   │   ├── dirichlet.jl           # Dirichlet-Categorical for transitions
│   │   └── abstraction.jl         # State abstraction / bisimulation
│   │
│   ├── sensors/
│   │   ├── binary.jl              # Binary yes/no sensors (Beta-Bernoulli)
│   │   ├── categorical.jl         # Multi-class sensors (Dirichlet)
│   │   ├── llm.jl                 # LLM oracle interface
│   │   └── composite.jl           # Combining multiple sensors
│   │
│   ├── planning/
│   │   ├── myopic.jl              # One-step lookahead
│   │   ├── mcts.jl                # Monte Carlo Tree Search
│   │   ├── thompson.jl            # Thompson sampling over trajectories
│   │   └── voi.jl                 # Value of information calculations
│   │
│   └── worlds/
│       ├── interface.jl           # World interface definition
│       ├── gridworld.jl           # Simple grid world for testing
│       └── jericho.jl             # Interactive Fiction via Jericho
│
├── test/
│   ├── runtests.jl
│   ├── test_beliefs.jl
│   ├── test_planning.jl
│   └── test_integration.jl
│
└── examples/
    ├── gridworld_agent.jl
    ├── if_agent.jl
    └── multi_armed_bandit.jl
```

---

## Key Components

### 1. Belief System

```julia
"""
Belief over discrete state transitions using Dirichlet-Categorical.

For each (state, action) pair, maintains a Dirichlet distribution over
possible next states. Conjugate updating gives closed-form posteriors.
"""
struct DirichletDynamicsBelief{S} <: AbstractBelief
    # For each (s, a): Dirichlet parameters over next states
    # α[s][a] is a Dict{S, Float64} of concentration parameters
    α::Dict{S, Dict{Any, Dict{S, Float64}}}
    
    # Prior concentration (pseudocounts)
    prior_α::Float64
end

function update!(belief::DirichletDynamicsBelief{S}, s::S, a, s_next::S) where S
    # Conjugate update: increment concentration for observed transition
    if !haskey(belief.α, s)
        belief.α[s] = Dict()
    end
    if !haskey(belief.α[s], a)
        belief.α[s][a] = Dict{S, Float64}()
    end
    
    current = get(belief.α[s][a], s_next, belief.prior_α)
    belief.α[s][a][s_next] = current + 1.0
end

function sample_transition(belief::DirichletDynamicsBelief{S}, s::S, a) where S
    # Sample from Dirichlet, then sample next state from resulting Categorical
    if !haskey(belief.α, s) || !haskey(belief.α[s], a)
        # No data: uniform over known states
        return rand(keys(belief.α))
    end
    
    α_vec = collect(values(belief.α[s][a]))
    states = collect(keys(belief.α[s][a]))
    
    # Sample from Dirichlet
    θ = rand(Dirichlet(α_vec))
    
    # Sample state from Categorical(θ)
    return states[rand(Categorical(θ))]
end
```

### 2. Binary Sensor (e.g., LLM Yes/No)

```julia
"""
Binary sensor with learned reliability via Beta-Bernoulli model.

Tracks:
- TPR: P(says yes | actually true)  ~ Beta(α_tp, β_tp)
- FPR: P(says yes | actually false) ~ Beta(α_fp, β_fp)
"""
@kwdef mutable struct BinarySensor <: Sensor{String, Bool}
    name::String
    
    # TPR parameters
    α_tp::Float64 = 2.0
    β_tp::Float64 = 1.0
    
    # FPR parameters  
    α_fp::Float64 = 1.0
    β_fp::Float64 = 2.0
    
    # Query function (injected)
    query_fn::Function
end

# Point estimates
tpr(s::BinarySensor) = s.α_tp / (s.α_tp + s.β_tp)
fpr(s::BinarySensor) = s.α_fp / (s.α_fp + s.β_fp)

"""
Bayesian update of belief given sensor response.
"""
function posterior_belief(sensor::BinarySensor, prior::Float64, said_yes::Bool)
    if said_yes
        # P(true | yes) ∝ P(yes | true) P(true)
        numerator = tpr(sensor) * prior
        denominator = tpr(sensor) * prior + fpr(sensor) * (1 - prior)
    else
        # P(true | no) ∝ P(no | true) P(true)
        numerator = (1 - tpr(sensor)) * prior
        denominator = (1 - tpr(sensor)) * prior + (1 - fpr(sensor)) * (1 - prior)
    end
    
    return denominator > 0 ? numerator / denominator : prior
end

"""
Update reliability parameters from ground truth.
"""
function update_reliability!(sensor::BinarySensor, said_yes::Bool, actually_true::Bool)
    if actually_true
        if said_yes
            sensor.α_tp += 1
        else
            sensor.β_tp += 1
        end
    else
        if said_yes
            sensor.α_fp += 1
        else
            sensor.β_fp += 1
        end
    end
end
```

### 3. Value of Information

```julia
"""
Compute VOI for querying a binary sensor about whether action `a` helps.

VOI = E[max EU after query] - max EU now
"""
function compute_voi(
    sensor::BinarySensor,
    prior_helps::Float64,
    reward_if_helps::Float64,
    other_action_eu::Float64
)
    # Current best EU (before asking)
    eu_action = prior_helps * reward_if_helps
    current_best = max(eu_action, other_action_eu)
    
    # P(sensor says yes)
    p_yes = tpr(sensor) * prior_helps + fpr(sensor) * (1 - prior_helps)
    p_no = 1 - p_yes
    
    # Posterior if yes
    posterior_if_yes = posterior_belief(sensor, prior_helps, true)
    eu_if_yes = posterior_if_yes * reward_if_helps
    best_if_yes = max(eu_if_yes, other_action_eu)
    
    # Posterior if no
    posterior_if_no = posterior_belief(sensor, prior_helps, false)
    eu_if_no = posterior_if_no * reward_if_helps
    best_if_no = max(eu_if_no, other_action_eu)
    
    # Expected best EU after asking
    expected_best_after = p_yes * best_if_yes + p_no * best_if_no
    
    # VOI is the improvement
    return expected_best_after - current_best
end
```

### 4. Thompson Sampling Planner

```julia
"""
Thompson sampling over world models with configurable planning depth.
"""
struct ThompsonPlanner{B<:AbstractBelief} <: AbstractPlanner
    belief::B
    planning_depth::Int
    discount::Float64
end

function select_action(planner::ThompsonPlanner, state, valid_actions)
    # 1. Sample a world model from belief
    sampled_dynamics = sample(planner.belief)
    
    # 2. Plan under sampled model
    best_action = nothing
    best_value = -Inf
    
    for action in valid_actions
        value = evaluate_action(
            planner, 
            sampled_dynamics, 
            state, 
            action, 
            planner.planning_depth
        )
        if value > best_value
            best_value = value
            best_action = action
        end
    end
    
    return best_action
end

function evaluate_action(planner, dynamics, state, action, depth)
    if depth == 0
        return 0.0
    end
    
    # Predict next state and reward under sampled dynamics
    next_state, reward = predict(dynamics, state, action)
    
    # Recurse with greedy policy under same sampled dynamics
    future_value = 0.0
    if depth > 1
        future_actions = get_actions(dynamics, next_state)
        if !isempty(future_actions)
            future_value = maximum(
                evaluate_action(planner, dynamics, next_state, a, depth - 1)
                for a in future_actions
            )
        end
    end
    
    return reward + planner.discount * future_value
end
```

### 5. State Abstraction via Bisimulation

```julia
"""
Learn state equivalence classes where states are equivalent iff
they have the same (action → outcome) signature.

Two states s1, s2 are bisimilar if:
  ∀a: P(r | s1, a) = P(r | s2, a) AND P(φ(s') | s1, a) = P(φ(s') | s2, a)
"""
struct BisimulationAbstractor{S}
    # Signature: state → Dict{action → (reward_dist, next_abstract_state_dist)}
    signatures::Dict{S, Dict{Any, Tuple{Vector{Float64}, Dict{Int, Int}}}}
    
    # Equivalence classes: abstract_id → Set{concrete_states}
    classes::Dict{Int, Set{S}}
    
    # Reverse mapping: concrete → abstract
    state_to_class::Dict{S, Int}
    
    # Counter for new class IDs
    next_class_id::Int
end

function observe_transition!(
    abstractor::BisimulationAbstractor{S}, 
    s::S, 
    a, 
    r::Float64, 
    s_next::S
) where S
    # Update signature for state s
    if !haskey(abstractor.signatures, s)
        abstractor.signatures[s] = Dict()
    end
    
    # Get abstract state of s_next
    abstract_next = get_abstract(abstractor, s_next)
    
    # Record this transition in signature
    # (In practice, maintain sufficient statistics for distributions)
    update_signature!(abstractor.signatures[s], a, r, abstract_next)
    
    # Potentially re-cluster if signatures changed
    recompute_classes!(abstractor)
end

function get_abstract(abstractor::BisimulationAbstractor{S}, s::S) where S
    if haskey(abstractor.state_to_class, s)
        return abstractor.state_to_class[s]
    end
    
    # New state: create singleton class
    new_id = abstractor.next_class_id
    abstractor.next_class_id += 1
    abstractor.classes[new_id] = Set([s])
    abstractor.state_to_class[s] = new_id
    
    return new_id
end
```

---

## World Adapters

### Jericho (Interactive Fiction)

```julia
"""
Adapter for Jericho IF games via PyCall.
"""
struct JerichoWorld <: World{String, String, String}
    env::PyObject  # Jericho FrotzEnv
    game_path::String
    
    # Cached state
    current_obs::String
    current_score::Int
    done::Bool
end

function JerichoWorld(game_path::String)
    jericho = pyimport("jericho")
    env = jericho.FrotzEnv(game_path)
    obs, info = env.reset()
    
    return JerichoWorld(env, game_path, obs, 0, false)
end

function reset!(world::JerichoWorld)
    obs, info = world.env.reset()
    world.current_obs = obs
    world.current_score = 0
    world.done = false
    return obs
end

function step!(world::JerichoWorld, action::String)
    obs, reward, done, info = world.env.step(action)
    world.current_obs = obs
    world.current_score += reward
    world.done = done
    return (obs, reward, done, info)
end

function valid_actions(world::JerichoWorld)
    return world.env.get_valid_actions()
end

function state_hash(world::JerichoWorld)
    # Jericho provides deterministic state hashing
    return world.env.get_world_state_hash()
end
```

### Grid World (Testing)

```julia
"""
Simple grid world for testing the framework.
"""
struct GridWorld <: World{Tuple{Int,Int}, Symbol, Tuple{Int,Int}}
    width::Int
    height::Int
    agent_pos::Tuple{Int, Int}
    goal_pos::Tuple{Int, Int}
    obstacles::Set{Tuple{Int, Int}}
    
    # Food items with true energy values (unknown to agent)
    food::Dict{Tuple{Int, Int}, Float64}
end

function step!(world::GridWorld, action::Symbol)
    dx, dy = Dict(:north => (0,1), :south => (0,-1), 
                  :east => (1,0), :west => (-1,0))[action]
    
    new_pos = (world.agent_pos[1] + dx, world.agent_pos[2] + dy)
    
    # Check validity
    if is_valid_pos(world, new_pos)
        world.agent_pos = new_pos
    end
    
    # Check for food
    reward = get(world.food, world.agent_pos, 0.0)
    if reward != 0.0
        delete!(world.food, world.agent_pos)
    end
    
    # Check for goal
    done = world.agent_pos == world.goal_pos
    
    return (world.agent_pos, reward, done, Dict())
end
```

---

## Agent Loop

```julia
"""
Main agent loop: observe → decide → act → update.
"""
function run_episode!(
    agent::BayesianAgent,
    world::World,
    max_steps::Int = 1000
)
    reset!(world)
    total_reward = 0.0
    
    for step in 1:max_steps
        # 1. Observe current state
        obs = observe(world)
        state = agent.abstractor(obs)  # Abstract if needed
        actions = valid_actions(world)
        
        # 2. Decide: gather info or act?
        while should_query(agent, state, actions)
            query, sensor = select_query(agent, state, actions)
            response = query(sensor, state, query)
            update_belief!(agent, query, response)
        end
        
        # 3. Select action via planning
        action = select_action(agent.planner, state, actions)
        
        # 4. Execute and observe outcome
        obs_next, reward, done, info = step!(world, action)
        state_next = agent.abstractor(obs_next)
        
        # 5. Update beliefs from outcome
        update_dynamics!(agent.belief, state, action, reward, state_next)
        update_sensor_reliability!(agent, state, action, reward)
        
        total_reward += reward
        
        if done
            break
        end
    end
    
    return total_reward
end

function should_query(agent, state, actions)
    # Check if any query has VOI > cost
    best_voi = 0.0
    
    for sensor in agent.sensors
        for action in actions
            voi = compute_voi(sensor, agent.belief, state, action)
            if voi > sensor.cost && voi > best_voi
                best_voi = voi
            end
        end
    end
    
    return best_voi > 0
end
```

---

## Design Principles (Enforced)

### DO

1. **Derive behaviour from EU maximisation**: Every action choice must be argmax E[U]
2. **Use conjugate priors where possible**: Dirichlet-Categorical, Beta-Bernoulli, Normal-Normal
3. **Learn sensor reliability**: All sensors have uncertainty that updates with experience
4. **Separate concerns**: World interface, belief system, planning, sensors are independent
5. **Test with simple domains first**: Grid world before IF, single sensor before bank

### DO NOT

1. **No ad-hoc exploration bonuses**: Exploration emerges from uncertainty
2. **No loop detection hacks**: If agent loops, the model is wrong—fix the model
3. **No hardcoded heuristics**: Everything must derive from the mathematics
4. **No premature optimisation**: Correctness first, then performance
5. **No black-box LLM decisions**: LLM is a sensor with learned reliability, not a decision-maker

---

## Next Steps

1. **Implement core types** in `src/core/types.jl`
2. **Implement Dirichlet dynamics** in `src/dynamics/dirichlet.jl`
3. **Implement binary sensor** in `src/sensors/binary.jl`
4. **Implement Thompson planner** in `src/planning/thompson.jl`
5. **Create grid world** for testing
6. **Integration test**: agent learns grid world dynamics and navigates to goal
7. **Add Jericho adapter** and test on simple IF game
8. **Add LLM sensor** with VOI-based querying

---

## Dependencies

```toml
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"  # For Jericho
POMDPs = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"  # Optional: for POMDP solvers
```
