# Role: brain-side application
"""
    Goal-Directed Planning (Stage 5)

Extract goals from game text and plan toward them.

Mathematical basis:
- Goal g = logical predicate over state: {V₁=v₁, V₂=v₂, ...}
- Goal progress: P(goal satisfied | state)
- Goal-biased rollout: bias action selection toward increasing goal progress
- Intrinsic motivation: information gain as exploration signal

Key insight: Goals provide high-level objective for planning without specifying how.
"""

"""
    Goal

Represents a game objective as a conjunction of state predicates.

Fields:
- predicates::Dict{String, Any}     # Variable → required value
- description::String               # Human-readable description
- priority::Float64                 # Importance (0-1)
- achieved::Bool                    # Whether already satisfied
"""
mutable struct Goal
    predicates::Dict{String,Any}
    description::String
    priority::Float64
    achieved::Bool
end

"""
    extract_goals_from_text(text::String) → Vector{Goal}

Parse game text to extract implicit goals.

Heuristic: Look for keywords like "need", "must", "find", "get", etc.

Examples:
- "You need a light to see here" → Goal(lamp_lit=true)
- "The door is locked - you need a key" → Goal(key_found=true)
- "Find the map" → Goal(map ∈ inventory)
"""
function extract_goals_from_text(text::String)::Vector{Goal}
    goals = Goal[]
    text_lower = lowercase(text)

    # Light-related
    if contains(text_lower, "light") || contains(text_lower, "dark") || contains(text_lower, "can't see")
        push!(goals, Goal(
            Dict("lamp_lit" => true),
            "Need light to see",
            0.8,
            false
        ))
    end

    # Key-related
    if contains(text_lower, "locked") || contains(text_lower, "need.*key") || contains(text_lower, "unlock")
        push!(goals, Goal(
            Dict("key_found" => true),
            "Need key to unlock",
            0.8,
            false
        ))
    end

    # Item collection
    if contains(text_lower, "find") || contains(text_lower, "get") || contains(text_lower, "need")
        for item in ["map", "book", "scroll", "gem", "treasure"]
            if contains(text_lower, item)
                push!(goals, Goal(
                    Dict("$(item)_in_inventory" => true),
                    "Need to find $item",
                    0.7,
                    false
                ))
            end
        end
    end

    # Goal completion
    if contains(text_lower, "won") || contains(text_lower, "victory") || contains(text_lower, "success")
        push!(goals, Goal(
            Dict("game_won" => true),
            "Win the game",
            1.0,
            false
        ))
    end

    return goals
end

"""
    compute_goal_progress(state::MinimalState, goal::Goal) → Float64

Compute how much of the goal is satisfied in this state.

Returns: 0 = not satisfied, 1 = fully satisfied, 0-1 = partially satisfied
"""
function compute_goal_progress(state::MinimalState, goal::Goal)::Float64
    satisfied = 0
    total = length(goal.predicates)

    if total == 0
        return 1.0  # Empty goal trivially satisfied
    end

    for (var, required_val) in goal.predicates
        if var == "lamp_lit"
            # Simplified: check if lamp-related item in inventory
            if "lamp" in state.inventory || "lantern" in state.inventory
                satisfied += 1
            end
        elseif var == "key_found"
            if "key" in state.inventory
                satisfied += 1
            end
        elseif contains(var, "_in_inventory")
            item = split(var, "_")[1]
            if item in state.inventory
                satisfied += 1
            end
        end
    end

    return satisfied / total
end

"""
    expected_goal_progress(state::MinimalState, action::String, goal::Goal, model::FactoredWorldModel) → Float64

Estimate expected goal progress after taking action.

Uses model to predict next state distribution.
"""
function expected_goal_progress(state::MinimalState, action::String, goal::Goal, model::FactoredWorldModel)::Float64
    # Sample next states from model
    current_progress = compute_goal_progress(state, goal)

    next_progresses = Float64[]
    for _ in 1:5  # Sample multiple trajectories
        sampled = sample_dynamics(model)
        next_state = sample_next_state(model, state, action, sampled)
        push!(next_progresses, compute_goal_progress(next_state, goal))
    end

    mean_next_progress = mean(next_progresses)
    delta = mean_next_progress - current_progress

    return delta
end

"""
    goal_biased_action_selection(state::MinimalState, available_actions::Vector{String}, goals::Vector{Goal}, model::FactoredWorldModel) → String

Select action biased toward goal progress.

Higher priority goals have more influence.
"""
function goal_biased_action_selection(state::MinimalState, available_actions::Vector{String}, goals::Vector{Goal}, model::FactoredWorldModel)::String
    action_scores = Dict(a => 0.0 for a in available_actions)

    # Score each action based on goal progress
    for goal in goals
        if goal.achieved
            continue  # Skip achieved goals
        end

        goal_weight = goal.priority

        for action in available_actions
            delta = expected_goal_progress(state, action, goal, model)
            action_scores[action] += goal_weight * delta
        end
    end

    # Softmax selection (temperature for exploration)
    tau = 0.5
    scores = [action_scores[a] for a in available_actions]
    if maximum(scores) - minimum(scores) > 1e-6
        scores = scores .- minimum(scores)
        probs = exp.(scores ./ tau) ./ sum(exp.(scores ./ tau))
    else
        probs = ones(length(available_actions)) ./ length(available_actions)
    end

    # Sample action
    return available_actions[rand(Categorical(probs))]
end

"""
    update_goal_status!(goals::Vector{Goal}, state::MinimalState)

Update which goals have been achieved.
"""
function update_goal_status!(goals::Vector{Goal}, state::MinimalState)
    for goal in goals
        progress = compute_goal_progress(state, goal)
        if progress >= 0.99
            goal.achieved = true
        end
    end
end

export Goal, extract_goals_from_text, compute_goal_progress, expected_goal_progress
export goal_biased_action_selection, update_goal_status!
