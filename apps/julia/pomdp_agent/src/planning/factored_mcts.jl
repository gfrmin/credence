"""
    Factored MCTS (Stage 1: MVBN)

Monte Carlo Tree Search with Thompson Sampling using factored dynamics model.

Key insight: Instead of maintaining one MCTS tree, sample one concrete MDP
(from sampled dynamics) and plan in that MDP. This naturally balances
exploration (uncertain transitions) and exploitation (learned dynamics).

Algorithm:
1. Sample dynamics: θ ~ P(θ | history), G ~ P(G | history) [Thompson Sampling]
2. Sample initial state: s₀ ~ P(s | observation)
3. Run MCTS in sampled MDP with UCB exploration
4. Execute first action of best trajectory
5. Observe outcome, update beliefs
"""

"""
    FactoredMCTSNode

MCTS tree node for factored planning.

Fields:
- state::MinimalState
- depth::Int
- visits::Int
- value::Float64                    # Cumulative return
- children::Dict{String, FactoredMCTSNode}  # action → child node
"""
mutable struct FactoredMCTSNode
    state::MinimalState
    depth::Int
    visits::Int
    value::Float64
    children::Dict{String, FactoredMCTSNode}

    function FactoredMCTSNode(state::MinimalState, depth::Int=0)
        new(state, depth, 0, 0.0, Dict{String, FactoredMCTSNode}())
    end
end

"""
    FactoredMCTS

Thompson Sampling MCTS for factored world model.

Fields:
- model::FactoredWorldModel
- state_belief::StateBelief
- horizon::Int                      # Planning horizon
- n_iterations::Int                 # MCTS iterations per decision
- ucb_c::Float64                   # UCB exploration constant
- discount::Float64                # Discount factor γ
"""
struct FactoredMCTS <: Planner
    model::FactoredWorldModel
    state_belief::StateBelief
    horizon::Int
    n_iterations::Int
    ucb_c::Float64
    discount::Float64

    function FactoredMCTS(model::FactoredWorldModel, state_belief::StateBelief;
                        horizon::Int=10, n_iterations::Int=100, ucb_c::Float64=2.0, discount::Float64=0.99)
        new(model, state_belief, horizon, n_iterations, ucb_c, discount)
    end
end

"""
    select_action_ucb(node::FactoredMCTSNode, ucb_c::Float64) → String

Select child action using UCB1 criterion.
UCB(a) = Q(a) + c * sqrt(ln(N) / n(a))
"""
function select_action_ucb(node::FactoredMCTSNode, ucb_c::Float64)::String
    if isempty(node.children)
        return ""  # No children (leaf)
    end

    best_action = ""
    best_ucb = -Inf

    for (action, child) in node.children
        q = if child.visits > 0
            child.value / child.visits
        else
            0.0
        end

        ucb = q + ucb_c * sqrt(log(node.visits) / max(child.visits, 1))

        if ucb > best_ucb
            best_ucb = ucb
            best_action = action
        end
    end

    return best_action
end

"""
    mcts_search(planner::FactoredMCTS, root_state::MinimalState, sampled::SampledFactoredDynamics, available_actions::Vector{String})

Run MCTS and return best action.
"""
function mcts_search(planner::FactoredMCTS, root_state::MinimalState, sampled::SampledFactoredDynamics, available_actions::Vector{String})::String
    root = FactoredMCTSNode(root_state)

    for _ in 1:planner.n_iterations
        # Selection & Expansion
        node = root
        gamma_factor = 1.0

        for depth in 1:planner.horizon
            if node.depth > 0 && isempty(node.children)
                break  # Leaf node
            end

            # Expand if needed
            if isempty(node.children)
                for action in available_actions
                    next_state = sample_next_state(planner.model, node.state, action, sampled)
                    node.children[action] = FactoredMCTSNode(next_state, depth)
                end
            end

            # Select action by UCB
            action = select_action_ucb(node, planner.ucb_c)
            if isempty(action)
                break
            end

            # Get reward and transition
            rd = reward_dist(planner.model, node.state, action)
            reward = isfinite(mean(rd)) ? mean(rd) : 0.0

            # Move to child
            node = node.children[action]
            node.value += gamma_factor * reward
            node.visits += 1

            gamma_factor *= planner.discount
        end
    end

    # Return best action at root
    best_action = ""
    best_value = -Inf

    for (action, child) in root.children
        value = if child.visits > 0
            child.value / child.visits
        else
            0.0
        end

        if value > best_value
            best_value = value
            best_action = action
        end
    end

    return best_action
end

"""
    plan(planner::FactoredMCTS, state::MinimalState, available_actions::Vector{String}) → String

Plan via Thompson Sampling: sample dynamics, sample initial state, run MCTS.
"""
function BayesianAgents.plan(planner::FactoredMCTS, state::MinimalState, available_actions::Vector{String})::String
    # Thompson Sampling: sample one concrete MDP
    sampled_dynamics = sample_dynamics(planner.model)

    # Run MCTS in sampled MDP
    best_action = mcts_search(planner, state, sampled_dynamics, available_actions)

    if isempty(best_action)
        # Fallback: random action
        return rand(available_actions)
    end

    return best_action
end

export FactoredMCTS, FactoredMCTSNode, mcts_search
