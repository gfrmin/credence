# Role: brain-side application
"""
    BayesianAgents

A framework for Bayes-Adaptive POMDPs — agents that maintain uncertainty over
both world state and world dynamics, planning in a way that naturally balances
exploration and exploitation.

All behaviour derives from expected utility maximisation. No hacks.
"""
module BayesianAgents

using Random
using Distributions
using LinearAlgebra
using Logging
using Credence

# ============================================================================
# CORE ABSTRACT TYPES
# ============================================================================

"""
    World

Abstract interface for environments the agent can interact with.
Implementations: JerichoWorld, GridWorld, GymnasiumWorld, etc.
"""
abstract type World end

"""
    Sensor

Abstract interface for information sources with learnable reliability.
Implementations: LLMSensor, HeuristicSensor, OracleSensor, etc.
"""
abstract type Sensor end

"""
    WorldModel

Abstract interface for Bayesian models of world dynamics.
Implementations: TabularWorldModel, GPWorldModel, NeuralWorldModel, etc.
"""
abstract type WorldModel end

"""
    Planner

Abstract interface for planning algorithms.
Implementations: ThompsonMCTS, POMCP, ValueIteration, etc.
"""
abstract type Planner end

"""
    StateAbstractor

Abstract interface for learning state equivalence classes.
Implementations: BisimulationAbstractor, IdentityAbstractor, etc.
"""
abstract type StateAbstractor end

# ============================================================================
# WORLD INTERFACE
# ============================================================================

"""
    reset!(world::World) → observation

Reset the world to its initial state and return the initial observation.
"""
function reset! end

"""
    step!(world::World, action) → (observation, reward, done, info)

Execute an action in the world and return the result.
"""
function step! end

"""
    actions(world::World, observation) → Vector

Return the available actions given the current observation.
"""
function actions end

"""
    render(world::World) → String

Return a human-readable representation of the world state.
Optional — defaults to empty string.
"""
render(::World) = ""

"""
    seed!(world::World, seed::Int)

Set the random seed for the world. Optional.
"""
seed!(::World, ::Int) = nothing

# ============================================================================
# SENSOR INTERFACE
# ============================================================================

"""
    query(sensor::Sensor, state, question) → answer

Query the sensor about a question given the current state.
"""
function query end

"""
    tpr(sensor::Sensor) → Float64

Return the true positive rate: P(positive | true).
"""
function tpr end

"""
    fpr(sensor::Sensor) → Float64

Return the false positive rate: P(positive | false).
"""
function fpr end

"""
    update_reliability!(sensor::Sensor, predicted::Bool, actual::Bool)

Update the sensor's reliability estimates from ground truth.
"""
function update_reliability! end

"""
    posterior(sensor::Sensor, prior::Float64, answer::Bool) → Float64

Compute the posterior probability given prior and sensor answer.
Uses Bayes' rule with learned TPR/FPR.
"""
function posterior(sensor::Sensor, prior::Float64, answer::Bool)::Float64
    t = tpr(sensor)
    f = fpr(sensor)
    
    if answer  # Sensor said "yes"
        numerator = t * prior
        denominator = t * prior + f * (1 - prior)
    else  # Sensor said "no"
        numerator = (1 - t) * prior
        denominator = (1 - t) * prior + (1 - f) * (1 - prior)
    end
    
    return denominator > 0 ? numerator / denominator : prior
end

# ============================================================================
# WORLD MODEL INTERFACE
# ============================================================================

"""
    sample_dynamics(model::WorldModel) → sampled_model

Sample a concrete dynamics model from the posterior for Thompson Sampling.
"""
function sample_dynamics end

"""
    update!(model::WorldModel, s, a, r, s′)

Update the model posterior with an observed transition.
"""
function update! end

"""
    transition_dist(model::WorldModel, s, a) → Distribution

Return the posterior predictive distribution over next states.
"""
function transition_dist end

"""
    reward_dist(model::WorldModel, s, a) → Distribution

Return the posterior predictive distribution over rewards.
"""
function reward_dist end

"""
    entropy(model::WorldModel) → Float64

Return the entropy of the model posterior (uncertainty measure).
"""
function entropy end

# ============================================================================
# STATE ABSTRACTOR INTERFACE
# ============================================================================

"""
    abstract_state(abstractor::StateAbstractor, observation) → abstract_state

Map a concrete observation to an abstract state.
"""
function abstract_state end

"""
    record_transition!(abstractor::StateAbstractor, s, a, r, s′)

Record a transition for learning equivalence classes.
"""
function record_transition! end

"""
    check_contradiction(abstractor::StateAbstractor) → Option{Contradiction}

Check for contradictions (same abstract state, different outcomes).
"""
function check_contradiction end

"""
    refine!(abstractor::StateAbstractor, contradiction)

Refine the abstraction to resolve a contradiction.
"""
function refine! end

# ============================================================================
# PLANNER INTERFACE
# ============================================================================

"""
    plan(planner::Planner, belief, world_model, actions) → action

Plan and return the best action given current belief and model.
"""
function plan end

# ============================================================================
# VALUE OF INFORMATION
# ============================================================================

"""
    _reward_mean(rd) → Float64

Extract the mean from a reward distribution. Handles both credence's
NormalGammaMeasure (returns μ) and Distributions.jl types.
"""
_reward_mean(rd::NormalGammaMeasure) = Credence.mean(rd)
_reward_mean(rd) = _reward_mean(rd)

"""
    get_kappa(model::WorldModel, state, action) → Float64

Extract κ (precision / pseudo-observation count) from the reward posterior
for a given (state, action) pair. Returns the prior κ if no observations exist.
"""
function get_kappa(model::WorldModel, state, action)
    key = (state, action)
    if hasfield(typeof(model), :reward_posterior) && haskey(model.reward_posterior, key)
        return model.reward_posterior[key].κ
    elseif hasfield(typeof(model), :reward_prior)
        return model.reward_prior.κ
    else
        return 1.0
    end
end

"""
    compute_voi(sensor::Sensor, oracle_beliefs::Dict, obs_key, target_action,
                model::WorldModel, state, actions, n_actions::Int) → Float64

Compute the value of information for querying the sensor about a specific action.

Uses separate likelihood models:
- Reward posterior (Normal-Gamma): updated by real (s,a,r) only
- Oracle beliefs (scalar [0,1]): updated by sensor queries only

EU(a) = μ_reward(s,a) + oracle_belief(a) × C × κ_prior/κ(s,a)

The oracle bonus decays as real rewards accumulate (κ grows), providing
automatic transition from oracle-dominated to data-dominated decisions.
"""
function compute_voi(sensor::Sensor, oracle_beliefs::Dict, obs_key, target_action,
                     model::WorldModel, state, actions, n_actions::Int)
    w = tpr(sensor) - fpr(sensor)
    w <= 0 && return 0.0  # Uninformative sensor

    C = 1.0  # oracle bonus scale
    default_belief = 1.0 / n_actions
    κ_prior = hasfield(typeof(model), :reward_prior) ? model.reward_prior.κ : 1.0

    # EU for each action: reward posterior mean + oracle bonus
    function eu(a)
        rd = reward_dist(model, state, a)
        μ = _reward_mean(rd)
        μ = isfinite(μ) ? μ : 0.0
        belief = get(oracle_beliefs, (obs_key, string(a)), default_belief)
        κ_a = get_kappa(model, state, a)
        return μ + belief * C * κ_prior / κ_a
    end

    current_best = maximum(eu, actions)

    # Current belief for target action
    target_str = string(target_action)
    belief_a = get(oracle_beliefs, (obs_key, target_str), default_belief)

    # P(sensor says yes) for target action
    t = tpr(sensor)
    f = fpr(sensor)
    p_yes = t * belief_a + f * (1 - belief_a)
    p_no = 1.0 - p_yes

    # Hypothetical beliefs after yes/no
    belief_yes = posterior(sensor, belief_a, true)
    belief_no = posterior(sensor, belief_a, false)

    # Only target action's EU changes after query
    rd_target = reward_dist(model, state, target_action)
    μ_target = let m = Distributions.mean(rd_target); isfinite(m) ? m : 0.0 end
    κ_target = get_kappa(model, state, target_action)

    eu_target_yes = μ_target + belief_yes * C * κ_prior / κ_target
    eu_target_no = μ_target + belief_no * C * κ_prior / κ_target

    other_best = maximum(a -> a == target_action ? -Inf : eu(a), actions; init=-Inf)

    best_after_yes = max(other_best, eu_target_yes)
    best_after_no = max(other_best, eu_target_no)

    expected_best = p_yes * best_after_yes + p_no * best_after_no
    return max(expected_best - current_best, 0.0)
end

# ============================================================================
# CONFIGURATION
# ============================================================================

"""
    AgentConfig

Configuration for the Bayesian agent.
"""
Base.@kwdef struct AgentConfig
    # Planning
    planning_depth::Int = 10
    mcts_iterations::Int = 100
    discount::Float64 = 0.99
    ucb_c::Float64 = 2.0

    # Sensors
    sensor_cost::Float64 = 0.01
    max_queries_per_step::Int = 10

    # Priors
    transition_prior_strength::Float64 = 0.1
    reward_prior_mean::Float64 = 0.0
    reward_prior_variance::Float64 = 1.0

    # State abstraction
    abstraction_threshold::Float64 = 0.95

    # Learning mechanism frequencies (computational efficiency tuning)
    # All mechanisms run unconditionally; these control HOW OFTEN
    variable_discovery_frequency::Int = 10      # Run every N steps
    variable_bic_threshold::Float64 = 0.0       # BIC improvement threshold for new variables
    structure_learning_frequency::Int = 50      # Run every N transitions
    max_parents::Int = 3                        # Max parents in learned causal structure
    schema_discovery_frequency::Int = 100       # Run every N transitions
    goal_rollout_bias::Float64 = 0.5           # Blend: (1-bias)*random + bias*goal-directed
end

# ============================================================================
# DECISION INFO (per-step logging)
# ============================================================================

"""
    DecisionInfo

Diagnostic information from a single act! call.
Populated each step for logging/debugging without needing --debug.
"""
struct DecisionInfo
    llm_selected::Union{Nothing, Any}     # What the LLM recommended
    action_chosen::Any                     # What MCTS actually picked
    top_reward_means::Vector{Pair{Any,Float64}} # Top 3 (action, posterior mean) pairs
    top_oracle_beliefs::Vector{Pair{Any,Float64}} # Top 3 (action, oracle belief) pairs
    n_selfloops::Int                      # Confirmed selfloops in current state
    n_actions_available::Int               # Total actions
    n_voi_queries::Int                    # How many oracle beliefs were updated from LLM
    n_sensor_queries::Int                  # Number of sensor queries this step
    planner_overrode_llm::Bool             # MCTS picked something different from LLM
end

# ============================================================================
# AGENT
# ============================================================================

"""
    BayesianAgent

The main agent that ties everything together.
"""
mutable struct BayesianAgent{W<:World, M<:WorldModel, P<:Planner, A<:StateAbstractor}
    world::W
    model::M
    planner::P
    abstractor::A
    sensors::Vector{Sensor}
    config::AgentConfig

    # State
    current_observation::Any
    current_abstract_state::Any
    step_count::Int
    episode_count::Int
    total_reward::Float64

    # History for credit assignment
    trajectory::Vector{NamedTuple{(:s, :a, :r, :s′), Tuple{Any, Any, Float64, Any}}}

    # Pending sensor queries awaiting ground truth: (sensor, action, said_yes, step)
    pending_sensor_queries::Vector{Tuple{Sensor, Any, Bool, Int}}

    # Step of last nonzero reward (for windowed credit assignment)
    last_reward_step::Int

    # Observation history: (action, observation_text) for building LLM context
    observation_history::Vector{Tuple{Any, String}}

    # State analysis cache: observation_hash → StateAnalysis
    state_analysis_cache::Dict{UInt64, Any}

    # Tracks actions boosted by state analysis (for ground truth learning)
    # Maps step → Set of actions that analysis marked as promising
    analysis_boosted_actions::Dict{Int, Set{Any}}

    # Oracle beliefs: (observable_key, action) → P(action helps | LLM observations)
    # Separate from reward posterior — updated by sensor queries only, not real rewards.
    oracle_beliefs::Dict{Tuple{Any, String}, Float64}

    # Decision info from last act! call (for logging/debugging)
    last_decision::Union{Nothing, DecisionInfo}
end

"""
    BayesianAgent(world, model, planner, abstractor; sensors=[], config=AgentConfig())

Construct a new Bayesian agent.
"""
function BayesianAgent(
    world::World,
    model::WorldModel,
    planner::Planner,
    abstractor::StateAbstractor;
    sensors::Vector{<:Sensor} = Sensor[],
    config::AgentConfig = AgentConfig()
)
    return BayesianAgent(
        world, model, planner, abstractor, convert(Vector{Sensor}, sensors), config,
        nothing, nothing, 0, 0, 0.0,
        NamedTuple{(:s, :a, :r, :s′), Tuple{Any, Any, Float64, Any}}[],
        Tuple{Sensor, Any, Bool, Int}[],
        0,
        Tuple{Any, String}[],
        Dict{UInt64, Any}(),
        Dict{Int, Set{Any}}(),
        Dict{Tuple{Any, String}, Float64}(),  # oracle_beliefs
        nothing  # last_decision
    )
end

"""
    reset!(agent::BayesianAgent)

Reset the agent for a new episode.
"""
function reset!(agent::BayesianAgent)
    agent.current_observation = reset!(agent.world)
    agent.current_abstract_state = abstract_state(agent.abstractor, agent.current_observation)
    agent.step_count = 0
    agent.episode_count += 1
    agent.total_reward = 0.0
    empty!(agent.trajectory)
    empty!(agent.pending_sensor_queries)
    agent.last_reward_step = 0
    empty!(agent.observation_history)
    empty!(agent.state_analysis_cache)
    empty!(agent.analysis_boosted_actions)
    return agent.current_observation
end

"""
    extract_observation_text(obs) → String

Extract a text representation from an observation for storing in history.
"""
function extract_observation_text(obs)
    if obs isa NamedTuple && hasproperty(obs, :text)
        return string(obs.text)
    else
        return string(obs)
    end
end

"""
    build_llm_context(agent::BayesianAgent) → String

Build rich context string for LLM selection queries. Includes:
- Current observation (location, text, inventory, score)
- Recent trajectory with outcomes (what happened when actions were tried)
- World model knowledge for current state (tried actions, observed rewards)
- Game progress (step count, score, states discovered)
"""
function build_llm_context(agent::BayesianAgent)
    parts = String[]
    s = agent.current_abstract_state

    # Current observation
    push!(parts, format_observation_for_llm(agent.current_observation))

    # Recent history with outcomes
    if !isempty(agent.observation_history)
        push!(parts, "")
        push!(parts, "Recent history:")
        n = min(15, length(agent.observation_history))
        for (action, outcome) in agent.observation_history[end-n+1:end]
            short = length(outcome) > 120 ? outcome[1:120] * "..." : outcome
            # Collapse whitespace for readability
            short = replace(short, r"\s+" => " ")
            push!(parts, "  > $action → $short")
        end
    end

    # Confirmed useless actions from this state (null outcomes)
    confirmed_useless = String[]
    obs_key = (s isa MinimalState) ? observable_key(s) : s
    for (state_key, action) in agent.model.confirmed_selfloops
        if state_key == obs_key
            push!(confirmed_useless, "'$action'")
        end
    end
    if !isempty(confirmed_useless)
        push!(parts, "")
        push!(parts, "Confirmed useless from this location:")
        for a in confirmed_useless
            push!(parts, "  $a (no effect)")
        end
    end

    # Globally effective actions: (state, action) pairs with positive average reward anywhere
    globally_effective = String[]
    for (key, p) in agent.model.reward_posterior
        # κ represents effective observations in the posterior (may be fractional from sensor updates)
        n_obs = floor(Int, p.κ - agent.model.reward_prior.κ)
        if n_obs > 0 && p.μ > 0.1  # Saw positive rewards
            action_name = key[2]
            state_name = key[1]
            avg_r = round(p.μ, digits=1)
            push!(globally_effective, "'$action_name' at $state_name → avg reward $avg_r")
        end
    end
    if !isempty(globally_effective)
        push!(parts, "")
        push!(parts, "Actions that worked elsewhere:")
        for e in globally_effective[1:min(5, length(globally_effective))]
            push!(parts, "  $e")
        end
    end

    # World model knowledge: what actions have been tried from this state
    tried = String[]
    for (key, p) in agent.model.reward_posterior
        state_match = (s isa MinimalState && key[1] isa MinimalState) ?
            (observable_key(key[1]) == observable_key(s)) : (key[1] == s)
        if state_match
            # κ represents effective observations (may be fractional from sensor updates)
            n_obs = floor(Int, p.κ - agent.model.reward_prior.κ)
            if n_obs > 0
                avg_r = round(p.μ, digits=2)
                push!(tried, "'$(key[2])' (tried $(n_obs)×, avg reward $avg_r)")
            end
        end
    end
    if !isempty(tried)
        push!(parts, "")
        push!(parts, "Previously tried from this location:")
        for t in tried
            push!(parts, "  $t")
        end
    end

    # Game progress
    push!(parts, "")
    # Count states: works for both TabularWorldModel and FactoredWorldModel
    n_states = if hasfield(typeof(agent.model), :known_states)
        length(agent.model.known_states)
    else
        length(agent.model.known_locations) + length(agent.model.known_objects)
    end
    push!(parts, "Step $(agent.step_count), total reward $(agent.total_reward), $n_states state elements discovered")

    return join(parts, "\n")
end

"""
    act!(agent::BayesianAgent) → (action, observation, reward, done)

Choose and execute an action via expected utility maximisation.
"""
function act!(agent::BayesianAgent)
    s = agent.current_abstract_state
    available_actions = actions(agent.world, agent.current_observation)
    n_total_actions = length(available_actions)
    n_actions = length(available_actions)

    @debug "act! start" state=s n_actions step=agent.step_count

    # Track queries for ground truth updates: (sensor, action, answer)
    sensor_queries = Tuple{Sensor, Any, Bool}[]
    _llm_selected = nothing
    _n_voi_queries = 0

    # Count selfloops for diagnostics (not filtering)
    n_selfloops = if isa(agent.model, FactoredWorldModel) && s isa MinimalState
        count(a -> is_selfloop(agent.model, s, a), available_actions)
    else
        0
    end

    # --- State analysis from LLM (search heuristic for MCTS priors) ---
    analysis_applied = false
    for sensor in agent.sensors
        sensor isa LLMSensor || continue

        obs_hash = hash(extract_observation_text(agent.current_observation))
        local current_analysis
        if haskey(agent.state_analysis_cache, obs_hash)
            current_analysis = agent.state_analysis_cache[obs_hash]
            analysis_applied = true
        else
            context = build_llm_context(agent)
            current_analysis = query_state_analysis(sensor, context, available_actions)
            agent.state_analysis_cache[obs_hash] = current_analysis
            analysis_applied = true
        end

        # Record which actions analysis boosted (for ground truth learning)
        if analysis_applied && @isdefined(current_analysis)
            promising_lower = [lowercase(d) for d in current_analysis.promising_directions]
            boosted = Set{Any}()
            for a in available_actions
                a_lower = lowercase(string(a))
                if any(d -> occursin(d, a_lower) || occursin(a_lower, d), promising_lower)
                    push!(boosted, a)
                end
            end
            if !isempty(boosted)
                agent.analysis_boosted_actions[agent.step_count] = boosted
            end
        end
        break  # Only use state analysis from first LLM sensor
    end

    # --- VOI-gated sensor queries: update oracle beliefs (NOT reward posteriors) ---
    obs_key = (s isa MinimalState) ? observable_key(s) : s
    for sensor in agent.sensors
        if sensor isa LLMSensor
            # Compute VOI for each action using oracle beliefs
            best_voi = 0.0
            best_action_for_voi = nothing
            for a in available_actions
                voi = compute_voi(sensor, agent.oracle_beliefs, obs_key, a,
                                  agent.model, s, available_actions, n_actions)
                if voi > best_voi
                    best_voi = voi
                    best_action_for_voi = a
                end
            end

            if best_voi > agent.config.sensor_cost
                # Query LLM for action selection
                context = build_llm_context(agent)
                selected = query_selection(sensor, context, available_actions)

                if !isnothing(selected)
                    _llm_selected = selected
                    _n_voi_queries += 1
                    push!(sensor_queries, (sensor, selected, true))

                    # Extract local beliefs for current obs_key, update via Categorical model
                    local_beliefs = Dict{Any, Float64}()
                    default_belief = 1.0 / n_actions
                    for a in available_actions
                        local_beliefs[a] = get(agent.oracle_beliefs, (obs_key, string(a)), default_belief)
                    end
                    update_beliefs_from_selection!(sensor, available_actions, selected, local_beliefs)

                    # Write updated beliefs back to agent.oracle_beliefs
                    for a in available_actions
                        agent.oracle_beliefs[(obs_key, string(a))] = local_beliefs[a]
                    end

                    @debug "LLM selection → oracle beliefs" sensor=sensor.name selected voi=best_voi beliefs=local_beliefs
                end
            end
        else
            # Binary sensors: VOI-gated yes/no queries → update oracle beliefs
            queries_made = 0
            while queries_made < agent.config.max_queries_per_step
                best_voi = 0.0
                best_action_to_ask = nothing

                for a in available_actions
                    voi = compute_voi(sensor, agent.oracle_beliefs, obs_key, a,
                                      agent.model, s, available_actions, n_actions)
                    if voi > best_voi
                        best_voi = voi
                        best_action_to_ask = a
                    end
                end

                (best_voi <= agent.config.sensor_cost || isnothing(best_action_to_ask)) && break

                question = "Will action \"$(best_action_to_ask)\" help make progress?"
                answer = query(sensor, agent.current_observation, question)

                # Update oracle belief for this action via Bayes rule with TPR/FPR
                action_key = (obs_key, string(best_action_to_ask))
                default_belief = 1.0 / n_actions
                prior_belief = get(agent.oracle_beliefs, action_key, default_belief)
                agent.oracle_beliefs[action_key] = posterior(sensor, prior_belief, answer)

                push!(sensor_queries, (sensor, best_action_to_ask, answer))
                queries_made += 1
                _n_voi_queries += 1
                @debug "binary query → oracle belief" sensor=sensor.name action=best_action_to_ask answer voi=best_voi belief=agent.oracle_beliefs[action_key]
            end
        end
    end

    # --- Derive PUCT priors from oracle beliefs ---
    action_priors = Dict{Any, Float64}()
    default_belief = 1.0 / n_actions
    floor_val = 1.0 / (10 * n_actions)
    for a in available_actions
        belief = get(agent.oracle_beliefs, (obs_key, string(a)), default_belief)
        action_priors[a] = belief + floor_val
    end

    # Boost priors from state analysis (search heuristic — affects MCTS exploration order)
    if analysis_applied && @isdefined(current_analysis)
        promising_lower = [lowercase(d) for d in current_analysis.promising_directions]
        for a in available_actions
            a_lower = lowercase(string(a))
            if any(d -> occursin(d, a_lower) || occursin(a_lower, d), promising_lower)
                action_priors[a] *= 2.0  # Mild boost
            end
        end
    end

    # Normalize
    total = sum(values(action_priors))
    if total > 0
        for a in available_actions
            action_priors[a] /= total
        end
    end

    action = plan_with_priors(agent.planner, s, agent.model, available_actions, action_priors)
    @debug "action selected" action n_sensor_queries=length(sensor_queries)

    # Snapshot top reward posterior means and oracle beliefs for decision logging
    _top_reward_means = sort(
        [a => let rd = reward_dist(agent.model, s, a); μ = _reward_mean(rd); isfinite(μ) ? μ : 0.0 end
         for a in available_actions],
        by=last, rev=true
    )[1:min(3, length(available_actions))]

    _top_oracle_beliefs = sort(
        [a => get(agent.oracle_beliefs, (obs_key, string(a)), 1.0 / n_actions)
         for a in available_actions],
        by=last, rev=true
    )[1:min(3, length(available_actions))]

    # If the planner picks a different action than the LLM selected, store a
    # negative prediction for the executed action
    llm_selected = isempty(sensor_queries) ? nothing : sensor_queries[end][2]
    if !isnothing(llm_selected) && action != llm_selected
        llm_sensor = sensor_queries[end][1]
        push!(sensor_queries, (llm_sensor, action, false))
        @debug "implicit negative" sensor=llm_sensor.name selected=llm_selected executed=action
    end

    # Execute action
    obs, reward, done, info = step!(agent.world, action)
    s′ = abstract_state(agent.abstractor, obs)

    # Persist hidden variables from previous state: knowledge accumulates
    # across steps. extract_minimal_state() creates empty hidden vars;
    # we carry forward what was learned, then the heuristic/LLM adds new.
    if s isa MinimalState && s′ isa MinimalState
        union!(s′.spells_known, s.spells_known)
        merge!(s′.object_states, s.object_states)
        union!(s′.knowledge_gained, s.knowledge_gained)
    end

    @debug "step result" action reward new_state=s′ done

    # Store observation text for LLM context history
    push!(agent.observation_history, (action, extract_observation_text(obs)))

    # ================================================================
    # PHASE 2: HIDDEN VARIABLE INFERENCE (if LLM sensor available)
    # Run BEFORE null-outcome detection so hidden variable changes
    # prevent false self-loop marking (e.g. "open box" changes
    # object_states even when observation text looks identical).
    # ================================================================
    if !isempty(agent.sensors)
        llm_idx = findfirst(s -> s isa LLMSensor, agent.sensors)
        if !isnothing(llm_idx)
            obs_text = extract_observation_text(obs)
            try
                infer_hidden_variables!(s′, obs_text, agent.sensors[llm_idx])
                @debug "Hidden vars inferred" spells=s′.spells_known objects=s′.object_states knowledge=s′.knowledge_gained
            catch e
                @debug "Hidden variable inference failed" error=e
            end
        end
    end

    # Also infer hidden variables from text heuristics (no LLM needed)
    if s′ isa MinimalState
        obs_text = extract_observation_text(obs)
        infer_hidden_variables_heuristic!(s′, obs_text)
    end

    # Null-outcome detection: use BOTH text comparison and state comparison.
    # Text unchanged → definitely a selfloop (same observation produced).
    # State unchanged → also a selfloop (location, inventory, hidden vars same).
    # Either signal suffices; this prevents false negatives from heuristic noise.
    text_unchanged = is_null_outcome(agent.current_observation, obs)
    state_unchanged = (s isa MinimalState && s′ isa MinimalState) ?
        (observable_key(s) == observable_key(s′)) : text_unchanged
    if (text_unchanged || state_unchanged) && reward == 0.0
        resolve_null_action_queries!(agent, action)
        mark_selfloop!(agent.model, s, action)
    end

    # Oscillation detection: if this action returns us to a state visited
    # 1-3 steps ago with zero reward, mark it as confirmed zero-value.
    # This is a generalization of selfloop to 2-step cycles (e.g., east↔sw).
    if reward == 0.0 && s′ isa MinimalState && !isempty(agent.trajectory)
        obs_key_prime = observable_key(s′)
        s_obs_key_here = (s isa MinimalState) ? observable_key(s) : s
        for i in max(1, length(agent.trajectory) - 3):length(agent.trajectory)
            prev_s = agent.trajectory[i].s
            if prev_s isa MinimalState && observable_key(prev_s) == obs_key_prime &&
               obs_key_prime != s_obs_key_here && agent.trajectory[i].r == 0.0
                mark_selfloop!(agent.model, s, action)
                @debug "Oscillation detected" from=s action to=s′ returned_to=prev_s
                break
            end
        end
    end

    # Update model
    update!(agent.model, s, action, reward, s′)

    # Record for abstraction learning
    record_transition!(agent.abstractor, s, action, reward, s′)

    # ================================================================
    # UNIFIED LEARNING MECHANISMS (No toggles - all run at configured frequency)
    # ================================================================
    # The agent uses ALL available mechanisms to maximize expected utility.
    # These are gated by computational frequency only, not by enable flags.

    # Structure Learning: Action scope identification (periodic)
    if agent.step_count % agent.config.structure_learning_frequency == 0
        try
            if isa(agent.model, FactoredWorldModel)
                scope = compute_action_scope(agent.model, action)
                if !isempty(scope)
                    @debug "Structure Learning: Action Scope" action scope=scope
                end
            end
        catch e
            @debug "Structure Learning error" exception=e
        end
    end

    # Action Schemas: Clustering similar actions (periodic)
    if agent.step_count % agent.config.schema_discovery_frequency == 0
        try
            if isa(agent.model, FactoredWorldModel)
                schemas = discover_schemas(agent.model)
                if !isempty(schemas)
                    @debug "Action Schemas discovered" num_schemas=length(schemas)
                end
            end
        catch e
            @debug "Schema discovery error" exception=e
        end
    end

    # Goal Planning: Extract and track goals from observation text
    try
        obs_text = extract_observation_text(obs)
        goals = extract_goals_from_text(obs_text)
        if !isempty(goals)
            # Update goal achievement status based on current state
            if isa(s′, MinimalState)
                update_goal_status!(goals, s′)
                achieved_count = count(g -> g.achieved for g in goals)
                @debug "Goal Planning" num_goals=length(goals) achieved=achieved_count
            end
        end
    catch e
        @debug "Goal planning error" exception=e
    end

    # Check for contradictions and refine if needed
    contradiction = check_contradiction(agent.abstractor)
    if !isnothing(contradiction)
        refine!(agent.abstractor, contradiction)
    end

    # Store sensor queries for delayed credit assignment
    for (sensor, queried_action, said_yes) in sensor_queries
        push!(agent.pending_sensor_queries, (sensor, queried_action, said_yes, agent.step_count))
    end

    # When reward != 0, resolve pending queries with discounted trajectory credit
    if reward != 0.0
        resolve_pending_queries!(agent, reward)
        agent.last_reward_step = agent.step_count

        # Update analysis accuracy: check if the executed action was boosted
        # by state analysis within a recent window (last 5 steps)
        for llm_sensor in agent.sensors
            llm_sensor isa LLMSensor || continue
            for step_k in max(0, agent.step_count - 5):agent.step_count
                if haskey(agent.analysis_boosted_actions, step_k)
                    boosted = agent.analysis_boosted_actions[step_k]
                    llm_sensor.analysis_total += 1
                    if action in boosted
                        llm_sensor.analysis_correct += 1
                    end
                    delete!(agent.analysis_boosted_actions, step_k)
                end
            end
            break
        end
    end

    # Record transition
    push!(agent.trajectory, (s=s, a=action, r=reward, s′=s′))

    # Update state
    agent.current_observation = obs
    agent.current_abstract_state = s′
    agent.step_count += 1
    agent.total_reward += reward

    # Populate decision info for logging
    agent.last_decision = DecisionInfo(
        _llm_selected,
        action,
        _top_reward_means,
        _top_oracle_beliefs,
        n_selfloops,
        n_total_actions,
        _n_voi_queries,
        length(sensor_queries),
        !isnothing(_llm_selected) && action != _llm_selected
    )

    return action, obs, reward, done
end

"""
    resolve_pending_queries!(agent::BayesianAgent, reward::Float64)

Resolve pending sensor queries using discounted trajectory credit.

Actions close to the reward event get strong credit (γ^1), distant actions get
weak credit (γ^20). The proposition was "action helps make progress" — temporal
proximity determines how much credit the action receives. Very distant actions
(|discounted| < 0.001) are skipped entirely to avoid noise.

For LLMSensor selection queries (said_yes=true means "LLM selected this action"),
also updates the selection-specific accuracy tracker separately from TPR/FPR.
"""
function resolve_pending_queries!(agent::BayesianAgent, reward::Float64)
    γ = agent.config.discount
    resolved = Int[]
    for (i, (sensor, action, said_yes, step)) in enumerate(agent.pending_sensor_queries)
        if step > agent.last_reward_step
            steps_elapsed = agent.step_count - step
            discounted = reward * γ^steps_elapsed
            # Skip if credit is negligible (avoids noise from very distant actions)
            if abs(discounted) < 0.001
                continue
            end
            actually_helped = discounted > 0.0
            update_reliability!(sensor, said_yes, actually_helped)

            # Update selection accuracy for LLMSensor selection queries.
            # said_yes=true means this was the LLM's selected action.
            if sensor isa LLMSensor && said_yes
                sensor.selection_total += 1
                if actually_helped
                    sensor.selection_correct += 1
                end
            end

            push!(resolved, i)
            @debug "trajectory credit" action said_yes actually_helped discount=γ^steps_elapsed
        end
    end
    deleteat!(agent.pending_sensor_queries, sort(resolved))
end

"""
    resolve_null_action_queries!(agent::BayesianAgent, null_action)

Resolve pending sensor queries about an action that produced no change.

When observation text is identical before and after, the action was definitively
unhelpful: P(obs_unchanged | action_helpful) ≈ 0. This provides negative ground
truth to calibrate the sensor's FPR without waiting for sparse rewards.
"""
function resolve_null_action_queries!(agent::BayesianAgent, null_action)
    resolved = Int[]
    for (i, (sensor, queried_action, said_yes, step)) in enumerate(agent.pending_sensor_queries)
        if queried_action == null_action
            update_reliability!(sensor, said_yes, false)
            push!(resolved, i)
            @debug "null action ground truth" action=null_action said_yes
        end
    end
    deleteat!(agent.pending_sensor_queries, sort(resolved))
end

"""
    run_episode!(agent::BayesianAgent; max_steps=1000) → total_reward

Run a complete episode and return the total reward.
"""
function run_episode!(agent::BayesianAgent; max_steps::Int = 1000)
    reset!(agent)
    
    for _ in 1:max_steps
        _, _, _, done = act!(agent)
        done && break
    end
    
    return agent.total_reward
end

# ============================================================================
# EXPORTS
# ============================================================================

export World, Sensor, WorldModel, Planner, StateAbstractor
export reset!, step!, actions, render, seed!
export query, tpr, fpr, update_reliability!, posterior
export sample_dynamics, update!, transition_dist, reward_dist, entropy
export abstract_state, record_transition!, check_contradiction, refine!
export plan
export compute_voi, get_kappa
export AgentConfig, BayesianAgent, DecisionInfo, act!, run_episode!, resolve_pending_queries!, resolve_null_action_queries!

# ============================================================================
# FOUNDATIONAL COMPONENTS (needed by both legacy and Stage 1)
# ============================================================================

# Legacy components (depended on by Stage 1)
include("models/tabular_world_model.jl")
include("models/binary_sensor.jl")
include("planners/thompson_mcts.jl")
include("abstractors/identity_abstractor.jl")
include("abstractors/bisimulation_abstractor.jl")
include("abstractors/minimal_state_abstractor.jl")

# ============================================================================
# STAGE 1: MVBN (Minimum Viable Bayesian Network) - NEW COMPONENTS
# ============================================================================

# Probability foundations
include("probability/cpd.jl")

# State representation
include("state/minimal_state.jl")
include("state/state_belief.jl")
include("state/variable_discovery.jl")  # Stage 2

# Factored world model
include("models/factored_world_model.jl")

# State inference
include("inference/bayesian_update.jl")
include("inference/hidden_variable_inference.jl")

# Factored planning
include("planning/factored_mcts.jl")

# Stage 3: Structure Learning
include("structure/structure_learning.jl")

# Stage 4: Action Schemas
include("actions/action_schema.jl")

# Stage 5: Goal-Directed Planning
include("planning/goal_planning.jl")

# World adapters
include("worlds/gridworld.jl")

# Jericho requires PyCall - load conditionally
try
    include("worlds/jericho.jl")
catch e
    @warn "Jericho world not available (PyCall or Jericho not installed)" exception=e
end

# Stage 1: MVBN exports
export DirichletCategorical, update!, predict, entropy, mode
export MinimalState, extract_minimal_state
export StateBelief, add_object!, update_from_state!, sample_state, predict_state, posterior_prob, loglikelihood
export FactoredWorldModel, SampledFactoredDynamics, add_location!, sample_next_state, mark_selfloop!, is_selfloop, observable_key
export FactoredMCTS, FactoredMCTSNode, mcts_search
export update_location_belief!, update_inventory_belief!, bayesian_update_belief!, predict_from_likelihood

# Stage 2: Variable Discovery exports
export VariableCandidate, extract_candidate_variables, compute_bic, compute_bic_delta
export should_accept_variable, discover_variables!, update_state_belief_with_discovery!

# Hidden variable inference (Phase 2)
export infer_hidden_variables!, infer_hidden_variables_heuristic!, extract_spell_name, extract_object_states, extract_knowledge

# Stage 3: Structure Learning exports
export DirectedGraph, add_edge!, remove_edge!, get_parents, is_acyclic
export bde_score, learn_structure_greedy, LearnedStructure
export learn_action_structure, compute_action_scope

# Stage 4: Action Schemas exports
export ActionSchema, ActionInstance
export extract_action_type, cluster_actions, infer_schema_from_cluster
export discover_schemas, apply_schema, zero_shot_transfer_likelihood

# Stage 5: Goal-Directed Planning exports
export Goal, extract_goals_from_text, compute_goal_progress, expected_goal_progress
export goal_biased_action_selection, update_goal_status!

# Legacy exports
export GridWorld, spawn_food!
export TabularWorldModel, NormalGammaPosterior, NormalGammaMeasure, SampledDynamics, get_reward
export ThompsonMCTS, MCTSNode, plan_with_priors, select_rollout_action
export IdentityAbstractor, BisimulationAbstractor, MinimalStateAbstractor, abstraction_summary
export BinarySensor, LLMSensor, format_observation_for_llm, query_selection, update_beliefs_from_selection!, is_null_outcome, StateAnalysis, query_state_analysis, parse_state_analysis, apply_state_analysis_priors!, selection_accuracy, analysis_accuracy
export extract_observation_text, build_llm_context
export action_features, collect_posteriors, combine_posteriors

end # module
