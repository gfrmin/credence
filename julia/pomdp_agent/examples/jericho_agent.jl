"""
    Unified Bayesian Agent: Interactive Fiction via Jericho

A world-agnostic Bayesian agent that maximizes expected utility using all available
mechanisms:

- **Factored State Representation**: MinimalState (location, inventory, hidden variables)
- **Dirichlet-Categorical Dynamics**: Learns P(state'|state, action) incrementally
- **Thompson Sampling MCTS**: Plans ahead with posterior-sampled dynamics
- **Hidden Variable Inference**: If LLM available, infers spell knowledge, object states
- **Variable Discovery**: Auto-discovers new state variables via BIC model selection
- **Structure Learning**: Learns causal dependencies (action scopes)
- **Action Schemas**: Generalizes across action instances
- **Goal Planning**: Extracts and tracks objectives from observation text

All mechanisms run unconditionally. No toggle flags. Just one agent, any world.

Requirements:
    pip install jericho

Optional (for LLM sensor):
    ollama serve && ollama pull llama3.2

Run with:
    julia --project=. examples/jericho_agent.jl path/to/enchanter.z3 --episodes 5 --steps 300
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Random
using Printf

include("../src/BayesianAgents.jl")
using .BayesianAgents

# ============================================================================
# JERICHO STATE ABSTRACTOR
# ============================================================================

"""
Maps JerichoWorld NamedTuple observations to compact state keys.

The raw observation contains text, score, steps, location, inventory, and
state_hash. Without abstraction the TabularWorldModel sees every observation
as unique (text and steps change each step). This abstractor reduces the
observation to "location|inv_hash" so the model can learn transitions.

TODO: Use MinimalState for FactoredWorldModel integration once planner is updated.
"""
struct JerichoStateAbstractor <: StateAbstractor end

function BayesianAgents.abstract_state(::JerichoStateAbstractor, obs)
    loc = hasproperty(obs, :location) ? obs.location : "unknown"
    inv = hasproperty(obs, :inventory) ? obs.inventory : ""
    inv_hash = string(hash(inv), base=16)
    inv_short = length(inv_hash) >= 8 ? inv_hash[1:8] : inv_hash
    if loc == "unknown"
        h = hasproperty(obs, :state_hash) ? obs.state_hash : hash(obs)
        return "hash_$(string(h, base=16))"
    end
    return "$(loc)|$(inv_short)"
end

BayesianAgents.record_transition!(::JerichoStateAbstractor, s, a, r, s′) = nothing
BayesianAgents.check_contradiction(::JerichoStateAbstractor) = nothing
BayesianAgents.refine!(::JerichoStateAbstractor, contradiction) = nothing

# ============================================================================
# JERICHO FEATURE EXTRACTOR (for factored reward model)
# ============================================================================

"""
Extract feature keys from a Jericho abstract state (e.g. "Kitchen|a3f2b1c8").

Returns features like [(:location, "Kitchen")] so the factored reward model
can generalise across states sharing the same location.
"""
function jericho_features(abstract_state)
    parts = split(string(abstract_state), "|")
    features = Any[(:location, parts[1])]
    return features
end

# ============================================================================
# OLLAMA CLIENT (pure Julia, stdlib Downloads + JSON.jl)
# ============================================================================

using Downloads
using JSON

"""
    get_available_ollama_models(; base_url) → Vector{String}

Query the Ollama /api/tags endpoint to list installed models.
Returns empty vector if Ollama is not running or unreachable.
"""
function get_available_ollama_models(; base_url::String = "http://localhost:11434")
    try
        io = IOBuffer()
        Downloads.request("$base_url/api/tags";
            method = "GET",
            output = io,
            timeout = 5)
        data = JSON.parse(String(take!(io)))
        models = get(data, "models", [])
        return String[get(m, "name", "") for m in models if !isempty(get(m, "name", ""))]
    catch
        return String[]
    end
end

"""
    make_ollama_client(; base_url, model, temperature, timeout)

Create a named tuple with a `query` function that calls the local Ollama
HTTP API. Compatible with the LLMSensor `llm_client.query(prompt)` contract.
"""
function make_ollama_client(;
    base_url::String = "http://localhost:11434",
    model::String = "llama3.2",
    temperature::Float64 = 0.1,
    timeout::Int = 30
)
    query_fn = function(prompt::String)
        url = "$base_url/api/generate"
        body = JSON.json(Dict(
            "model" => model,
            "prompt" => prompt,
            "temperature" => temperature,
            "stream" => false
        ))
        io = IOBuffer()
        try
            Downloads.request(url;
                method = "POST",
                headers = ["Content-Type" => "application/json"],
                input = IOBuffer(body),
                output = io,
                timeout = timeout)
            data = JSON.parse(String(take!(io)))
            return get(data, "response", "")
        catch e
            @warn "Ollama query failed" exception=e
            return ""
        end
    end
    return (query = query_fn,)
end

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

function run_jericho_experiment(;
    game_path::String,
    n_episodes::Int = 5,
    max_steps::Int = 300,
    ollama_model::String = "llama3.2",
    mcts_iterations::Int = 60,
    mcts_depth::Int = 12,
    verbose::Bool = true,
    use_llm::Bool = true
)
    println("=" ^ 60)
    println("UNIFIED BAYESIAN AGENT - Interactive Fiction")
    println("=" ^ 60)
    println("Game:     $(basename(game_path))")
    println("Episodes: $n_episodes")
    println("Steps:    $max_steps per episode")
    println("MCTS:     $(mcts_iterations) iterations, depth $(mcts_depth)")
    println()

    # Create world
    world = JerichoWorld(game_path; max_steps=max_steps)
    println("Max score: $(world.max_score)")
    println()

    # Stage 1: MVBN with factored state representation
    # Learns P(location' | location, action) and P(obj' | obj, action)
    model = FactoredWorldModel(0.1;
                              reward_prior_mean=0.0,
                              reward_prior_variance=1.0)

    # Create planner
    planner = ThompsonMCTS(
        iterations = mcts_iterations,
        depth = mcts_depth,
        discount = 0.99,
        ucb_c = 2.0
    )

    # Stage 1: Factored state extraction (location, inventory)
    abstractor = MinimalStateAbstractor()

    # Create sensors (auto-detect LLM availability)
    sensors = Sensor[]
    if !use_llm
        println("LLM sensor disabled (--no-llm)")
    else
    try
        println("Detecting Ollama models...")
        available_models = get_available_ollama_models()
        if isempty(available_models)
            println("⊘ Ollama not responding on localhost:11434")
            println("  To enable: ollama serve && ollama pull llama3.2")
        else
            println("  Available models: $(join(available_models, ", "))")

            # Try requested model first, then fall back to whatever is available
            # Strip :latest suffix for matching since Ollama returns "model:latest"
            normalize_model(m) = replace(m, r":latest$" => "")
            candidates = vcat(
                [ollama_model],
                sort(available_models, by=m -> normalize_model(m) == ollama_model ? 0 : 1)
            )
            seen = Set{String}()

            connected = false
            for model_name in candidates
                norm = normalize_model(model_name)
                norm ∈ seen && continue
                push!(seen, norm)

                try
                    client = make_ollama_client(model=model_name, timeout=60)
                    test_response = client.query("Say 'ok'.")
                    if !isempty(test_response)
                        println("✓ Ollama connected with model: $model_name")
                        sensor = LLMSensor("llm", client;
                            prompt_template = "{question}",
                            tp_prior = (2.0, 1.0),
                            fp_prior = (1.0, 2.0))
                        push!(sensors, sensor)
                        connected = true
                        break
                    end
                catch e
                    println("  ✗ Model $model_name failed: $(sprint(showerror, e))")
                end
            end

            if !connected
                println("⊘ No Ollama models responded — proceeding without LLM sensor")
                println("  Models found but not working: $(join(available_models, ", "))")
            end
        end
    catch e
        println("⊘ LLM sensor unavailable — proceeding without it")
        println("  Error: $(sprint(showerror, e))")
    end
    end # use_llm
    println()

    # Create agent with unified Bayesian framework
    agent = BayesianAgent(
        world, model, planner, abstractor;
        sensors = sensors,
        config = AgentConfig(
            planning_depth = mcts_depth,
            mcts_iterations = mcts_iterations,
            sensor_cost = 0.01,
            max_queries_per_step = 3,
            variable_discovery_frequency = 20,
            variable_bic_threshold = 0.0,
            structure_learning_frequency = 50,
            schema_discovery_frequency = 100,
            goal_rollout_bias = 0.5
        )
    )

    # Run episodes - Agent created ONCE, model accumulates knowledge across episodes
    episode_rewards = Float64[]
    episode_scores = Int[]

    for episode in 1:n_episodes
        println("-" ^ 60)
        println("Episode $episode / $n_episodes")
        println("-" ^ 60)
        flush(stdout)

        # Reset world state and episode tracking, but NOT the model
        agent.current_observation = reset!(world)
        agent.current_abstract_state = abstract_state(agent.abstractor, agent.current_observation)
        agent.step_count = 0
        agent.total_reward = 0.0
        agent.trajectory = []
        # NOTE: agent.model is NOT reset - it accumulates knowledge across episodes!

        # Manual step loop for per-step output
        for step in 1:max_steps
            action, obs, reward, done = act!(agent)
            r_str = reward != 0 ? " → reward $(reward)" : ""
            sq = !isempty(agent.sensors) ? " [queries: $(agent.sensors[1].n_queries)]" : ""
            if verbose
                # Decision context: LLM pick, top beliefs, selfloop filtering
                di = agent.last_decision
                di_str = ""
                if !isnothing(di)
                    parts = String[]
                    if di.n_voi_queries > 0 && !isnothing(di.llm_selected)
                        llm_tag = di.planner_overrode_llm ? "LLM→✗$(di.llm_selected)" : "LLM→$(di.llm_selected)"
                        push!(parts, llm_tag)
                    elseif di.n_voi_queries == 0
                        push!(parts, "no-VOI")
                    end
                    if di.n_selfloops > 0
                        push!(parts, "sl:$(di.n_selfloops)/$(di.n_actions_available)")
                    end
                    # Top actions: reward posterior mean + oracle belief
                    if !isempty(di.top_reward_means)
                        top_parts = String[]
                        for (i, (a, μ)) in enumerate(di.top_reward_means)
                            b = i <= length(di.top_oracle_beliefs) ? di.top_oracle_beliefs[i][2] : 0.0
                            push!(top_parts, "$(a)=μ$(round(μ, digits=2)),b$(round(b, digits=2))")
                        end
                        push!(parts, "top:" * join(top_parts, " | "))
                    end
                    if !isempty(parts)
                        di_str = " {" * join(parts, ", ") * "}"
                    end
                end
                print(@sprintf("  %3d. %-30s%s%s%s\n", step, action, r_str, sq, di_str))
                flush(stdout)
            end
            done && break
        end

        push!(episode_rewards, agent.total_reward)
        push!(episode_scores, world.current_score)

        println()
        println(@sprintf("  Score:  %d / %d", world.current_score, world.max_score))
        println(@sprintf("  Steps:  %d", agent.step_count))
        println(@sprintf("  Reward: %.1f", agent.total_reward))

        # Model-agnostic state/transition reporting
        if isa(model, TabularWorldModel)
            println(@sprintf("  States learned: %d", length(model.known_states)))
            println(@sprintf("  Transitions:    %d", length(model.transition_counts)))
        elseif isa(model, FactoredWorldModel)
            num_states = length(model.known_locations) * length(model.known_objects)
            println(@sprintf("  State space:    %d", num_states))
            println(@sprintf("  Transitions:    %d", length(model.transitions)))
        end
        println()
        flush(stdout)
    end

    # Summary
    println("=" ^ 60)
    println("RESULTS")
    println("=" ^ 60)
    println(@sprintf("Episodes:         %d", n_episodes))
    println(@sprintf("Average reward:   %.2f", _mean(episode_rewards)))
    println(@sprintf("Best score:       %d / %d", maximum(episode_scores), world.max_score))

    # Model-agnostic reporting
    if isa(model, TabularWorldModel)
        println(@sprintf("States learned:   %d", length(model.known_states)))
        println(@sprintf("Transitions:      %d", length(model.transition_counts)))
    elseif isa(model, FactoredWorldModel)
        num_states = length(model.known_locations) * length(model.known_objects)
        println(@sprintf("State space:      %d", num_states))
        println(@sprintf("Transitions:      %d", length(model.transitions)))
    end

    if !isempty(sensors)
        for sensor in sensors
            println()
            println("Sensor: $(sensor.name)")
            println(@sprintf("  Queries: %d", sensor.n_queries))
            println(@sprintf("  TPR:     %.3f", tpr(sensor)))
            println(@sprintf("  FPR:     %.3f", fpr(sensor)))
            if sensor isa LLMSensor
                println(@sprintf("  Selection accuracy: %.3f (%d/%d)",
                    selection_accuracy(sensor), sensor.selection_correct, sensor.selection_total))
                println(@sprintf("  Analysis accuracy:  %.3f (%d/%d)",
                    analysis_accuracy(sensor), sensor.analysis_correct, sensor.analysis_total))
            end
        end
    end

    # Learning curve
    println()
    println("Score by episode:")
    for (i, sc) in enumerate(episode_scores)
        bar = "█" ^ max(0, sc)
        println(@sprintf("  Episode %2d: %s %d", i, bar, sc))
    end

    return (rewards = episode_rewards, scores = episode_scores, agent = agent)
end

# ============================================================================
# HELPERS
# ============================================================================

_mean(x) = isempty(x) ? 0.0 : sum(x) / length(x)

# ============================================================================
# CLI
# ============================================================================

function parse_args(args)
    game_path = nothing
    n_episodes = 5
    max_steps = 300
    ollama_model = "llama3.2"
    verbose = true
    mcts_iterations = 60
    mcts_depth = 12
    use_llm = true

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--no-llm"
            use_llm = false
        elseif arg == "--episodes" && i < length(args)
            i += 1
            n_episodes = parse(Int, args[i])
        elseif arg == "--steps" && i < length(args)
            i += 1
            max_steps = parse(Int, args[i])
        elseif arg == "--model" && i < length(args)
            i += 1
            ollama_model = args[i]
        elseif arg == "--mcts-iter" && i < length(args)
            i += 1
            mcts_iterations = parse(Int, args[i])
        elseif arg == "--mcts-depth" && i < length(args)
            i += 1
            mcts_depth = parse(Int, args[i])
        elseif arg == "--quiet"
            verbose = false
        elseif arg == "--verbose"
            verbose = true
        elseif arg == "--debug"
            ENV["JULIA_DEBUG"] = "Main.BayesianAgents"
        elseif !startswith(arg, "-") && isnothing(game_path)
            game_path = arg
        else
            println("Unknown argument: $arg")
        end
        i += 1
    end

    if isnothing(game_path)
        # Default: look for games in sibling project
        candidates = [
            joinpath(@__DIR__, "..", "..", "bayesian-if-agent", "games", "905.z5"),
            joinpath(@__DIR__, "..", "..", "bayesian-if-agent", "games", "pentari.z5"),
            joinpath(@__DIR__, "..", "..", "bayesian-if-agent", "games", "zork1.z5"),
        ]
        for c in candidates
            if isfile(c)
                game_path = c
                break
            end
        end
        if isnothing(game_path)
            error("""No game file specified. Usage:
  julia --project=. examples/jericho_agent.jl path/to/game.z5 [--episodes N] [--steps N] [--model NAME] [--quiet]""")
        end
    end

    if !isfile(game_path)
        error("Game file not found: $game_path")
    end

    return (
        game_path = game_path,
        n_episodes = n_episodes,
        max_steps = max_steps,
        ollama_model = ollama_model,
        verbose = verbose,
        mcts_iterations = mcts_iterations,
        mcts_depth = mcts_depth,
        use_llm = use_llm,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    opts = parse_args(ARGS)
    run_jericho_experiment(;
        game_path = opts.game_path,
        n_episodes = opts.n_episodes,
        max_steps = opts.max_steps,
        ollama_model = opts.ollama_model,
        verbose = opts.verbose,
        mcts_iterations = opts.mcts_iterations,
        mcts_depth = opts.mcts_depth,
        use_llm = opts.use_llm
    )
end
