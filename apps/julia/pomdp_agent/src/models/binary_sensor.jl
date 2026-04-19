# Role: brain-side application
"""
    BinarySensor

A yes/no sensor with learned reliability (TPR and FPR).
Uses credence's BetaMeasure for both true positive rate and false positive rate posteriors.
"""

using Distributions

# Bernoulli kernel for Beta-Bernoulli conjugate updates via credence
const _BERNOULLI_KERNEL = Kernel(
    Interval(0.0, 1.0),
    Finite([true, false]),
    θ -> (o -> o == true ? log(θ) : log(1.0 - θ)),
    (θ, o) -> o == true ? log(θ) : log(1.0 - θ);
    likelihood_family = BetaBernoulli())

"""
    BinarySensor

A sensor that provides binary (yes/no) answers with learnable reliability.
Uses credence's BetaMeasure for TPR and FPR posteriors.
"""
mutable struct BinarySensor <: Sensor
    name::String

    # TPR: P(yes | true) ~ Beta(α, β) via credence
    tp_measure::BetaMeasure

    # FPR: P(yes | false) ~ Beta(α, β) via credence
    fp_measure::BetaMeasure

    # Query function: (state, question) → Bool
    query_fn::Function

    # Statistics
    n_queries::Int
    n_correct::Int
end

"""
    BinarySensor(name, query_fn; tp_prior=(2,1), fp_prior=(1,2))

Create a binary sensor with specified query function and priors.

The default priors encode a weak belief that the sensor is somewhat reliable:
- TPR prior: Beta(2,1) → mean 0.67, believes sensor usually detects true positives
- FPR prior: Beta(1,2) → mean 0.33, believes sensor usually avoids false positives
"""
function BinarySensor(
    name::String,
    query_fn::Function;
    tp_prior::Tuple{Float64,Float64} = (2.0, 1.0),
    fp_prior::Tuple{Float64,Float64} = (1.0, 2.0)
)
    return BinarySensor(
        name,
        BetaMeasure(tp_prior[1], tp_prior[2]),
        BetaMeasure(fp_prior[1], fp_prior[2]),
        query_fn,
        0, 0
    )
end

"""
    query(sensor::BinarySensor, state, question; action_history=nothing) → Bool

Query the sensor and return its answer.
"""
function query(sensor::BinarySensor, state, question; action_history=nothing)
    sensor.n_queries += 1
    return sensor.query_fn(state, question)
end

"""
    tpr(sensor::BinarySensor) → Float64

Return the expected true positive rate: E[P(yes | true)] via credence's mean().
"""
function tpr(sensor::BinarySensor)
    return mean(sensor.tp_measure)
end

"""
    fpr(sensor::BinarySensor) → Float64

Return the expected false positive rate: E[P(yes | false)] via credence's mean().
"""
function fpr(sensor::BinarySensor)
    return mean(sensor.fp_measure)
end

"""
    tpr_dist(sensor::BinarySensor) → BetaMeasure

Return the full posterior distribution over TPR.
"""
function tpr_dist(sensor::BinarySensor)
    return sensor.tp_measure
end

"""
    fpr_dist(sensor::BinarySensor) → BetaMeasure

Return the full posterior distribution over FPR.
"""
function fpr_dist(sensor::BinarySensor)
    return sensor.fp_measure
end

"""
    update_reliability!(sensor::BinarySensor, said_yes::Bool, actual::Bool)

Update the sensor's reliability estimates from ground truth.
Uses credence's condition() for Beta-Bernoulli conjugate update.

Arguments:
- said_yes: What the sensor predicted
- actual: The ground truth
"""
function update_reliability!(sensor::BinarySensor, said_yes::Bool, actual::Bool)
    if actual
        # Ground truth was positive — update TPR posterior
        sensor.tp_measure = condition(sensor.tp_measure, _BERNOULLI_KERNEL, said_yes)
        if said_yes
            sensor.n_correct += 1
        end
    else
        # Ground truth was negative — update FPR posterior
        sensor.fp_measure = condition(sensor.fp_measure, _BERNOULLI_KERNEL, said_yes)
        if !said_yes
            sensor.n_correct += 1
        end
    end
end

"""
    posterior(sensor::BinarySensor, prior::Float64, answer::Bool) → Float64

Compute the posterior probability that the proposition is true,
given the prior and the sensor's answer.

Uses Bayes' rule with expected TPR/FPR: P(true | answer) = P(answer | true) P(true) / P(answer).
This is a scalar computation on the expected TPR/FPR — consumer code, not inference.
"""
function posterior(sensor::BinarySensor, prior::Float64, answer::Bool)
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

"""
    accuracy(sensor::BinarySensor) → Float64

Return the empirical accuracy of the sensor.
"""
function accuracy(sensor::BinarySensor)
    return sensor.n_queries > 0 ? sensor.n_correct / sensor.n_queries : 0.5
end

"""
    reliability_summary(sensor::BinarySensor) → String

Return a human-readable summary of the sensor's reliability.
"""
function reliability_summary(sensor::BinarySensor)
    return """
    Sensor: $(sensor.name)
      TPR: $(round(tpr(sensor), digits=3)) [Beta($(sensor.tp_measure.alpha), $(sensor.tp_measure.beta))]
      FPR: $(round(fpr(sensor), digits=3)) [Beta($(sensor.fp_measure.alpha), $(sensor.fp_measure.beta))]
      Queries: $(sensor.n_queries)
      Accuracy: $(round(accuracy(sensor), digits=3))
    """
end

# ============================================================================
# LLM SENSOR (specialised for language model queries)
# ============================================================================

"""
    LLMSensor

A binary sensor backed by a large language model.
Wraps an LLM client and converts questions to yes/no queries.
Uses credence's BetaMeasure for TPR/FPR posteriors.
"""
mutable struct LLMSensor <: Sensor
    name::String

    # Reliability tracking via credence's BetaMeasure
    tp_measure::BetaMeasure
    fp_measure::BetaMeasure

    # LLM interface
    llm_client::Any  # Duck-typed: must have query(client, prompt) → String
    prompt_template::String

    # Statistics
    n_queries::Int
    n_correct::Int

    # Selection query reliability: separate from binary TPR/FPR
    # Tracks how often the LLM's selected action led to reward
    selection_correct::Int
    selection_total::Int

    # State analysis reliability: separate from binary and selection
    # Tracks how often analysis-boosted actions led to reward
    analysis_correct::Int
    analysis_total::Int
end

"""
    LLMSensor(name, llm_client; prompt_template="...", tp_prior=(2,1), fp_prior=(1,2))

Create an LLM-backed binary sensor.
"""
function LLMSensor(
    name::String,
    llm_client;
    prompt_template::String = "Answer with only 'yes' or 'no': {question}",
    tp_prior::Tuple{Float64,Float64} = (2.0, 1.0),
    fp_prior::Tuple{Float64,Float64} = (1.0, 2.0)
)
    return LLMSensor(
        name,
        BetaMeasure(tp_prior[1], tp_prior[2]),
        BetaMeasure(fp_prior[1], fp_prior[2]),
        llm_client,
        prompt_template,
        0, 0,
        0, 0,  # selection_correct, selection_total
        0, 0   # analysis_correct, analysis_total
    )
end

"""
    selection_accuracy(sensor::LLMSensor) → Float64

Posterior mean accuracy of the LLM's action selection.
Beta(1,1) prior (Laplace smoothing) → learns from reward signal.
"""
selection_accuracy(s::LLMSensor) = (s.selection_correct + 1) / (s.selection_total + 2)

"""
    analysis_accuracy(sensor::LLMSensor) → Float64

Posterior mean accuracy of the LLM's state analysis.
Beta(1,1) prior (Laplace smoothing) → learns from reward signal.
"""
analysis_accuracy(s::LLMSensor) = (s.analysis_correct + 1) / (s.analysis_total + 2)

"""
    format_observation_for_llm(obs) → String

Format an observation for use as LLM context.
Handles NamedTuples with :text/:location/:inventory/:score fields (e.g. Jericho),
and falls back to string() for other types.
"""
function format_observation_for_llm(obs)
    if obs isa NamedTuple
        parts = String[]
        hasproperty(obs, :location) && push!(parts, "Location: $(obs.location)")
        hasproperty(obs, :text) && push!(parts, "Observation: $(obs.text)")
        hasproperty(obs, :inventory) && push!(parts, "Inventory: $(obs.inventory)")
        hasproperty(obs, :score) && push!(parts, "Score: $(obs.score)")
        if !isempty(parts)
            return join(parts, "\n")
        end
    end
    return string(obs)
end

"""
    query(sensor::LLMSensor, state, question::String; action_history=nothing) → Bool

Query the LLM and parse the response as yes/no.

When `action_history` is provided (a vector of recent action strings), it is
included in the prompt so the LLM can avoid recommending already-tried actions.
"""
function query(sensor::LLMSensor, state, question::String; action_history=nothing)
    sensor.n_queries += 1

    # Format prompt
    prompt = replace(sensor.prompt_template, "{question}" => question)

    # Add state context if available, formatted for LLM readability
    if !isnothing(state)
        context = format_observation_for_llm(state)
        prompt = "Context:\n$context\n\n$prompt"
    end

    # Add action history if available
    if !isnothing(action_history) && !isempty(action_history)
        history_str = join(string.(action_history), ", ")
        prompt = "$prompt\n\nRecent actions tried: $history_str"
    end

    # Query LLM
    response = lowercase(strip(sensor.llm_client.query(prompt)))

    # Parse response
    if startswith(response, "yes")
        return true
    elseif startswith(response, "no")
        return false
    else
        # Ambiguous — try harder
        if occursin("yes", response) && !occursin("no", response)
            return true
        elseif occursin("no", response) && !occursin("yes", response)
            return false
        else
            # Truly ambiguous — default to no (conservative)
            return false
        end
    end
end

# Reliability methods for LLMSensor — use credence's mean()
tpr(s::LLMSensor) = mean(s.tp_measure)
fpr(s::LLMSensor) = mean(s.fp_measure)

function update_reliability!(sensor::LLMSensor, said_yes::Bool, actual::Bool)
    if actual
        sensor.tp_measure = condition(sensor.tp_measure, _BERNOULLI_KERNEL, said_yes)
        if said_yes
            sensor.n_correct += 1
        end
    else
        sensor.fp_measure = condition(sensor.fp_measure, _BERNOULLI_KERNEL, said_yes)
        if !said_yes
            sensor.n_correct += 1
        end
    end
end

# ============================================================================
# STATE ANALYSIS (extract structured knowledge from game text)
# ============================================================================

"""
    StateAnalysis

Structured analysis of the current game state extracted from observation text.
"""
struct StateAnalysis
    visible_objects::Vector{String}
    current_obstacle::String
    promising_directions::Vector{String}
    situation_summary::String
    confidence::Float64
end

"""
    query_state_analysis(sensor::LLMSensor, context::String, actions) → StateAnalysis

Ask the LLM to analyze the current game situation and extract structured information.
"""
function query_state_analysis(sensor::LLMSensor, context::String, actions)
    sensor.n_queries += 1

    action_list = join(["  - $a" for a in actions], "\n")

    prompt = """Analyze this text adventure game situation. Extract structured information.

$context

Available actions:
$action_list

Respond in EXACTLY this format (one item per line after the label):
OBJECTS: [comma-separated visible objects or things you can interact with]
OBSTACLE: [the main challenge or problem you need to solve right now]
PROMISING: [comma-separated action types that seem relevant to the situation, e.g. 'examine', 'take', 'cast spell', 'open door']
SUMMARY: [one sentence describing the current situation]
CONFIDENCE: [number from 0.0 to 1.0 indicating your confidence in this analysis]"""

    response = strip(sensor.llm_client.query(prompt))
    return parse_state_analysis(response)
end

"""
    parse_state_analysis(response::AbstractString) → StateAnalysis

Parse the structured response from state analysis query.
Robust to malformed output — uses empty lists and 0.5 confidence as fallback.
"""
function parse_state_analysis(response::AbstractString)
    visible_objects = String[]
    current_obstacle = ""
    promising_directions = String[]
    situation_summary = ""
    confidence = 0.5

    lines = split(response, "\n")
    for line in lines
        line = strip(line)

        if startswith(lowercase(line), "objects:")
            content = strip(line[9:end])
            visible_objects = [strip(s) for s in split(content, ",") if !isempty(strip(s))]
        elseif startswith(lowercase(line), "obstacle:")
            current_obstacle = strip(line[10:end])
        elseif startswith(lowercase(line), "promising:")
            content = strip(line[11:end])
            promising_directions = [strip(s) for s in split(content, ",") if !isempty(strip(s))]
        elseif startswith(lowercase(line), "summary:")
            situation_summary = strip(line[9:end])
        elseif startswith(lowercase(line), "confidence:")
            try
                conf_str = strip(line[12:end])
                confidence = parse(Float64, conf_str)
                confidence = clamp(confidence, 0.0, 1.0)
            catch
                confidence = 0.5
            end
        end
    end

    return StateAnalysis(visible_objects, current_obstacle, promising_directions, situation_summary, confidence)
end

"""
    apply_state_analysis_priors!(analysis::StateAnalysis, actions, action_beliefs, sensor)

Apply Bayesian update to action beliefs based on state analysis.
Promising directions boost action beliefs using the analysis-specific accuracy
(separate from binary TPR/FPR and selection accuracy).
"""
function apply_state_analysis_priors!(analysis::StateAnalysis, actions, action_beliefs::Dict, sensor)
    promising_lower = [lowercase(d) for d in analysis.promising_directions]
    α = sensor isa LLMSensor ? analysis_accuracy(sensor) : 0.5
    Z = 0.0

    for a in actions
        a_lower = lowercase(string(a))
        is_promising = any(d -> occursin(d, a_lower) || occursin(a_lower, d), promising_lower)
        prior = get(action_beliefs, a, 1.0 / length(actions))

        if is_promising
            action_beliefs[a] = prior * α
        else
            action_beliefs[a] = prior * (1 - α)
        end
        Z += action_beliefs[a]
    end
    # Renormalize
    if Z > 0
        for a in actions
            action_beliefs[a] /= Z
        end
    end
end

# ============================================================================
# SELECTION QUERY (ask LLM to pick best action from a list)
# ============================================================================

"""
    query_selection(sensor::LLMSensor, context::String, actions) → Union{action, Nothing}

Ask the LLM to select the most promising action from the full action list,
given rich context about the current game state, recent history, and
accumulated knowledge.

Returns the matched action, or nothing if the response couldn't be parsed.
"""
function query_selection(sensor::LLMSensor, context::String, actions)
    sensor.n_queries += 1

    action_list = join(["  $(i). $(a)" for (i, a) in enumerate(actions)], "\n")

    prompt = """You are advising an agent playing a text adventure game.

The agent needs to make REAL progress toward winning the game. Based on the situation below, which single action from the list is most likely to advance the game?

Avoid actions that have already been tried with no effect (listed in confirmed useless).
Prefer actions that explore new areas or interact with new objects over re-examining things already seen.

$context

Available actions:
$action_list

Reply with ONLY the action text, nothing else."""

    response = strip(sensor.llm_client.query(prompt))
    response_lower = lowercase(response)

    # Match response to an action (exact match first, then substring)
    for a in actions
        if lowercase(string(a)) == response_lower
            return a
        end
    end
    for a in actions
        a_lower = lowercase(string(a))
        if occursin(a_lower, response_lower) || occursin(response_lower, a_lower)
            return a
        end
    end

    return nothing
end

"""
    update_beliefs_from_selection!(sensor, actions, selected, action_beliefs) → Dict

Apply Bayesian update from a selection query result using the correct
Categorical observation model.

The sensor picks action j with accuracy α (learned). For each action i:
    P(sensor picks j | i is best) = α       if i == j
    P(sensor picks j | i is best) = (1-α)/(N-1)  if i ≠ j

Then by Bayes' rule:
    P(i is best | sensor picked j) ∝ P(sensor picked j | i is best) × π_i
"""
function update_beliefs_from_selection!(sensor, actions, selected, action_beliefs::Dict)
    α = sensor isa LLMSensor ? selection_accuracy(sensor) : 0.5
    N = length(actions)
    Z = 0.0

    for a in actions
        π = get(action_beliefs, a, 1.0 / N)
        if a == selected
            action_beliefs[a] = π * α
        else
            action_beliefs[a] = π * (1 - α) / max(N - 1, 1)
        end
        Z += action_beliefs[a]
    end
    # Renormalize
    if Z > 0
        for a in actions
            action_beliefs[a] /= Z
        end
    end
    return action_beliefs
end

# ============================================================================
# NULL OUTCOME DETECTION
# ============================================================================

"""
    is_null_outcome(obs_before, obs_after) → Bool

Detect whether an action produced no change — same command + same observation
text means the transition is S→S with reward 0.

This is a world model fact: P(obs_unchanged | action_helpful) ≈ 0. Used to
provide negative ground truth to sensors without waiting for sparse rewards.
"""
function is_null_outcome(obs_before, obs_after)
    text_before = if obs_before isa NamedTuple && hasproperty(obs_before, :text)
        obs_before.text
    else
        string(obs_before)
    end
    text_after = if obs_after isa NamedTuple && hasproperty(obs_after, :text)
        obs_after.text
    else
        string(obs_after)
    end
    return strip(text_before) == strip(text_after)
end
