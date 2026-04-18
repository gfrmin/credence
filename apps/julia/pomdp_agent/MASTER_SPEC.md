# MASTER_SPEC.md — Complete Bayesian Agent Framework

## Executive Summary

A principled framework for agents that:
1. Maintain **hierarchical uncertainty** — over states, dynamics, AND model structures
2. Reason about **trajectories**, not just single steps
3. Treat **information gathering and acting** as choices in the same decision space
4. Learn **state equivalence classes** to avoid loops on functionally identical states
5. Derive ALL behaviour from **expected utility maximisation** — no hacks

This document synthesises all discoveries from the design process.

---

# PART I: PHILOSOPHICAL FOUNDATIONS

## 1.1 The Core Principle

**Every agent behaviour must derive from expected utility maximisation.**

This means:
- No ε-greedy exploration (exploration emerges from uncertainty)
- No loop detection heuristics (loops indicate model failure)
- No hardcoded "ask first, then act" (both compete on EU)
- No exploration bonuses (Thompson sampling handles this)

If the agent behaves suboptimally, the model is wrong. Fix the model.

## 1.2 What Is Fixed vs Learned vs Discovered

### Fixed (The Physics)
- Bayesian inference as the belief update rule
- Expected utility maximisation as the decision criterion
- Probability theory as the language of uncertainty

### Learned (Within a Model)
- Transition probabilities P(s'|s,a)
- Reward distributions P(r|s,a)
- Sensor reliability (TPR, FPR)
- State equivalence classes

### Discovered (Meta-Level)
- Which state variables matter (model structure)
- Which questions are informative
- Subgoal decomposition
- Temporal abstractions (options)

## 1.3 The Hierarchy of Uncertainty

```
Level 3: Model Structure    P(M | history)     "What kind of world is this?"
           │
           ▼
Level 2: Parameters         P(θ | M, history)  "Given this world type, what are the dynamics?"
           │
           ▼
Level 1: State              P(s | θ, M, obs)   "Where am I right now?"
           │
           ▼
Level 0: Trajectories       P(τ | s, θ, M)     "What paths are possible from here?"
```

A fully Bayesian agent maintains uncertainty at ALL levels.

---

# PART II: MATHEMATICAL FRAMEWORK

## 2.1 The State Space Problem

### The Naïve Approach (What Failed)
```
State = hash(game_state)
```
Problem: Functionally equivalent states get different hashes.
- "Wearing trousers" ≠ "Not wearing trousers" (different hash)
- But if neither affects goal achievement, they're equivalent!
- Agent loops: take off trousers → put on trousers → take off...

### The Solution: Behavioural Equivalence (Bisimulation)

States s₁ and s₂ are **behaviourally equivalent** iff:

$$s_1 \sim s_2 \iff \forall a \in \mathcal{A}: P(r|s_1,a) = P(r|s_2,a) \land P([\cdot]|s_1,a) = P([\cdot]|s_2,a)$$

where [·] denotes equivalence class.

**Key insight**: We don't need to know the true equivalence relation. We LEARN it from experience by tracking behavioural signatures.

### Behavioural Signature

For state s, its signature is the observed (action → outcome) mapping:

$$\text{sig}(s) = \{a \mapsto \{(r_i, s'_i, n_i)\}\}$$

where $n_i$ is the count of times we observed outcome $(r_i, s'_i)$ after taking action $a$ in state $s$.

States with matching signatures (up to statistical noise) are equivalent.

## 2.2 The Trajectory Space

### Why Single-Step Reasoning Fails

In IF games:
- ~22 actions needed to complete a puzzle
- Rewards are sparse (0 for most actions)
- The right action now might look useless without seeing its role in the trajectory

Single-step EU: "This action gives 0 reward. That action gives 0 reward. They're equal."
Trajectory EU: "This action opens a path to +100 reward in 15 steps."

### Trajectory Definition

A trajectory τ is a sequence:

$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$$

### Expected Utility Over Trajectories

$$\mathbb{E}[U | s, \pi, \theta] = \mathbb{E}_{\tau \sim P(\tau | s, \pi, \theta)} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

The agent chooses policy π to maximise this.

### Thompson Sampling Over Trajectories

Rather than Thompson sampling a single transition, we sample an entire world model and plan optimally under it:

```
1. Sample world model: θ̂ ~ P(θ | history)
2. Plan optimal trajectory: τ* = argmax_τ E[U | τ, θ̂]
3. Execute first action of τ*
4. Observe outcome, update P(θ | history)
5. Repeat
```

This naturally balances exploration (uncertain models get sampled, leading to information-gathering trajectories) and exploitation (confident models get sampled, leading to reward-maximising trajectories).

## 2.3 The Unified Decision Space

### The Critical Insight

At each moment, the agent chooses from:

$$\mathcal{O} = \underbrace{\{take(a) : a \in \mathcal{A}\}}_{\text{game actions}} \cup \underbrace{\{ask(q, k) : q \in \mathcal{Q}, k \in \mathcal{K}\}}_{\text{sensor queries}}$$

Both are evaluated by expected utility. The agent picks:

$$o^* = \arg\max_{o \in \mathcal{O}} \mathbb{E}[U | o]$$

### Expected Utility of Game Actions (Trajectory-Aware)

$$\mathbb{E}[U | take(a)] = \mathbb{E}_{\theta \sim P(\theta|H)} \left[ r(s, a, \theta) + \gamma \cdot V^*(s', \theta) \right]$$

where $V^*(s', \theta)$ is the optimal value-to-go under sampled dynamics θ.

For computational tractability, approximate via MCTS:
- Sample θ once
- Build search tree under θ
- Estimate V* from tree statistics

### Expected Utility of Asking (Trajectory-Aware)

$$\mathbb{E}[U | ask(q, k)] = \mathbb{E}_{y \sim P(y|q,k)} \left[ V^*_{\text{after}}(s, \theta | y) \right] - c_k$$

where:
- $y$ is the sensor's response
- $V^*_{\text{after}}$ is the optimal value AFTER updating beliefs with answer $y$
- $c_k$ is the cost of querying sensor $k$

### Trajectory-Level Value of Information

$$\text{TVOI}(q, k) = \mathbb{E}_y \left[ V^*_{\text{after}}(s | y) \right] - V^*_{\text{before}}(s)$$

**Key difference from myopic VOI**: This considers how the answer improves the ENTIRE trajectory, not just the next action.

Computing TVOI exactly is expensive. Approximations:
1. **Sample-based**: Run MCTS before and after hypothetical answers, compare
2. **Heuristic**: Weight by planning depth — questions about distant states weighted less
3. **Decomposed**: If using subgoals, compute VOI per subgoal

## 2.4 Meta-Models: Uncertainty Over Structure

### The Problem

Standard Bayesian RL assumes a fixed model structure (e.g., tabular MDP) and learns parameters. But in IF:
- We don't know what state variables matter
- We don't know the causal structure
- We don't know the goal decomposition

### Model Space

Let M denote a model structure (e.g., "location + inventory determines transitions" vs "location + inventory + time of day determines transitions").

Full Bayesian treatment:

$$P(\theta, M | H) = P(\theta | M, H) \cdot P(M | H)$$

$$P(M | H) \propto P(H | M) \cdot P(M)$$

where $P(H | M) = \int P(H | \theta, M) P(\theta | M) d\theta$ is the marginal likelihood.

### Practical Approximation: Model Selection

Instead of full posterior over M, use:
1. Start with simple model (e.g., location only)
2. Detect contradictions (same state, different outcomes)
3. Expand model (add variables) when contradictions accumulate
4. Prune variables that don't affect predictions

This is **adaptive state abstraction** — the model structure itself is learned.

### Structure Learning Triggers

Expand the model when:
$$\frac{\text{contradictions}}{\text{observations}} > \epsilon_{\text{expand}}$$

Contract the model when:
$$I(\text{variable}; \text{outcome} | \text{other variables}) < \epsilon_{\text{prune}}$$

---

# PART III: SENSOR FRAMEWORK

## 3.1 Sensors as Noisy Information Sources

Every sensor k provides observations with characteristic reliability:

$$P(\text{response} | \text{true state}, k) = f_k(\text{true state})$$

For binary sensors:
- **TPR** (True Positive Rate): $P(\text{yes} | \text{true})$
- **FPR** (False Positive Rate): $P(\text{yes} | \text{false})$

### Key Principle: Reliability is LEARNED

We don't assume we know sensor reliability. We learn it from ground truth.

$$\text{TPR}_k \sim \text{Beta}(\alpha_{tp}, \beta_{tp})$$
$$\text{FPR}_k \sim \text{Beta}(\alpha_{fp}, \beta_{fp})$$

Update after observing ground truth:
- Sensor said yes, actually true: $\alpha_{tp} \leftarrow \alpha_{tp} + 1$
- Sensor said no, actually true: $\beta_{tp} \leftarrow \beta_{tp} + 1$
- Sensor said yes, actually false: $\alpha_{fp} \leftarrow \alpha_{fp} + 1$
- Sensor said no, actually false: $\beta_{fp} \leftarrow \beta_{fp} + 1$

## 3.2 Ground Truth Definition

**CRITICAL**: This was a major source of bugs.

**CORRECT**:
$$\text{action helped} \iff \text{reward} > 0$$

**INCORRECT** (caused failures):
- ❌ State changed
- ❌ New room reached
- ❌ Item acquired
- ❌ Any non-zero outcome

**Rationale**: Many state changes are neutral or harmful. Only positive reward indicates genuine progress toward the goal.

## 3.3 LLM as Sensor

The LLM is treated as a sensor, not a decision-maker.

```julia
struct LLMSensor <: Sensor
    client::LLMClient      # Ollama, OpenAI, etc.
    model::String          # "llama3.1", etc.
    
    # Learned reliability
    tp_α::Float64
    tp_β::Float64
    fp_α::Float64
    fp_β::Float64
    
    # Cost (in utility units)
    cost::Float64
    
    # Question templates
    templates::Dict{Symbol, String}
end
```

### Question Types

1. **Action queries**: "Will action X help make progress?"
2. **State queries**: "Am I in danger here?"
3. **Goal queries**: "What's the immediate objective?"
4. **Comparison queries**: "Is action X better than action Y?"

### Why LLM-as-Sensor Works

- The LLM has relevant knowledge (game walkthroughs, common puzzle patterns)
- But it's imperfect (hallucinations, context limitations)
- By learning TPR/FPR, the agent calibrates how much to trust it
- A useless LLM (TPR ≈ FPR) gets ignored automatically (VOI → 0)

## 3.4 Multi-Sensor Fusion

With multiple sensors, combine evidence:

$$P(\text{prop} | y_1, \ldots, y_K) = \frac{P(y_1, \ldots, y_K | \text{prop}) P(\text{prop})}{P(y_1, \ldots, y_K)}$$

Assuming conditional independence given truth:

$$P(\text{prop} | y_1, \ldots, y_K) \propto P(\text{prop}) \prod_k P(y_k | \text{prop})$$

---

# PART IV: PLANNING

## 4.1 Planning Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│  Level 2: Subgoal Selection                                 │
│  "Which subgoal should I pursue next?"                      │
│  Options: {pursue(g) for g in subgoals}                     │
├─────────────────────────────────────────────────────────────┤
│  Level 1: Trajectory Planning                               │
│  "What sequence of actions achieves subgoal g?"             │
│  Thompson MCTS with sampled dynamics                        │
├─────────────────────────────────────────────────────────────┤
│  Level 0: Unified Decision                                  │
│  "Should I act or ask?"                                     │
│  Compare EU(take(a)) vs EU(ask(q,k))                        │
└─────────────────────────────────────────────────────────────┘
```

## 4.2 Thompson MCTS

### Algorithm

```python
def thompson_mcts(state, model, config):
    root = MCTSNode(state)
    
    for iteration in range(config.mcts_iterations):
        # 1. Sample world model from posterior
        θ = model.sample_dynamics()
        
        # 2. Simulate trajectory under sampled model
        node = root
        path = []
        
        # Selection
        while node.is_expanded() and not node.is_terminal:
            action = node.select_ucb(config.ucb_c)
            path.append((node, action))
            node = node.children[action]
        
        # Expansion
        if not node.is_terminal:
            action = node.select_unexpanded()
            next_state = θ.sample_transition(node.state, action)
            reward = θ.sample_reward(node.state, action)
            child = MCTSNode(next_state)
            node.children[action] = child
            path.append((node, action))
            node = child
        
        # Rollout
        value = rollout(node.state, θ, config.rollout_depth)
        
        # Backpropagation
        for (n, a) in reversed(path):
            value = reward + config.discount * value
            n.update(a, value)
    
    return root.best_action(criterion='visit_count')
```

### Why Thompson + MCTS?

- **Thompson**: Samples world model → explores uncertain dynamics
- **MCTS**: Plans multi-step trajectories → sees long-term consequences
- **Combined**: Explores trajectories that might reveal information about uncertain dynamics

## 4.3 Subgoal Discovery

### Automatic Subgoal Detection

Subgoals are states that:
1. Are reached repeatedly on successful trajectories
2. Significantly increase expected value-to-go
3. Partition the state space into "before" and "after"

### Subgoal Representation

```julia
struct Subgoal
    description::String          # "Get the lamp"
    predicate::Function          # state -> Bool, is subgoal achieved?
    value_increase::Float64      # Expected V(after) - V(before)
    reachability::Float64        # P(reach this subgoal)
end
```

### Subgoal-Conditioned Planning

Given subgoals G₁, G₂, ..., Gₙ:

1. Estimate value of achieving each: $V(G_i)$
2. Estimate cost of achieving each from current state: $C(s, G_i)$
3. Select subgoal: $G^* = \arg\max_G V(G) - C(s, G)$
4. Plan trajectory to $G^*$

This decomposes the long-horizon problem into shorter segments.

---

# PART V: IMPLEMENTATION DETAILS

## 5.1 Prior Beliefs

### Action Success Prior

**WRONG** (caused failures):
```julia
prior_helps = 0.5  # Maximum entropy, but unrealistic
```

**CORRECT**:
```julia
prior_helps = 1.0 / n_actions  # Realistic: most actions don't help
```

**Rationale**: In IF games with 50 actions, typically 1-2 lead to progress. Prior of 0.5 is absurdly optimistic.

### Sensor Reliability Prior

```julia
TPR ~ Beta(2, 1)   # Mean 0.67: weakly believe sensor detects true positives
FPR ~ Beta(1, 2)   # Mean 0.33: weakly believe sensor avoids false positives
```

### Dynamics Prior

```julia
# Dirichlet concentration for transitions
α₀ = 0.1  # Weak prior, quickly overwhelmed by data

# Normal-Gamma for rewards
μ₀ = 0.0   # Prior mean
κ₀ = 0.1   # Prior strength
α₀ = 1.0   # Shape
β₀ = 1.0   # Rate
```

## 5.2 The Unified Decision Function

```julia
function decide(
    state::AbstractState,
    actions::Vector{Action},
    sensors::Vector{Sensor},
    model::WorldModel,
    config::Config
)::Decision
    
    # === PHASE 1: Evaluate game actions (trajectory-aware) ===
    
    # Sample world model for Thompson
    θ = sample_dynamics(model)
    
    # Run MCTS to get action values
    action_values = mcts(state, θ, actions, config)
    
    best_action = argmax(action_values)
    best_action_value = action_values[best_action]
    
    # === PHASE 2: Evaluate questions (trajectory-aware VOI) ===
    
    best_query = nothing
    best_query_value = -Inf
    
    for sensor in sensors
        for question in generate_questions(state, actions, config)
            # Compute trajectory-level VOI
            tvoi = trajectory_voi(
                state, question, sensor, 
                action_values, model, config
            )
            
            if tvoi > sensor.cost
                query_value = best_action_value + tvoi - sensor.cost
                
                if query_value > best_query_value
                    best_query_value = query_value
                    best_query = (question, sensor)
                end
            end
        end
    end
    
    # === PHASE 3: Unified decision ===
    
    if best_query !== nothing && best_query_value > best_action_value
        return AskDecision(best_query...)
    else
        return ActDecision(best_action)
    end
end
```

## 5.3 Trajectory-Level VOI

```julia
function trajectory_voi(
    state, question, sensor,
    action_values, model, config
)
    # Current best trajectory value
    V_before = maximum(values(action_values))
    
    # Get prior for the proposition this question is about
    prior = get_proposition_prior(question, state, model)
    
    # Sensor parameters
    tpr_est = tpr(sensor)
    fpr_est = fpr(sensor)
    
    # P(sensor says yes)
    p_yes = tpr_est * prior + fpr_est * (1 - prior)
    p_no = 1 - p_yes
    
    # Posterior beliefs
    post_yes = bayes_update(prior, tpr_est, fpr_est, true)
    post_no = bayes_update(prior, tpr_est, fpr_est, false)
    
    # Re-plan trajectories under updated beliefs
    # This is the expensive part — we re-run MCTS with modified beliefs
    
    model_yes = update_model_belief(model, question, post_yes)
    model_no = update_model_belief(model, question, post_no)
    
    θ_yes = sample_dynamics(model_yes)
    θ_no = sample_dynamics(model_no)
    
    values_yes = mcts(state, θ_yes, actions, config)
    values_no = mcts(state, θ_no, actions, config)
    
    V_after_yes = maximum(values(values_yes))
    V_after_no = maximum(values(values_no))
    
    # Expected value after asking
    E_V_after = p_yes * V_after_yes + p_no * V_after_no
    
    # VOI = improvement in trajectory value
    return E_V_after - V_before
end
```

## 5.4 State Abstraction Implementation

```julia
mutable struct BisimulationAbstractor
    # Concrete state → abstract state ID
    state_to_abstract::Dict{Any, Int}
    
    # Abstract ID → set of concrete states
    abstract_to_states::Dict{Int, Set{Any}}
    
    # Behavioural signatures: state → {action → [(reward, next_abstract, count)]}
    signatures::Dict{Any, Dict{Any, Vector{Tuple{Float64, Int, Int}}}}
    
    # Similarity threshold for merging
    threshold::Float64
    
    # Next abstract ID
    next_id::Int
end

function get_abstract(abs::BisimulationAbstractor, state)
    if haskey(abs.state_to_abstract, state)
        return abs.state_to_abstract[state]
    end
    
    # Try to find matching signature
    if haskey(abs.signatures, state)
        sig = abs.signatures[state]
        for (abstract_id, states) in abs.abstract_to_states
            for existing in states
                if haskey(abs.signatures, existing)
                    if signatures_match(sig, abs.signatures[existing], abs.threshold)
                        # Found match — merge
                        abs.state_to_abstract[state] = abstract_id
                        push!(states, state)
                        return abstract_id
                    end
                end
            end
        end
    end
    
    # No match — create new abstract state
    id = abs.next_id
    abs.next_id += 1
    abs.state_to_abstract[state] = id
    abs.abstract_to_states[id] = Set([state])
    return id
end

function record_transition!(abs::BisimulationAbstractor, s, a, r, s′)
    # Record in signature
    if !haskey(abs.signatures, s)
        abs.signatures[s] = Dict()
    end
    if !haskey(abs.signatures[s], a)
        abs.signatures[s][a] = []
    end
    
    s′_abstract = get_abstract(abs, s′)
    
    # Find or create entry
    found = false
    for (i, (r_old, s′_old, n)) in enumerate(abs.signatures[s][a])
        if r_old ≈ r && s′_old == s′_abstract
            abs.signatures[s][a][i] = (r_old, s′_old, n + 1)
            found = true
            break
        end
    end
    if !found
        push!(abs.signatures[s][a], (r, s′_abstract, 1))
    end
    
    # Check for contradictions and refine if needed
    check_and_refine!(abs)
end
```

---

# PART VI: FORBIDDEN PATTERNS

These patterns caused failures. **Never use them.**

## ❌ Exploration Bonuses

```julia
# WRONG
eu = belief * reward + exploration_bonus

# RIGHT
eu = belief * reward  # Exploration via Thompson sampling
```

## ❌ Separate Ask/Act Logic

```julia
# WRONG
if should_ask():
    ask()
else:
    act()

# RIGHT
decision = argmax(EU(actions) ∪ EU(questions))
```

## ❌ LLM as Decision-Maker

```julia
# WRONG
action = llm.choose_action(state, actions)

# RIGHT
answer = llm.answer_question(state, question)
beliefs = update(beliefs, answer)
action = argmax_eu(beliefs, actions)
```

## ❌ Loop Detection

```julia
# WRONG
if action in recent_actions:
    action = random_choice()

# RIGHT
# If looping, fix the state abstraction (bisimulation)
```

## ❌ State Change as Ground Truth

```julia
# WRONG
helped = (new_state != old_state)

# RIGHT
helped = (reward > 0)
```

## ❌ Uniform Prior Over Action Success

```julia
# WRONG
prior = 0.5

# RIGHT
prior = 1.0 / n_actions
```

## ❌ Myopic VOI for Trajectory Problems

```julia
# WRONG
voi = E[best_next_action | answer] - best_next_action

# RIGHT
voi = E[best_trajectory_value | answer] - best_trajectory_value
```

---

# PART VII: DEBUGGING GUIDE

## Symptom: Agent Always Asks, Never Acts

**Check**: Is sensor cost too low? Is VOI computed correctly?

**Fix**: 
- Set `sensor_cost = 0.01` or higher
- Verify VOI computes *improvement*, not raw probability
- Check that P(yes) and posteriors are reasonable

## Symptom: Agent Never Asks

**Check**: Is sensor cost too high? Is TPR/FPR prior too pessimistic?

**Fix**:
- Lower sensor cost
- Use prior TPR Beta(2,1), FPR Beta(1,2)
- Verify question generation produces relevant questions

## Symptom: Agent Loops on Equivalent States

**Check**: Is bisimulation enabled? Are signatures being recorded?

**Fix**:
- Enable bisimulation abstractor
- Verify `record_transition!` is called
- Check `signatures_match` threshold (try 0.9)

## Symptom: Sensor Reliability Doesn't Improve

**Check**: Is ground truth determined? Are updates happening?

**Fix**:
- Only use `reward > 0` as ground truth
- Add logging to `update_reliability!`
- Verify question-action correspondence is tracked

## Symptom: Poor Performance Despite Asking

**Check**: Is the LLM actually helpful? Are questions informative?

**Fix**:
- Check LLM raw outputs
- Verify prompt includes sufficient context
- Try different question formulations
- Consider LLM might be unhelpful (learned TPR ≈ FPR means ignore it)

## Symptom: MCTS Takes Too Long

**Check**: Planning depth, iteration count, action branching

**Fix**:
- Reduce `mcts_iterations` (try 50)
- Reduce `planning_depth` (try 5)
- Prune unpromising actions using sensor priors
- Cache repeated computations

---

# PART VIII: COMPLETE TYPE HIERARCHY

```julia
# === Core Abstract Types ===

abstract type World end
abstract type WorldModel end
abstract type Sensor end
abstract type Planner end
abstract type StateAbstractor end

# === Decision Types ===

struct ActDecision
    action::Any
end

struct AskDecision
    question::String
    sensor::Sensor
end

const Decision = Union{ActDecision, AskDecision}

# === World Interface ===

reset!(w::World) → observation
step!(w::World, action) → (observation, reward, done, info)
actions(w::World, observation) → Vector{Action}
render(w::World) → String

# === WorldModel Interface ===

update!(m::WorldModel, s, a, r, s′)
sample_dynamics(m::WorldModel) → SampledDynamics
transition_dist(m::WorldModel, s, a) → Distribution
reward_dist(m::WorldModel, s, a) → Distribution
entropy(m::WorldModel) → Float64

# === Sensor Interface ===

query(sensor::Sensor, state, question) → response
tpr(sensor::Sensor) → Float64
fpr(sensor::Sensor) → Float64
cost(sensor::Sensor) → Float64
update_reliability!(sensor::Sensor, predicted, actual)

# === Planner Interface ===

plan(p::Planner, state, model, actions) → action
plan_trajectory(p::Planner, state, model, horizon) → Vector{Action}

# === StateAbstractor Interface ===

get_abstract(a::StateAbstractor, state) → AbstractState
record_transition!(a::StateAbstractor, s, a, r, s′)
check_contradiction(a::StateAbstractor) → Option{Contradiction}
refine!(a::StateAbstractor, contradiction)
```

---

# PART IX: CONFIGURATION

```julia
@kwdef struct AgentConfig
    # === Planning ===
    planning_depth::Int = 10
    mcts_iterations::Int = 100
    rollout_depth::Int = 20
    discount::Float64 = 0.99
    ucb_c::Float64 = 2.0
    
    # === Sensors ===
    sensor_cost::Float64 = 0.01
    max_questions_per_step::Int = 5
    question_budget::Int = 20
    
    # === Priors ===
    action_success_prior::Function = n -> 1.0/n  # P(action helps)
    sensor_tpr_prior::Tuple{Float64,Float64} = (2.0, 1.0)
    sensor_fpr_prior::Tuple{Float64,Float64} = (1.0, 2.0)
    
    # === World Model ===
    transition_prior::Float64 = 0.1  # Dirichlet concentration
    reward_prior_mean::Float64 = 0.0
    reward_prior_var::Float64 = 1.0
    
    # === State Abstraction ===
    use_bisimulation::Bool = true
    signature_similarity_threshold::Float64 = 0.9
    
    # === Meta-Learning ===
    model_expansion_threshold::Float64 = 0.1  # Contradiction rate
    model_pruning_threshold::Float64 = 0.01   # Mutual information
    
    # === Subgoals ===
    use_subgoals::Bool = true
    subgoal_detection_threshold::Float64 = 0.5
end
```

---

# PART X: IMPLEMENTATION ORDER

1. **Core types** — `types.jl`
2. **Binary sensor + VOI** — `sensors/binary.jl`, `sensors/voi.jl`
3. **Tabular world model** — `models/tabular.jl`
4. **Unified decision function** — `core/decision.jl`
5. **Agent loop** — `core/agent.jl`
6. **GridWorld** — `worlds/gridworld.jl`
7. **Unit tests** — VOI, sensor learning, unified decision
8. **Thompson MCTS** — `planning/thompson_mcts.jl`
9. **Trajectory VOI** — `sensors/trajectory_voi.jl`
10. **Bisimulation** — `abstraction/bisimulation.jl`
11. **Integration tests**
12. **LLM sensor** — `sensors/llm.jl`
13. **Jericho world** — `worlds/jericho.jl`
14. **Subgoal discovery** — `planning/subgoals.jl`
15. **Meta-model adaptation** — `models/adaptive.jl`
16. **Full integration tests**

---

# PART XI: KEY EQUATIONS SUMMARY

**Unified Decision:**
$$o^* = \arg\max_{o \in \mathcal{A} \cup \mathcal{Q}} \mathbb{E}[U | o]$$

**Trajectory Value:**
$$V^\pi(s) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^T \gamma^t r_t \mid s_0 = s \right]$$

**Thompson Sampling:**
$$\theta \sim P(\theta | \mathcal{H}), \quad a^* = \arg\max_a Q(s, a; \theta)$$

**Trajectory VOI:**
$$\text{TVOI}(q, k) = \mathbb{E}_{y \sim P(y|q,k)} \left[ V^*(s | y) \right] - V^*(s)$$

**Bayes Update (Sensor):**
$$P(\text{prop} | y) = \frac{P(y | \text{prop}) P(\text{prop})}{P(y)}$$

**Bisimulation:**
$$s_1 \sim s_2 \iff \forall a: P(r|s_1,a) = P(r|s_2,a) \land P([\cdot]|s_1,a) = P([\cdot]|s_2,a)$$

**Model Selection:**
$$P(M | \mathcal{H}) \propto P(\mathcal{H} | M) P(M)$$

---

# APPENDIX A: Glossary

- **Bisimulation**: State equivalence based on identical behaviour
- **MCTS**: Monte Carlo Tree Search
- **POMDP**: Partially Observable Markov Decision Process
- **Thompson Sampling**: Select actions by sampling from posterior over models
- **TPR**: True Positive Rate = P(yes | true)
- **FPR**: False Positive Rate = P(yes | false)
- **VOI**: Value of Information
- **TVOI**: Trajectory-level Value of Information

---

# APPENDIX B: References

1. **Bayes-Adaptive POMDPs**: Ross, Chaib-draa, Pineau (2008)
2. **Thompson Sampling**: Russo et al. (2018) "Tutorial on Thompson Sampling"
3. **MCTS**: Browne et al. (2012) "A Survey of MCTS Methods"
4. **Bisimulation**: Givan, Dean, Greig (2003) "Equivalence Notions and Model Minimization"
5. **Value of Information**: Howard (1966) "Information Value Theory"
6. **Bayesian RL**: Ghavamzadeh et al. (2015) "Bayesian Reinforcement Learning: A Survey"
