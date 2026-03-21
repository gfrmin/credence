# Program-Space Bayesian Agent

## Research Spec for the Credence DSL

---

## 1. Thesis

An agent that maintains a complexity-weighted MixtureMeasure over programs, and conditions it on observations via Bayes' rule, will exhibit feature discovery, model selection, change-point detection, meta-learning, grammar evolution, and sensor adaptation — all as emergent behaviour of a single mechanism. No dedicated subsystem is needed for any of these capabilities.

The mechanism is: `condition` over a nested MixtureMeasure (grammars × programs), weighted by description length (Solomonoff prior / Bayesian Occam's Razor). The five invariant components are `condition`, `expect`, `optimise`, the nested program space with complexity prior, and the raw sensor stream. Everything else is learned.

A population of such agents is a superorganism doing the same inference at a higher level: the population's distribution over agent-grammars is itself a MixtureMeasure conditioned on fitness. This spec focuses on the single-agent case. The population extension requires no new theory — only a host-level loop that manages reproduction and selection.

---

## 2. The Unified Mechanism

### 2.1 The belief state

The agent's entire belief about the world is a single flat MixtureMeasure:

```
belief : MixtureMeasure
  components: [ TaggedBetaMeasure(tag=j, beta=Beta_j) for j in 1..M ]
  weights:    [ w_j ∝ 2^(-|G_j|) × 2^(-|P_j|) × likelihood_j ]
```

Each component is a `TaggedBetaMeasure` — a BetaMeasure tracking P(enemy | this program's predicate matches), tagged with the component's index. The tag is immutable: conditioning updates the Beta parameters but never changes which program a component represents. The tag is what allows a single kernel to dispatch per-component: it extracts the tag, looks up the corresponding compiled predicate, and computes the appropriate likelihood.

The mixture is flat by design. The existing `condition(MixtureMeasure, kernel, obs)` dispatch in ontology.jl flattens nested mixtures as a side effect. Rather than fighting this, the architecture embraces it — grammar × program structure is maintained by the host in parallel arrays bundled with the belief in an `AgentState` struct:

```
AgentState:
  belief           : MixtureMeasure of TaggedBetaMeasures
  metadata         : Vector{Tuple{Int,Int}}    # (grammar_id, program_id) per component
  compiled_kernels : Vector{CompiledKernel}     # precompiled closures for conditioning
  all_programs     : Vector{Program}            # ASTs retained for subtree extraction
```

The four arrays are always the same length and aligned by index. `sync_prune!` and `sync_truncate!` maintain this invariant by pruning/truncating all arrays together and reindexing tags so that `belief.components[i].tag == i` for all `i`. Since kernels are rebuilt each step, stale tag references from previous steps cannot persist.

Each component's prior weight is the product of its grammar's prior (`2^(-|G|)`) and its program's prior within that grammar (`2^(-|P|)`). Grammar-level inference is recovered by the host as needed:

```
weight(Grammar_i) = Σ_j w_j  where metadata[j].grammar_id == i
```

This gives exact grammar-level posteriors without maintaining nested structure.

### 2.2 Conditioning (the only learning rule)

When the agent observes `(sensor_vector, energy_delta)`:

The host constructs a single kernel whose `log_density(component_j, obs)` function encodes program-specific logic: it looks up the program associated with component `j` (via the metadata index), evaluates that program's predicate against the sensor vector (projected through the program's grammar's sensor config), and computes the likelihood of the observed energy delta given the program's Beta. One kernel, applied uniformly across the flat mixture, but producing different likelihoods per component because the program identity is encoded in the hypothesis.

The `condition(belief, kernel, obs)` call reweights every component simultaneously. Programs whose predicates fire correctly and predict the right entity type gain weight. Programs that mispredict lose weight. Grammar-level selection is a side effect: if all programs belonging to Grammar_i predict poorly, the aggregate weight `Σ_j w_j` for `grammar_id == i` drops. No separate grammar-level conditioning step is needed.

The kernel is rebuilt each timestep, closing over the current temporal state (a sliding window of recent sensor readings maintained by the host). This means temporal operators like `CHANGED` and `PERSISTS` evaluate against the window captured at kernel construction time. The DSL never needs to know about temporal state — it just calls `condition` with a kernel that already has the history baked in. This is consistent with the existing pattern in `host_credence_agent.jl` where kernels are constructed fresh each step.

This is Bayesian Occam's Razor operating simultaneously on model structure (which program?) and representational structure (which grammar?), via a single flat `condition` call.

### 2.3 Why every capability emerges

**Feature discovery.** A grammar with a nonterminal `RED := GT(0, 0.9) ∧ LT(1, 0.15) ∧ LT(2, 0.15)` lets programs reference "redness" at cost 1 instead of cost 5. If redness is predictive, programs using it are short and accurate, so this grammar's marginal likelihood is high. The agent "discovers" the concept RED by shifting posterior mass toward grammars that define it.

**Model selection.** Programs within a grammar compete. `CLASSIFY(RED, enemy)` and `CLASSIFY(MOVING, enemy)` coexist in the program mixture. Conditioning on data selects the one that predicts better. This is standard Bayesian model comparison; no special mechanism needed.

**Change-point detection.** Programs with temporal operators compete with stable programs. `CLASSIFY(RED, enemy)` competes with `CLASSIFY(IF_BEFORE(50, RED, MOVING), enemy)` (red means enemy before step 50, then movement means enemy). If the world changes its rule at step 50, the stable program starts predicting poorly and the temporal program rises. The agent detects the regime change not via a dedicated change-point mechanism but because programs encoding temporal structure become more credible.

**Meta-learning.** Across regime changes within a continuous lifetime, grammars whose nonterminals capture recurrent abstractions (e.g., colour features that keep being informative across regimes) maintain high marginal likelihood. The agent's effective prior at the start of each new regime is shaped by all previous regimes — it learns faster because its grammar has been refined by experience.

**Sensor adaptation.** A grammar's sensor config determines what the agent can perceive. A grammar with only two sensor channels (say, channels 0 and 5 — hue and speed) cannot express programs that reference wall distance. If wall distance matters, this grammar's programs predict poorly, and the grammar loses posterior mass. Grammars with richer sensors that happen to capture the informative dimensions rise. The agent's "perceptual apparatus" adapts to the environment through the same inference that adapts its beliefs.

**Strategy.** `optimise` over the action space, with `voi` determining the value of observation, gives the agent a decision policy. As beliefs sharpen (high-posterior programs are confident), VOI drops and the agent exploits. When beliefs are disrupted (regime change detected via temporal programs), VOI rises and the agent explores. No epsilon-greedy, no exploration bonus. Just EU maximisation with endogenous information value.

---

## 3. Formal Definitions

### 3.1 Sensor configuration

```
SensorConfig = Vector{SensorChannel}

SensorChannel:
  source_index :: Int           # which dimension of the true environment state
  transform    :: Symbol        # :identity, :threshold, :delta (change since last step),
                                #   :windowed_mean (rolling average over n steps)
  noise_σ      :: Float64       # observation noise (higher = less precise)
  cost         :: Float64       # contributes to grammar complexity
```

The `source_index` maps into the environment's true state vector. The `transform` determines preprocessing. The `noise_σ` determines how reliably the agent perceives that channel. The `cost` penalises the grammar for maintaining the channel.

A minimal sensor config might be a single noisy proximity channel. A rich one might have 8+ channels covering colour, position, speed, and wall distance. The complexity prior penalises richness; the data rewards informativeness. The posterior finds the balance.

The environment exposes a fixed-dimensionality true state vector for each entity. Every grammar's sensor config is a projection of this vector — selecting dimensions, transforming them, and adding noise. Different grammars can "see" different things.

### 3.2 Terminal alphabet

For an agent with sensor config `S` of `k` channels:

**Predicates** (per channel `i`, per threshold `t ∈ T`):
```
GT(i, t)  →  reading_i > t       cost: 1
LT(i, t)  →  reading_i < t       cost: 1
```

Threshold set `T = {0.1, 0.3, 0.5, 0.7, 0.9}` (5 values). Total predicate terminals: `2 × k × |T|`.

**Connectives:**
```
AND(p, q)         cost: 1
OR(p, q)          cost: 1
NOT(p)            cost: 1
```

**Temporal operators:**
```
PERSISTS(p, n)    cost: 1     # p true for last n steps (n ∈ {2, 3, 5})
CHANGED(p)        cost: 1     # p flipped since last step
SINCE(p, q)       cost: 1     # p has been true since q was last true
```

**Output:**
```
PREDICT(predicate) → P(enemy | predicate true) with Beta-Bernoulli conjugate
```

A program is a PREDICT expression whose predicate is built from terminals, connectives, and grammar nonterminals.

### 3.3 Grammar

```
Grammar:
  sensor_config   :: SensorConfig
  rules           :: Vector{ProductionRule}
  
ProductionRule:
  name  :: Symbol                # nonterminal name
  body  :: Program               # expansion (over terminals + earlier nonterminals)
```

Grammar complexity:
```
|G| = Σ(channel.cost for channel in sensor_config) + Σ(1 + |body_i| for rule_i in rules)
```

Grammar prior weight: `2^(-|G|)`.

The empty grammar (no rules, minimal sensor config) has the highest prior. Every nonterminal and every sensor channel must earn its keep by enabling shorter, more predictive programs.

### 3.4 Program

A program under grammar `G` is a PREDICT expression built from `G`'s terminals and nonterminals:

```
Program:
  predicate   :: Expression      # composed from grammar
  complexity  :: Int             # derivation length using G's nonterminals at cost 1
```

Program prior weight within grammar `G`: `2^(-complexity)`.

The same logical predicate can have different complexity under different grammars. Under the empty grammar, "red AND moving" costs ~9 (raw sensor predicates). Under a grammar with `RED` and `MOVING` nonterminals, it costs ~3. This is the compression payoff that drives grammar evolution.

### 3.5 Prediction and likelihood

Each program contains a Beta distribution (conjugate with Bernoulli) tracking its confidence that the predicate identifies enemies:

```
program.beta : BetaMeasure       # P(enemy | predicate matches)
```

When the agent interacts with an entity:
1. Evaluate the program's predicate against the entity's sensor reading (through the grammar's sensor config).
2. The predicate is either true or false for this entity.
3. The interaction reveals whether the entity was food (energy > 0) or enemy (energy < 0).
4. If predicate was true: condition the Beta on the observed type (enemy → success, food → failure).
5. If predicate was false: condition a complementary Beta (or simply don't update — the predicate made no claim).

The program's likelihood for this observation is:
```
P(outcome | program) = expect(program.beta, outcome) if predicate matched
                     = base_rate                      if predicate didn't match
```

Programs whose predicates fire on the right entities and correctly predict their type accumulate likelihood. Programs that fire indiscriminately or predict incorrectly lose weight.

---

## 4. The Grid World

### 4.1 Environment

5×5 grid. Terrain: ground, wall, mud (mud slows movement by 1 turn).

Entities have hidden types (FOOD, ENEMY, NEUTRAL) and observable properties. The environment maintains a true state vector per entity:

```
EntityState (hidden from agent, used to generate sensor readings):
  rgb         :: Tuple{Float64, Float64, Float64}   # (r, g, b) ∈ [0,1]³
  pos         :: Tuple{Int, Int}                     # grid position
  speed       :: Float64                             # 0.0 = stationary, 1.0 = max speed
  wall_dist   :: Float64                             # distance to nearest wall
  kind        :: EntityKind                          # FOOD, ENEMY, NEUTRAL (hidden)
  energy      :: Float64                             # energy on interaction (hidden)
```

The agent never sees EntityState directly. It sees sensor readings, which are noisy projections through its grammar's SensorConfig.

### 4.2 World rules

5 rule sets, one active per regime (the agent is not told which, nor when it changes):

| Rule | Classification logic | Key discriminating sensors |
|------|---------------------|--------------------------|
| all-food | Everything is food | None needed |
| colour-typed | Red=enemy, blue=food, green=neutral | Colour channels |
| motion-typed | Moving=enemy, stationary=food | Speed channel |
| territorial | Near-wall=enemy, centre=food | Wall-distance channel |
| mixed | Red AND moving=enemy, else food | Colour + speed channels |

The world may change its active rule mid-run (without notification). The agent must detect this via inference — programs with temporal structure that model regime changes will outperform stable programs.

### 4.3 Actions

| Action | Effect | Information gained | Cost |
|--------|--------|--------------------|------|
| move_N/S/E/W | Move one cell (mud: skip next turn) | See new entities | -1 energy (movement cost) |
| interact | Touch adjacent entity | Energy delta reveals type | Variable (positive or negative) |
| observe | Stay put, watch for one turn | See entity movement patterns | -1 energy (opportunity cost) |
| wait | Do nothing | Nothing new | 0 |

### 4.4 Sensor projection

Each entity's true state is projected through the agent's SensorConfig to produce a sensor vector:

```julia
function project(entity::EntityState, config::SensorConfig)::Vector{Float64}
    readings = Float64[]
    for channel in config.channels
        raw = entity_state_field(entity, channel.source_index)
        transformed = apply_transform(raw, channel.transform)
        noisy = transformed + randn() * channel.noise_σ
        push!(readings, clamp(noisy, 0.0, 1.0))
    end
    return readings
end
```

An agent whose grammar lacks a colour channel literally cannot distinguish red from blue entities. Its programs can only reference the channels its grammar provides. This is how sensor adaptation works: grammars with informative sensor configs enable short, predictive programs and accumulate posterior mass.

---

## 5. Implementation Plan

### 5.1 Guiding principles

The DSL already has `condition`, `expect`, `optimise`, `voi`, `MixtureMeasure`, `ProductMeasure`, `BetaMeasure`, and `GaussianMeasure`. The implementation should maximise reuse of these existing primitives. New types should be added only when structurally necessary for the nesting, not for convenience.

The split between DSL and host follows one rule: the DSL handles inference (conditioning, prediction, decision). The host handles the world (simulation, sensor projection, grammar enumeration) and the interface between them (compiling grammars into kernels the DSL can condition on). The host constructs the belief structure; the DSL reasons over it.

**Enforcement of design invariants.** The DSL enforces correct agent behaviour by construction — you can't add an exploration bonus because the primitive doesn't exist. But the host-side Julia code (kernel compilation, grammar perturbation) is unconstrained by the DSL's type system, which is where implementation shortcuts are most likely to appear. Design invariants in the host are enforced at three levels:

1. **Type-level constraints (primary).** Design structs so the wrong thing is structurally unrepresentable. `CompiledKernel` has no AST field — you can't interpret a tree that isn't there. `propose_nonterminal` requires a `SubprogramFrequencyTable` — you can't propose nonterminals without posterior analysis. The type system makes shortcuts fail at compile time, not at review time.

2. **Observable-consequence tests (secondary).** Tests that verify the *effects* of correct implementation without inspecting internals. Kernel evaluation must complete 10,000 calls in < 1ms (catches AST interpretation). Proposed nonterminals must appear as subtrees in high-posterior programs (catches random mutation). Compression payoff must be measurable (catches vacuous promotions).

3. **Documentation in CLAUDE.md (tertiary).** Forbidden patterns with diagnostic criteria. This is the weakest enforcement — it relies on the implementer reading and obeying. It exists to explain *why* the type constraints and tests are designed the way they are, not as the primary defence.

### 5.2 DSL types and ontology.jl changes

**`TaggedBetaMeasure`** (~30 lines in ontology.jl). The program index is genuinely part of the hypothesis — each component isn't just "P(enemy) is Beta(α, β)" but "program #47 is the correct world model, and under #47, P(enemy | predicate matches) is Beta(α, β)." This requires a measure type that carries the hypothesis identity alongside the belief parameters:

```julia
struct TaggedBetaMeasure <: Measure
    space::Interval           # always [0,1]
    tag::Int                  # program/component index — immutable, never updated
    beta::BetaMeasure         # P(enemy | predicate matches) — updated by conditioning
end
```

The tag is immutable: conditioning never changes which program a component represents, only its belief parameters. Dispatches:

- `_predictive_ll(m::TaggedBetaMeasure, k, obs)` — passes the tagged measure to the kernel's `log_density`, which extracts `m.tag` to dispatch to the right compiled predicate. If the predicate fires, returns Beta-Bernoulli log-likelihood. If not, returns `log(0.5)` — the program is implicitly predicting the base rate ("I don't know about this entity"), and that prediction is scored against the observation. This ensures non-firing programs lose weight relative to firing-and-correct programs (informed and right > uninformed > informed and wrong). The predicate result is cached in a side-channel `Dict{Int, Bool}` captured by the kernel closure, avoiding double evaluation during the subsequent `condition` call.
- `condition(m::TaggedBetaMeasure, k, obs)` — reads the predicate cache. If the predicate didn't fire, returns `m` unchanged (no update — the program made no prediction about this entity). If the predicate fired, performs a standard Beta-Bernoulli conjugate update on `m.beta`, wraps the result in a new `TaggedBetaMeasure` preserving the tag.
- `expect`, `draw`, `mean`, `variance`, `log_density_at` — delegate to `m.beta`.

The predicate cache is a `Dict{Int, Bool}` captured by the kernel closure at construction time. Since kernels are rebuilt each step, the cache is always fresh — stale entries from previous observations cannot persist.

**`_predictive_ll(MixtureMeasure, k, obs)` exact dispatch** (~5 lines). The existing fallback uses Monte Carlo (200 samples). Add an exact dispatch that computes the weighted sum of component likelihoods:

```julia
function _predictive_ll(m::MixtureMeasure, k::Kernel, obs)
    w = weights(m)
    total = sum(w[i] * exp(_predictive_ll(m.components[i], k, obs)) for i in eachindex(w))
    log(max(total, 1e-300))
end
```

This delegates to each component's specialised `_predictive_ll` — exact for TaggedBetaMeasure components.

**Per-component kernel dispatch.** The kernel's `log_density` function is a closure that captures `compiled_kernels`, `grammar_sensor_vectors` (per-grammar projections of the current entity's state), `temporal_window`, and the predicate cache. When called with a `TaggedBetaMeasure`:

```julia
(tagged_beta, obs) -> begin
    ck = compiled_kernels[tagged_beta.tag]
    sv = grammar_sensor_vectors[ck.grammar_id]
    fires = ck.evaluate(sv, temporal_window)
    predicate_cache[tagged_beta.tag] = fires  # cache for condition dispatch
    
    if fires
        p = mean(tagged_beta.beta)
        obs == 1.0 ? log(p) : log(1 - p)
    else
        log(0.5)  # base rate — program predicts 50/50, scored against observation
    end
end
```

The existing `condition(MixtureMeasure, kernel, obs)` dispatch works unmodified. It iterates over components, calls `_predictive_ll` on each `TaggedBetaMeasure`, gets back per-component log-likelihoods, reweights, then calls `condition` on each component to update beliefs. The `TaggedBetaMeasure` dispatches handle all per-component logic. No bypass, no manual posterior construction.

### 5.3 `fold-init` in eval.jl

As specified in the original plan: add `fold-init` special form (~4 lines) after existing `fold` at line ~165. Takes `(f, init, list)` and folds with explicit initial accumulator. Needed for sequential conditioning over observation lists within the DSL.

### 5.4 Grid world simulation (Julia)

**File:** `examples/grid_world.jl` (~250 lines)

Types: `Pos`, `Entity`, `EntityState`, `WorldConfig`, `WorldState`, `TerrainType`, `EntityKind`.

World rules: 5 configuration functions, one per rule set. A `create_world(rule_name)` constructor and a `set_rule!(state, rule_name)` mutator for mid-run regime changes.

Simulation: `world_step!(state, action)` advances one timestep — entity movement, collision, interaction resolution. Returns energy delta if interaction occurred.

Entity movement: stationary, patrol (follow fixed path), chase (step toward agent), wander (random walk). Movement rules are per-entity and hidden from the agent.

### 5.5 Sensor projection layer (Julia)

**File:** `examples/sensor_projection.jl` (~80 lines)

Types: `SensorChannel`, `SensorConfig`.

`project(entity_state, sensor_config) → Vector{Float64}` — produces the noisy sensor vector an agent actually receives.

`available_source_indices()` — returns the list of environment state dimensions that sensor channels can reference. For the grid world: 0=r, 1=g, 2=b, 3=x, 4=y, 5=speed, 6=wall_dist, 7=agent_energy.

### 5.6 Grammar and program enumeration (Julia)

**File:** `examples/grammar_programs.jl` (~300 lines)

This is the most substantial new Julia code. It handles:

**Grammar representation.**

```julia
struct ProductionRule
    name::Symbol
    body::ProgramExpr    # AST node
end

struct Grammar
    sensor_config::SensorConfig
    rules::Vector{ProductionRule}
    complexity::Float64   # precomputed |G|
end
```

**Program enumeration.** Given a grammar, enumerate programs up to a maximum derivation depth, pruning any program whose prior weight falls below a floor. The enumeration is bounded by two constraints: derivation depth (structural limit) and minimum log-prior (weight limit). The depth bound limits the combinatorial space; the weight floor ensures that programs the prior already judges as implausible are never materialised.

```julia
function enumerate_programs(
    grammar::Grammar, max_depth::Int;
    min_log_prior::Float64 = -20.0    # skip programs with log2(prior) below this
)::Vector{Program}
    # Bottom-up enumeration: depth-1 programs are single predicates/nonterminals,
    # depth-2 programs compose depth-1 with connectives, etc.
    # At each depth, skip any candidate whose -complexity < min_log_prior.
    # Returns programs with precomputed complexity scores.
end
```

The weight floor is not an arbitrary cap — it is Occam's Razor applied at enumeration time. A program with complexity 20 has prior weight `2^(-20) ≈ 1e-6`; it would need overwhelming likelihood to overcome this penalty, and in a 200-step game it almost certainly won't. Enumerating it wastes memory without affecting the posterior.

Critically, the weight floor creates the correct incentive for grammar evolution. Under the empty grammar, a depth-3 conjunction of raw sensor predicates might have complexity 12 and prior weight `2^(-12) ≈ 2.4e-4` — enumerated. Under a grammar with a `RED` nonterminal, the same logical predicate has complexity 5 and prior weight `2^(-5) ≈ 0.03` — much higher. More complex programs *become reachable* as the grammar improves, because abstractions lower their complexity below the floor. The enumeration space grows not by increasing depth but by adding nonterminals. This is the mechanism by which grammar evolution expands the agent's effective hypothesis space.

**Practical scale.** With 80 terminal predicates (2 comparisons × 8 channels × 5 thresholds), depth 2 produces ~20K programs per grammar before the weight floor. The floor prunes this to a few thousand. Depth 3 under the empty grammar would produce millions, but the floor prunes aggressively — only short compositions survive. Depth 3 under a grammar with good nonterminals produces many more viable programs, because nonterminal references cost 1 regardless of their expansion's complexity. Default: `max_depth=2, min_log_prior=-20.0`.

**Grammar enumeration.** The space of grammars is large but structured. For the initial implementation, use a curated seed population plus perturbation:

Seed grammars:
  - Empty grammar (no rules, minimal 2-channel sensor config). Highest prior, worst expressiveness.
  - Full-sensor grammar (all 8 channels, no rules). Moderate prior, full raw expressiveness.
  - One nonterminal grammars: for each "natural" abstraction (colour predicates, movement predicates, spatial predicates), a grammar that defines it. These are the hypotheses the agent evaluates.
  - Multi-nonterminal grammars: pairwise combinations of the single-nonterminal grammars.

Perturbation operators (for generating new grammar candidates):
  - Add/remove a sensor channel.
  - Add a production rule (nonterminal) that names a high-posterior subprogram.
  - Remove a production rule.
  - Change a threshold in a production rule's body.

The grammar space is explored incrementally: after each conditioning step, the host can propose new grammars by perturbing high-posterior ones. These are added to the PopulationMeasure as new mixture components with prior weight `2^(-|G|)`. Low-posterior grammars are pruned to keep the mixture tractable.

**Kernel compilation.** Each program, under its grammar's sensor config, implies a prediction function: given a sensor vector, does the predicate match, and what is the predicted entity type?

The type system enforces precompilation. `CompiledKernel` contains only a closure and metadata — no AST field. Since the AST is structurally absent at evaluation time, it cannot be interpreted.

```julia
struct CompiledKernel
    # NO AST FIELD. The predicate is compiled into the closure.
    # This is a type-level constraint: if the AST isn't here, it can't be interpreted.
    evaluate::Function             # sensor_vector, temporal_state → Bool
    complexity::Int                # precomputed derivation length
    grammar_id::Int
    program_id::Int
end

function compile_kernel(program::Program, grammar::Grammar)::CompiledKernel
    # Walks the AST once at compile time, producing a closure that
    # captures threshold values, channel indices, and nonterminal
    # expansions as closed-over constants. The returned CompiledKernel
    # has no reference to the original AST.
    #
    # The closure signature is:
    #   (sensor_vector::Vector{Float64}, temporal_state::TemporalWindow) → Bool
    #
    # Example: for program AND(GT(0, 0.9), GT(5, 0.5)) the closure is:
    #   (sv, _) -> sv[1] > 0.9 && sv[6] > 0.5
    # No tree walk at evaluation time.
end
```

The `Program` struct retains its AST for complexity analysis, subprogram extraction (Phase 4), and display. But `CompiledKernel` — the object used at conditioning time — has no access to it. The separation is enforced by the type system.

```julia
struct Program
    predicate::ProgramExpr         # AST retained for analysis and display
    complexity::Int                # precomputed derivation length
    grammar_id::Int
end

# Program → CompiledKernel is a one-way transformation.
# CompiledKernel → Program is not possible (AST is gone).
```

The compiled kernel is what the host uses to build the per-component `log_density` function. The DSL conditions on a kernel whose inner dispatch calls `compiled_kernel.evaluate` — a closure, not an interpreter.

### 5.7 DSL agent

**File:** `examples/program_agent.bdsl` (~80 lines)

The DSL agent is structurally simple because all the complexity lives in the belief structure the host constructs.

```scheme
;; Spaces
(define sensor-space (finite-space "sensor-vector"))
(define outcome-space (finite-space "enemy" "food" "neutral"))
(define action-space (finite-space "move-n" "move-s" "move-e" "move-w"
                                     "interact" "observe" "wait"))

;; Core inference: condition belief on observation
(define (update-belief belief kernel observation)
  (condition belief kernel observation))

;; Sequential conditioning over a list of observations
(define (update-belief-seq belief kernel observations)
  (fold-init
    (lambda (b obs) (condition b kernel obs))
    belief
    observations))

;; Prediction: expected entity type given sensor reading
(define (predict-type belief sensor-reading)
  (expect belief (lambda (program) (program sensor-reading))))

;; EU of interacting with an entity
(define (eu-interact belief sensor-reading energy-pref)
  (expect belief
    (lambda (program)
      (expect (program sensor-reading) energy-pref))))

;; VOI of observing (watching without interacting)
(define (observe-voi belief observe-kernel possible-obs pref)
  (voi belief observe-kernel possible-obs pref))

;; Choose best action
(define (choose-action belief action-eus)
  (optimise action-space (lambda (a) (action-eus a))))
```

The agent doesn't know about grammars, sensors, or programs. It receives a belief (the flat MixtureMeasure), kernels (compiled by the host, closing over temporal state), and observations (energy deltas). It does inference. The host manages everything else.

### 5.8 Host driver

**File:** `examples/host_program_agent.jl` (~400 lines)

This is the main orchestration loop. Pattern follows `examples/host_credence_agent.jl`.

**Key architectural decisions:**

The belief is a flat MixtureMeasure. Each component is a BetaMeasure indexed by `(grammar_id, program_id)` in a host-maintained metadata table. Grammar-level weights are recovered by aggregation whenever needed (e.g., for perturbation decisions or metrics).

Kernels are rebuilt every step, closing over the current temporal state — a sliding window of past sensor readings per entity. This means temporal operators (`CHANGED`, `PERSISTS`, `SINCE`) evaluate against the window captured at kernel construction time. The DSL never touches temporal state; it receives a kernel that already has the history baked in. This is consistent with the existing pattern in `host_credence_agent.jl`.

The single observation kernel encodes program identity: its `log_density(component_j, obs)` looks up the precompiled closure for component `j` from the `compiled_kernels` array, calls `compiled_kernel.evaluate(sensor_vector, temporal_state)`, and computes the Beta-Bernoulli likelihood from the result. One kernel handles all components because the program dispatch happens inside the kernel via closure lookup — no AST interpretation.

Program ASTs are stored separately in `all_programs` for subtree analysis during grammar perturbation (Phase 4). The two representations — `CompiledKernel` for fast conditioning, `Program` for structural analysis — serve different purposes and are never confused. `CompiledKernel` has no AST field; `Program` has no closure. The separation is enforced by the type system.

```julia
function run_agent(;
    world_rules::Vector{Symbol} = [:color_typed],
    max_steps::Int = 200,
    regime_change_steps::Vector{Int} = Int[],  # when to silently change the rule
    grammar_pool_size::Int = 50,
    program_max_depth::Int = 6,
    grammar_perturbation_interval::Int = 20    # how often to propose new grammars
)
    # 1. INITIALISE
    world = create_world(world_rules[1])
    grammar_pool = generate_seed_grammars()
    
    # Enumerate all (grammar, program) pairs and build flat mixture
    components = BetaMeasure[]     # one Beta(1,1) per (grammar, program)
    prior_weights = Float64[]
    metadata = Tuple{Int,Int}[]    # (grammar_id, program_id) per component
    compiled_kernels = CompiledKernel[]  # precompiled closures for conditioning
    all_programs = Program[]       # ASTs retained for analysis (Phase 4 subtree extraction)
    
    for (gi, g) in enumerate(grammar_pool)
        programs = enumerate_programs(g, program_max_depth)
        for (pi, p) in enumerate(programs)
            push!(components, BetaMeasure(1.0, 1.0))
            push!(prior_weights, 2.0^(-g.complexity) * 2.0^(-p.complexity))
            push!(metadata, (gi, pi))
            push!(compiled_kernels, compile_kernel(p, g))  # AST compiled away here
            push!(all_programs, p)                          # AST retained here
        end
    end
    
    belief = build_mixture_measure(components, normalize(prior_weights))
    
    # Temporal state: sliding window of past sensor readings per entity
    temporal_window = TemporalWindow(max_history=10)
    
    # Load DSL agent
    dsl = load_dsl("examples/program_agent.bdsl")
    
    # 2. MAIN LOOP
    regime_idx = 1
    metrics = MetricsTracker()
    
    for step in 1:max_steps
        # Regime change (silent)
        if step in regime_change_steps
            regime_idx += 1
            set_rule!(world, world_rules[regime_idx])
        end
        
        # Observe
        entity_states = get_visible_entities(world)
        update!(temporal_window, entity_states)
        
        # Build observation kernel from precompiled closures
        # Closes over temporal state; dispatches per-component via compiled_kernels
        kernel = build_observation_kernel(
            compiled_kernels, metadata, temporal_window, entity_states)
        
        # Condition belief on interaction feedback (if any)
        if has_feedback(world)
            feedback = get_feedback(world)  # energy_delta
            belief = dsl.call("update-belief", belief, kernel, feedback)
        end
        
        # Decide: compute EU for each action
        action_eus = compute_action_eus(dsl, belief, kernel, world)
        chosen_action = dsl.call("choose-action", belief, action_eus)
        
        # Act
        energy_delta = world_step!(world, chosen_action)
        
        # Track metrics (aggregate grammar-level weights from metadata)
        grammar_weights = aggregate_grammar_weights(belief, metadata)
        record!(metrics, step, belief, grammar_weights, energy_delta, chosen_action)
        
        # Periodically: propose new grammars by perturbing high-posterior ones
        if step % grammar_perturbation_interval == 0
            top_grammars = select_top_grammars(grammar_weights, grammar_pool, k=5)
            
            # Analyse posterior subtrees FIRST — required by type system
            # for nonterminal proposal. Cannot be skipped.
            freq_table = analyse_posterior_subtrees(
                all_programs, weights(belief),
                min_frequency=0.1, min_complexity=2)
            
            # perturb_grammar requires freq_table — enforced by method signature
            new_grammars = [perturb_grammar(g, freq_table) for g in top_grammars]
            
            for g in new_grammars
                gi = length(grammar_pool) + 1
                push!(grammar_pool, g)
                programs = enumerate_programs(g, program_max_depth)
                for (pi, p) in enumerate(programs)
                    push!(components, BetaMeasure(1.0, 1.0))
                    push!(prior_weights, 2.0^(-g.complexity) * 2.0^(-p.complexity))
                    push!(metadata, (gi, pi))
                end
            end
            
            # Prune low-weight grammars (aggregate weight < 1e-10)
            prune_grammars!(belief, grammar_pool, metadata, components, min_weight=1e-10)
            
            # Rebuild flat mixture with current weights
            belief = rebuild_mixture(components, belief, metadata)
        end
    end
    
    return metrics
end
```

### 5.9 Grammar perturbation (the "evolution" that isn't separate)

The perturbation step is where grammar evolution happens, but it's not a separate mechanism — it's proposal generation for the grammar-level inference. The host proposes new grammars; `condition` evaluates them against data.

**Type-level enforcement of principled subprogram extraction.** The `propose_nonterminal` function requires a `SubprogramFrequencyTable` as input — a struct that can only be constructed by `analyse_posterior_subtrees`. This makes it structurally impossible to propose nonterminals without first having analysed the posterior. Random mutation of production rule bodies is not merely forbidden; the type signatures make it unrepresentable.

```julia
struct SubprogramFrequencyTable
    # Constructed ONLY by analyse_posterior_subtrees.
    # No public constructor. Cannot be faked.
    subtrees::Vector{ProgramExpr}           # candidate subtrees
    weighted_frequency::Vector{Float64}     # posterior-weighted occurrence count
    source_programs::Vector{Vector{Int}}    # which programs contained each subtree
end

function analyse_posterior_subtrees(
    programs::Vector{Program},
    weights::Vector{Float64};
    min_frequency::Float64 = 0.1,
    min_complexity::Int = 2          # don't promote trivially small subtrees
)::SubprogramFrequencyTable
    # Walk each program's AST, extract all subtrees of depth ≥ 2.
    # Weight each subtree occurrence by the program's posterior weight.
    # Aggregate across programs. Filter by min_frequency and min_complexity.
    # Return the frequency table — the ONLY input that propose_nonterminal accepts.
end

function propose_nonterminal(table::SubprogramFrequencyTable)::Union{ProductionRule, Nothing}
    # Select the subtree with highest weighted frequency.
    # If its compression payoff (frequency × complexity reduction) exceeds
    # the cost of adding a grammar rule (1 + subtree complexity), propose it.
    # Otherwise return nothing — no nonterminal is worth adding.
end
```

Perturbation operators:

```julia
function perturb_grammar(g::Grammar, freq_table::SubprogramFrequencyTable)::Grammar
    # Note: freq_table is a REQUIRED argument, enforcing the dependency.
    op = rand([:add_channel, :remove_channel, :add_rule, :remove_rule, :modify_threshold])
    
    if op == :add_channel
        # Add a sensor channel for a source index not yet covered
        available = setdiff(all_source_indices(), [c.source_index for c in g.sensor_config])
        isempty(available) && return g
        new_channel = SensorChannel(rand(available), :identity, 0.1, 1.0)
        return Grammar([g.sensor_config; new_channel], g.rules)
        
    elseif op == :add_rule
        # Promote a frequent subprogram to a nonterminal.
        # MUST use the frequency table — cannot bypass analysis.
        proposed = propose_nonterminal(freq_table)
        isnothing(proposed) && return g
        return Grammar(g.sensor_config, [g.rules; proposed])
        
    # ... other operators
    end
end
```

The `analyse_posterior_subtrees` → `propose_nonterminal` pipeline is the DreamCoder compression step. The type constraint ensures it cannot be skipped: `perturb_grammar` demands a `SubprogramFrequencyTable`, and there is no way to construct one without doing the posterior analysis. The proposed grammar enters the mixture at its prior weight and must earn posterior mass through predictive performance.

### 5.10 Metrics and verification

**File:** `examples/metrics.jl` (~100 lines)

Track per-step:
  - Top-5 grammar posterior weights (does the right grammar rise?)
  - Top-5 program posterior weights within the leading grammar
  - Prediction accuracy (does the agent correctly classify entities?)
  - Cumulative energy (is the agent accumulating reward?)
  - VOI of observe action (does it decrease as beliefs sharpen?)
  - Surprise (negative log-likelihood of observations under the posterior)
  - `time_to_convergence(start_step, end_step; accuracy_threshold, window)` — steps from `start_step` until rolling-window accuracy hits threshold. Pure measurement.

**File:** `test/test_program_agent.jl` (~250 lines)

### Testing philosophy

Tests fall into three categories with different epistemic standards.

**Mechanism tests** verify that the mathematical machinery does what the theory says. These are deterministic, involve no agent or simulation, and are falsifiable by construction. A mechanism test failure means the code is wrong.

**Emergent-behaviour tests** verify that the whole system produces expected qualitative patterns. These are statistical and must be framed carefully to avoid attributing emergent outcomes to specific mechanisms when confounds exist. An emergent-behaviour test failure might mean the code is wrong, or it might mean the test's causal attribution is wrong. Emergent tests should assert *directional* properties (X increases, Y is higher than Z) rather than threshold-dependent properties (X > 0.7).

**Enforcement tests** verify that type-level and performance constraints hold. These are structural — they test the shape of the code, not its behaviour. An enforcement test failure means someone bypassed a design invariant.

### Mechanism tests

  - **Single regime, colour-typed world.** After N interactions, the posterior concentrates on programs that reference colour channels. Grammars without colour sensors lose mass. (Mechanism: conditioning reweights by likelihood; programs whose predicates match the true rule predict better.)

  - **Single regime, motion-typed world.** Posterior concentrates on speed-channel programs. Colour-sensor grammars are penalised for unnecessary sensor cost. (Mechanism: Occam's Razor — simpler grammars that predict equally well have higher prior weight.)

  - **Single regime, mixed-rule world.** Programs with conjunction (colour AND movement) outperform single-feature programs. (Mechanism: single-feature programs misclassify entities where the conjunction matters.)

  - **Occam's Razor.** In the all-food world, the simplest possible program (empty predicate → food) dominates. Rich grammars are penalised for unused complexity. (Mechanism: complexity prior penalises unnecessary structure; the simplest program that predicts the data dominates.)

  - **Temporal programs represent regime changes.** Hand-construct two programs: `PREDICT(RED)` (stable) and a temporal program that switches from colour to movement prediction. Feed both an observation sequence where colour predicts danger for 50 steps, then movement predicts danger. Compute cumulative log-likelihood of each program on the full sequence. Assert the temporal program has higher total likelihood after the switch. (Mechanism: temporal operators allow programs to represent non-stationary structure. No agent, no simulation — pure likelihood computation on hand-crafted data.)

  - **Compression pipeline produces real compression.** Run agent on colour-typed world for N steps. Extract posterior. Call `analyse_posterior_subtrees` and `propose_nonterminal`. Verify the proposed nonterminal's body references colour channels. Enumerate programs under the new grammar. Verify that the best colour-typed program has strictly lower complexity than its expansion under the old grammar. (Mechanism: the DreamCoder compression step extracts useful abstractions. No convergence-speed measurement — just structural verification of the compression.)

  - **Per-grammar sensor projection.** Two grammars with different sensor configs produce different sensor vectors for the same entity. Programs from each grammar evaluate against their own grammar's projection, not a global one. (Mechanism: different grammars see different things.)

  - **Complexity scoring reflects nonterminals.** Same logical predicate has different complexity under different grammars: `AND(GT(0,0.9), GT(5,0.5))` costs 3 under the empty grammar, 2 under a grammar with `RED := GT(0,0.9)`. (Mechanism: nonterminal references cost 1 regardless of expansion complexity. This is what makes grammar evolution pay off.)

### Emergent-behaviour tests

These test qualitative patterns of the whole system. Assertions are directional, not threshold-dependent. Confounds are controlled where possible.

  - **Regime change: posterior shifts toward correct feature set.** Run agent across colour→motion change at step 50. Assert: mean posterior weight of programs referencing the speed channel is higher in steps 60–80 than in steps 30–50. (This tests the right thing — the posterior is moving toward the correct features — without assuming surprise spikes, which are an indirect indicator that may or may not appear depending on how confident the agent was pre-change.)

  - **Regime change: re-learning occurs.** Additionally assert: mean surprise in steps 70–80 is lower than mean surprise in steps 51–60. (This tests that the agent recovers from the disruption, not just that it's disrupted. The comparison is within the post-change period, avoiding confounds from pre-change confidence levels.)

  - **Meta-learning: controlled comparison.** Run two agents on the same 3-regime sequence (colour→motion→colour, changes at steps 75 and 150): one with grammar perturbation enabled, one without. Both start with identical seed grammars and face identical observation sequences (fix the random seed). Assert: the perturbation agent's accuracy in regime 3 (steps 150–225) exceeds the no-perturbation agent's accuracy in regime 3. This controls for all confounds — both agents have the same initial pool, same observations, same pruning. The only difference is whether the grammar evolves. If the perturbation agent converges faster, it's because its grammar improved, not because of pool size or residual weights. (If this test fails, investigate perturbation quality first — `find_frequent_subprogram` may not be producing useful nonterminals — before questioning the meta-learning mechanism.)

  - **Meta-learning: grammar pool evolution is real.** After the perturbation agent's multi-regime run, verify that perturbed grammars with nonterminals exist in the pool and that at least one has non-trivial posterior weight (above median). This is observational — it reports that the perturbation pipeline produced grammars the inference found useful, without attributing convergence speed to them.

  - **Baseline comparison.** The program-space agent accumulates more energy than: (a) random action, (b) greedy (interact with everything). It approaches (c) oracle (knows the true rule). (Emergent, not mechanism: the full system outperforms naive strategies, but the specific margin depends on many factors.)

### Enforcement tests

  - **Kernel precompilation speed.** Time 10,000 calls to `compiled_kernel.evaluate(sensor_vector, temporal_state)` for a depth-4 program. Assert < 1ms total. Catches AST interpretation at conditioning time (10-50× slower due to tree walks, allocation, and dispatch overhead). The test checks the observable consequence, not the implementation.

  - **CompiledKernel has no AST field.** Assert that `fieldnames(CompiledKernel)` does not contain any field of type `ProgramExpr` or any `AbstractArray{ProgramExpr}`. Structural test: catches AST fields added for "debugging convenience."

  - **SubprogramFrequencyTable required for nonterminal proposal.** Verify that `propose_nonterminal` has no method accepting anything other than `SubprogramFrequencyTable`. Verify that `perturb_grammar` requires a `SubprogramFrequencyTable` argument. Method-signature tests: catches convenience overloads that bypass posterior analysis.

  - **Proposed nonterminals are actual posterior subtrees.** After running the agent for N steps in a colour-typed world, call `analyse_posterior_subtrees` and `propose_nonterminal`. Verify that the proposed nonterminal's body is a subtree that actually appears in at least 2 of the top-10 posterior programs. Catches random AST generation masquerading as subprogram extraction.

  - **AgentState stays synced after prune/truncate.** After `sync_prune!` and `sync_truncate!`, verify `length(state.belief.components) == length(state.metadata) == length(state.compiled_kernels) == length(state.all_programs)` and `state.belief.components[i].tag == i` for all `i`. Catches index desynchronisation.

---

## 6. Implementation Sequence

The critical insight for ordering: grammar enumeration and kernel compilation (Phase 1) define the API that the host driver, perturbation operators, and test suite all depend on. Getting `enumerate_programs` and `compile_kernel` right — with solid tests — before touching anything else means the integration phase is about wiring, not about discovering the API is wrong.

### Phase 0: Verify conditioning + add primitives (1 day)

Confirm that `condition(MixtureMeasure, kernel, obs)` correctly reweights a flat mixture of BetaMeasures with a kernel that dispatches per-component. Write a minimal test: a MixtureMeasure of 6 BetaMeasures (simulating 2 grammars × 3 programs), a kernel whose `log_density` varies by component index, and an observation. Verify that weights update correctly and that grammar-level aggregation (summing by grammar_id) gives the expected grammar posteriors.

Add `_predictive_ll(MixtureMeasure, k, obs)` exact dispatch to ontology.jl (~5 lines).

Add `fold-init` special form to eval.jl (~4 lines).

If anything fails: stop, diagnose, fix before proceeding.

### Phase 1: Grammar and program enumeration (3-4 days)

**This is where the project lives or dies.** Implement `grammar_programs.jl`: terminal alphabet, grammar representation, program enumeration, complexity scoring, and kernel compilation. This defines the API everything else depends on.

Test independently: given a hand-constructed grammar and sensor vector, verify that `enumerate_programs` produces the expected set, that complexity scores are correct, and that `compile_kernel` produces functions that correctly evaluate predicates and return likelihoods.

### Phase 2: Grid world + sensors (2-3 days, parallelisable with Phase 1)

Implement `grid_world.jl` and `sensor_projection.jl`. Verify that different SensorConfigs produce different views of the same world. Test entity movement, interaction, and sensor projection independently.

### Phase 3: DSL agent + host driver (2-3 days)

Implement `program_agent.bdsl` and `host_program_agent.jl`. Wire everything together: grammar pool → program enumeration → kernel compilation → flat MixtureMeasure with metadata → DSL inference → action selection → world step → repeat.

Run against a single world rule (colour-typed) and verify that the posterior concentrates on the correct programs and grammars.

### Phase 4: Grammar perturbation (2-3 days)

Implement the perturbation operators. Add the periodic perturbation step to the host driver. Verify that new grammars proposed from high-posterior programs are sensible, and that the grammar pool evolves toward good representations.

Test the DreamCoder-style subprogram promotion: after the agent has learned in a colour-typed world, verify that the perturbation step proposes a grammar with a `RED` nonterminal (or equivalent sensor-predicate bundle).

### Phase 5: Regime changes and temporal programs (2-3 days)

Add regime change support to the host driver. Extend the terminal alphabet with temporal operators (CHANGED, PERSISTS, SINCE). Extend program enumeration to include temporal programs. The host's temporal window (sliding history of sensor readings per entity) is already integrated via kernel closures from Phase 3.

Test: run colour-typed → motion-typed mid-game. Verify surprise spike, temporal program rise, and re-learning.

### Phase 6: Meta-learning validation (1-2 days)

Run multi-regime experiments. Measure time-to-convergence across regimes. Verify that it decreases — the hallmark of learning to learn. Analyse which grammars survive across regimes and why.

### Phase 7: Population extension (deferred)

Run N agents with different initial grammar pools in the same environment. Aggregate fitness. Condition the population-level grammar distribution. Reproduce with perturbation. Verify that the population converges on good grammars faster than any individual agent.

This phase requires no new DSL primitives — only a host-level loop managing multiple agents.

### Critical path

```
Phase 0 (verify conditioning) → Phase 1 (grammars — HARDEST) ─┐
                                 Phase 2 (grid world + sensors) ├→ Phase 3 (host driver) → Phase 4 (perturbation)
                                                                ┘    → Phase 5 (temporal) → Phase 6 (meta-learning)
```

---

## 7. Critical Files

| File | Role | Lines (est.) |
|------|------|-------------|
| `src/ontology.jl` | Add `_predictive_ll(MixtureMeasure)` exact dispatch | ~5 new |
| `src/eval.jl` | Add `fold-init` | ~4 new |
| `examples/grid_world.jl` | World simulation | ~250 new |
| `examples/sensor_projection.jl` | Sensor config and projection | ~80 new |
| `examples/grammar_programs.jl` | Grammar/program enumeration, kernel compilation | ~300 new |
| `examples/program_agent.bdsl` | DSL agent | ~80 new |
| `examples/host_program_agent.jl` | Host driver + grammar perturbation + metadata management | ~400 new |
| `examples/metrics.jl` | Tracking and visualisation | ~100 new |
| `test/test_nested_mixture.jl` | Phase 0: flat mixture conditioning verification | ~80 new |
| `test/test_program_agent.jl` | Full agent test suite | ~200 new |

**Reference files** (read, don't modify):
  - `src/stdlib.bdsl` — `optimise`, `value`, `voi` (reuse as-is)
  - `examples/host_credence_agent.jl` — host driver pattern
  - `examples/grid_agent.bdsl` — existing minimal grid agent
  - `src/host_helpers.jl` — `update_beta_state`, `FactorSelector` pattern

---

## 8. Open Questions

**Tractability of grammar enumeration.** The space of grammars is combinatorially large. The seed-plus-perturbation approach avoids full enumeration, but the quality of the grammar pool depends on the perturbation operators finding good candidates. If this proves insufficient, consider MCMC over grammar space (propose-accept/reject) instead of maintaining an explicit mixture.

**The compression-exploration tradeoff.** Promoting a subprogram to a nonterminal creates a new grammar in the pool — the original grammar remains. If the promoted nonterminal turns out to be unhelpful in a new regime, the new grammar is penalised by its higher complexity cost without compensating predictive benefit, and it loses posterior mass while the original (or another better-fitting grammar) rises. This self-corrects naturally, but grammar perturbation should still be conservative about promotion. Require strong posterior evidence before promoting.

**Grammar pruning and revival.** The pruning threshold permanently removes grammars from the pool. If a pruned grammar's structure would have been useful after a future regime change, it's gone. One mitigation: maintain a "graveyard" of pruned grammars and periodically resurrect the most recently pruned ones at low prior weight, giving them a second chance if the world has changed. Whether this is necessary depends on how often regime changes invalidate the current grammar pool — verify empirically.

**Dynamic depth.** At depth 2, programs are too short (complexity 1–3) for meaningful subtree sharing — the DreamCoder compression step rarely finds subtrees with `min_complexity ≥ 2`. Grammar evolution via nonterminal promotion becomes meaningful at depth 3+, where longer programs share more structure. But depth 3 under the empty grammar is intractable (~34K programs per grammar). The resolution is dynamic depth: the host periodically checks whether the top grammar's nonterminals would enable substantially more depth-3 programs to pass the `min_log_prior` floor, and if so, re-enumerates at depth 3 for that grammar. Grammar evolution *creates the conditions* for deeper search — the system starts shallow, learns nonterminals, then deepens when the grammar can support it. This is a natural extension of the prior-weighted enumeration design: nonterminals lower program complexity, which pushes more depth-3 programs above the floor, which expands the hypothesis space, which enables richer learning. Defer until the depth-2 system is stable; implement as a host-side parameter adjustment, not a DSL change.

**Perturbation interaction with P(enemy).** Grammar perturbation adds new programs with Beta(1,1) priors (mean 0.5). In a mature mixture where established programs have learned low P(enemy) (~0.07), fresh components shift the global expectation toward 0.5, reducing the agent's willingness to interact. This is transient — new components either fire and learn (converging away from 0.5) or don't fire and receive the base-rate penalty. But it can temporarily disrupt learned behaviour. Consider initialising new components' Betas from the posterior mean of existing components in the same grammar, rather than from the uninformative Beta(1,1). This preserves the prior-weight penalty (the program is still new and complexity-penalised) while avoiding the P(enemy) shock.

### Resolved

- ~~Kernel compilation complexity~~ — resolved by `CompiledKernel` type constraint and precompilation into closures at enumeration time. Per-component dispatch via `TaggedBetaMeasure.tag`.
- ~~Temporal operator implementation~~ — resolved by kernel closures capturing temporal window, rebuilt each step.
- ~~Precompile predicate evaluation~~ — hard requirement, enforced by `CompiledKernel` having no AST field.
- ~~Program enumeration tractability~~ — resolved by prior-weighted enumeration with `min_log_prior` floor. Depth bounds + weight floor replace arbitrary count caps.
- ~~Scale~~ — empirically validated. 12 seed grammars at depth 2 produce ~46K components, pruned to ~2K. Conditioning is O(2K) per step — fast. The `min_log_prior` floor prunes Grammar 12 entirely (complexity 20, all ~13,600 programs eliminated). Depth 3 is tractable only for grammars with good nonterminals.
- ~~Base-rate sentinel for non-firing predicates~~ — **resolved: use `log(0.5)`, not `0.0`.** Empirical finding: with `ll = 0.0` for non-firing predicates, established programs that stop being relevant after a regime change coast indefinitely — they never fire, never get penalised, never lose weight. Meanwhile, newly relevant programs that *do* fire take learning penalties from Beta(1,1) priors. This inverts the intended dynamic: the posterior becomes sticky rather than adaptive. The correct semantics: a non-firing program is implicitly predicting the base rate ("I don't know about this entity, so I predict 50/50"). That prediction should be scored against the observation. The ranking becomes: informed and right > uninformed > informed and wrong. With `0.0`, the ranking is uninformed ≥ informed and right, which is perverse. The predicate cache (`Dict{Int, Bool}`) already resolves the sentinel-collision concern in `condition` — it reads the cache directly rather than using the return value. **Change the non-firing return from `0.0` to `log(0.5)` in the kernel closure and in `_predictive_ll(TaggedBetaMeasure, ...)`.**
