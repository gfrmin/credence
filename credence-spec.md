# Credence: A Program-Space Bayesian Agent Architecture

## Research Spec

---

## 1. Thesis

An agent that maintains a complexity-weighted MixtureMeasure over programs, and conditions it on observations via Bayes' rule, will exhibit feature discovery, model selection, change-point detection, meta-learning, grammar evolution, and preference adaptation — all as emergent behaviour of a single mechanism. No dedicated subsystem is needed for any of these capabilities.

The mechanism is: `condition` over a flat MixtureMeasure of `TaggedBetaMeasure` components (grammars × programs), weighted by description length (Solomonoff prior / Bayesian Occam's Razor). The five invariant components are `condition`, `expect`, `optimise`, the program space with complexity prior, and the observation stream. Everything else is learned.

The architecture is domain-independent. The DSL core handles inference. A reusable program-space layer handles grammar evolution and hypothesis management. Each domain provides only its terminal alphabet (what features exist), its observation source (where data comes from), and its host driver (how the loop runs). This spec describes the shared architecture and two domains: a grid world (validated testbed) and an email agent (first real application).

---

## 2. The Unified Mechanism

### 2.1 The belief state

The agent's entire belief is a single flat MixtureMeasure:

```
belief : MixtureMeasure
  components: [ TaggedBetaMeasure(tag=j, beta=Beta_j) for j in 1..M ]
  weights:    [ w_j ∝ 2^(-|G_j|) × 2^(-|P_j|) × likelihood_j ]
```

Each component is a `TaggedBetaMeasure` — a BetaMeasure tracking P(positive_outcome | this program's predicate matches), tagged with the component's index. The tag is immutable: conditioning updates the Beta parameters but never changes which program a component represents. The tag allows a single kernel to dispatch per-component: it extracts the tag, looks up the corresponding compiled predicate, and computes the appropriate likelihood.

The mixture is flat by design. Grammar × program structure is maintained by the host in parallel arrays bundled with the belief in an `AgentState` struct:

```
AgentState:
  belief           : MixtureMeasure of TaggedBetaMeasures
  metadata         : Vector{Tuple{Int,Int}}    # (grammar_id, program_id) per component
  compiled_kernels : Vector{CompiledKernel}     # precompiled closures for conditioning
  all_programs     : Vector{Program}            # ASTs retained for subtree extraction
```

The four arrays are always the same length and aligned by index. `sync_prune!` and `sync_truncate!` maintain this invariant by pruning/truncating all arrays together and reindexing tags so that `belief.components[i].tag == i` for all `i`. Since kernels are rebuilt each step, stale tag references cannot persist.

Grammar-level inference is recovered by aggregation:

```
weight(Grammar_i) = Σ_j w_j  where metadata[j].grammar_id == i
```

### 2.2 Conditioning (the only learning rule)

When the agent observes an outcome:

The host constructs a single kernel whose `log_density(component_j, obs)` encodes program-specific logic: it looks up the program for component `j` via the tag, evaluates that program's predicate against the current feature vector (projected through the program's grammar's feature config), and computes the likelihood. One kernel, applied uniformly across the flat mixture, producing different likelihoods per component because the program identity is encoded in the hypothesis.

When a program's predicate fires and the outcome matches: high likelihood, weight increases. When it fires and the outcome doesn't match: low likelihood, weight decreases. When the predicate doesn't fire: `log(0.5)` — the program is implicitly predicting the base rate, and that prediction is scored. The ranking is: informed and right > uninformed > informed and wrong.

The predicate result is cached in a `Dict{Int, Bool}` captured by the kernel closure, so the subsequent `condition` dispatch can read it without re-evaluating.

The kernel is rebuilt each step, closing over the current temporal state (a sliding window of recent observations maintained by the host). The DSL never needs to know about temporal state — it just calls `condition` with a kernel that has the history baked in.

### 2.3 Why every capability emerges

**Feature discovery.** A grammar with a nonterminal that names a frequent feature pattern lets programs reference it at cost 1 instead of the full expansion cost. If the pattern is predictive, those programs are short and accurate, so the grammar accumulates posterior mass. The agent discovers useful abstractions by shifting mass toward grammars that define them.

**Model selection.** Programs within a grammar compete. Conditioning on data selects the one that predicts better. Standard Bayesian model comparison; no special mechanism.

**Change-point detection.** Programs with temporal operators compete with stable programs. If the world changes, stable programs predict poorly while temporal programs that encode regime-shift structure rise. The agent detects the change because programs encoding temporal structure become more credible.

**Meta-learning.** Across regime changes, grammars whose nonterminals capture recurrent abstractions maintain high marginal likelihood. The agent's effective prior at the start of each new regime is shaped by all previous regimes — it learns faster because its grammar has been refined by experience.

**Feature adaptation.** A grammar's feature config determines what the agent can perceive. Grammars with informative features enable short, predictive programs; grammars with irrelevant features waste complexity. The agent's perceptual vocabulary adapts to the domain through the same inference.

**Strategy.** `optimise` over the action space, with `voi` determining the value of information-gathering, gives the agent a decision policy. Explore/exploit emerges from VOI. No epsilon-greedy, no exploration bonus.

**Preference alignment.** When the agent's goal is to satisfy a user, the user's preferences are just another hypothesis space. Preference-programs predict user reactions. Conditioning on observed reactions (approval, correction, override) selects preference-programs that model this user well. VOI tells the agent when to ask rather than guess.

---

## 3. The Five-Layer Stack

Concepts emerge from raw observations through five layers, each arising from inference under a complexity prior at a different timescale. The layers are not designed — they are descriptions of what the inference does, viewed from outside.

**Layer 0 (given): Raw features and minimal logic.** The agent receives real-valued feature vectors. It can form threshold predicates and compose them with `AND`/`OR`/`NOT`/temporal operators. This is the terminal alphabet, provided by the domain.

**Layer 1 (emerges): Feature abstraction.** The agent discovers that certain feature predicates recur usefully. Grammar perturbation promotes them to named nonterminals, extending the grammar. Concepts like "urgent" or "red" are born — not as given categories but as compressions of feature patterns. The grammar grows.

**Layer 2 (emerges): Model structure.** The agent discovers that certain combinations of its learned features recur. Models like "urgent AND from-manager → user responds immediately" are programs in the current grammar, weighted by complexity. Conditioning selects the best models.

**Layer 3 (emerges): Meta-regularities.** Across regime changes (or across users, for a multi-user agent), the agent notices that certain grammar extensions keep being useful. These abstractions are retained even after regime changes. The grammar evolves a stable core alongside a mutable periphery.

**Layer 4 (emerges): Inductive bias.** The stable core of the grammar IS the agent's inductive bias — the set of concepts it has learned are generally useful. New situations start with a prior concentrated on programs built from proven abstractions. The agent learns faster because its grammar has been shaped by experience.

Each layer uses the same mechanism: `condition` over a complexity-weighted mixture, with grammar perturbation extracting recurring structure. The output of each layer becomes the vocabulary for the layer above.

---

## 4. Formal Definitions

### 4.1 Feature configuration (domain-provided)

Each domain defines a feature configuration — the set of observable channels available to the agent.

```
FeatureConfig = Vector{FeatureChannel}

FeatureChannel:
  source_index :: Int           # which dimension of the raw observation
  transform    :: Symbol        # :identity, :threshold, :delta, :windowed_mean
  noise_σ      :: Float64       # observation noise
  cost         :: Float64       # contributes to grammar complexity
```

For the grid world, channels are sensor readings (RGB, position, speed, wall distance). For the email agent, channels are extracted features (sender frequency, topic classification, urgency score, etc.).

The feature config is part of the grammar. Different grammars can have different configs, and the complexity prior penalises richer configs. The agent's perceptual vocabulary adapts to the domain through grammar-level inference.

### 4.2 Terminal alphabet (domain-provided)

Each domain provides a terminal alphabet: the set of atomic predicates programs can reference. Terminals are typically threshold comparisons over feature channels:

```
GT(i, t)  →  feature_i > t       cost: 1
LT(i, t)  →  feature_i < t       cost: 1
```

The threshold set and number of channels are domain-specific. Domains may also define domain-specific terminals beyond threshold comparisons (e.g., `SENDER_IS(category)` for email).

**Connectives** (shared across all domains):
```
AND(p, q)         cost: 1
OR(p, q)          cost: 1
NOT(p)            cost: 1
```

**Temporal operators** (shared):
```
PERSISTS(p, n)    cost: 1     # p true for last n steps
CHANGED(p)        cost: 1     # p flipped since last step
SINCE(p, q)       cost: 1     # p has been true since q was last true
```

### 4.3 Grammar

```
Grammar:
  feature_config   :: FeatureConfig
  rules            :: Vector{ProductionRule}

ProductionRule:
  name  :: Symbol
  body  :: ProgramExpr
```

Grammar complexity: `|G| = Σ(channel.cost for channel in feature_config) + Σ(1 + |body_i| for rule_i in rules)`

Grammar prior weight: `2^(-|G|)`.

### 4.4 Program and CompiledKernel

```
Program:
  predicate   :: ProgramExpr      # AST retained for analysis
  complexity  :: Int              # derivation length under this grammar
  grammar_id  :: Int

CompiledKernel:
  # NO AST FIELD. Predicate compiled into closure.
  evaluate    :: Function         # (feature_vector, temporal_state) → Bool
  complexity  :: Int
  grammar_id  :: Int
  program_id  :: Int
```

`Program → CompiledKernel` is a one-way transformation. The AST is compiled away. The type system enforces this: `CompiledKernel` has no `ProgramExpr` field.

Program prior weight within grammar: `2^(-complexity)`. Joint prior: `2^(-|G|) × 2^(-complexity)`.

### 4.5 TaggedBetaMeasure

```julia
struct TaggedBetaMeasure <: Measure
    space::Interval           # [0,1]
    tag::Int                  # component index — immutable
    beta::BetaMeasure         # P(positive_outcome | predicate matches)
end
```

Dispatches:
- `_predictive_ll`: passes tagged measure to kernel. Predicate fires → Beta-Bernoulli ll. Doesn't fire → `log(0.5)`. Caches predicate result.
- `condition`: reads cache. Fired → conjugate update, preserve tag. Not fired → return unchanged.
- `expect`, `draw`, `mean`, `variance`, `log_density_at`: delegate to inner beta.

### 4.6 Program enumeration

```julia
function enumerate_programs(
    grammar::Grammar, max_depth::Int;
    min_log_prior::Float64 = -20.0
)::Vector{Program}
```

Bottom-up enumeration, bounded by depth AND prior weight floor. Programs whose `-(grammar_complexity + program_complexity)` falls below `min_log_prior` are never materialised.

The weight floor creates the correct incentive for grammar evolution: nonterminals lower program complexity, pushing more programs above the floor. The enumeration space grows by adding abstractions, not by increasing depth.

Default: `max_depth=2, min_log_prior=-20.0`.

### 4.7 Grammar perturbation (type-enforced)

```julia
struct SubprogramFrequencyTable
    # Only constructed by analyse_posterior_subtrees. No public constructor.
    subtrees::Vector{ProgramExpr}
    weighted_frequency::Vector{Float64}
    source_programs::Vector{Vector{Int}}
end

function analyse_posterior_subtrees(programs, weights; min_frequency, min_complexity)::SubprogramFrequencyTable
function propose_nonterminal(table::SubprogramFrequencyTable)::Union{ProductionRule, Nothing}
function perturb_grammar(g::Grammar, freq_table::SubprogramFrequencyTable)::Grammar
```

`perturb_grammar` requires `SubprogramFrequencyTable` — type system enforces posterior analysis before nonterminal proposal.

---

## 5. Three-Tier Architecture

### 5.1 Tier 1: DSL core (`src/`)

The Credence DSL. Measures, kernels, `condition`, `expect`, `optimise`, `voi`, `TaggedBetaMeasure`, the evaluator, the stdlib. Domain-independent.

```
src/
├── ontology.jl          # measures, kernels, condition dispatches, TaggedBetaMeasure
├── eval.jl              # DSL evaluator (includes fold-init)
├── stdlib.bdsl          # DSL standard library (optimise, value, voi)
├── host_helpers.jl      # host-side utilities
└── Credence.jl          # module definition + exports
```

### 5.2 Tier 2: Program-space inference (`src/program_space/`)

Reusable machinery for inference over complexity-weighted program mixtures. Depends on Tier 1. Domain-independent.

```
src/program_space/
├── types.jl             # ProgramExpr AST, Grammar, ProductionRule, Program, CompiledKernel
├── enumeration.jl       # enumerate_programs, complexity scoring, prior-weighted floor
├── compilation.jl       # compile_kernel (AST → closure, one-way)
├── perturbation.jl      # SubprogramFrequencyTable, analyse/propose/perturb
└── agent_state.jl       # AgentState, sync_prune!, sync_truncate!
```

### 5.3 Tier 3: Domain applications (`domains/`)

Each domain provides its feature config, terminal alphabet, observation source, outcome definition, action space, and host driver.

```
domains/
├── grid_world/
│   ├── simulation.jl       # grid, entities, world rules, world_step!
│   ├── sensors.jl          # SensorConfig, project()
│   ├── terminals.jl        # grid-world terminal alphabet
│   ├── agent.bdsl          # DSL agent
│   ├── host.jl             # host driver loop
│   └── metrics.jl          # tracking
│
└── email_agent/
    ├── features.jl         # email feature extraction
    ├── terminals.jl        # email terminal alphabet
    ├── preferences.jl      # preference-program types, user reaction model
    ├── agent.bdsl          # DSL agent
    ├── host.jl             # host driver loop
    └── metrics.jl          # tracking
```

### 5.4 Domain interface

A domain module exports:

```julia
struct DomainConfig
    seed_grammars::Vector{Grammar}
    max_depth::Int
    min_log_prior::Float64
    action_space::Vector{Symbol}
end

project(obs, grammar::Grammar)::Vector{Float64}
classify_outcome(event)::Float64     # 1.0 = positive, 0.0 = negative
compute_action_eu(belief, action, domain_state)::Float64
```

---

## 6. Domain: Grid World (Validated Testbed)

### 6.1 Environment

5×5 grid. Terrain: ground, wall, mud. Entities have hidden types (FOOD, ENEMY, NEUTRAL) and observable properties (RGB, position, speed, wall distance). The agent sees noisy projections through its grammar's feature config.

### 6.2 World rules

| Rule | Classification logic | Key features |
|------|---------------------|--------------|
| all-food | Everything is food | None needed |
| colour-typed | Red=enemy, blue=food, green=neutral | Colour channels |
| motion-typed | Moving=enemy, stationary=food | Speed channel |
| territorial | Near-wall=enemy, centre=food | Wall-distance channel |
| mixed | Red AND moving=enemy, else food | Colour + speed |

The world may change its active rule mid-run without notification.

### 6.3 Empirical results (Phases 0-6 complete)

72 tests passing (42 core + 6 flat mixture + 24 program agent). Key findings:

- **Single regime:** posterior concentrates on correct programs/grammars within ~30 interactions.
- **Occam's Razor:** in the all-food world, simplest program dominates; rich grammars penalised.
- **Regime change:** posterior shifts toward correct feature set. Surprise increases then decreases.
- **Meta-learning (controlled comparison):** perturbation-enabled agent outperforms no-perturbation agent in regime 3 accuracy (0.700 vs 0.667).
- **Grammar evolution:** perturbation produces 23 grammars, 4 with positive posterior weight.
- **Scale:** 12 seed grammars at depth 2 → ~46K initial components → ~2K after truncation.

---

## 7. Domain: Email Agent (First Real Application)

### 7.1 Alignment as preference inference

The agent's utility function is fixed and known: maximise user satisfaction. What's unknown is the mapping from actions to user reactions — the user's preference model. This is a hypothesis in the program space, inferred from observed behaviour.

```
U_agent(action) = E_preference [ preference(action) ]
```

VOI tells the agent when to ask. When the preference posterior is broad (early, or after preference change), VOI for "ask the user" is high. As preferences sharpen, VOI drops and the agent acts autonomously. If preferences change (regime change), surprise rises and the agent becomes consultative again.

### 7.2 Feature configuration

| Index | Feature | Range | Extraction |
|-------|---------|-------|------------|
| 0 | sender_frequency | [0, 1] | normalised email count from sender |
| 1 | sender_is_manager | {0, 1} | binary |
| 2 | sender_is_direct_report | {0, 1} | binary |
| 3 | sender_is_external | {0, 1} | binary |
| 4 | urgency_score | [0, 1] | LLM-extracted or keyword-based |
| 5 | topic_finance | [0, 1] | topic classifier confidence |
| 6 | topic_scheduling | [0, 1] | topic classifier confidence |
| 7 | topic_marketing | [0, 1] | topic classifier confidence |
| 8 | requires_action | [0, 1] | LLM-extracted |
| 9 | email_length | [0, 1] | normalised word count |
| 10 | has_attachment | {0, 1} | binary |
| 11 | time_of_day | [0, 1] | normalised hour |
| 12 | thread_depth | [0, 1] | normalised reply count |

Feature extraction may use LLMs as noisy sensors. The `noise_σ` reflects extraction reliability.

### 7.3 Terminal alphabet

Threshold predicates over 13 channels, plus domain-specific terminals:

```
SENDER_IS(category)     cost: 1     # manager, direct_report, external, frequent, rare
TOPIC_IS(topic)         cost: 1     # finance, scheduling, marketing, personal, technical
IS_REPLY                cost: 1
HAS_ATTACHMENT          cost: 1
```

Domain-specific terminals are syntactic sugar for common threshold predicates — they reduce program complexity by 1, giving the agent a head start in forming useful abstractions.

### 7.4 Outcome definition

The agent recommends an action. The user accepts (1.0) or overrides (0.0). The observation: (email features, recommended action, user reaction).

### 7.5 Action space

```
ARCHIVE, FLAG_URGENT, SCHEDULE_LATER, DRAFT_RESPONSE,
DELEGATE, SUMMARISE, ASK_USER
```

`ASK_USER` is the information-gathering action. Its EU includes VOI.

### 7.6 Two program spaces

**World-model programs:** predict email properties from other properties. "Emails from this sender are usually about finance." Conditioned on the email stream.

**Preference-model programs:** predict user reactions from email properties + recommended action. "For urgent emails from the manager, the user wants FLAG_URGENT." Conditioned on user reactions.

Both live in the same flat MixtureMeasure. Both evolve their grammars.

### 7.7 Multi-user meta-learning

**Population level.** The grammar vocabulary. Evolved across all users from anonymised behavioural patterns. Abstractions like "urgency-sensitive," "delegation-heavy." Shared infrastructure.

**User level.** The preference-program posterior for this user. Built from the population grammar, conditioned on this user's reactions.

**Interaction level.** Each email, each recommendation, each reaction.

The population grammar evolves slowly across users. Each user's posterior updates quickly. New users benefit from the evolved grammar (faster convergence). This is the decreasing time-to-convergence validated in the grid world, applied to users.

---

## 8. Design Invariant Enforcement

**Type-level constraints (primary).** `CompiledKernel` has no AST field. `propose_nonterminal` requires `SubprogramFrequencyTable`. Shortcuts fail at compile time.

**Observable-consequence tests (secondary).** 10K kernel evaluations < 1ms. Proposed nonterminals appear in top-10 posterior programs. Compression payoff is measurable.

**CLAUDE.md forbidden patterns (tertiary).** Documents the why behind the type constraints and tests.

---

## 9. Testing Philosophy

**Mechanism tests:** deterministic, agent-free where possible. A failure means the code is wrong.

**Emergent-behaviour tests:** statistical, directional assertions, controlled confounds. A failure might mean the code is wrong or the test's causal attribution is wrong.

**Enforcement tests:** structural. A failure means a design invariant was bypassed.

---

## 10. Implementation Sequence

### Completed: Grid world (Phases 0-6)

72 tests passing. Core machinery validated.

### Next: Refactor to three-tier architecture

Extract Tier 2 from `examples/grammar_programs.jl` → `src/program_space/`. Move grid world → `domains/grid_world/`. Split tests. No logic changes.

### Next: Email agent

**Phase E1:** Feature extraction from email data.
**Phase E2:** Terminal alphabet and seed grammars.
**Phase E3:** Host driver and preference learning loop.
**Phase E4:** Multi-user meta-learning.

### Future: Dynamic depth

Start depth 2. Escalate to depth 3 when grammar nonterminals enable it.

### Future: LLM integration

LLMs as noisy sensors for feature extraction, and as proposal distributions for generating hypothesis candidates. Calibrate `noise_σ` from observed extraction reliability. Maintain Beta over LLM accuracy per feature type, conditioned on user corrections.

---

## 11. Open Questions

**Grammar enumeration tractability.** Email domain has 13 features (vs grid world's 8). Verify the prior floor handles the larger terminal set.

**Dynamic depth.** Nonterminal promotion needs depth 3+. Implement depth escalation when grammar evolution creates the conditions.

**LLM noise calibration.** Feature extraction reliability varies by feature type. Consider per-feature Beta over LLM accuracy.

**Perturbation Beta initialisation.** Fresh Beta(1,1) components disrupt mature mixtures. Consider initialising from posterior mean of existing components.

**Grammar pruning and revival.** Consider a graveyard with periodic resurrection.

**Privacy in multi-user grammar evolution.** Population grammar from structural patterns only, not content.

### Resolved

- ~~Kernel compilation~~ — `CompiledKernel` type constraint, precompiled closures.
- ~~Temporal operators~~ — kernel closures capture temporal window.
- ~~Precompile predicates~~ — enforced by `CompiledKernel` having no AST field.
- ~~Program enumeration~~ — prior-weighted floor (`min_log_prior`).
- ~~Scale~~ — validated at depth 2 (46K → 2K components).
- ~~Base-rate sentinel~~ — `log(0.5)` for non-firing predicates.
- ~~Flat vs nested mixture~~ — flat by design.
- ~~Per-component dispatch~~ — `TaggedBetaMeasure.tag`.
