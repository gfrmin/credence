# Credence Brain Protocol

JSON-RPC 2.0 over stdio. Brain reads newline-delimited JSON from stdin,
writes newline-delimited JSON to stdout, logs to stderr.

## Framing

Each message is a single JSON object terminated by `\n`.
No Content-Length headers (unlike LSP). One request per line.

```
-> {"jsonrpc":"2.0","id":1,"method":"create_state","params":{...}}\n
<- {"jsonrpc":"2.0","id":1,"result":{...}}\n
```

## Lifecycle

### initialize

Called once after spawning the brain process. Loads the Credence module
and optionally preloads DSL programs and Julia plugin files.

```json
{"method": "initialize", "params": {
  "dsl_files": {
    "router": "/path/to/router.bdsl",
    "agent": "/path/to/credence_agent.bdsl"
  },
  "plugins": ["/path/to/domain_kernels.jl"]
}}
```

Response:

```json
{"result": {"version": "0.1.0"}}
```

`dsl_files`: optional map of name -> path. Each .bdsl file is loaded into
a named DSL environment. The name becomes the `env_id` for `call_dsl`.

`plugins`: optional list of .jl files to `include()`. These can register
custom kernel constructors in the brain's kernel registry.

### shutdown

Graceful shutdown. Brain process exits after responding.

```json
{"method": "shutdown", "params": {}}
```

```json
{"result": "ok"}
```

## State Management

The brain holds stateful objects (measures, agent states, DSL environments)
in a registry keyed by opaque string IDs. The host never sees the internal
representation -- only the ID.

### create_state

Create a new measure or agent state. Returns its ID.

**Categorical measure:**

```json
{"method": "create_state", "params": {
  "type": "categorical",
  "space": {"type": "finite", "values": [0, 1, 2, 3]}
}}
```

**Beta measure:**

```json
{"method": "create_state", "params": {
  "type": "beta",
  "alpha": 1.0, "beta": 1.0
}}
```

**Product measure:**

```json
{"method": "create_state", "params": {
  "type": "product",
  "factors": [
    {"type": "beta", "alpha": 1.0, "beta": 1.0},
    {"type": "gamma", "alpha": 2.0, "beta": 0.5}
  ]
}}
```

**Mixture measure:**

```json
{"method": "create_state", "params": {
  "type": "mixture",
  "components": [
    {"type": "beta", "alpha": 1.0, "beta": 1.0},
    {"type": "beta", "alpha": 2.0, "beta": 3.0}
  ],
  "log_weights": [0.0, -0.5]
}}
```

**Program-space agent state:**

```json
{"method": "create_state", "params": {
  "type": "program_space",
  "grammars": [
    {
      "features": ["colour", "speed", "wall_dist"],
      "rules": {
        "S": ["(if P A A)"],
        "P": ["(> F T)", "(< F T)", "(and P P)"],
        "A": ["food", "enemy"],
        "F": ["colour", "speed", "wall_dist"],
        "T": ["0.5"]
      }
    }
  ],
  "max_depth": 3,
  "action_space": ["food", "enemy"]
}}
```

Response (all types):

```json
{"result": {"state_id": "s_1", "n_components": 150}}
```

`n_components` is present only for mixture and program_space types.

### destroy_state

Release a state object. The ID becomes invalid.

```json
{"method": "destroy_state", "params": {"state_id": "s_1"}}
```

### snapshot_state

Serialize state to base64-encoded binary. For persistence across sessions.

```json
{"method": "snapshot_state", "params": {"state_id": "s_1"}}
```

```json
{"result": {"data": "<base64>"}}
```

### restore_state

Deserialize a snapshot. Returns a new state ID.

```json
{"method": "restore_state", "params": {"data": "<base64>"}}
```

```json
{"result": {"state_id": "s_6"}}
```

### transfer_beliefs

Create a new state with different embodiment but transfer compatible
beliefs from a source state. For when the skin's constraints change
(new features, new action space, new tools).

```json
{"method": "transfer_beliefs", "params": {
  "source_id": "s_5",
  "target_config": {
    "type": "program_space",
    "grammars": [{"features": ["colour", "speed", "wall_dist", "agent_dist"], ...}],
    "max_depth": 3,
    "action_space": ["food", "enemy", "neutral"]
  }
}}
```

Beliefs for hypotheses that exist in both source and target carry their
posteriors; new hypotheses get the prior.

```json
{"result": {"state_id": "s_7", "n_transferred": 120, "n_new": 45}}
```

## Core Inference (Tier 1)

### condition

Bayesian inversion. Updates the measure in place given a kernel and observation.

```json
{"method": "condition", "params": {
  "state_id": "s_1",
  "kernel": {"type": "bernoulli"},
  "observation": 1.0
}}
```

```json
{"result": {"state_id": "s_1", "log_marginal": -0.693}}
```

### weights

Normalized probability weights.

```json
{"method": "weights", "params": {"state_id": "s_1"}}
```

```json
{"result": {"weights": [0.25, 0.25, 0.25, 0.25]}}
```

### mean

Mean of a measure.

```json
{"method": "mean", "params": {"state_id": "s_1"}}
```

```json
{"result": {"mean": 0.5}}
```

### expect

Integration against a measure.

```json
{"method": "expect", "params": {
  "state_id": "s_1",
  "function": {"type": "identity"}
}}
```

```json
{"result": {"value": 0.5}}
```

### optimise

EU-maximizing action.

```json
{"method": "optimise", "params": {
  "state_id": "s_1",
  "actions": {"type": "finite", "values": [0, 1, 2, 3]},
  "preference": {"type": "tabular_2d", "matrix": [[1,-1,-1,-1],[-1,1,-1,-1],[-1,-1,1,-1],[-1,-1,-1,1]]}
}}
```

```json
{"result": {"action": 2, "eu": 0.85}}
```

### value

Maximum expected utility (without returning the action).

```json
{"method": "value", "params": {
  "state_id": "s_1",
  "actions": {"type": "finite", "values": [0, 1, 2, 3]},
  "preference": {"type": "tabular_2d", "matrix": [[1,-1,-1,-1],[-1,1,-1,-1],[-1,-1,1,-1],[-1,-1,-1,1]]}
}}
```

```json
{"result": {"value": 0.85}}
```

### draw

Sample from the measure. The only source of randomness.

```json
{"method": "draw", "params": {"state_id": "s_1"}}
```

```json
{"result": {"value": 2}}
```

## Program-Space Operations (Tier 2)

### enumerate

Enumerate programs from a grammar, compile kernels, add to state.

```json
{"method": "enumerate", "params": {
  "state_id": "s_5",
  "grammar": {
    "features": ["colour", "speed", "wall_dist"],
    "rules": {
      "S": ["(if P A A)"],
      "P": ["(> F T)", "(< F T)", "(and P P)"],
      "A": ["food", "enemy"],
      "F": ["colour", "speed", "wall_dist"],
      "T": ["0.5"]
    }
  },
  "max_depth": 3,
  "action_space": ["food", "enemy"]
}}
```

```json
{"result": {"n_added": 42, "grammar_id": 3, "n_components": 192}}
```

### perturb_grammar

Analyse posterior subtrees and create a perturbed grammar.

```json
{"method": "perturb_grammar", "params": {
  "state_id": "s_5",
  "grammar_id": 1,
  "all_features": ["colour", "speed", "wall_dist", "agent_dist", "x", "y"]
}}
```

```json
{"result": {"new_grammar_id": 4, "n_new_rules": 3}}
```

### add_programs

Enumerate from an existing grammar at a new depth.

```json
{"method": "add_programs", "params": {
  "state_id": "s_5",
  "grammar_id": 1,
  "max_depth": 4
}}
```

```json
{"result": {"n_added": 87, "n_components": 279}}
```

### sync_prune

Remove negligible-weight components.

```json
{"method": "sync_prune", "params": {
  "state_id": "s_5",
  "threshold": -30.0
}}
```

```json
{"result": {"n_remaining": 250}}
```

### sync_truncate

Keep only top-weighted components.

```json
{"method": "sync_truncate", "params": {
  "state_id": "s_5",
  "max_components": 2000
}}
```

```json
{"result": {"n_remaining": 250}}
```

### top_grammars

Top-k grammars by aggregated posterior weight.

```json
{"method": "top_grammars", "params": {
  "state_id": "s_5",
  "k": 3
}}
```

```json
{"result": {"grammars": [
  {"grammar_id": 1, "weight": 0.45},
  {"grammar_id": 3, "weight": 0.32},
  {"grammar_id": 2, "weight": 0.23}
]}}
```

## Batch Operations

Convenience calls for latency. **Every batch operation is decomposable
into primitive calls.** The body can always orchestrate at the primitive
level instead.

### belief_summary

```json
{"method": "belief_summary", "params": {"state_id": "s_5"}}
```

```json
{"result": {
  "n_components": 250,
  "weights": [0.05, 0.03, ...],
  "means": [0.7, 0.4, ...],
  "grammar_weights": {"1": 0.45, "3": 0.32, "2": 0.23},
  "top_programs": [
    {"index": 0, "weight": 0.05, "grammar_id": 1, "program_id": 7,
     "mean": 0.7, "expr": "(if (> colour 0.5) food enemy)"}
  ]
}}
```

### condition_and_prune

The hot path for program-space agents. Combines condition + sync_prune +
sync_truncate in one round-trip.

```json
{"method": "condition_and_prune", "params": {
  "state_id": "s_5",
  "kernel": {"type": "program_observation", "features": {"colour": 0.8, "speed": 0.3}, "true_label": "food"},
  "observation": 1.0,
  "prune_threshold": -30.0,
  "max_components": 2000
}}
```

```json
{"result": {"n_remaining": 248, "log_marginal": -1.2}}
```

### eu_interact

EU of interacting for a program-space agent. Evaluates all compiled
kernels on features, returns weighted EU.

```json
{"method": "eu_interact", "params": {
  "state_id": "s_5",
  "features": {"colour": 0.8, "speed": 0.3},
  "rewards": {"food": 5.0, "enemy": -5.0}
}}
```

```json
{"result": {"eu": 3.2, "p_labels": {"food": 0.82, "enemy": 0.18}}}
```

## DSL Operations

### call_dsl

Call a named function from a loaded DSL environment.

```json
{"method": "call_dsl", "params": {
  "env_id": "router",
  "function": "make-router-state",
  "args": [3, 5]
}}
```

For DSL functions that return opaque Julia objects (measures, lists of
measures), the brain wraps them in state IDs automatically. For scalar
returns (numbers, booleans), the value is returned directly.

```json
{"result": {"state_id": "s_7"}}
```

Arguments can be:
- **Scalars**: numbers, booleans, strings
- **Lists**: JSON arrays (converted to Julia vectors)
- **State references**: `{"ref": "state_id"}` (resolved to the Julia object)

Example with state references:

```json
{"method": "call_dsl", "params": {
  "env_id": "router",
  "function": "router-decide",
  "args": [{"ref": "s_7"}, [0.3, 0.5, 0.2], [0.01, 0.02, 0.005], 1.0]
}}
```

```json
{"result": {"value": 1}}
```

## Declarative Specifications

### Space specs

| Type | Params | Julia |
|------|--------|-------|
| `finite` | `values` | `Finite(values)` |
| `interval` | `lo, hi` | `Interval(lo, hi)` |
| `product` | `factors: [space, ...]` | `ProductSpace(factors)` |
| `euclidean` | `dim` | `Euclidean(dim)` |
| `positive_reals` | -- | `PositiveReals()` |
| `simplex` | `k` | `Simplex(k)` |

### Measure specs

| Type | Params | Julia |
|------|--------|-------|
| `categorical` | `space`, optional `log_weights` | `CategoricalMeasure` |
| `beta` | `alpha, beta` | `BetaMeasure` |
| `gaussian` | `mu, sigma` | `GaussianMeasure` |
| `gamma` | `alpha, beta` | `GammaMeasure` |
| `dirichlet` | `alpha: [...]` | `DirichletMeasure` |
| `normal_gamma` | `kappa, mu, alpha, beta` | `NormalGammaMeasure` |
| `product` | `factors: [measure, ...]` | `ProductMeasure` |
| `mixture` | `components, log_weights` | `MixtureMeasure` |
| `tagged_beta` | `tag, alpha, beta` | `TaggedBetaMeasure` |

### Kernel specs

| Type | Params | Description |
|------|--------|-------------|
| `bernoulli` | -- | theta -> Bernoulli(theta). Beta-Bernoulli conjugate. |
| `gaussian_known_var` | `variance` | mu -> N(mu, var). Gaussian conjugate. |
| `gaussian_unknown_var` | -- | (mu, tau) -> N(mu, 1/tau). Normal-Gamma conjugate. |
| `program_observation` | `features`, `true_label` | Per-component CompiledKernel dispatch. |
| `answer_kernel` | `reliability_state_id`, `n_answers` | Tool reliability -> answer distribution. |
| `plackett_luce` | `chosen_features`, `pool_features` | Ranking observation model. |
| `quality` | -- | (theta, k) -> Beta(theta*k, (1-theta)*k). Continuous quality. |
| `tabular_log_density` | `source_vals`, `target_vals`, `densities` | Precomputed log-density matrix. |
| `dsl` | `env_id`, `fn_name`, `args` | Calls DSL function to construct kernel. General escape hatch. |

### Function specs (for expect)

Function specs are serialised `Functional` types. The brain's `expect`
dispatches on the Functional subtype to select the optimal computation
(closed-form for leaf measures, recursive decomposition for ProductMeasure,
weighted sums for CategoricalMeasure, Monte Carlo fallback for
`opaque_bdsl`). Structure enables fast paths — bare closures forfeit them.

| Type | Params | Description |
|------|--------|-------------|
| `identity` | -- | `f(x) = x`. Closed-form on Beta/Gamma/Gaussian leaves. |
| `projection` | `index` (0-based) | `f(x) = x[index]`. Decomposes on ProductMeasure. `project` accepted as alias. |
| `nested_projection` | `indices: [Int]` (0-based) | Recursive navigation through nested ProductMeasures. |
| `tabular` | `values: [Float]` | Weighted sum over CategoricalMeasure atoms. |
| `linear_combination` | `terms: [[coeff, sub_fn_spec]]`, `offset?` | Linearity of expectation: sum of (coeff · sub-functional expectations) + offset. Sub-specs are recursive function specs. |
| `opaque_bdsl` | `env_id`, `expr` | DSL lambda; delegates to the bare-function `expect` method for the measure type (closed-form / quadrature / Monte Carlo depending on measure). `bdsl` accepted as alias. |

Example — router preference for provider 0 over 2 categories:
```json
{"type": "linear_combination",
 "terms": [
   [0.5, {"type": "nested_projection", "indices": [0, 0, 0]}],
   [0.5, {"type": "nested_projection", "indices": [0, 1, 0]}]
 ],
 "offset": -0.01}
```

### Preference specs (for optimise/value)

| Type | Params | Description |
|------|--------|-------------|
| `functional_per_action` | `actions: {"0": fn_spec, "1": fn_spec, ...}` | Maps each action (as a string key) to an arbitrary Functional spec. `optimise` picks the argmax; `value` returns the max EU. Orthogonal to the Functional hierarchy — any function spec type can be used per action. |
| `tabular_2d` | `matrix` | `pref(h_i, a_j) = matrix[i][j]`. Legacy; use `functional_per_action` with `tabular` specs instead. |
| `bdsl` | `env_id`, `expr` | DSL lambda `(h, a) -> utility`. Legacy. |

### State manipulation for ProductMeasure

These enable a body to decompose nested ProductMeasures, condition a
specific leaf, and reassemble — without any DSL wrapper.

`factor`:
```json
{"method": "factor", "params": {"state_id": "s_1", "index": 0}}
-> {"result": {"state_id": "s_5"}}
```
Register factor `index` (0-based) as a new state. The original state is
unchanged; the factor is referenced by a new id.

`replace_factor`:
```json
{"method": "replace_factor", "params": {"state_id": "s_1", "index": 0, "new_factor_id": "s_5"}}
-> {"result": {"state_id": "s_6"}}
```
Build a new ProductMeasure with factor `index` replaced by the measure
referenced by `new_factor_id`. Returns a new state id; the source states
are unchanged.

`n_factors`:
```json
{"method": "n_factors", "params": {"state_id": "s_1"}}
-> {"result": {"n_factors": 3}}
```

### Grammar specs (for enumerate)

```json
{
  "features": ["colour", "speed"],
  "rules": {
    "S": ["(if P A A)"],
    "P": ["(> F T)", "(< F T)", "(and P P)", "(or P P)", "(not P)"],
    "A": ["food", "enemy"],
    "F": ["colour", "speed"],
    "T": ["0.3", "0.5", "0.7"]
  }
}
```

## Error Handling

Standard JSON-RPC 2.0 error responses:

```json
{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"state not found: s_99"}}
```

| Code | Meaning |
|------|---------|
| -32700 | Parse error (malformed JSON) |
| -32600 | Invalid request (missing method/id) |
| -32601 | Method not found |
| -32603 | Internal error (uncategorised handler failure) |
| -32000 | State not found |
