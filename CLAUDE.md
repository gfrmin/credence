# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Constitution for Credence

> Read this entire document before writing any code.

## What this project is

A monorepo containing the Credence Bayesian decision-making DSL (Julia)
and all Python packages that build on it. S-expression syntax. Compiles
to Julia. Grounded in axiomatic decision theory.

### Python workspace (uv)

The `python/` directory is a uv workspace with four packages:

| Directory | PyPI package | Import path | Description |
|-----------|-------------|-------------|-------------|
| `apps/python/credence_bindings/` | `credence` | `import credence` | Low-level Python bindings (Space, Measure, Kernel) |
| `apps/python/credence_agents/` | `credence-agents` | `import credence_agents` | Agent library + Julia bridge + benchmark |
| `apps/python/credence_router/` | `credence-router` | `import credence_router` | Tool routing via EU maximisation |
| `apps/python/bayesian_if/` | — | `import bayesian_if` | Interactive fiction agent application |

Install all: `uv sync` from repo root. `credence_agents`, `credence_router`,
and `bayesian_if` each have their own CLAUDE.md; read the relevant one when
working inside that package.

### The product surface

The repo has two framings. CLAUDE.md is DSL-focused (axioms, frozen layer,
forbidden patterns). README.md is product-focused: the public face is
**credence-proxy** (in `apps/python/credence_router/`), a drop-in
OpenAI-compatible gateway that Bayesian-routes requests across LLM
providers, packaged as a Docker image. The DSL program that drives it
lives at `examples/router.bdsl`. When in doubt about user intent on the
`apps/python/credence_router/` package, the product framing is the gateway, not
the library.

## The axioms (mathematical truths, not design choices)

    A1. Beliefs are probability measures                  (Cox)
    A2. Rational action maximises expected utility        (Savage)
    A3. Learning is conditioning on evidence              (Bayes, de Finetti)

A1–A3 are theorems: each forces a unique answer under mild
consistency requirements. The next item is engineering
discipline — not proved by A1–A3, but what it takes to
implement them without creating internal incoherence:

    A4. One learning mechanism, one decision mechanism.

Rationale: if condition and some second function can both
modify beliefs, the implementation can disagree with itself
— the situation Dutch book arguments rule out at the agent
level. A4 is the invariant that prevents the theorems from
being violated in code.

## The frozen layer: three types

The DSL has exactly three kinds of object. These are the
ontology of Bayesian decision theory. They do not change.

    Space   — a set of possibilities
    Measure — a probability distribution over a space
    Kernel  — a conditional distribution between two spaces

Everything else — every operation, every combinator, every
named concept — is a function over these three types.

What is frozen: the three types and their semantics. What
is NOT frozen: the constructor roster below. Named
distributions and space types may be added (and are),
provided the added item respects the semantics of its type.
The lists below are current vocabulary, not a closed set.

### Space constructors

    (space :finite a b c ...)       finite discrete set
    (space :interval lo hi)         bounded real interval
    (space :product S1 S2 ...)      cartesian product
    (space :euclidean n)            R^n (honest domain for Gaussians)
    (space :positive-reals)         R+ (domain for precision/variance)

### Measure constructors

    (measure S :uniform)            maximum entropy
    (measure S :categorical w1 w2)  explicit weights
    (measure S :beta α β)           Beta distribution on [0,1]
    (measure S :gaussian μ σ)       Gaussian on Euclidean space
    (measure S :gamma α β)          Gamma on positive reals
    (measure S :exponential rate)   Exponential on positive reals

### Kernel constructors

    (kernel H O generator)          H → Measure(O)
        H: source space (hypothesis space)
        O: target space (observation space)
        generator: (lambda (h) distribution-spec)

A kernel is a morphism in the Markov category. It specifies,
for each hypothesis, what distribution over observations the
agent expects. It is the agent's theory of how hypotheses
generate data.

## The axiom-constrained functions

These functions are in the standard library, not the frozen layer.
Their INTERFACE may change. Their BEHAVIOUR is constrained by the
axioms and must not be violated.

    condition : Measure(H) × Kernel(H,O) × O → Measure(H)
        Bayesian inversion. The unique coherent way to update
        a belief given evidence. Must implement Bayes' rule.
        No other function may modify a measure's weights.

    expect : Measure(S) × (S → ℝ) → ℝ
        Integration against a measure. This is what a measure IS:
        a thing that assigns expected values to functions. EU, VOI,
        predictive probability — all are expectations.

    push : Measure(H) × Kernel(H,T) → Measure(T)
        Pushforward / composition. Given a distribution over
        hypotheses and a kernel to another space, produce the
        induced distribution. expect is push to ℝ.

    density : Kernel(H,O) × H × O → ℝ
        The kernel's density at a point. What condition needs
        to compute the likelihood ratio.

## The derived functions (stdlib)

These are compositions of the axiom-constrained functions with
ordinary computation. They are convenience, not capability.
Their interfaces are negotiable and will evolve.

    optimise   = argmax over actions of expect(measure, pref)
    value      = max over actions of expect(measure, pref)
    predictive = expect(measure, density(kernel, ·, obs))
    voi        = expect over obs of [value after condition] - value before
    model      = packaging of measure + kernel (convenience)
    problem    = packaging of model + actions + preference (convenience)

## What Claude Code may change

The standard library: function signatures, new combinators,
packaging of types into convenient bundles.

The Julia implementations: how condition dispatches (conjugate,
particle, quadrature). How expect computes (summation, quadrature,
Monte Carlo). How push constructs new measures. The computational
strategy is determined by the types of the spaces involved and
is invisible to the DSL user.

The set of named distributions and space types.

Tests, examples, documentation, host drivers.

## What Claude Code may NOT change

The three types: Space, Measure, Kernel.

The axiom constraints: condition must be Bayesian inversion,
expect must be integration, no other function may modify
a measure's weights.

The purity of the DSL: no side effects, no IO, no mutation.
The host provides observations and executes actions.

The S-expression grammar: expr = atom | '(' expr* ')'

## Forbidden patterns

These are mathematical errors, not style preferences.

### No AST interpretation at conditioning time
Programs are compiled into closures at enumeration time via `compile_kernel`.
The `CompiledKernel` struct has no AST field — enforced by the type system.
Kernel evaluation calls a closure, never walks a tree. If you need to
analyse program structure (for grammar perturbation), use the `Program`
struct which retains the AST. The two representations serve different
purposes and must not be conflated.

### No random mutation in place of subprogram extraction
Nonterminal proposals must be grounded in posterior analysis.
`propose_nonterminal` requires a `SubprogramFrequencyTable`, which can
only be constructed by `analyse_posterior_subtrees`. There is no shortcut:
the type system enforces the dependency. Random AST generation or mutation
without posterior analysis violates the principled compression pipeline.

### No second learning mechanism
Only condition modifies beliefs. Any function that produces a
measure with altered weights without conditioning on an
observation violates A4. This includes: forget, decay, exploration
bonuses, ad-hoc reweighting. If the world changes, include
drift-rate in the hypothesis space.

### No second decision mechanism
Only EU maximisation (via expect + argmax) selects actions.
Epsilon-greedy, UCB bonuses, and similar heuristics are
not forbidden as CONCEPTS — they may emerge as EU-maximising
strategies when computational cost enters the utility function.
But they must not be hard-coded as mechanisms outside of EU
maximisation.

### No opaque likelihood functions
Likelihoods are kernels, not bare lambdas. A kernel declares
its source space, target space, and generative structure. This
enables the compiler to select computational backends and detect
conjugate structure. (lambda (h o) ...) as a likelihood is a
v1 pattern that should not appear.

### Declare kernel structure at construction, not at dispatch
A kernel's per-θ algebraic form (BetaBernoulli, Flat, etc.) is
declared via the likelihood_family field at Kernel construction, not
inferred at dispatch by probing log-density values or return types.
This is the condition-side analogue of the Functional hierarchy for
expect: structure enables dispatch, declaration is the mechanism.
Kernel construction requires a likelihood_family keyword argument —
omission raises UndefKeywordError at construction time, not later.
condition() additionally rejects PushOnly and any unrecognised family
with a clear remediation error. Probing a
kernel's output at chosen inputs to infer structure (e.g. treating
log_density == 0.0 as "flat") is forbidden — it misfires on
legitimate edge cases and hides the assumption from the type system.

Per-component routing in mixtures has a declarative vocabulary:
FiringByTag(fires::Set{Int}, when_fires, when_not) for the dominant
"some predicates fire, some don't" pattern. DispatchByComponent takes
a classify(measure) -> LikelihoodFamily closure; it is the typed-return
escape hatch (analogous to OpaqueClosure for Functional) and should
only be used when no declarative subtype fits. Adding a new declarative
subtype is preferred over reaching for DispatchByComponent.

### No opaque functions passed to expect
Functions passed to expect are Functionals, not bare lambdas. A
Functional declares its algebraic structure (Identity, Projection,
NestedProjection, Tabular, LinearCombination) so expect can dispatch
to the optimal computation. This is the same principle as Kernels
for condition: structure enables dispatch. OpaqueClosure is the
fallback — it works but forfeits fast paths.

Functional types must compose. LinearCombination carries
Vector{Tuple{Float64, Functional}}, not flat coefficient arrays.
Each sub-functional navigates its own structure. Flat indexing
schemes encode stride conventions that are invisible to the type
system and forbidden.

### No DSL wrappers around axiom-constrained operations
A domain DSL file contains data (spaces, kernels, priors). It must
not define functions that are thin wrappers around optimise, condition,
expect, or push with domain-specific list navigation. Those wrappers
exist only because state was a list needing navigation; with
ProductMeasure state + factor/replace-factor, they disappear. Preferences
are protocol-level Functional specs (functional_per_action with
LinearCombination of NestedProjections), not DSL lambdas. Wrapping an
axiom-constrained operation in a DSL function forces its arguments
through an opaque closure bottleneck that defeats Functional dispatch.

### Indifference implies exploration
When EU of interacting equals EU of waiting (both 0), interact.
Indifference means VOI from the interaction outcome is positive.
The select_action threshold is >= 0, not > 0. Do not "fix" this
back to strict inequality.

### Non-firing predicates predict the base rate
Programs whose predicates don't fire return log(0.5), not 0.0.
A non-firing program is implicitly predicting "I don't know, so
50/50." That prediction is scored against the observation. The
ranking: informed-and-right > uninformed > informed-and-wrong.
Returning 0.0 makes non-firing programs unbeatable (no information
= no penalty), creating weight rigidity after regime changes.

### Heuristics are EU maximisation, not approximations
When computational cost enters the utility function, a faster
approximate strategy may have higher EU than the exact Bayesian
computation. This is not an approximation — it IS the optimal
strategy. Heuristics belong in the Julia execution layer as
alternative implementations of expect/condition/push, selected
by the same EU-maximisation machinery the agent uses for
everything else. The DSL specification does not change.

## The host boundary

The DSL constructs mathematical objects (measures, kernels, numbers).
The host realises them (draws values, executes actions, drives loops).

draw : Measure(S) → S is the ONLY source of randomness. It lives
in the ontology module, exported for Julia callers, NOT in the
DSL's default_env.

optimise and value live in the ontology module alongside expect
and condition. One implementation per operation. The host calls
them; the host does not reimplement them.

If you need randomness → use draw in the host.
If you need decisions → use optimise/value from the ontology, or
call the DSL's optimise via run_dsl.

Do not reimplement expect, condition, optimise, or value in the
host. Call the ontology module's exports.

## Rejected patterns (with reasoning)

PROPOSED: Add (sample measure) to the DSL for Thompson sampling.
REJECTED: sample is randomness, randomness is a side effect,
the DSL is pure. Construct the posterior in the DSL, call
draw() in the host.

PROPOSED: Add host_decide() / host_optimise() in the host driver.
REJECTED: optimise and value belong in the ontology alongside
expect and condition. One implementation per operation. The host
driver is pure orchestration — it calls ontology functions.

PROPOSED: (thompson-sample m actions pref) in stdlib that calls sample.
REJECTED: Compounds both errors above. Thompson sampling is
draw (host) + argmax (ordinary computation). The DSL computed
the posterior; its job is done.

## Development commands

Run tests:
    julia test/test_core.jl             # Core DSL tests
    julia test/test_flat_mixture.jl     # Flat mixture tests
    julia test/test_host.jl             # Host helper tests
    julia test/test_program_space.jl    # Tier 2 tests
    julia test/test_grid_world.jl       # Tier 3 grid-world tests
    julia test/test_email_agent.jl      # Email agent domain tests
    julia test/test_rss.jl              # RSS domain tests

Run POMDP agent tests:
    cd apps/julia/pomdp_agent && julia --project=. -e 'using Pkg; Pkg.test()'

Run the Jericho IF agent:
    cd apps/julia/pomdp_agent && julia --project=. examples/jericho_agent.jl /path/to/game.z3

Run an example:
    julia -e 'push!(LOAD_PATH, "src"); using Credence; run_dsl(read("examples/coin.bdsl", String))'

Run the grid-world agent:
    julia apps/julia/grid_world/host.jl

Run the credence agent (host-driven):
    julia examples/host_credence_agent.jl

Use the module from Julia REPL:
    push!(LOAD_PATH, "src")
    using Credence

Load DSL and get callable closures (host-driver pattern):
    env = load_dsl(read("examples/credence_agent.bdsl", String))
    agent_step = env[Symbol("agent-step")]

Requires Julia >=1.9 (stdlib only for the DSL core). CI pins Julia 1.11.
External deps (for full workspace): HTTP, JSON3, Serialization.

Python workspace:
    uv sync                                         # install all 4 packages
    uv sync --extra server --extra search           # what CI installs
    PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run pytest apps/python/

Run a single Python test file (example):
    PYTHON_JULIACALL_HANDLE_SIGNALS=yes \
      uv run pytest apps/python/credence_router/tests/test_routing.py -x

`apps/python/credence_router/tests/test_live.py` is excluded from CI (hits
real provider APIs); run it manually when changing live paths.

Brain server (language-agnostic host interface):
    julia apps/brain/server.jl                      # holds Measures, evaluates DSL via JSON-RPC
    python -m brain.test_brain                      # smoke tests (from repo root)

credence-proxy (production gateway):
    PYTHON_JULIACALL_HANDLE_SIGNALS=yes credence-router serve
    docker build -t credence-proxy .                # same image CI publishes

CI: `.github/workflows/publish-image.yml` runs core Julia tests,
`uv sync --extra server --extra search --no-dev`, then
`credence_router` + `credence_agents` pytest (excluding test_live.py),
and publishes the Docker image.

## Project structure

    src/                          Tier 1: DSL core
      Credence.jl                 Module entry point
      parse.jl                    S-expression parser
      ontology.jl                 Three types + axiom-constrained functions
      eval.jl                     Evaluator / compiler (DSL → Julia calls)
      stdlib.bdsl                 Standard library (optimise, value, voi, etc.)
      persistence.jl              Save/load agent state across sessions
      host_helpers.jl             Host-level reliability/coverage helpers
      program_space/              Tier 2: program-space inference
        ProgramSpace.jl           Module entry point
        types.jl                  AST types, Grammar, Program, CompiledKernel
        enumeration.jl            Bottom-up enumeration, complexity scoring
        compilation.jl            AST → closure compilation
        perturbation.jl           Posterior subtree analysis, grammar perturbation
        agent_state.jl            AgentState, sync_prune!, sync_truncate!
    examples/                     Runnable DSL programs
      coin.bdsl                   Biased coin learning
      credence_agent.bdsl         Agent DSL (pure functions, host-driven)
      grid_agent.bdsl             Grid agent DSL
      router.bdsl                 Bayesian LLM/search routing (drives credence-proxy)
      host_credence_agent.jl      Julia host driver for credence agent
    test/
      test_core.jl                Core DSL tests (42 tests)
      test_flat_mixture.jl        Flat mixture conditioning tests
      test_host.jl                Host helper tests
      test_program_space.jl       Tier 2 tests (enumeration, compilation, perturbation)
      test_grid_world.jl          Tier 3 tests (full agent, regime change, meta-learning)
      test_email_agent.jl         Email agent domain tests
      test_rss.jl                 RSS domain tests
    apps/                         Everything built on top of the DSL
      brain/                      Language-agnostic host interface (JSON-RPC)
        protocol.md               JSON-RPC protocol spec
        server.jl                 Julia server (holds Measures, evaluates DSL)
        client.py                 Python client (spawns subprocess, sends RPC)
        test_brain.py             Smoke tests
      julia/                      Julia applications of the DSL
        DOMAIN_INTERFACE.md       Contract for Tier 3 domains
        grid_world/               Grid-world domain (simulation, host, terminals, metrics)
        email_agent/              Email domain (JMAP integration, LLM prosthetic)
        qa_benchmark/             QA benchmark domain (LLM comparison harness)
        rss/                      RSS article ranking domain (Postgres-backed)
        pomdp_agent/              POMDP agent package (MCTS, factored models)
          Project.toml            Separate Julia package depending on Credence
          src/, examples/, test/, CLAUDE.md
      python/                     Python applications (uv workspace; Python >=3.11)
        credence_bindings/        Low-level Python bindings
        credence_agents/          Agent library + Julia bridge + benchmark
        credence_router/          credence-proxy (LLM/search routing gateway)
        bayesian_if/              Interactive fiction agent
    docs/                         Additional documentation
      rss-preference-learning.md  RSS preference learning design
    papers/                       Publication (credence.tex)
    data/                         Eval output artefacts (gitignored)
    Dockerfile                    credence-proxy container (published by CI)
    SPEC.md                       Authoritative three-tier architecture spec
    pyproject.toml                uv workspace root (4 members under apps/python/)
    .github/workflows/            CI: publish-image.yml (tests + Docker publish)

DSL source files use the `.bdsl` extension.

Weights are stored in log-space internally (`logw` field). Use
`weights(m)` to get normalized probabilities. This is a cross-file
invariant — never exponentiate manually.

Constructors: `CategoricalMeasure(Finite(vals))` for uniform prior,
`CategoricalMeasure{T}(Finite{T}(vals), logw)` for explicit log-weights
(auto-normalises). `BetaMeasure(α, β)` for Beta on [0,1].

## Architecture

Three-tier architecture. See SPEC.md for details.

- Tier 1 (src/): DSL core — condition, expect, optimise, voi, TaggedBetaMeasure
- Tier 2 (src/program_space/): program-space inference — grammars, enumeration, compilation, perturbation
- Tier 3 (apps/julia/): domain applications — each provides features, terminals, host driver (see apps/julia/DOMAIN_INTERFACE.md)

    ┌─────────────────────────────┐
    │  Three types                │  FROZEN
    │  Space, Measure, Kernel     │
    ├─────────────────────────────┤
    │  Axiom-constrained fns      │  Behaviour frozen,
    │  condition, expect, push    │  interface negotiable
    ├─────────────────────────────┤
    │  Standard library           │  MUTABLE
    │  optimise, voi, model, etc  │
    ├─────────────────────────────┤
    │  Program-space inference    │  Tier 2, MUTABLE
    │  Grammars, enumeration,     │
    │  compilation, perturbation  │
    ├─────────────────────────────┤
    │  Domain applications        │  Tier 3, MUTABLE
    │  Grid world, email agent    │
    ├─────────────────────────────┤
    │  Julia execution layer      │  MUTABLE
    │  Conjugate, quadrature,     │
    │  particle, heuristic        │
    └─────────────────────────────┘

    Host (Julia): provides observations, executes actions,
    drives loops, manages persistent state. The DSL is pure.

## Authoritative spec

credence-spec.md (in repo root)

## On metareasoning

There is no separate metareasoning layer. The choice of how
much to compute is itself a decision problem: beliefs about
computational cost, expected improvement from further
computation, EU maximisation over the choice of strategy.
This is expressible in the same DSL the agent uses for
everything else. The Julia layer implements the strategies;
the DSL specifies the choice among them.

## Key references

    Cox (1946)                — probability from consistency
    Savage (1954)             — utility + probability from preferences
    Jaynes (2003)             — Probability Theory: The Logic of Science
    Hutter (2005)             — AIXI: the Platonic limit
    Russell & Wefald (1991)   — rational metareasoning
    Lieder & Griffiths (2020) — resource-rationality
    Fritz (2020)              — Markov categories
    Staton (2017)             — measure-theoretic PPL semantics
    McCarthy (1960)           — why S-expressions

## Performance problems

If a test run is too slow, if the agent hangs, or if exact inference
is computationally expensive: STOP. Do not replace exact inference
with an approximation to fix a performance problem. Instead:

1. Report the performance problem and where it occurs.
2. Propose the approximation explicitly, explaining what
   exactness is lost.
3. Wait for approval before implementing.

Approximations are sometimes necessary. Undocumented approximations
are always bugs. The distinction is consent, not capability.

## One-line summary

Three types, axiom-constrained functions, everything else is stdlib.
