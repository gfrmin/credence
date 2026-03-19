# CLAUDE.md — Constitution for Credence

> Read this entire document before writing any code.

## What this project is

A minimal DSL for Bayesian decision-making agents. S-expression
syntax. Compiles to Julia. Grounded in axiomatic decision theory.

## The axioms (mathematical truths, not design choices)

    A1. Beliefs are probability measures                  (Cox)
    A2. Rational action maximises expected utility        (Savage)
    A3. Learning is conditioning on evidence              (Bayes, de Finetti)
    A4. There is one learning mechanism and one
        decision mechanism                                (Dutch book coherence)

## The frozen layer: three types

The DSL has exactly three kinds of object. These are the
ontology of Bayesian decision theory. They do not change.

    Space   — a set of possibilities
    Measure — a probability distribution over a space
    Kernel  — a conditional distribution between two spaces

Everything else — every operation, every combinator, every
named concept — is a function over these three types.

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
    julia test/test.jl

Run an example:
    julia -e 'push!(LOAD_PATH, "src"); using Credence; run_dsl(read("examples/coin.bdsl", String))'

Run the credence agent (host-driven):
    julia examples/host_credence_agent.jl

Use the module from Julia REPL:
    push!(LOAD_PATH, "src")
    using Credence

Load DSL and get callable closures (host-driver pattern):
    env = load_dsl(read("examples/credence_agent.bdsl", String))
    agent_step = env[Symbol("agent-step")]

No external dependencies — Julia stdlib only. No Project.toml yet.

## Project structure

    src/
      parse.jl            S-expression parser (the entire front-end)
      ontology.jl         Three types + axiom-constrained functions + draw/optimise/value
      eval.jl             Evaluator / compiler (DSL → Julia calls)
      stdlib.bdsl         Standard library (optimise, value, voi, etc.)
      persistence.jl      Save/load agent state across sessions
      Credence.jl         Module entry point
    examples/
      coin.bdsl                 Biased coin learning
      credence_agent.bdsl       Agent DSL (pure functions, host-driven)
      grid_agent.bdsl           Bayesian grid-world navigation
      host_credence_agent.jl    Julia host driver for credence agent
    test/
      test.jl                   End-to-end validation

DSL source files use the `.bdsl` extension.

Weights are stored in log-space internally (`logw` field). Use
`weights(m)` to get normalized probabilities. This is a cross-file
invariant — never exponentiate manually.

Constructors: `CategoricalMeasure(Finite(vals))` for uniform prior,
`CategoricalMeasure{T}(Finite{T}(vals), logw)` for explicit log-weights
(auto-normalises). `BetaMeasure(α, β)` for Beta on [0,1].

## Architecture

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
    │  Julia execution layer      │  MUTABLE
    │  Conjugate, quadrature,     │
    │  particle, heuristic        │
    └─────────────────────────────┘

    Host (Julia): provides observations, executes actions,
    drives loops, manages persistent state. The DSL is pure.

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
