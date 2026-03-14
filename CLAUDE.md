This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CLAUDE.md — Constitution for the Bayesian Agent DSL

> Read this entire document before writing any code.
> Every design decision was derived from first principles.

---

## What this project is

A minimal domain-specific language for specifying Bayesian
decision-making agents. S-expression syntax. Compiles to Julia.
Grounded in axiomatic decision theory (Cox, Savage, de Finetti).

## Development commands

Run tests:
    julia test/test_vertical_slice.jl

Run an example:
    julia -e "push!(LOAD_PATH, \"src\"); using BayesianDSL; run_dsl(read(\"examples/coin.bdsl\", String))"

Use the module from Julia REPL:
    push!(LOAD_PATH, "src")
    using BayesianDSL

No external dependencies — Julia stdlib only. No Project.toml yet.

## The axioms (not negotiable, not derivable, not changeable)

These are the metal. They are asserted, not computed:

    A1. Preferences are complete and transitive     (Savage P1)
    A2. Beliefs satisfy the product rule             (Cox)
    A3. No sure loss                                 (Dutch book coherence)

From which UNIQUELY follow:

    → Beliefs must be probabilities                  (Cox's theorem)
    → Learning must be Bayesian conditioning          (Bayes' rule)
    → Action must maximise expected utility           (Savage's theorem)

## The DSL (frozen — do not modify)

The grammar:

    expr = atom | '(' expr* ')'

The three primitives:

    (belief h1 h2 ...)              weighted hypotheses (uniform prior)
    (update <belief> <obs> <lik>)   Bayesian conditioning
    (decide <belief> <acts> <util>) expected utility maximisation

Everything else is a derived combinator. The DSL grammar and primitives
are IMMUTABLE. They are forced by the axioms. Adding a fourth primitive
is a mathematical error, not a design choice.

## The supporting forms (frozen — do not modify)

These exist to make the DSL Turing-complete but introduce
no new decision-theoretic capability.

A supporting form may read from beliefs (like weighted-sum) or
construct new beliefs from hypotheses (like belief), but it may
NOT modify the weights of an existing belief. Only update modifies
weights, because only Bayesian conditioning is sanctioned by the
axioms. Any operation that produces a belief with altered weights
without conditioning on an observation is a second learning mechanism
competing with update — and there is only one learning mechanism.

    (let <name> <expr> <body>)      binding
    (define <name> <expr>)          top-level binding (mutates env)
    (lambda (<params>) <body>)      abstraction
    (if <cond> <then> <else>)       conditional
    (do <e1> ... <en>)              sequence
    (list <e1> ... <en>)            data construction
    (map <fn> <lst>)                apply fn to each element
    (fold <fn> <lst>)               reduce (first element is init)
    (first <lst>)                   first element of a list
    (weighted-sum <belief> <fn>)    Σ_i w_i · fn(h_i)
    (max <a> <b> ...)               maximum of 2+ values
    Arithmetic: + * (variadic), - / (binary), log exp
    Comparison: = > <               for conditional logic
    String literals: "..."          for print messages

## What Claude Code may change

The SEMANTICS — how primitives compile to Julia:

    - Improve update to dispatch to conjugate fast-paths
    - Improve decide to use bounded-depth planning
    - Add particle resampling when ESS drops
    - Add backends (Gen.jl, POMDPs.jl interop)
    - Optimise: caching, parallelism, SIMD

The STANDARD LIBRARY — derived combinators built from the three primitives
(defined in `src/stdlib.bdsl`, auto-loaded by `run_dsl`):

    eu              = Σ_i w_i · u(h_i, a)              (expected utility)
    best-eu         = max_a eu(b, a, u)                 (best EU over actions)
    predictive-prob = Σ_i w_i · exp(lik(h_i, o))       (observation probability)
    voi             = E_o[best-eu(update(b,o))] - best-eu(b)
    predict         = marginalise hypotheses' observation models (TODO)
    fuse            = update(update(b, o1), o2) (TODO)
    thompson-sample = sample h ~ weights, decide as if h is true (TODO)

Tests, examples, documentation, tooling.

## What Claude Code may NOT change

- The grammar
- The three primitives
- The axioms
- The separation between DSL and compilation target
- The principle that hypotheses are S-expressions (inspectable data)

## Forbidden patterns

These are not style preferences. They are mathematical errors.
If you find yourself reaching for one, the model is wrong — fix that.

### Never add an explore primitive
Exploration is a CONSEQUENCE of value of information, which is a
COMPOSITION of update and decide. If the agent isn't exploring,
its observation model is wrong or its utility doesn't value
information. Fix the model. Do not bolt on exploration.

### Never add loop detection
If the agent loops, it means update is not changing the beliefs,
which means the observations carry no information under the current
hypotheses. The hypothesis space is inadequate. Expand it.
Do not patch the symptom.

### Never add exploration bonuses
Epsilon-greedy, UCB bonuses, optimism-in-the-face-of-uncertainty
heuristics — these are all approximations to proper Bayesian
exploration (Thompson sampling, VOI, Bayes-adaptive planning).
Use the principled version or document why the approximation
is necessary with a complexity argument.

### Never make update optional or skippable
If there is an observation, it must update beliefs. Ignoring
evidence violates A3 (Dutch book coherence). If the observation
is uninformative, the likelihood will be flat and weights won't
change — that's fine. But you don't get to skip the step.

### Never use 0.5 as a default prior for binary hypotheses
With N hypotheses, the prior is 1/N for each. For 2 hypotheses,
that happens to be 0.5, but the reasoning is "maximum entropy
over N states" not "binary so fifty-fifty." This matters when
N changes.

### Never compare (VOI - cost) > expected_utility
The correct comparison is VOI > cost. The value of information
is the improvement over current best EU, not an absolute value
to be compared against the current best EU.

### Ground truth for sensor learning: positive rewards only
When updating beliefs about sensor reliability, only count
observations where the ground truth is known (positive reward
received). State changes alone do not constitute ground truth.

### Never add a forget or decay mechanism
Non-stationarity is a property of the world, not a computational
operation on beliefs. If reliability might drift, include drift-rate
in the hypothesis space — hypotheses that predict stability get
upweighted when the world is stable, and hypotheses that predict
change get upweighted when the world shifts. The effective forgetting
rate emerges from the posterior rather than being imposed as a
parameter. Any mechanism that modifies belief weights outside of
Bayesian conditioning (update) is a second learning channel that
violates the axioms.

## Architecture

    ┌─────────────────────────────┐
    │  DSL (S-expressions)        │  ← what the user writes
    │  Three primitives.          │     IMMUTABLE
    │  Composition.               │
    ├─────────────────────────────┤
    │  Semantics layer            │  ← enforces axioms
    │  Validates well-formedness  │     IMMUTABLE constraints
    │  Selects backend            │     MUTABLE dispatch
    ├─────────────────────────────┤
    │  Julia compilation target   │  ← where computation happens
    │  Conjugate fast-paths       │     MUTABLE (improve freely)
    │  Particle methods           │
    │  Gen.jl / POMDPs.jl        │
    ├─────────────────────────────┤
    │  Metareasoning              │  ← controls compute budget
    │  VOC-based stopping         │     MUTABLE (improve freely)
    │  BMPS-style learned weights │
    └─────────────────────────────┘

## Project structure

    src/
      parse.jl          S-expression parser (the entire front-end)
      primitives.jl     belief, update, decide, weighted_sum (the theory)
      eval.jl           Evaluator / compiler (DSL → Julia calls)
      stdlib.bdsl       Standard library (eu, best-eu, voi, etc.)
      BayesianDSL.jl    Module entry point
    examples/
      coin.bdsl             Biased coin learning
      tool_selection.bdsl   VOI-based tool choice (mini Credence)
      credence_engine.bdsl  Full credence-engine decision loop
    test/
      test_vertical_slice.jl  End-to-end validation

DSL source files use the `.bdsl` extension.

Weights are stored in log-space internally (`logw` field). Use
`weights(b)` to get normalized probabilities. This is a cross-file
invariant — never exponentiate manually.

## Key references

    Cox (1946)          — probability from consistency
    Savage (1954)       — utility + probability from preferences
    Jaynes (2003)       — Probability Theory: The Logic of Science
    Hutter (2005)       — AIXI: the Platonic limit
    Leike & Hutter (2015) — AIXI's optimality is vacuous
    Russell & Wefald (1991) — rational metareasoning
    Lieder & Griffiths (2017) — strategy selection as metareasoning
    McCarthy (1960)     — why S-expressions

## One-line summary

Three axioms force three primitives; everything else is composition.
