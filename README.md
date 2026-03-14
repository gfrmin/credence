# Credence

A minimal DSL for Bayesian decision agents. Three axioms, three primitives, everything else is composition.

## Core idea

Three axioms (complete preferences, product rule for beliefs, no sure loss) uniquely force three primitives:

```
(belief h1 h2 ...)              — weighted hypotheses
(update <belief> <obs> <lik>)   — Bayesian conditioning
(decide <belief> <acts> <util>) — expected utility maximisation
```

There is no fourth primitive. This is not a design choice — it's a mathematical consequence.

## Quick start

Requires Julia (stdlib only, no packages).

```bash
julia -e 'push!(LOAD_PATH, "src"); using BayesianDSL; run_dsl(read("examples/coin.bdsl", String))'
```

Run tests:

```bash
julia test/test_vertical_slice.jl
```

## Example: learning a biased coin

```scheme
(let prior (belief 0.1 0.3 0.5 0.7 0.9)
  (let lik (lambda (theta obs)
             (if (= obs 1) (log theta) (log (- 1.0 theta))))
    (let posterior (update (update prior 1 lik) 0 lik)
      (decide posterior (list 1 0)
        (lambda (theta action)
          (if (= action 1)
            (if (> theta 0.5) 1.0 -1.0)
            (if (< theta 0.5) 1.0 -1.0)))))))
```

See `examples/` for more: VOI-based tool selection, a full credence engine decision loop.

## Standard library

Derived combinators built from the three primitives (defined in `src/stdlib.bdsl`):

| Combinator | Definition |
|---|---|
| `eu` | `Σ_i w_i · u(h_i, a)` — expected utility |
| `best-eu` | `max_a eu(b, a, u)` — best EU over actions |
| `predictive-prob` | `Σ_i w_i · exp(lik(h_i, o))` — observation probability |
| `voi` | `E_o[best-eu(update(b,o))] - best-eu(b)` — value of information |

## Architecture

```
DSL (S-expressions)        ← what the user writes (immutable)
  ↓
Semantics layer            ← enforces axioms, selects backend
  ↓
Julia compilation target   ← where computation happens (mutable)
```

## References

- Cox (1946) — probability from consistency
- Savage (1954) — utility + probability from preferences
- Jaynes (2003) — *Probability Theory: The Logic of Science*
- McCarthy (1960) — why S-expressions

## License

AGPL-3.0
