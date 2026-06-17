# answer-brain — master plan

A new credence "brain" that governs **answering** the way `credence-pi` governs tool-call
safety. It is a `credence-pi` sibling under `apps/answer-brain` (Julia daemon on `src/Credence`;
a pi-mono TS body arrives in Stage 2). This file is the durable in-repo record; the originating
session plan lives at the author's `~/.claude/plans/this-repo-is-supposed-mutable-taco.md`, which
is session-external and not guaranteed available to future sessions (CLAUDE.md §"Session memory").

## Why

The consumer (the `life-agent` repo) has a single-pass lookup answerer — a static
`threshold(pool(extract(retrieve(q))))` with no feedback. Every failure was met by hand-coding one
more covariate, with the author as the gradient (the "patch-treadmill"). The canonical failure: a
current-state point fact (e.g. a mobile number) whose true value **is** retrieved and **is**
extracted cleanly, but loses at low credence to a superseded value, because the posterior pools
candidates by *corroboration count* and the owner out-documents his current life with his old one.
No retrieval/extraction knob fixes a *semantic* current-vs-stale problem; recency/whose-document
are decision-theoretic covariates, not ranking heuristics.

The fix is to make answering a **governed, multi-step** decision: gather the discriminating
evidence the question's words missed, value it under one posterior, and choose answer / ask / abstain
by EU — `condition`/`optimise`/`net_voi`, no magic thresholds. That is exactly what this repo's
axioms already are; the brain *is* credence.

## Relationship to the existing skin

`life-agent`'s live ask path **already** runs its candidate posterior + EU decision through this
repo's **skin** (`apps/skin/server.jl`) over JSON-RPC: it builds a `categorical` state
(`build_prevision`, server.jl:153), conditions a `tabular_log_density` PushOnly kernel per
observation (`build_kernel`, server.jl:295; `condition(::CategoricalMeasure, …)`, ontology.jl:861),
and decides via `functional_per_action` `optimise` over `Tabular` utility vectors (server.jl:870;
`expect(::CategoricalMeasure, ::Tabular)`, ontology.jl:690).

The skin is a **stateless RPC evaluator** — the Python body builds every spec and ships it; the
skin holds no belief between calls and makes no decision of its own. The answer-brain is the
**stateful governor**: it *holds* the candidate posterior, receives evidence as it arrives
(observations from probes), and emits the effector (answer/ask/abstain/gather). **Same Credence
primitives, different control flow.** The math the body currently constructs in Python ports into
native Julia in the brain; the skin remains the in-process evaluator life-agent keeps using until
the body cuts over (Stage 2).

## Architecture (three parts; pi-mono is reactive)

pi-mono's loop is LLM-driven: the model proposes every tool call; extensions can only gate, rewrite
a result, or inject a steering message. So the brain **governs + steers**, it does not drive.

1. **pi-mono answering agent (TS, Stage 2).** Proposes `retrieve` / `extract` / `probe-*` / `answer`.
2. **answer-brain (this app).** Belief = posterior over K candidate values + an explicit NONE atom
   (the ported `lookup_posterior` + tempered observation kernels). Effectors = `answer` / `ask-user`
   / `abstain` / `gather`. Decide = one EU comparison; `gather`/`ask-user` priced by `net_voi − cost`
   against the terminal payoffs, the terminal payoffs evaluated under the owner's **utility
   posterior** (supplied by the body as Ū). Learn = `condition` on probe-result observations + owner
   verdicts; the observation log replays the posterior on restart.
3. **life-agent capabilities (Python).** pkm retrieval + grounded extraction + the **probe library**
   (`core/probes.py`: recency / authority / subject / corroborate). Each probe gathers a
   discriminating signal the brain selects by VOI. These stay in life-agent; the brain never derives.

## The parity boundary (what the brain owns)

Once a grounded `Observation` exists, its covariate factors (`authority`, `subject_factor`,
`time_factor`) and its ancestry-group key are **fixed numbers**. From there the math is pure and
deterministic:

    observations + rho + Ū  ──►  candidates, posterior weights, chosen action, EU

Retrieval, extraction, gather-retrieval, and covariate *projection* are the body's job (Ollama +
DuckDB); they are **not** ported. The brain owns the boxed function — and that is exactly what
Stage-1 parity tests pin against the validated Stage-0 Python.

## Staging

- **Stage 0 — DONE (in `life-agent`, committed `4b336db`).** The probe library
  (`core/probes.py`) + the gather-augmented loop (`core/gather.py`) + the shared
  `lookup.decide_and_record` tail, wired behind `ask --gather` and the adoption gate. Gate result:
  monotonic improvement on every axis with **zero confident-wrong** (the hard gate held), still
  FAIL on the 0.90 bar — the dominant blocker is the width of the owner's `u_wrong` prior, not the
  mechanism. The owner-scoped guard (gather augments only what whose-document can protect) removed
  the one confident-wrong gather introduced.
- **Stage 1 — NOW (this app).** Port the Stage-0 posterior + the `net_voi` decide to native Julia
  (`answer_brain.jl`); declare effectors/utility in `bdsl/*`; adapt the observation log + replay.
  **Parity tests vs Stage 0 on shared fixtures (same posterior, same chosen effector); replay
  reconstructs the posterior.** See `move-1-design.md`.
- **Stage 2 — pi-mono body + app.** Update the upstream pi-mono checkout first; build the TS body
  extension (effectors + feature extractors), the HTTP/SSE daemon surface (`server.jl`, deferred
  from Stage 1 because its sensor-event schema is defined by the body), and a minimal pi-mono
  answering app on the life-agent retrieval/extraction/probe tools. End-to-end on the eval;
  gate-measured.
- **Stage 3 — learning + breadth.** Owner-verdict folding via the observation log (replayable);
  widen the eval (frozen-blind).

## Final-state architecture

A standing `answer-brain` daemon (Tailscale-only) that the pi-mono answering app steers: the app
proposes retrieve/extract/probe/answer; the brain holds one candidate posterior, gathers by VOI
until a terminal effector wins under Ū, and emits answer/ask/abstain. Owner verdicts and probe
outcomes condition the brain through its replayable observation log. The skin stays the in-process
evaluator for any caller that wants stateless RPC; the brain is the stateful governor for answering.

## Discipline (non-negotiable)

Frozen-blind at every stage (never tune utility priors / eval golds to the gate); **zero new
confident-wrong is a hard gate** (revert any increment that adds one); the three credence invariants
bind every decision-causing computation (Invariant 1: apps declare data and call Tier-1 primitives —
no arithmetic on weights to select an action; Invariant 2: typed kernels/functionals; Invariant 3:
one role per representation); never fabricate owner verdicts; PII stays in `$LIFE_AGENT_KB`, never in
this repo; the brain **reuses** life-agent retrieval/extraction, it does not rebuild them; pi-mono is
upstream (update before building on it, contribute changes upstream).
