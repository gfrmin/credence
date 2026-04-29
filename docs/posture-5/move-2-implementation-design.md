# Move 2 — Implementation design

## 1. Strategic context

The Move 2 design doc (PR #86) settled six architectural decisions for the
governance plugin v0.1. Three amendments (PR #87) refined the threshold
derivations, race-window semantics, and instruction-decay self-tuning
property. The architecture is settled; what remains is translating the
settled decisions into a buildable sequence of sub-PRs.

This document specifies six sub-PRs, their scopes, dependencies, and
acceptance criteria. It does not revisit architectural decisions — those
live in `docs/posture-5/move-2-design.md` and its amendments.

### Existing code

Move 1 (PR #85) shipped a prototype sidecar (`apps/credence-governance-
sidecar/server.jl`) and plugin (`apps/openclaw-plugin/`). The prototype
demonstrates loop-detection veto via repetition counting. Move 2's sub-PRs
replace the prototype's logic with the full Bayesian brain while preserving
the HTTP IPC contract and plugin registration shape.

### IPC contract

The JSON shapes exchanged between plugin and sidecar are the architectural
seam between TypeScript and Julia. The contract evolves across sub-PRs:

| Endpoint | Sub-PR 1 | Sub-PR 4 |
|---|---|---|
| `POST /evaluate` | Adds `decision`, `signals` to response | Unchanged |
| `POST /observe` | Adds `outcome` field to request | Unchanged |
| `POST /compaction-preview` | Stub (accepts, returns ok) | Full implementation |
| `GET /health` | Adds `posterior_summary` | Unchanged |

Each sub-PR that changes the contract documents the change in its PR
description. The plugin is backwards-compatible: it handles missing
response fields gracefully (the Move 1 `action`/`reason` shape remains
valid as a subset of the enriched response).

## 2. Scope

Six sub-PR specifications with scopes, dependencies, acceptance criteria,
and testing approaches. A phasing summary with the dependency DAG.

## 3. Out of scope

- Code changes. This PR is the implementation-design doc only.
- Amendments to the Move 2 design doc (PR #86 + PR #87).
- Move 3 (targeted demonstration evaluation).
- Post-MVP roadmap (publication, multi-harness, personal-agent).

## 4. Language-side coordination (Risk 4)

| Sub-PR | Language | Rationale |
|---|---|---|
| 1 (brain) | Julia only | Plugin unchanged; enriched response is backwards-compatible |
| 2 (vocabulary) | TypeScript primarily | Consumes sub-PR 1's response shape |
| 3 (persistence) | Julia only | Sidecar-internal; plugin unaware |
| 4 (compaction) | Both | Feature meaningless without both halves; one PR |
| 5 (fail-open) | TypeScript only | Plugin wraps its own HTTP calls |
| 6 (detectors) | Julia only | Posterior-signature checks inside `/evaluate` |

## 5. Sub-PR specifications

### §5.1 — Sub-PR 1: Sidecar brain

**Scope.** Replace the prototype's repetition-counting logic with the
Bayesian EU calculation. The sidecar becomes a Credence application: it
loads the DSL substrate, maintains a posterior over tool-call outcomes per
(tool-name, category) pair, and computes EU for the five actions on each
`/evaluate` request.

Concrete deliverables:

- Posterior state structure: per-(tool-name, category) Beta distributions
  for tool outcomes; per-(model-id, category) Beta distributions for model
  quality. In-memory only (persistence is sub-PR 3).
- EU calculation: for each `/evaluate`, compute EU(proceed), EU(halt),
  EU(downgrade(alt)), EU(route(model)), EU(escalate) using `expect` over
  the relevant posteriors. Return the argmax decision and the signal values
  that drove it (alpha/beta, comparison probability, CV).
- Posterior update: `/observe` calls `condition` with a BetaBernoulli kernel
  to update the relevant (tool-name, category) posterior. Outcome is binary:
  success (no error, duration within budget) vs failure (error present or
  timeout). v0.1 uses a fixed per-tool duration budget table (e.g., `Read`
  500ms, `Bash` 30s, `Edit` 2s). Learned budgets are post-MVP.
- Task-category inference: fixed feature-driven mapping (tool name + file
  extension + command pattern → category). Categories: code, documentation,
  delete, deploy, privileged-exec, dependency, version-control, generic.
- `/compaction-preview` stub: accepts the request, returns `{"status": "ok"}`,
  does nothing. The endpoint exists so sub-PR 4 doesn't need to add a new
  route.
- Enriched `/evaluate` response shape:
  ```json
  {
    "action": "proceed|block|escalate",
    "decision": "proceed|halt|downgrade|route|escalate",
    "reason": "...",
    "signals": {
      "alpha": 1.0, "beta": 1.0,
      "comparison_p": 0.0,
      "cv": 0.0,
      "eu_proceed": 0.0, "eu_halt": 0.0,
      "eu_downgrade": 0.0, "eu_escalate": 0.0
    },
    "requireApproval": null
  }
  ```
  The `action` field preserves backwards compatibility with the Move 1
  plugin (which checks `action === "block"`). The `decision` field carries
  the five-way argmax. `requireApproval` is populated when escalation wins
  (sub-PR 2 reads it).

**Dependencies.** None. Foundational.

**Out of scope.** Plugin-side intervention rendering (sub-PR 2). Persistence
(sub-PR 3). Compaction-survival pattern matching (sub-PR 4). Fail-open
behaviour (sub-PR 5). The three failure-mode detectors (sub-PR 6) — the
brain computes EU, but the detector-specific posterior-signature checks
are sub-PR 6's scope.

**Acceptance criteria.**

- Sidecar starts and all four endpoints respond.
- EU calculation returns sensible values under test posteriors: a fresh
  Beta(1,1) prior yields EU(escalate) as the argmax (high uncertainty →
  ask); a concentrated Beta(50,2) yields EU(proceed) as the argmax (high
  confidence → proceed); a concentrated Beta(2,50) yields EU(halt) as the
  argmax (high failure rate → stop).
- `condition` with BetaBernoulli kernel produces exact alpha/beta updates
  (α+1 on success, β+1 on failure).
- Task-category inference correctly classifies: `Bash` + `git` → version-
  control; `Edit` + `.py` → code; `Bash` + `rm` → delete.
- Move 1 plugin continues working unchanged against the enriched sidecar
  (backwards compatibility).

**Testing approach.** Julia `@assert`/`check` tests: EU calculation against
known posteriors (exact values), category inference table, backwards
compatibility (Move 1 response shape valid as subset).

### §5.2 — Sub-PR 2: Intervention vocabulary

**Scope.** The plugin learns to interpret the sidecar's enriched response
and render all four intervention types as OpenClaw hook return values.

Concrete deliverables:

- Parse the `decision` field from the `/evaluate` response.
- Render veto-and-halt: `{ block: true, blockReason: "..." }` with the
  sidecar's reason string.
- Render veto-and-downgrade: `{ block: true, blockReason: "..." }` with
  the alternative suggestion embedded in the reason.
- Render escalate-to-user-confirmation: `{ requireApproval: { title,
  description, severity, timeoutMs, timeoutBehavior } }` using the
  sidecar's `requireApproval` payload.
- Route-to-cheaper-model: no plugin change needed; routing happens at the
  credence-router proxy level. This sub-PR documents the routing
  configuration in the plugin's README.
- Enriched `/observe` request: include `userApproval` field
  (`boolean | null`) so the sidecar can update the preference posterior
  from escalation responses. Default is `null` (most tool calls are not
  escalation responses); only escalation approve/deny produces `true`/
  `false`. Consumers must not treat `null` as `false`.

**Dependencies.** Sub-PR 1 (the enriched `/evaluate` response to consume).

**Out of scope.** Persistence. Compaction-survival. Fail-open (sub-PR 5
handles unavailability; sub-PR 2 assumes the sidecar is available).

**Acceptance criteria.**

- Each of the four intervention types renders the correct OpenClaw return
  value when triggered by the sidecar's decision.
- The `after_tool_call` handler forwards `userApproval` to the sidecar.
- The plugin gracefully handles a sidecar response missing the enriched
  fields (falls back to Move 1's `action`/`reason` interpretation).
- TypeScript type-checks clean (`tsc --noEmit`).

**Testing approach.** TypeScript tests with a mock HTTP server returning
each decision type; verify exact OpenClaw return value shapes. Integration
test against real sidecar with test posterior.

### §5.3 — Sub-PR 3: Persistence machinery

**Scope.** The sidecar gains load/save for per-user posterior state.

Concrete deliverables:

- State file at `~/.credence/state/posterior.json` (or `CREDENCE_STATE_DIR`
  override). Schema version 1 per the Move 2 design doc's §5.3 spec.
- Bootstrap: on first start with no state file, generate UUID, initialise
  Beta(1,1) priors, write state file with mode 0600.
- Load: on start with existing state file, deserialise and populate the
  in-memory posterior.
- Save: after every `/observe` and `/compaction-preview`, serialise and
  write the state file. Atomic write (write to temp file, rename) to
  prevent torn writes.
- Concurrency: `ReentrantLock` around posterior-modifying operations.
  `/evaluate` reads do not acquire the write lock.
- Schema validation: reject state files with `schema_version > 1` (future
  sidecar version) with a clear error message. Accept `schema_version == 1`.

**Dependencies.** Sub-PR 1 (the posterior state structure to persist).

**Out of scope.** Schema migration logic (v1 → v2 is planned capability,
not v0.1 scope). `registered_instructions` field is present in the schema
but written as empty — sub-PR 4 populates it. Cross-machine sync.

**Acceptance criteria.**

- Sidecar starts cleanly with no state file (bootstrap).
- Sidecar starts cleanly with a valid state file (load).
- Posterior survives sidecar restart: send observations, restart, verify
  alpha/beta values match.
- Concurrent `/observe` requests do not produce torn writes or corrupt
  state.
- State file permissions are 0600.
- State file with `schema_version: 2` is rejected with a clear error.

**Testing approach.** Julia tests for round-trip, bootstrap, and atomic
write. Integration test: observations → restart → verify exact alpha/beta.

### §5.4 — Sub-PR 4: Compaction-survival pattern matching

**Scope.** The Issue #1084 mitigation. The sidecar's `/compaction-preview`
endpoint does real work; the plugin registers on `before_compaction`.

Concrete deliverables (sidecar side):

- Seven regex patterns from the design doc's §5.4 table, implemented as a
  pattern table with (regex, action-class) pairs.
- `/compaction-preview` handler: scan the messages array for pattern matches;
  for each match, register the instruction in `registered_instructions` in
  the state file (deduplicated by pattern + action-class).
- Escalation-threshold elevation: for each registered instruction, shift
  the action-class preference posterior toward uncertainty (equivalent to
  adding prior evidence of "user wants confirmation").
- Instruction decay: when `/observe` reports a user approval for a guarded
  action class, update the posterior toward "user approves this". Retirement
  fires when posterior precision exceeds 10× the prior's precision at
  registration time — i.e., the sidecar has accumulated ten times more
  evidence than the instruction contributed. This mirrors Amendment 1's
  threshold derivation: not a hardcoded floor, but a ratio of posterior
  precision to prior precision.

Concrete deliverables (plugin side):

- Register on `before_compaction` hook.
- Forward the `messages` array to `POST /compaction-preview`.
- The `await` is internal to the handler; OpenClaw does not wait (void hook).

**Dependencies.** Sub-PR 1 (brain), sub-PR 2 (intervention vocabulary —
the elevated threshold triggers escalation, which sub-PR 2 renders),
sub-PR 3 (persistence — registered instructions must survive restarts).

**Out of scope.** LLM-based instruction extraction (v0.2). Cross-machine
instruction sync. Compaction blocking (the hook is void).

**Acceptance criteria.**

- All seven patterns match their canonical examples from the design doc.
- All seven patterns reject plausible non-matches (e.g., "I confirmed the
  deletion" does not match the "confirm before deleting" pattern).
- Registered instructions persist across sidecar restarts.
- Post-registration, a tool call in the guarded action class triggers
  escalation (elevated threshold fires the escalate intervention).
- User approvals decay the instruction: after N approvals, the instruction
  retires and escalation stops firing for the action class.
- The self-tuning property holds: user denials strengthen the instruction
  (posterior shifts toward "user does not want this").

**Testing approach.** Pattern-match unit tests (positive + negative per
pattern). Full-lifecycle integration test: register → escalate → approve →
decay → retire. Three commits: (1) pattern matching, (2) instruction
registration + persistence, (3) decay logic.

### §5.5 — Sub-PR 5: Fail-open behaviour

**Scope.** The plugin's behaviour when the sidecar is unavailable.

Concrete deliverables:

- HTTP timeout of 50ms on `/evaluate` calls (configurable via `timeoutMs`).
- On first sidecar unavailability per session: log a warning via OpenClaw's
  logging mechanism with the sidecar URL and start command.
- On subsequent unavailability in the same session: silent (no warning spam).
- On sidecar becoming available mid-session: log a one-time "governance
  resumed" message.
- State transitions: available → unavailable → available → unavailable all
  handled without crashes or memory leaks.
- Permanent unavailability: degrades to permanent no-op. Plugin does not
  crash OpenClaw.

**Dependencies.** Sub-PR 2 (the plugin's hook implementation that fail-open
wraps). Sub-PR 3 is a soft dependency: the persistence layer determines
what the sidecar's state looks like when the plugin reconnects, but the
plugin doesn't need to know — it just calls `/evaluate` and gets a response.

**Out of scope.** Sidecar process management (auto-start, auto-restart).
Sidecar health monitoring beyond per-call timeout. Separate reconnection
polling.

**Acceptance criteria.**

- Plugin operates correctly when the sidecar is unavailable from session
  start (all tool calls proceed, one warning logged).
- Plugin warns once per session, not per tool call.
- Plugin re-engages cleanly when the sidecar comes back mid-session.
- Plugin handles rapid available↔unavailable transitions without state
  corruption.
- Plugin does not crash OpenClaw under any sidecar-availability sequence.

**Testing approach.** TypeScript tests with a controllable mock HTTP server
covering: unavailable from start, available→unavailable, unavailable→
available ("resumed" message), and rapid toggling (no crashes).

### §5.6 — Sub-PR 6: Three failure-mode detectors

**Scope.** The posterior-signature checks that translate posterior state into
intervention triggers. Each detector is evaluated inside the `/evaluate`
handler and contributes to the EU calculation.

Concrete deliverables (one commit per detector):

**Commit 1: Exec-repetition (#34574).**

- Stationarity condition on per-(tool-name, argument-hash) posterior.
- KL divergence between the posterior before and after the last K
  observations. Threshold proportional to posterior variance.
- K is derived from the posterior: K = ceil(1 / variance) — more
  concentrated posteriors require more observations to confirm stationarity.
- Read-tool exemption list (Read, Grep, LS — configurable).
- When stationarity fires, EU(halt) is elevated to dominate EU(proceed).
  The argmax naturally shifts to halt.

**Commit 2: Compaction-wipes-confirm-instruction (#1084).**

- Integration with sub-PR 4's instruction registry.
- When a registered instruction exists for the candidate's action class,
  EU(escalate) is elevated. The elevation magnitude is derived from the
  instruction's posterior (fresh instruction → strong elevation; decayed
  instruction → weak elevation).
- This detector is thin glue: sub-PR 4 manages the registry; this commit
  wires the registry into the EU calculation.

**Commit 3: No-confidence dreaming loops (#65550).**

- Coefficient-of-variation condition on EU(proceed) across recent
  evaluations.
- The sidecar maintains a sliding window of recent EU(proceed) values
  (window size configurable, default 10).
- When CV exceeds 1/√(α+β) for a sustained span (default 5 consecutive
  evaluations), the no-confidence condition fires.
- When no-confidence fires, EU(halt) is elevated with a different reason
  string than #34574 ("posterior over next-action value is flat" vs
  "posterior stationary on repeated tool call").

**Dependencies.** Sub-PR 1 (brain — the EU calculation these detectors
feed into), sub-PR 3 (persistence — stationarity operates over
observation history), sub-PR 4 (partial — the #1084 detector consumes
the instruction registry).

Sub-PR 2 is a transitive dependency (detectors produce EU values →
brain selects argmax → plugin renders the intervention). Sub-PR 5 is
independent (fail-open wraps the whole plugin; detector internals don't
affect it).

**Out of scope.** Detectors for failure modes beyond the three named.
Semantic task categorisation (the detectors use the feature-driven
categories from sub-PR 1).

**Acceptance criteria.**

- #34574 detector fires on synthetic stationary posterior (10 identical
  observations) and does not fire on 9.
- #34574 detector does not fire on Read/Grep/LS tool calls regardless of
  repetition count.
- #1084 detector fires when a registered instruction exists for the
  candidate's action class and does not fire when no instruction exists.
- #65550 detector fires when CV exceeds threshold for 5+ consecutive
  evaluations and does not fire at 4.
- All three detectors operate independently: one detector's firing does
  not prevent another's. A tool call can trigger both #34574 and #65550
  simultaneously (the argmax resolves which intervention fires).
- End-to-end: a synthetic trajectory exercising each failure mode produces
  the correct intervention.

**Testing approach.** Per-commit unit tests: each detector tested at trigger
and near-miss (one below threshold). End-to-end integration test spanning
all three detectors in a single sidecar session.

## 6. Risks

### Risk 1: Sub-PR 1's scope

The sidecar brain encompasses EU calculation, posterior updates, task-
category inference, and the enriched response shape. This is the largest
single sub-PR.

**Assessment.** The scope is large but cohesive — the components are tightly
coupled (the EU calculation needs the posterior, which needs category
inference, which determines the posterior's key space). Splitting would
produce sub-PRs that each test against stub state and then re-test against
real state when composed. The prototype's server.jl is ~150 lines; the
brain replaces its logic wholesale. A single sub-PR keeps the replacement
atomic.

**Mitigation.** If the sub-PR exceeds ~500 lines of Julia during
implementation, split the EU calculation into its own module file
(`brain.jl`) loaded by `server.jl`. The split is a file-organisation
concern, not a sub-PR-boundary concern — both files land in the same PR.

### Risk 2: Sub-PR 4's coupling

Compaction-survival depends on three prior sub-PRs (brain, vocabulary,
persistence). It is the most coupling-heavy sub-PR.

**Assessment.** The coupling is inherent: the feature spans both languages
(plugin registers on hook, sidecar processes messages), reads persistent
state (registered instructions), and produces an intervention (elevated
escalation threshold). Splitting would not reduce coupling — it would
distribute it across PRs that each test only their half.

**Mitigation.** Sub-PR 4 lands as one PR with three logical commits:
(1) pattern matching + unit tests, (2) instruction registration +
persistence + integration test, (3) decay logic + lifecycle test. Each
commit is independently reviewable; the PR is independently revertable
as a unit.

### Risk 3: Cross-detector dependencies in sub-PR 6

The #1084 detector depends on sub-PR 4's instruction registry. The
#34574 detector depends on sub-PR 3's observation persistence. These
are cross-sub-PR dependencies within the "last" sub-PR.

**Assessment.** The dependencies flow in one direction (detectors read
state that prior sub-PRs produce). No circular dependencies. The risk
is sequencing: if sub-PR 4 ships with a different registry interface
than sub-PR 6 expects, the #1084 detector commit needs rework.

**Mitigation.** Sub-PR 4's acceptance criteria include the instruction
registry's API contract (a function that returns registered instructions
for a given action class). Sub-PR 6's #1084 commit consumes this
contract. The contract is specified in sub-PR 4's PR description and
tested by sub-PR 4's own tests.

### Risk 4: Language-side coordination

TypeScript plugin and Julia sidecar evolve in parallel across sub-PRs.
The IPC contract is the seam.

**Assessment.** The contract changes are additive (new fields, not changed
fields) and backwards-compatible (the Move 1 `action`/`reason` shape
remains valid). The main coordination risk is sub-PR 4, which changes
both sides simultaneously.

**Mitigation.** See §4 above. Sub-PR 4 is the only dual-language sub-PR
with tightly coupled changes. Its three-commit structure separates the
Julia-side pattern matching (commit 1) from the TypeScript-side hook
registration (commit 2) from the decay logic (commit 3, Julia only).

### Risk 5: Prototype replacement regression

Sub-PR 1 replaces the prototype's repetition-counting logic. The
stationarity detector (sub-PR 6) subsumes it, but ships later. Sub-PR 1
preserves the repetition counter as a gated fallback, removed when
sub-PR 6 ships the stationarity detector.

## 7. Phasing summary

```
Sub-PR 1 (sidecar brain) ─── foundational, no dependencies
    │
    ├── Sub-PR 2 (intervention vocabulary) ─── depends on 1
    │       │
    │       └── Sub-PR 5 (fail-open behaviour) ─── depends on 2
    │
    ├── Sub-PR 3 (persistence machinery) ─── depends on 1
    │       │
    │       └──┐
    │          │
    │   Sub-PR 4 (compaction-survival) ─── depends on 1, 2, 3
    │          │
    │          └── Sub-PR 6 (three detectors) ─── depends on 1, 3, 4
    │
    └── (sub-PRs 2 and 3 are parallelisable after 1)
```

**Proposed ordering:**

1. Sub-PR 1: Sidecar brain
2. Sub-PRs 2 + 3: Intervention vocabulary + Persistence (parallel)
3. Sub-PR 4: Compaction-survival
4. Sub-PR 5: Fail-open (can land any time after 2; sequenced here for
   narrative clarity, but can be parallelised with 4)
5. Sub-PR 6: Three failure-mode detectors

**Validation.** Sub-PR 2's tests use a mock sidecar with stub posteriors,
so persistence is orthogonal — the parallel ordering of 2+3 stands. Sub-PR
5 wraps the plugin's HTTP call, independent of sidecar internals; it can
land any time after 2. No circular dependencies — the DAG is strict.

## 8. Acceptance criteria for this document

- Six §5.x sub-PR specifications with scope, dependencies, out-of-scope,
  acceptance criteria, and testing approach.
- Phasing summary with dependency DAG, validated against the proposed
  ordering.
- Risks named with assessments and mitigations.
- The document cites Move 2's design doc (PR #86) and amendments (PR #87).
- No code changes.
