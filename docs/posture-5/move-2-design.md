# Move 2 — Plugin v0.1

## 1. Strategic context

Move 1 (PR #85) settled the integration path: OpenClaw's `before_tool_call`
hook natively supports block/veto and user-confirmation escalation; the sidecar
IPC architecture works at 5% of the latency budget (4.7ms vs 100ms); plugins
cannot write to MEMORY.md or SQLite, so posterior persistence is sidecar-managed.
The prototype demonstrated loop-detection veto end-to-end.

Move 2 converts the prototype into a product. It is the substantial design
conversation of Posture 5 — the integration path is settled; what remains is the
architecture that gives the integration path its brain. After Move 2, there is
something a user can install and use.

The strategic frame is unchanged: Bayesian governance sidecar for OpenClaw, not
a routing gateway. Routing is a feature inside governance. The empty competitive
quadrant is in-loop Bayesian governance: decision-theoretic intervention on
individual tool-call decisions with a continuously-updated posterior, operating
between gateway-level routers and harness-level guardrails.

One non-negotiable settled in conversation: **the agent is per-user.** A single
continuous belief state accumulates across every OpenClaw session, every
workspace, every task. Not per-session, not per-workspace. This is the
architectural claim that makes the product longitudinally valuable — the agent
improves over time because its posterior encodes the user's cumulative
experience.

## 2. Scope

1. Specify EU semantics for each of the four interventions (§5.1).
2. Specify detection logic for the three MVP failure modes (§5.2).
3. Design the sidecar persistence architecture for per-user state (§5.3).
4. Design the compaction-survival mechanism for Issue #1084 (§5.4).
5. Specify user identification and privacy properties (§5.5).
6. Specify failure-open behaviour and re-engagement (§5.6).

## 3. Out of scope

- Move 2 implementation. This PR is the design doc only.
- Publication track (Papers One and Two).
- Multi-harness adapters (Anthropic Agent SDK, OpenHands V1).
- Personal-agent direction (Posture 6+).
- Cache-aware governance via native Messages API.
- Learned-pattern instruction extraction (§5.4 v0.2 direction).
- Cross-machine state sync (§5.3 v2 direction).
- Semantic task categorisation (v0.2; §5.2 uses feature-driven categorisation).
- Changes to Credence's existing Julia substrate.

## 4. Dependencies on Move 1

This design builds on six findings from Move 1 (PR #85):

| Finding | Source | Implication |
|---|---|---|
| `before_tool_call` supports `block: true` and `requireApproval` | §5.1 hook analysis | Veto and escalate are native; no workarounds |
| `after_tool_call` is void/fire-and-forget | §5.1 hook analysis | Posterior updates don't add tool-call latency |
| `before_compaction` receives the full messages array | Callsite verification | Plugin can inspect about-to-be-compacted context |
| Plugins cannot write MEMORY.md or SQLite | §5.1 persistence analysis | Posterior persistence is sidecar-managed |
| Sidecar latency at 4.7ms per call | Prototype measurement | 95% of 100ms budget available for the brain |
| OpenAI-compatible proxy is a first-class provider | §5.1 model provider analysis | credence-router slots in for model routing |

## 5. Design decisions

### §5.1 — The four interventions: EU semantics

Each intervention is a decision the sidecar makes by comparing the expected
utility of the candidate tool call against the expected utilities of the
alternatives. The sidecar maintains a posterior over tool-call outcomes; the
EU calculation integrates over this posterior with the user's (inferred)
utility function.

The sidecar's decision problem for each `POST /evaluate` request:

```
Space:   A = {proceed, halt, downgrade(alt), route(model), escalate}
Measure: posterior over (tool-outcome | candidate, history, user)
Kernel:  outcome-likelihood model conditioned on tool identity and history
Preference: utility function over outcomes (user-inferred)
Decision: argmax_{a ∈ A} EU(a)
```

#### Veto-and-halt

**Posterior signal.** Fires when EU(halt) > EU(proceed) and EU(halt) > EU(a)
for all alternatives a. The brain's posterior over "expected value of the next
tool call" has degraded — the agent's recent history shows diminishing returns
(repetition without progress, error accumulation, or belief convergence to
low-value outcomes).

The degradation signal is a stationarity condition on the posterior, not a
hardcoded count: if the posterior over tool-call outcome value has not shifted
across the last K observations (where K is the number of observations needed
to detect stationarity at the current posterior variance), the evidence favours
halt. This subsumes the prototype's repetition counter — identical tool calls
produce identical observations, which produce a stationary posterior, which
triggers halt.

**OpenClaw return value.**
```typescript
{
  block: true,
  blockReason: "Credence: expected utility of continuing has degraded below
    idle. The posterior over tool-call value has been stationary for [K]
    observations. Halting to prevent runaway loop."
}
```

**User-facing behaviour.** The agent stops its current tool-call trajectory.
The user sees the block reason in their OpenClaw session. The agent may
re-plan with a different approach if it has the context to do so; the
governance plugin does not prevent re-planning, only the specific tool call
that was vetoed.

#### Veto-and-downgrade

**Posterior signal.** Fires when an alternative tool call's EU exceeds the
candidate's: EU(downgrade(alt)) > EU(proceed) by a margin that exceeds the
posterior's uncertainty about the comparison. The margin condition prevents
triggering on noise — the sidecar only vetoes when the alternative is
confidently better, not when the comparison is ambiguous.

The margin is derived from the posterior itself: the sidecar computes
P(EU(alt) > EU(proceed)) under the current posterior. Downgrade fires when
this probability exceeds 1 − 1/(α + β), where α and β are the Beta
parameters of the tool-success belief for the candidate action class. A
fresh prior (α = β = 1) yields a threshold of 0.5 — downgrade fires on
any evidence of alternative superiority, which is correct: the sidecar has
no reason to prefer the candidate. As observations accumulate, the
threshold rises toward 1, requiring stronger evidence to override a
well-established tool. This is not a tuning parameter; it is a consequence
of demanding that the downgrade decision's confidence scale with the
posterior's own precision.

In v0.1, the alternative space is limited: the sidecar considers only the
"read instead of exec" substitution (the dominant pattern from Issue #34574)
and "less-privileged tool" substitutions (e.g., `Bash` → `Read` when the
command is a read operation). The alternative space grows in v0.2 with learned
tool-substitution patterns.

**OpenClaw return value.**
```typescript
{
  block: true,
  blockReason: "Credence: EU of [alternative] exceeds EU of [candidate]
    (P=0.93). Suggestion: [specific alternative description]."
}
```

**User-facing behaviour.** The agent sees the block with a suggestion. OpenClaw
doesn't automatically execute the alternative — it blocks the candidate and the
agent re-plans. The suggestion in the block reason guides re-planning. This is
deliberate: the governance plugin advises, the agent decides.

#### Route-to-cheaper-model

**Posterior signal.** Fires when the posterior over model quality for the
current task category supports using a cheaper model without expected quality
degradation. The comparison is EU(route(cheaper)) ≥ EU(proceed) — note ≥, not
strictly >. Routing to a cheaper model is preferred when EU is equal because
the cost saving is a tiebreaker in the utility function.

This intervention operates at the model-provider level, not the tool-call level.
credence-router, configured as an OpenAI-compatible provider in OpenClaw's
`models.providers`, receives all LLM requests and routes them based on its own
posterior. The governance plugin's `before_tool_call` hook is not involved in
model routing — routing happens transparently at the provider layer.

The routing posterior is the same posterior that drives the other interventions.
The per-model, per-category beliefs about quality are updated after each tool
call's outcome observation. When the posterior's belief about Haiku-quality-on-
this-category is high enough that EU(Haiku) ≥ EU(Sonnet), routing selects Haiku.

**OpenClaw return value.** None from the plugin. The routing happens at the
credence-router proxy level, transparent to OpenClaw. The user configures
credence-router as their model provider and routing happens automatically.

**User-facing behaviour.** Invisible. The user's tool calls work as before; only
the per-call cost changes. The user can observe routing decisions in
credence-router's logs.

#### Escalate-to-user-confirmation

**Posterior signal.** Fires when the EU calculation is dominated by uncertainty
about user preference. The sidecar computes the posterior variance of the utility
function for the proposed action's class. When the variance-to-mean ratio of
EU(proceed) exceeds a threshold, the sidecar cannot confidently determine whether
the user would want this action — the correct response is to ask.

The threshold is derived from the posterior's own concentration. The sidecar
computes the coefficient of variation (CV) of EU(proceed) under the current
posterior. When CV exceeds 1/√(α + β) — where α and β are the Beta
parameters of the action-class preference belief — escalation fires. A fresh
prior (α = β = 1) yields CV threshold 1/√2 ≈ 0.71, making escalation easy
to trigger, which is correct: the sidecar has little evidence about user
preference and should ask. As observations accumulate, √(α + β) grows and
the threshold tightens, requiring proportionally less uncertainty to stay
silent. This mirrors the downgrade threshold's derivation: both are
consequences of scaling decision confidence with posterior precision, not
independent tuning parameters.

The escalation threshold is also elevated by the compaction-survival mechanism
(§5.4): when the sidecar detects a lost user instruction, it raises the
threshold for the relevant action class, making escalation more likely for
actions the lost instruction would have guarded.

**OpenClaw return value.**
```typescript
{
  requireApproval: {
    title: "Credence governance check",
    description: "The proposed action [tool call description] has uncertain
      expected utility. [Specific reason — e.g., 'this action class has not
      been observed before' or 'a confirm-before instruction was recently
      compacted']. Confirm to proceed.",
    severity: "warning",
    timeoutMs: 120000,
    timeoutBehavior: "deny"
  }
}
```

**User-facing behaviour.** OpenClaw renders its native confirmation dialog. The
user sees the title, description, and can approve or deny. Timeout defaults to
deny (conservative — if the user doesn't respond, the action is blocked). The
user's response is observed by the sidecar via `after_tool_call` and updates
the posterior: approval increases the belief that this action class is desirable;
denial decreases it.

### §5.2 — The three failure modes: detection logic

Each failure mode has a posterior signature (the belief state that characterises
the failure) and an intervention pattern (which of the four interventions
responds to it). The architecture supports adding new failure-mode detectors as
additional patterns without rearchitecture — each detector reads the same
posterior and triggers the same intervention vocabulary.

#### Issue #34574: exec repetition

**Posterior signature.** The posterior over tool-call outcome value for a
specific (tool-name, argument-hash) pair has converged to a stationary
distribution — repeated observations of the same tool call producing the same
outcome have not shifted the belief. The stationarity condition is: the KL
divergence between the posterior before and after the last K observations is
below a threshold proportional to the posterior's current variance.

**Detection logic.** The sidecar tracks, for each (tool-name, argument-hash)
pair, the posterior over outcome value. After each `POST /observe`, it updates
this posterior via `condition`. On each `POST /evaluate`, it checks the
stationarity condition on the candidate's (tool-name, argument-hash) pair.

**Intervention.** Veto-and-halt when the stationarity condition fires. The
block reason names the specific tool call and the number of observations since
the posterior last shifted.

**Exemption.** `Read` tool calls are exempted from the repetition detector (as
in the prototype). Reading a file repeatedly is normal agent behaviour; the
detector should not fire on `Read(path)` even if the path is the same. Other
read-like tools (e.g., `Grep`, `LS`) are similarly exempted. The exemption
list is configurable.

**Task-category inference.** The sidecar infers the task category from
feature-driven signals: tool name, file extension in arguments (`.py` → code,
`.md` → documentation), command patterns in `Bash` arguments (`git` → version
control, `npm`/`pip` → dependency management). v0.1 uses a fixed mapping; v0.2
may add semantic inference.

#### Issue #1084: compaction-wipes-confirm-instruction

**Posterior signature.** The sidecar's internal registry of "active user
instructions" has at least one entry whose source message is in the about-to-be-
compacted portion of the context. Post-compaction, the agent will not have access
to this instruction, but the sidecar's posterior still encodes it.

**Detection logic.** The `before_compaction` hook fires with the full messages
array before compaction begins. The plugin sends the messages to the sidecar
via a new `POST /compaction-preview` endpoint. The sidecar scans the messages
for instruction patterns (§5.4) and registers any found instructions in its
persistent state. For each registered instruction, the sidecar elevates the
escalation threshold for the relevant action class.

Post-compaction, when the agent attempts an action that would have been guarded
by a registered instruction, the elevated threshold triggers
escalate-to-user-confirmation. The user sees the confirmation prompt that the
lost instruction would have produced.

**Intervention.** Escalate-to-user-confirmation. The confirmation description
names the registered instruction that triggered the escalation (e.g., "a
confirm-before-deleting instruction was recently compacted; confirming before
this delete operation").

**Instruction decay.** Registered instructions have a half-life: their
influence on the escalation threshold decays over time (measured in
observations, not wall-clock). After enough observations where the user
approves actions in the guarded class, the sidecar's posterior converges to
the user's revealed preference and the elevated threshold subsides. This is
Bayesian — the instruction is prior evidence, not a permanent constraint.

#### Issue #65550: no-confidence dreaming loops

**Posterior signature.** The posterior over "expected utility of the next
action" has high variance and low absolute values — the brain's belief about
what to do next is essentially flat. The agent is not stuck in a loop (that
would be #34574); it is thrashing between different low-value actions without
converging.

**Detection logic.** The sidecar tracks the running variance of EU(proceed)
across recent evaluations. When the coefficient of variation (stddev / |mean|)
of EU(proceed) exceeds a threshold for a sustained span (configurable, default
5 consecutive evaluations), the no-confidence condition fires.

**Intervention.** Veto-and-halt with a different block reason than #34574. The
block reason says "the posterior over next-action value is flat — the agent is
exploring without converging. Halting to prevent resource waste." If the user
restarts the agent with clearer instructions, the sidecar's posterior updates
accordingly.

**Distinction from #34574.** Exec-repetition fires on a specific
(tool-name, argument-hash) pair's stationarity. No-confidence fires on the
global EU distribution's flatness. An agent that calls different tools but none
with positive EU triggers #65550 but not #34574.

### §5.3 — Sidecar persistence architecture

#### Storage location

**Default:** `~/.credence/state/` on all platforms. Per-OS-user by default
(each OS user's home directory is separate). This follows the user-home
convention used by OpenClaw itself (`~/.openclaw/`), Credence's existing config
(`~/.credence/`), and most CLI tools.

**Override:** `CREDENCE_STATE_DIR` environment variable. Allows operational
flexibility (shared server, custom directory, external volume mount).

**State file:** `~/.credence/state/posterior.json` (schema-versioned). A single
file containing the complete posterior state. The file is small (the posterior
is parametric — alpha/beta pairs, not sample arrays) and serialises/deserialises
in sub-millisecond time.

#### Storage format

Reuse Credence's existing schema-versioned persistence machinery from Posture 3.
The state file carries a `schema_version` field; the sidecar's loader dispatches
on this field to apply migrations. This is the same pattern used for the v3
fixtures in `test/fixtures/`.

v0.1 schema (version 1):

```json
{
  "schema_version": 1,
  "user_id": "uuid",
  "created_at": "iso8601",
  "updated_at": "iso8601",
  "posterior": {
    "tool_outcomes": {
      "<tool-name>": {
        "<category>": { "alpha": 1.0, "beta": 1.0 }
      }
    },
    "model_quality": {
      "<model-id>": {
        "<category>": { "alpha": 1.0, "beta": 1.0 }
      }
    }
  },
  "registered_instructions": [
    {
      "pattern": "confirm before deleting",
      "action_class": "delete",
      "source_session": "session-id",
      "registered_at": "iso8601",
      "observations_since": 0
    }
  ],
  "observation_count": 0,
  "session_count": 0
}
```

The posterior uses Beta distributions throughout — the natural conjugate prior
for binary outcome signals (tool call succeeded/failed, model quality
adequate/inadequate). Each (tool-name, category) and (model-id, category) pair
has its own Beta posterior, updated via Credence's `condition` with a
BetaBernoulli kernel.

#### Concurrency model

**Serialised updates with in-process mutex.** The sidecar is a single Julia
process. Multiple OpenClaw instances on the same machine connect to the same
sidecar via HTTP. The sidecar serialises all posterior-modifying operations
(`POST /observe`, `POST /compaction-preview`) behind a `ReentrantLock`. Reads
(`POST /evaluate`) acquire a read lock. The locking cost is sub-microsecond —
invisible within the 100ms latency budget.

If two OpenClaw instances send concurrent `/observe` requests, one blocks
until the other completes. At 4.7ms per request, the maximum queue delay is
one request's duration — acceptable for v0.1.

**Consistency model.** Read-your-writes within a single OpenClaw instance
(the instance's `/evaluate` after its `/observe` sees the updated posterior).
Eventually consistent across instances (instance A's `/observe` may not be
visible to instance B's immediately-following `/evaluate` if B's request
arrives during A's write lock).

#### Multi-machine state

v0.1: **one sidecar per machine, posterior per machine.** A user with multiple
machines has approximately-the-same-agent — each machine's sidecar accumulates
observations independently. The posteriors diverge as the machines see different
workloads.

Cross-machine sync is post-MVP. When it's needed, the architecture supports
it: the state file is a single JSON file with a schema version. A sync
mechanism (rsync, cloud storage, remote sidecar) could merge posteriors from
multiple machines. The merge operation is well-defined for Beta distributions:
two Beta posteriors with the same prior can be merged by summing their
sufficient statistics (total alphas and betas minus the shared prior).

#### Bootstrap

First-time sidecar start with no existing state file:

1. Generate a new UUID for `user_id`.
2. Initialise all posteriors to Beta(1, 1) — the uniform prior, expressing
   no opinion about tool-call outcomes or model quality.
3. Initialise `registered_instructions` to empty.
4. Write the state file.

The cold-start period has analogous behaviour to the routing benchmark's
early workloads: the agent explores without confident intervention, and
interventions become more calibrated as observations accumulate. The user
should expect a settling-in period of approximately 20–50 tool calls before
the posterior is informative enough for confident veto decisions (this
estimate comes from the routing benchmark's observation that Haiku's posteriors
moved from 0.5 to 0.7+ within 5 workloads of ~5 turns each).

User-facing documentation should set expectations: "Credence improves over
time. During the first few sessions, interventions are less specifically
calibrated as the agent learns your patterns."

#### Migration

When the sidecar is upgraded to a new schema version, the loader detects the
version mismatch and applies migrations. The migration path is:

1. Read the state file.
2. Check `schema_version`.
3. If version < current, apply migrations sequentially (v1 → v2 → v3 → ...).
4. Write the migrated state file.

v0.1 → v0.2 migration is a planned capability. The state file's
schema-versioning ensures that upgrades never lose state. Downgrades are not
supported (the sidecar refuses to load a state file with a higher schema
version than it understands and reports a clear error).

### §5.4 — Compaction-survival mechanism

#### What the sidecar receives

The plugin registers on `before_compaction`. When the hook fires, it receives
the full messages array (confirmed by callsite inspection: the `messages` field
at line 29 of `pi-embedded-subscribe.handlers.compaction.ts` passes
`ctx.params.session.messages` directly). The plugin forwards this array to the
sidecar via `POST /compaction-preview`.

The `before_compaction` hook is void/fire-and-forget, which means the plugin
cannot block or modify the compaction. The plugin's role is observational: it
inspects the context and updates the sidecar's state before the compaction
removes the context. The sidecar must process the compaction preview
asynchronously — the compaction proceeds regardless of whether the sidecar has
finished processing.

#### What the sidecar does

1. **Scan for instruction patterns.** The sidecar scans the messages for
   user-instruction patterns. v0.1 uses a fixed set of regex patterns:

   | Pattern | Action class | Example match |
   |---|---|---|
   | `confirm before (deleting\|removing\|dropping)` | delete | "please confirm before deleting any files" |
   | `(don't\|do not\|never) (delete\|remove\|drop)` | delete | "don't delete anything without asking" |
   | `(always\|must) ask before` | any-destructive | "always ask before running destructive commands" |
   | `confirm before (pushing\|deploying\|merging)` | deploy | "confirm before pushing to production" |
   | `(don't\|do not\|never) (push\|deploy\|merge) (to\|into) (main\|master\|prod)` | deploy | "never push to main without review" |
   | `(don't\|do not\|never) run .* (sudo\|as root)` | privileged-exec | "don't run anything as root" |
   | `(don't\|do not\|never) (install\|add\|upgrade) .*package` | dependency | "don't install new packages without asking" |

   The pattern set is deliberately small and conservative for v0.1. False
   negatives (missing a real instruction) are the expected failure mode; false
   positives (registering a non-instruction) are low-cost because the
   instruction decays via Bayesian updating.

2. **Register matched instructions.** For each matched instruction, the sidecar
   adds an entry to `registered_instructions` in the state file (if not already
   present). Each entry records the pattern, the action class it guards, the
   source session, and the observation count since registration.

3. **Elevate escalation threshold.** For each registered instruction, the
   sidecar raises the escalation threshold for the relevant action class. The
   mechanism: a registered instruction acts as prior evidence that the user
   wants confirmation for this action class. This shifts the posterior toward
   higher uncertainty about user preference, making the "two-sided uncertainty"
   condition (§5.1, escalate) more likely to fire.

4. **Persist.** The sidecar writes the updated state file. Since
   `before_compaction` fires before the compaction removes context, the
   sidecar's state captures the instruction before it's lost.

#### Instruction decay

Registered instructions are not permanent constraints. They are prior evidence
in the Bayesian sense — the instruction shifts the posterior, but subsequent
observations can shift it back. The decay mechanism:

- Each time the user approves an action in the guarded class (via the
  escalation confirmation dialog), the sidecar observes "user wants this action"
  and updates the posterior accordingly.
- After enough approvals, the posterior converges away from the elevated
  threshold and escalation no longer fires for that action class.
- The `observations_since` counter in the registered instruction tracks this.
  When the posterior's belief about user preference for this class converges
  (posterior variance drops below a threshold), the registered instruction is
  retired from the active list.

This is the correct Bayesian treatment: the instruction is evidence, not
dogma. If the user's behaviour after compaction consistently approves actions
in the guarded class, the agent learns that the instruction is no longer
load-bearing and stops escalating.

#### Known limitations

v0.1's pattern matching misses instructions phrased in unusual ways. Examples:
- "be careful with file system operations" (no keyword match)
- "treat this repo like production" (implied constraint, no explicit action)
- "I'm nervous about this — double-check everything" (emotional signal, not
  instruction pattern)

v0.2 direction: use an LLM (local or via the sidecar's model provider) to
extract instruction semantics from the about-to-be-compacted context. This adds
latency (~500ms for a local model, ~2s for a remote model) but catches
instructions that regex cannot. The `before_compaction` hook is fire-and-forget,
so the latency doesn't block compaction — but the sidecar must handle the case
where compaction completes before the LLM finishes processing.

### §5.5 — User identification

#### Default identification

On first sidecar launch, the sidecar generates a UUID v4 and writes it to the
state file as `user_id`. All subsequent sidecar starts read this ID from the
state file. The user ID is an opaque identifier with no PII — it does not
encode the user's name, email, machine, or any other identifying information.

"The user" is defined as "whoever has access to this state file." This is the
correct security property: the agent's belief state is a private artefact of
the user, protected by filesystem permissions. The state file is created with
mode 0600 (owner read/write only).

#### Multiple users on one machine

Each OS user has their own home directory and therefore their own
`~/.credence/state/` directory. Two OS users on the same machine get separate
sidecars and separate posteriors automatically. No additional configuration
needed.

For finer-grained separation within a single OS user (e.g., work vs. personal
OpenClaw configurations), the `CREDENCE_STATE_DIR` override provides the
mechanism: set different state directories in different shell profiles or
OpenClaw configurations.

#### State file portability

A user moving to a new machine copies `~/.credence/state/posterior.json` to the
new machine's `~/.credence/state/`. The sidecar on the new machine loads the
state file and continues with the existing posterior. The `user_id` stays the
same — it is a property of the agent, not the machine.

No export/import ceremony. The state file is a single JSON file; `cp` or
`rsync` suffices. The schema-versioned format ensures that a state file created
by sidecar v0.1 is loadable by sidecar v0.2 (with automatic migration).

#### Privacy properties

- The state file contains no PII. It stores parametric posterior parameters
  (alpha/beta values), tool-call category statistics, and instruction patterns.
  It does not store tool-call arguments, file contents, conversation transcripts,
  or any other content from the user's sessions.
- The state file is not transmitted anywhere. The sidecar is a local process;
  no telemetry, no cloud sync, no remote API calls (except to model providers
  configured by the user).
- The state file's `user_id` is an opaque UUID. It cannot be used to identify
  the user externally.
- Deleting the state file resets the agent to cold-start. The user can "forget
  everything" by deleting one file.

### §5.6 — Failure-open behaviour

#### Detection

The plugin detects sidecar unavailability via HTTP connection timeout. The
timeout budget is 50ms (configurable via `timeoutMs` in plugin config),
well within the 100ms `before_tool_call` latency budget. If the sidecar
doesn't respond within the timeout, the plugin treats it as unavailable.

No process-not-running check. The HTTP timeout catches all failure modes
(process not running, process hung, network failure for a remote sidecar)
with a single mechanism.

#### User notification

On first sidecar unavailability in a session, the plugin logs a warning via
OpenClaw's logging mechanism:

```
[credence] Governance sidecar unavailable at [url]. OpenClaw is running
without Credence governance protection. Start the sidecar with:
  julia ~/.credence/sidecar/server.jl
```

The warning appears once per session. Subsequent unavailability in the same
session is silently absorbed (no warning spam). If the sidecar becomes
available later in the session, the plugin logs a one-time "governance
resumed" message.

#### Persistent failure

If the sidecar is permanently unavailable (uninstalled, state corruption,
configuration error), the plugin degrades to a permanent no-op. It does not
crash OpenClaw. The user's agent runs without governance protection indefinitely.

The plugin does not attempt to repair the sidecar or auto-install it. The
user is responsible for sidecar lifecycle. This is the correct boundary for
v0.1 — the plugin is a thin integration shim, not a process manager.

#### Re-engagement

The plugin checks sidecar availability on every `before_tool_call` invocation.
There is no separate polling or reconnection mechanism. If the sidecar comes
back during a running session, the very next tool call triggers a successful
`POST /evaluate`, the plugin resumes governance, and the "governance resumed"
log message fires.

This means re-engagement latency is bounded by the inter-tool-call interval
(typically seconds). No background polling needed.

#### Partial failure

If the sidecar responds to `/evaluate` but the subsequent `/observe` fails
(e.g., the sidecar crashes between the two calls), the governance decision
was already made and the observation is lost. The posterior is slightly stale
(one observation behind). This is acceptable — the posterior is robust to
individual lost observations.

## 6. Risks

### Risk 1: Pattern-match coverage is incomplete

v0.1's regex approach to identifying user instructions in compacted context
will miss instructions phrased in unusual ways. The v0.1 pattern set (§5.4)
covers the seven most common instruction phrasings; unusual phrasings are not
caught.

**Mitigation.** The pattern set is conservative (low false-positive rate).
Missing an instruction degrades to OpenClaw's default behaviour (no governance
protection for that instruction), which is the pre-Credence baseline.
User-facing documentation names this limitation explicitly. v0.2's LLM-based
instruction extractor addresses it.

### Risk 2: Task-category inference is feature-driven only

The four-intervention vocabulary depends on the brain having beliefs about
specific (tool, task-category) pairs. v0.1's category inference uses
feature-driven signals (tool name, file extension, command patterns). Semantic
categorisation (understanding what the tool call is *doing*, not just what
tool it uses) is v0.2.

**Mitigation.** Feature-driven categorisation covers the dominant patterns:
`Bash` with `git` commands → version-control, `Edit` on `.py` files → code,
`Bash` with `rm` → delete. The category space is small (code, documentation,
delete, deploy, privileged-exec, dependency, version-control, generic). Most
tool calls fall into a category that feature-driven inference identifies
correctly. Misclassified tool calls use the "generic" category's posterior,
which is less specifically calibrated but not wrong.

### Risk 3: Concurrent posterior-update conflicts

Multiple OpenClaw instances sending concurrent `/observe` requests to the same
sidecar. The serialised-update design (§5.3) handles this correctly but
introduces queuing delay.

**Mitigation.** Maximum queue delay is one request's duration (~5ms). For
typical usage (1–3 concurrent instances), the delay is invisible. If a user
runs many concurrent instances (>10), the queue delay may become noticeable
(~50ms). This is an edge case; the design doc names it but does not optimise
for it.

### Risk 4: Escalation UX is OpenClaw's responsibility

The plugin returns the `requireApproval` shape; OpenClaw renders the
confirmation dialog. The plugin cannot control the rendering quality.

**Mitigation.** Move 1's investigation confirmed that OpenClaw's confirmation
rendering includes title, description, and severity. This is sufficient for
v0.1. The `timeoutBehavior: "deny"` default ensures that unanswered
escalations are conservative (block rather than proceed). If OpenClaw's UX
for confirmation dialogs changes in a future version, the plugin's rendering
may need adjustment.

### Risk 5: Cold-start behaviour

A freshly-installed Credence has Beta(1,1) priors throughout. During the
cold-start period (~20–50 tool calls), the posterior is not informative enough
for confident interventions. The agent may under-intervene (missing loops that
a calibrated posterior would catch) or, less likely, over-intervene (triggering
escalation on benign actions due to high prior uncertainty).

**Mitigation.** User-facing documentation sets expectations for the settling-in
period. The cold-start behaviour is analogous to a new colleague who asks more
questions initially and becomes more confident over time. The Beta(1,1) prior
is the correct uninformative prior — it expresses genuine ignorance, which is
the honest state of a freshly-installed agent.

### Risk 6: Multi-machine posterior divergence

A user with multiple machines has approximately-the-same-agent. Each machine's
sidecar accumulates observations independently, producing divergent posteriors
over time.

**Mitigation.** The divergence is bounded: both machines see the same user
and the same general workload patterns. The posteriors converge to similar
regions of the parameter space even without sync. Cross-machine sync is
post-MVP; the state file's schema-versioning and Beta-distribution sufficient
statistics make future merge well-defined.

### Risk 7: `before_compaction` hook is fire-and-forget

The plugin cannot block compaction to ensure the sidecar has finished
processing. If compaction completes before the sidecar processes the
compaction preview, the instruction registration may race with the
post-compaction tool calls.

**Mitigation.** The sidecar processes the compaction preview synchronously
within the `/compaction-preview` handler. The HTTP request completes when
processing finishes. The plugin sends the request and awaits the response
before returning from the hook handler. Since the hook is void, OpenClaw
doesn't wait for the plugin's return — but the sidecar's state is updated
as soon as the HTTP response returns. The race window is the time between
the sidecar's response and the next `before_tool_call` evaluation, which
is typically milliseconds. If a tool call arrives during the race window,
it evaluates against a stale (pre-compaction) posterior, which is
conservative (the elevated threshold isn't yet active, so the tool call
proceeds without escalation). The next tool call sees the updated posterior.

## 7. Test plan

### Unit tests (sidecar)

- Posterior update: verify that `condition` with a BetaBernoulli kernel produces
  the expected alpha/beta values after N observations.
- Stationarity detection: verify that the KL-divergence-based stationarity
  condition fires after K identical observations and does not fire after
  K - 1.
- No-confidence detection: verify that the coefficient-of-variation condition
  fires when EU variance is high and absolute values are low.
- Instruction pattern matching: verify each of the seven patterns against
  positive and negative examples.
- Instruction decay: verify that registered instructions retire after
  sufficient user approvals.
- State file round-trip: verify that serialise → deserialise → serialise
  produces identical output.
- Schema migration: verify that a v0 state file (or missing state file)
  migrates cleanly to v1.

### Integration tests (plugin + sidecar)

- Loop detection end-to-end: same tool call repeated N+1 times, verify
  that the (N+1)th is blocked.
- Compaction survival: register an instruction via compaction preview,
  verify that the post-compaction tool call triggers escalation.
- Fail-open: sidecar not running, verify that tool calls proceed without
  intervention and the warning is logged once.
- Re-engagement: start sidecar mid-session, verify that the next tool call
  triggers governance and the "resumed" message fires.
- Concurrent instances: two plugin instances sending alternating requests,
  verify that both see consistent posterior updates.

### Manual verification

- Install the plugin in a real OpenClaw instance.
- Run a task that triggers each of the three failure modes.
- Verify that the intervention fires and the user experience matches the
  design.

## 8. Acceptance criteria for Move 2 implementation

- All four interventions implemented: veto-and-halt, veto-and-downgrade,
  route-to-cheaper-model (via credence-router config), escalate-to-user-
  confirmation.
- All three failure-mode detectors implemented: exec-repetition (#34574),
  compaction-wipes-confirm-instruction (#1084), no-confidence dreaming
  loops (#65550).
- Sidecar persistence: per-user state file at `~/.credence/state/`, schema-
  versioned, concurrent-safe, bootstrap from empty, migration-ready.
- Compaction survival: `before_compaction` hook scans for instruction
  patterns, registers them in sidecar state, post-compaction escalation
  fires for guarded action classes.
- Fail-open: sidecar unavailability → no intervention + one-time warning;
  re-engagement on sidecar restart.
- Installable plugin with documented setup instructions.
- CI passes; existing functionality unchanged.
- Prototype's loop-detection behaviour preserved as a special case of
  the stationarity detector.
