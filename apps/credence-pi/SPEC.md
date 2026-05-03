# Credence-Pi — Pass 1 Specification

## Status

This document is the binding specification for `apps/credence-pi/`. It is written from first principles around the body-brain architecture; it is not an extension of any prior credence integration, and `apps/credence-governance-sidecar/` (the OpenClaw integration) is not a model. That sidecar remains in place untouched as an artefact, but its design — host-side EU arithmetic with pragma armour, hand-rolled feature partitions, multi-endpoint HTTP — is precisely what this branch avoids.

## First principles

Before any code, the architecture. Five commitments, in dependency order. Each is binding.

### 1. The brain is opaque to the body

The body sends sensor data to the brain. The brain returns effector signals. The body has no access to the brain's posterior, no access to its EU calculations, no access to its working hypothesis set, no access to whether it has a single Beta posterior or a richer structure. The wire carries observations and actions, nothing else.

This is the only commitment that makes Pass 2 architecturally invisible. If the body knew anything about how decisions were made, replacing the single-Beta with a CEG-and-meta-actions in Pass 2 would force a wire change. With opacity it doesn't: the brain replaces its representation of belief, the action signals continue to mean what they meant, the body is unaffected. Same again at Pass 3 with continuous features and structural search. The discipline pays its price now and rebates it forever.

### 2. The body has tentacles

The body's effector capabilities are determined by its embodiment. For pi the body is the TypeScript extension; the tentacles are what pi's `tool_call` hook plus the SDK's `ctx.ui` permit. Specifically: ask the user a question, refuse a tool call, allow a tool call to proceed. Three tentacles because that's what pi mechanically allows. A different body — an email agent, a robot — has different tentacles.

The brain does not invent tentacles. It selects from those the body declares. If a future deployment doesn't allow user prompts, the body's manifest omits `ask` and the brain's action-space shrinks accordingly; no special-casing in the brain. Body preconditions — the constitution's "consent as embodiment" — are operationalised as the contents of the manifest, not as a separate concept.

### 3. The capability manifest is in BDSL

The body publishes its tentacles as a BDSL file. Both sides read it: the brain to construct its action-space, the body to know which effector implementations it must register. Single source of truth, declarative, structural. Not a runtime handshake (the manifest is design-time information, fixed at deployment); not a TypeScript-side declaration that the brain has to parse (BDSL is the right home for structural data per the constitution).

### 4. Three layers: periphery, sensor protocol, brain

The brain doesn't process raw photons. The retina does enormous preprocessing — edge detection, motion, colour-opponent — before the optic nerve carries anything cortical. Feature extraction is *sensory*, not cortical: it knows about the modality (timestamps, message arrays, network protocols, file formats) and produces the brain's declared sensory vocabulary.

Three layers:

1. **Sensory periphery** (raw → features). Owned by the body. For credence-pi, the TypeScript extension. Knows about pi's data shapes, timestamps, ISO8601 parsing, message arrays. Produces feature dicts whose values are members of the brain's declared spaces.

2. **Sensor protocol** (features → brain). The wire format. What flows is feature dicts, not raw bundles. Declared spaces define the brain's expected sensory vocabulary; the periphery is contractually obliged to produce values in them.

3. **Brain** (features → action). BDSL. Conditions on observations, optimises over actions. No string manipulation, no datetime arithmetic, no array indexing — only mathematical computation over declared spaces.

BDSL is for mathematical objects only: spaces, measures, kernels, expectations, optimisations. Anything that requires knowing about the world's physical or computational structure is sensory or motor system, owned by the body. Spaces are declared in BDSL because they're the brain's representation; what populates them is a sensory choice owned by the body.

A consequence worth being explicit about: the brain cannot unilaterally invent a feature any more than it can invent an effector. A new feature in Pass 2 is a deployment event — declared in BDSL, the body's startup verifier flags the missing extractor, the body's developer writes one. Bounded by the body's perception, exactly as for an organism.

### 5. Decisions come from EU-maximisation, period

There is one decision mechanism: `optimise` over the action-space (the manifest's effectors) under preferences over (action, observation-of-approval). EVPI for ask is `voi` from the stdlib, no magic numbers, no exploration bonuses, no fallback to "always ask". At cold-start the posterior is uninformative, voi(ask) is positive and exceeds the symmetric EU of proceed/block, so ask wins by computation. As the posterior concentrates, proceed and block become live by computation. Pass 1 has the full three-action decision space; the structural learning is what waits for Pass 2.

These five commitments are not separable. Together they mean: the body's job is sensors-and-effectors; the brain's job is condition-extract-decide; the wire carries raw observations and named actions; the manifest mediates what's possible.

## What follows from this

The wire schema. Two message types, both serialised as JSON over HTTP.

**Inbound to the brain (sensor events):** the body emits these as it observes the world. Each event has a type tag and a payload. Pass 1 declares three event types in BDSL: `tool-proposed`, `user-responded`, `tool-completed`. The body emits whichever applies whenever it applies; correlation between events is by `event_id`.

**Outbound from the brain (effector signals):** when the brain's decision logic produces an action, the brain emits a signal naming an effector from the manifest, with parameters matching that effector's signature. Effector signals carry the `event_id` of the sensor event that prompted them, so the body can route the signal back to the correct dispatch.

These are the only two message types. There are no `/decide` and `/observe` endpoints. There is one inbound channel and one outbound channel; both flow asynchronously; correlation is by id.

The body's pi integration. Pi's `tool_call` hook is synchronous — pi expects a return value from the hook before it does anything else. The body bridges this to the asynchronous brain interface locally: on `tool_call`, emit a `tool-proposed` sensor event, await an effector signal correlated by id, dispatch the signal to the registered effector implementation, return the appropriate value to pi. The asynchronicity is hidden inside the body; the wire stays clean.

The brain's daemon. A Julia HTTP server that loads the BDSL programs at startup, holds the posterior as a Measure value bound to a name in the loaded environment, receives sensor events on one POST endpoint, and emits effector signals on another (or via WebSocket — see below). On each sensor event the brain calls `condition` (or doesn't, if the event is informative but not belief-updating in some sense the BDSL decides), then calls `optimise` with the current posterior over the action-space (or doesn't, if the event doesn't warrant deciding), and possibly emits a signal.

The architecture is identical for the OpenClaw body, the email agent body, the robot body. Replacing pi with another integration is replacing the TypeScript extension; the brain doesn't notice.

## The capability manifest

`apps/credence-pi/bdsl/capabilities.bdsl`. This file is the source of truth. It's read by the brain's BDSL environment to construct the action-space, and by the body's startup code to verify that all declared effectors have registered implementations.

```scheme
; ============================================================
; capabilities.bdsl — the body's effector manifest.
;
; Each (effector ...) form declares one tentacle the body has.
; The brain's action-space is exactly the set of declared
; effectors. The body's startup verifies every declared
; effector has a registered implementation and exits with an
; error if any is missing.
;
; This file is read by both sides. Changes to it are
; deployment events, not runtime events.
; ============================================================

(define manifest
  (list

    ; --- ask: synchronously query the user for yes/no approval ---
    ;
    ; Parameters:
    ;   text (string) — the question shown to the user
    ;
    ; Returns to the brain (as a subsequent user-responded event):
    ;   one of: yes, no, timeout
    ;
    ; The body's implementation calls ctx.ui.confirm with the text
    ; and emits a user-responded event when the dialog resolves.
    ; The brain may or may not produce a follow-up signal in
    ; response to the user-responded event.
    (effector ask
      (parameters (text string)))

    ; --- proceed: allow the proposed tool call to execute ---
    ;
    ; No parameters. Body returns undefined to pi's hook,
    ; letting the tool call proceed unmodified.
    (effector proceed
      (parameters))

    ; --- block: refuse the proposed tool call ---
    ;
    ; Parameters:
    ;   reason (string) — shown to pi/the user as the refusal reason
    (effector block
      (parameters (reason string)))))
```

The `effector` form is a BDSL macro (or, in Pass 1, a regular function with a specific shape) that registers one entry in the manifest. The brain's action-space construction reads `manifest`, extracts the effector names, and produces a Finite space of action symbols.

The body, at startup, loads `capabilities.bdsl` as plain text, extracts the declared effectors via a small s-expression parser (the body does not run a full BDSL evaluator — it only reads the manifest), and verifies its effector dispatch table has an entry for each. Missing effector implementation: hard error at startup, body exits.

Adding a fourth effector — Pass 2 may add `substitute`, for example — is a change to `capabilities.bdsl` plus a new TypeScript implementation in the body. The brain's action-space picks up the addition automatically. The wire schema doesn't change; effector signals already carry effector names by string.

## Sensor events

Three event types in Pass 1, all declared in BDSL. The body's periphery has already extracted features; the wire carries the brain's declared sensory vocabulary.

### `tool-proposed`

Emitted when pi's `tool_call` hook fires. The body has observed a proposed tool call and extracted its features.

Wire shape:
```json
{
  "event_type": "tool-proposed",
  "event_id": "evt_a1b2c3",
  "session_id": "sess_xyz",
  "timestamp": "2026-05-03T14:22:11.421Z",
  "features": {
    "tool_name": "bash",
    "working_directory_relative": "subdirectory",
    "parent_tool_call_name": "read",
    "recent_repetition_count": "rep_1",
    "time_since_last_user_message": "lt_2m"
  },
  "proposed_call": {
    "tool_name": "bash",
    "input": { "command": "ls -la /tmp" }
  }
}
```

The `features` dict is the body's sensory output. Each value is a symbol from a space declared in `bdsl/features.bdsl`. The body's startup-time verifier ensures every declared feature has a registered extractor.

The `proposed_call` field rides along not as input to feature extraction (extraction has already happened) but for the daemon to render `ask` text from. Pass 2 may move ask-text rendering into the body and drop this field; Pass 1 keeps it for daemon-side templating.

Notably absent: `cwd`, `messages` array, raw timestamps. Those are sensory-periphery concerns the body has already processed. The brain receives features.

### `user-responded`

Emitted when an `ask` effector's dialog resolves. Carries the user's answer and the `event_id` of the original `tool-proposed` so the brain can correlate.

Wire shape:
```json
{
  "event_type": "user-responded",
  "event_id": "evt_d4e5f6",
  "in_response_to": "evt_a1b2c3",
  "timestamp": "2026-05-03T14:22:18.943Z",
  "response": "yes"
}
```

`response` is one of `yes`, `no`, `timeout`. The brain decides what to make of it. In Pass 1 the BDSL conditions the posterior on `yes` or `no` and ignores `timeout`; Pass 2 will likely treat `timeout` as informative.

After processing, the brain may emit a follow-up effector signal. In Pass 1 the only follow-up needed is: if the user said `yes`, the brain emits `proceed`; if `no`, the brain emits `block`. (Pass 1 architecturally, since Pass 1's preferences allow the brain to do this. Earlier drafts had the body short-circuit this — "yes means proceed, body knows that" — but that's a body-decides-things violation. The body waits for the signal.)

### `tool-completed`

Emitted when a tool call that was allowed to proceed has finished executing. The body observed an outcome.

Wire shape:
```json
{
  "event_type": "tool-completed",
  "event_id": "evt_g7h8i9",
  "in_response_to": "evt_a1b2c3",
  "timestamp": "2026-05-03T14:22:31.117Z",
  "outcome": {
    "success": true,
    "duration_ms": 12450,
    "result_summary": null,
    "error": null
  }
}
```

Pass 1's BDSL does nothing with this event — it's collected for forward compatibility with Pass 2's secondary-signal observation model. The brain logs it and returns no signal. Worth shipping it now so the observation log contains the data Pass 2 will want.

## Effector signals

One outbound message type, parameterised by effector name.

```json
{
  "signal_type": "effector",
  "signal_id": "sig_j1k2l3",
  "in_response_to": "evt_a1b2c3",
  "effector": "ask",
  "parameters": {
    "text": "Allow `bash` to run `ls -la /tmp`?"
  }
}
```

`effector` is a name from the manifest. `parameters` is a dict matching the effector's declared signature. The body looks up `effector` in its dispatch table, calls the registered implementation with `parameters`, and the implementation handles the rest.

The brain may emit zero, one, or more signals per sensor event. `tool-proposed` typically produces one signal (`ask` or `proceed` or `block`). `user-responded` may produce one (a follow-up `proceed` or `block` after an `ask` resolves). `tool-completed` produces zero in Pass 1.

## Wire transport

Two HTTP endpoints, both under the daemon.

**`POST /sensor`**: the body sends a sensor event. The body MUST send the event regardless of whether it expects a signal. The brain receives, processes, and the response is acknowledgement only — the response body is `{ "ack": true, "event_id": "..." }`. Effector signals do not flow on this connection.

**`GET /signals` (Server-Sent Events)**: the body opens a long-lived SSE connection at session startup and listens for effector signals. Each signal is one SSE message. The body's local correlation table dispatches signals to the right pi-hook awaiter.

SSE rather than WebSocket because the body→brain channel is request/response (the body never receives an "ask" mid-request to, e.g., re-emit a sensor event), so a unidirectional brain→body push is sufficient and SSE is the simpler tool. The body never sends anything on the SSE connection; it only receives.

If the SSE connection drops, the body reconnects with exponential backoff. Effector signals emitted during the disconnection are buffered server-side (Pass 1 uses an in-memory bounded queue, size 100, oldest-dropped-on-overflow; if a signal is dropped, the corresponding pi-hook awaiter eventually times out and the body fails open). Pass 2 may want a more durable channel; Pass 1 is best-effort.

## Pass 1 BDSL

The brain. Five files under `apps/credence-pi/bdsl/`. All small.

### `capabilities.bdsl`

The manifest. Shown above.

### `prior.bdsl`

```scheme
; ============================================================
; prior.bdsl — prior over P(approve)
;
; The Pass 1 belief: a single Beta(2, 2) over the probability
; that the user would approve a proposed tool call. No
; conditioning on features; the posterior is global.
;
; Pass 2 will replace this with a richer structure (a CEG over
; features). The public name make-prior is stable; consumers do
; not change.
; ============================================================

(define p-approve-space (space :interval 0 1))

(define make-prior
  (lambda ()
    (measure p-approve-space :beta 2.0 2.0)))
```

### `kernel.bdsl`

```scheme
; ============================================================
; kernel.bdsl — observation kernel for user responses
;
; P(observation | theta), Bernoulli with parameter theta.
; Conjugate to Beta prior; closed-form update via the
; BetaBernoulli family.
; ============================================================

(define response-space (space :finite 0 1))

(define approve-kernel
  (kernel p-approve-space response-space
    (lambda (theta)
      (lambda (obs)
        (if (= obs 1)
          (log theta)
          (log (- 1.0 theta)))))
    :family bernoulli))
```

### `features.bdsl`

The brain's declared sensory vocabulary. Each feature is a Finite space; the body's periphery is contractually obliged to produce values that are members of these spaces.

```scheme
; ============================================================
; features.bdsl — the brain's sensory vocabulary.
;
; Each (feature ...) form declares one feature the brain
; expects to receive in tool-proposed events. The body's
; startup verifies every declared feature has a registered
; extractor and exits with an error if any is missing.
;
; Pass 1 declares the features but does not condition on them
; (the posterior is global). Pass 2 will use them in the
; structure-learning machinery; declaring them now means the
; observation log already contains the data Pass 2 will want.
; ============================================================

(define tool-name-space
  (space :finite read write edit bash grep find ls other))

(define wd-relative-space
  (space :finite project-root subdirectory outside-project no-path))

(define rep-count-space
  (space :finite rep-0 rep-1 rep-2 rep-3plus))

(define time-since-user-space
  (space :finite lt-30s lt-2m lt-10m gt-10m))

(define parent-tool-name-space
  (space :finite read write edit bash grep find ls other none))

(define features
  (list
    (feature tool-name                    tool-name-space)
    (feature working-directory-relative   wd-relative-space)
    (feature parent-tool-call-name        parent-tool-name-space)
    (feature recent-repetition-count      rep-count-space)
    (feature time-since-last-user-message time-since-user-space)))
```

The `feature` form (analogous to `effector` in the manifest) declares one feature with its associated space. The body parses this file at startup and verifies its extractor table has an entry for each declared feature; missing extractor implementation: hard error at startup, body exits.

`project_id` is *not* declared as a feature in Pass 1. It's an opaque session identifier the body produces and the daemon records in the observation log; Pass 1 doesn't condition on it. Pass 2 will declare an appropriate space (likely a growing categorical) when the structure-learning machinery is in place.

### `decide.bdsl`

The decision program.

```scheme
; ============================================================
; decide.bdsl — Pass 1 decision program
;
; Action space: constructed from the manifest's effectors.
; Preferences: symmetric over (approve, refuse) for proceed/
; block; ask is competing against them via voi.
;
; At cold-start (Beta(2,2)), proceed and block have EU = 0
; under symmetric utilities; voi(ask) > 0; ask wins by
; computation. As observations accumulate, posterior
; concentrates and proceed or block win in their respective
; regimes.
; ============================================================

; The action space comes from the manifest. effector-names
; extracts the names from the (effector ...) forms.
(define action-space
  (apply space :finite (effector-names manifest)))

; Pass 1 preferences:
;   pref(theta, proceed) =  1.0  if user would approve
;                        = -1.0  if user would refuse
;   pref(theta, block)   = -1.0  if user would approve  (refusing what they wanted)
;                        =  1.0  if user would refuse   (correctly refused)
;   pref(theta, ask)     =  0.0  unconditional (the cost of interruption is
;                                 implicit; voi handles the gain)
;
; theta is the unknown P(approve). The preference takes theta
; and returns expected payoff at that theta — proceed pays
; +1·theta + (-1)·(1-theta) = 2*theta - 1.
;
; This is honest EU-max with the three actions live. Pass 2
; refines.
(define pass1-pref
  (lambda (theta action)
    (cond
      ((= action (quote proceed)) (- (* 2.0 theta) 1.0))
      ((= action (quote block))   (- 1.0 (* 2.0 theta)))
      ((= action (quote ask))     0.0)
      (else (error "unknown action")))))

; ── Public entry points ──

; decide-action: returns the chosen action symbol, or one of
; them if there is a tie (optimise's behaviour).
;
; voi is computed inline: optimise alone would pick proceed or
; block by EU and ignore ask. To compete ask fairly, we compute
; voi over the response kernel and add ask's voi to its base
; preference of 0. This is the textbook EVPI gate as a
; computation, not a magic constant.
(define decide-action
  (lambda (posterior)
    (let voi-of-asking
        (voi posterior approve-kernel action-space pass1-pref
             (list 0 1))                  ; possible observations
      (let pref-with-voi
          (lambda (theta action)
            (if (= action (quote ask))
              voi-of-asking
              (pass1-pref theta action)))
        (optimise posterior action-space pref-with-voi)))))

; observe-response: returns the updated posterior given a
; binary response (1 = yes, 0 = no).
(define observe-response
  (lambda (posterior obs)
    (condition posterior approve-kernel obs)))
```

The voi computation is the cold-start fix from your original review document, written honestly as the textbook EVPI rather than a magic-number proxy. At Beta(2,2) with the symmetric preferences, voi(ask) computes to a positive number that exceeds the EU of proceed (= 0) and block (= 0), so ask wins. As observations accumulate and the posterior concentrates away from 0.5, voi(ask) decreases and EU(proceed) or EU(block) eventually exceeds it. The behaviour falls out of the maths.

## The Julia daemon

`apps/credence-pi/daemon/`. Pure transport plus observation log management.

### Responsibilities

- HTTP server: `POST /sensor`, `GET /signals` (SSE).
- BDSL environment: load the five `bdsl/*.bdsl` files at startup; hold the resulting environment.
- Posterior state: a `Ref{Measure}` holding the current posterior; reconstructed from the observation log at startup; replaced via `condition` on `user-responded` events.
- Observation log: append-only JSONL at `~/.credence-pi/observations.jsonl`; one line per sensor event; written before any signal is emitted in response.
- Effector signal emission: when the BDSL's `decide-action` returns a signal-worthy action, serialise to JSON and push to all SSE-connected clients.

The daemon does no probabilistic reasoning, computes no EU, makes no decisions. It receives sensor events, calls into the BDSL environment via the loaded named entry points (`decide-action`, `observe-response`, `followup-after-response`), serialises the BDSL's return values to JSON, manages I/O.

### Sensor event handling

```julia
# Pseudocode in apps/credence-pi/daemon/server.jl

function handle_sensor_event(event::Dict)
    append_to_observation_log(event)

    if event["event_type"] == "tool-proposed"
        # Brain decides on an action.
        action = (ENV[Symbol("decide-action")])(POSTERIOR[], event)
        emit_signal(event["event_id"], action_to_signal(action, event))

    elseif event["event_type"] == "user-responded"
        if event["response"] in ("yes", "no")
            obs = event["response"] == "yes" ? 1 : 0
            POSTERIOR[] = (ENV[Symbol("observe-response")])(POSTERIOR[], obs)
        end
        # Follow-up: yes -> proceed, no -> block.
        # The brain decides the follow-up; the daemon does not infer it.
        followup = (ENV[Symbol("followup-after-response")])(event)
        if followup !== nothing
            emit_signal(event["in_response_to"], action_to_signal(followup, event))
        end

    elseif event["event_type"] == "tool-completed"
        # Pass 1: log only.
    end

    return Dict("ack" => true, "event_id" => event["event_id"])
end
```

The `followup-after-response` BDSL function is referenced here and needs to exist:

```scheme
; In decide.bdsl:
(define followup-after-response
  (lambda (event)
    (let response (lookup event (quote response))
      (cond
        ((= response "yes") (quote proceed))
        ((= response "no")  (quote block))
        (else nothing)))))
```

This looks like the body could short-circuit it ("the body knows yes means proceed"), but the discipline is the brain decides. The brain's choice happens to be deterministic in Pass 1; in Pass 2 the response might be one of several actions depending on broader context. Putting the logic in BDSL from the start keeps the architecture clean.

### Observation log

JSONL at `~/.credence-pi/observations.jsonl`. Format:

```json
{ "schema": "credence-pi/v1", "received_at": "...", "event": { ... } }
```

The `event` field is the sensor event verbatim. `schema` is `credence-pi/v1`; Pass 2 may bump.

On startup the daemon reads the log in full, replays each `user-responded` event with `response ∈ {yes, no}` through the BDSL's `observe-response`, and the resulting posterior becomes the runtime state. `tool-proposed` and `tool-completed` events are logged but don't update belief; they're collected for Pass 2.

The daemon opens the log in append mode; one writer per daemon process (Pass 1 doesn't support concurrent daemons). `fsync` after each append (correctness over throughput).

## The TypeScript extension

`apps/credence-pi/extension/`. Pure body responsibilities.

### Responsibilities

- Pi extension lifecycle: register an extension factory; subscribe to pi events.
- Sensor event emission: on `tool_call` emit `tool-proposed`; on confirm-resolution emit `user-responded`; on `tool_execution_end` emit `tool-completed`.
- Effector implementation registration: a dispatch table from effector name to TypeScript implementation.
- Effector signal handling: maintain an SSE connection to the daemon; on signal arrival, dispatch to the registered implementation; correlate signals to awaiting pi-hook callbacks via `event_id`.
- Manifest verification at startup: parse `apps/credence-pi/bdsl/capabilities.bdsl`, verify every declared effector has a registered implementation, exit with error if not.
- Fail-open on daemon unavailability: if the daemon is unreachable, log a warning and let pi proceed without governance.

### Sketch

```typescript
// apps/credence-pi/extension/src/index.ts (sketch)

import { parseManifest } from "./manifest.js";
import { connectSignalsStream, postSensor } from "./client.js";
import { effectors } from "./effectors.js";

export default function credencePiExtension(pi: ExtensionAPI) {
  const manifest = parseManifest(MANIFEST_PATH);
  for (const eff of manifest) {
    if (!(eff.name in effectors)) {
      throw new Error(`No implementation for declared effector: ${eff.name}`);
    }
  }

  const pendingHooks = new Map<string, {
    resolve: (value: HookReturn) => void,
    reject: (err: Error) => void,
  }>();

  // SSE connection: route signals to hook awaiters via event_id.
  connectSignalsStream(DAEMON_URL, (signal) => {
    const eventId = signal.in_response_to;
    const pending = pendingHooks.get(eventId);
    if (!pending) {
      // Signal arrived after timeout, or an unsolicited followup.
      // For followups (proceed/block after user-responded),
      // the original tool_call hook is already long since
      // returned. The body has nothing to do.
      return;
    }
    const impl = effectors[signal.effector];
    impl(signal.parameters, pending);
  });

  pi.on("tool_call", async (event, ctx) => {
    const eventId = generateEventId();
    return new Promise<HookReturn>((resolve, reject) => {
      pendingHooks.set(eventId, { resolve, reject });
      postSensor(DAEMON_URL, {
        event_type: "tool-proposed",
        event_id: eventId,
        session_id: ctx.session.sessionId,
        timestamp: new Date().toISOString(),
        features: extractFeatures(event, ctx.session),
        proposed_call: { tool_name: event.toolName, input: event.input },
      }).catch(() => {
        // Daemon down: fail open.
        pendingHooks.delete(eventId);
        ctx.logger?.warn("credence-pi daemon unreachable; proceeding without governance");
        resolve(undefined);
      });
      // Timeout in case no signal arrives (daemon dropped, etc.).
      setTimeout(() => {
        if (pendingHooks.has(eventId)) {
          pendingHooks.delete(eventId);
          ctx.logger?.warn("credence-pi: no signal received in time; proceeding");
          resolve(undefined);
        }
      }, 30_000);
    });
  });

  pi.on("tool_execution_end", async (event, ctx) => {
    postSensor(DAEMON_URL, {
      event_type: "tool-completed",
      event_id: generateEventId(),
      in_response_to: lookupOriginatingEventId(event),
      timestamp: new Date().toISOString(),
      outcome: { success: !event.isError, duration_ms: event.durationMs,
                 result_summary: null, error: event.error ?? null },
    }).catch(() => {});
  });
}
```

### Effector implementations

A dispatch table. One file per effector for clarity; one map at registration.

```typescript
// apps/credence-pi/extension/src/effectors.ts

import type { Effector } from "./types.js";
import { ask } from "./effectors/ask.js";
import { proceed } from "./effectors/proceed.js";
import { block } from "./effectors/block.js";

export const effectors: Record<string, Effector> = { ask, proceed, block };
```

`ask` implementation: invoke `ctx.ui.confirm` with the text parameter; when it resolves, post a `user-responded` sensor event; do *not* resolve the originating tool_call hook yet (the brain's followup signal will). If `ctx.hasUI === false`, post a `user-responded` with `response: "timeout"` and let the brain's followup decide; if no followup arrives within the timeout, fail open.

`proceed` implementation: resolve the hook with `undefined` (pi proceeds with the tool call).

`block` implementation: resolve the hook with `{ block: true, reason: parameters.reason }`.

### Feature extraction (the periphery)

The body's sensory periphery. Each declared feature has a corresponding extractor function. The dispatch table is verified against `bdsl/features.bdsl` at startup.

```typescript
// apps/credence-pi/extension/src/features/index.ts

import type { Features, ToolCallEvent, Session } from "../types.js";
import { extractToolName } from "./tool_name.js";
import { extractWorkingDirectoryRelative } from "./working_directory.js";
import { extractParentToolCallName } from "./parent_tool.js";
import { extractRecentRepetitionCount } from "./repetition.js";
import { extractTimeSinceLastUserMessage } from "./time_since_user.js";

export const extractors = {
  "tool-name": extractToolName,
  "working-directory-relative": extractWorkingDirectoryRelative,
  "parent-tool-call-name": extractParentToolCallName,
  "recent-repetition-count": extractRecentRepetitionCount,
  "time-since-last-user-message": extractTimeSinceLastUserMessage,
};

export function extractFeatures(event: ToolCallEvent, session: Session): Features {
  return {
    "tool_name": extractors["tool-name"](event, session),
    "working_directory_relative": extractors["working-directory-relative"](event, session),
    "parent_tool_call_name": extractors["parent-tool-call-name"](event, session),
    "recent_repetition_count": extractors["recent-repetition-count"](event, session),
    "time_since_last_user_message": extractors["time-since-last-user-message"](event, session),
  };
}
```

Each extractor is a small pure function. Bucketing logic lives here:

```typescript
// apps/credence-pi/extension/src/features/repetition.ts

const KNOWN_TOOLS = new Set(["read", "write", "edit", "bash", "grep", "find", "ls"]);

export function extractRecentRepetitionCount(
  event: ToolCallEvent,
  session: Session,
): string {
  const target = KNOWN_TOOLS.has(event.toolName.toLowerCase())
    ? event.toolName.toLowerCase()
    : "other";
  const recentToolCalls = session.agent.state.messages
    .slice(-20)
    .filter(m => m.role === "tool_call")
    .slice(-5);
  const matches = recentToolCalls.filter(
    m => (KNOWN_TOOLS.has(m.toolName.toLowerCase()) ? m.toolName.toLowerCase() : "other") === target
  ).length;
  if (matches === 0) return "rep_0";
  if (matches === 1) return "rep_1";
  if (matches === 2) return "rep_2";
  return "rep_3plus";
}
```

```typescript
// apps/credence-pi/extension/src/features/time_since_user.ts

export function extractTimeSinceLastUserMessage(
  event: ToolCallEvent,
  session: Session,
): string {
  const userMessages = session.agent.state.messages.filter(m => m.role === "user");
  if (userMessages.length === 0) return "gt_10m";
  const last = userMessages[userMessages.length - 1];
  const elapsed = (Date.now() - new Date(last.timestamp).getTime()) / 1000;
  if (elapsed < 30)  return "lt_30s";
  if (elapsed < 120) return "lt_2m";
  if (elapsed < 600) return "lt_10m";
  return "gt_10m";
}
```

The remaining extractors (`extractToolName`, `extractWorkingDirectoryRelative`, `extractParentToolCallName`) are similarly small and live in their respective files.

### Manifest and feature parsing

The body parses two BDSL files at startup: `capabilities.bdsl` for effectors and `features.bdsl` for features. A small s-expression reader covers both — the body does not run a full BDSL evaluator, it only extracts top-level declarations.

```typescript
// apps/credence-pi/extension/src/manifest.ts (sketch)

export interface EffectorDecl { name: string; parameters: ParameterDecl[]; }
export interface FeatureDecl { name: string; spaceName: string; }

export function parseCapabilities(path: string): EffectorDecl[] { /* ... */ }
export function parseFeatures(path: string): FeatureDecl[] { /* ... */ }

// Verification at extension factory startup:
function verifyManifests() {
  for (const eff of parseCapabilities(CAPABILITIES_PATH)) {
    if (!(eff.name in effectors)) {
      throw new Error(`No implementation for declared effector: ${eff.name}`);
    }
  }
  for (const feat of parseFeatures(FEATURES_PATH)) {
    if (!(feat.name in extractors)) {
      throw new Error(`No extractor for declared feature: ${feat.name}`);
    }
  }
}
```

Either error at startup is fatal: the body cannot satisfy a contract it has declared.

## File structure

Exactly these files. No others without conversation approval.

```
apps/credence-pi/
├── README.md                       # 1-page run/dev/test pointer
├── SPEC.md                         # this document, copied verbatim
├── NOTES-ON-EXISTING-SIDECAR.md    # short note: see "Status" section above
├── bdsl/
│   ├── capabilities.bdsl           # the body's effector manifest
│   ├── features.bdsl               # the brain's sensory vocabulary
│   ├── prior.bdsl                  # prior over P(approve)
│   ├── kernel.bdsl                 # observation kernel
│   └── decide.bdsl                 # decision program; voi gate
├── daemon/
│   ├── server.jl                   # HTTP server, SSE, BDSL env
│   ├── observation_log.jl          # JSONL append/replay
│   ├── Project.toml                # HTTP, JSON3
│   └── README.md                   # 1-page run instructions
├── extension/
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   │   ├── index.ts                # extension factory
│   │   ├── manifest.ts             # parses capabilities.bdsl, features.bdsl
│   │   ├── client.ts               # SSE + POST sensor
│   │   ├── features/
│   │   │   ├── index.ts            # extractor dispatch + extractFeatures
│   │   │   ├── tool_name.ts
│   │   │   ├── working_directory.ts
│   │   │   ├── parent_tool.ts
│   │   │   ├── repetition.ts
│   │   │   └── time_since_user.ts
│   │   ├── effectors.ts            # dispatch table
│   │   ├── effectors/
│   │   │   ├── ask.ts
│   │   │   ├── proceed.ts
│   │   │   └── block.ts
│   │   └── types.ts                # shared types
│   └── README.md
└── tests/
    ├── julia/
    │   ├── test_bdsl.jl            # programs load; cold-start picks ask;
    │   │                           # observations update posterior; voi
    │   │                           # decreases as posterior concentrates
    │   ├── test_observation_log.jl # round trip; replay; schema rejection
    │   └── test_server.jl          # /sensor + /signals end-to-end
    └── typescript/
        ├── manifest.test.ts        # parse capabilities + features;
        │                           # error on missing implementation/extractor
        ├── features.test.ts        # bucketing edges per extractor
        ├── client.test.ts          # SSE reconnection; POST timeout
        ├── effectors.test.ts       # dispatch correctness per effector
        ├── index.test.ts           # full hook flow via mocks
        └── fixtures/               # session-state fixture JSON files
```

## Lint pragma policy

Code under `apps/credence-pi/` MUST land with zero `# credence-lint: allow` pragmas. If during implementation Claude Code is tempted to add a pragma, it MUST stop and report the situation to conversation rather than proceed. This is a stricter rule than the rest of the repo (which permits sanctioned pragmas under documented precedents); it applies here because the existing sidecar shows precisely what happens when the constraint is "review later".

Two narrow exceptions are pre-sanctioned:

1. Tests under `tests/julia/` that need `precedent:test-oracle` pragmas to assert equality on a posterior's expected value computed via an analytical oracle. Required per line.
2. Display-only arithmetic in single-line `@info` log statements (e.g. `@info "voi=$(round(v, digits=3))"`). The arithmetic and the log call must be on the same line; multi-line composition is forbidden. `precedent:display-arithmetic`.

Anything else: stop and report.

## CI

Update `.github/workflows/publish-image.yml` to add to `unit-tests`:

1. `for f in apps/credence-pi/tests/julia/*.jl; do julia "$f"; done` after the existing Python pytest step.
2. `cd apps/credence-pi/extension && npm install && npm run build && cd ../tests/typescript && npm test`.
3. `python tools/credence-lint/credence_lint.py apps/credence-pi/` (zero violations expected).

## Sequencing

Each step ends in a working commit; conversation review between steps. Stop and report if any step uncovers a structural question this spec does not anticipate.

1. **BDSL primitives audit.** Verify which of `cond`, `apply`, `effector-names`, `feature-names`, the `effector` form, and the `feature` form exist or can be expressed in `src/eval.jl`'s default environment. Add what's missing. The list is short because the brain's BDSL no longer manipulates raw data — only declares structure and reasons mathematically. Stop and report.

2. **BDSL programs.** `capabilities.bdsl`, `features.bdsl`, `prior.bdsl`, `kernel.bdsl`, `decide.bdsl`. Plus `tests/julia/test_bdsl.jl`. Verify: programs load; manifest yields three-action space; cold-start `decide-action` returns `:ask` (because voi(ask) > 0 and EU(proceed) = EU(block) = 0); observations update posterior; voi decreases as posterior concentrates; followup-after-response returns proceed/block correctly. Stop and report.

3. **Observation log.** `daemon/observation_log.jl`, `tests/julia/test_observation_log.jl`. Append, replay, schema versioning. Stop and report.

4. **Daemon transport.** `daemon/server.jl` with `/sensor` and `/signals`. SSE plumbing. `tests/julia/test_server.jl`. Stop and report.

5. **Extension scaffolding and manifest parsing.** `extension/package.json`, `tsconfig.json`, `src/manifest.ts`, `tests/typescript/manifest.test.ts`. Verify build succeeds; both `capabilities.bdsl` and `features.bdsl` parse correctly; missing-implementation and missing-extractor errors fire. Stop and report.

6. **Feature extractors and client.** `src/features/*.ts`, `src/client.ts`, `tests/typescript/features.test.ts`, `tests/typescript/client.test.ts`. Stop and report.

7. **Effector implementations and integration.** `src/effectors/*.ts`, `src/index.ts`, full hook flow. `tests/typescript/effectors.test.ts`, `tests/typescript/index.test.ts`. Stop and report.

8. **CI integration.** Workflow edits, READMEs. Verify the full thing builds clean. Stop and report.

## Out of scope (Pass 1)

- Meta-actions, working set management, structural learning. Single Beta posterior, global, no feature conditioning at decision time.
- Embeddings, continuous features. Pass 1 has no continuous variables in the brain's belief state.
- Tool argument substitution. The `substitute` effector does not exist in the manifest.
- Outcome-based posterior updating. `tool-completed` events are logged but the BDSL does not condition on them.
- Detectors (stationarity, no-confidence). Not ported from the existing sidecar.
- Multi-user, multi-session brain isolation. One daemon, one observation log; `session_id` is recorded but does not partition the posterior.
- Diagnostic / inspection endpoints. Pass 2 concern; Pass 1 inspection is by reading the observation log and replaying offline.
- Configuration files. Prior parameters live in BDSL source.
- Robust SSE delivery semantics. Best-effort with reconnection; lost signals fail open.

## Success criteria

1. Observation log contains ≥ 200 well-formed sensor events from real sessions over two weeks.
2. No pi crash, hang, or unrecoverable state caused by the extension.
3. Daemon-down: pi proceeds normally; extension fails open with a single warning per outage.
4. Replayed posterior matches an analytical computation from the observation log to floating-point tolerance.
5. Round-trip latency for sensor → signal is below 50ms p95 in normal conditions.
6. `tools/credence-lint/credence_lint.py apps/credence-pi/` reports zero violations.
7. The Julia and TypeScript test suites pass under CI.

These are integration criteria, not capability criteria. Capability evaluation — does the agent learn anything useful, do the partition assumptions hold, does the structure-learning work — is Pass 2's empirical case to make, on the foundation Pass 1 establishes.

## Decisions deferred to Pass 2

Recorded here so they are not silently made in Pass 1.

- The proposal distribution and triggering policy for meta-actions over CEG move-set.
- The complexity prior's exact form over CEG structures.
- Promotion of `project_id`, `parent_tool_call_name`, and tool names beyond the core seven from opaque strings to first-class space members.
- Continuous features and embeddings (Pass 3, possibly).
- Headless mode policy (`ctx.hasUI === false` proper behaviour).
- Substitute and additional effectors.
- Risk-premium / utility-curvature parameters.
- Persistence schema migration from `v1`.
- Whether to deprecate `apps/credence-governance-sidecar/`.
- Diagnostic / inspection endpoints.
- Robust SSE delivery semantics; durable signal queue.
