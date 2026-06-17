# answer-brain — Move 4 design (the body: a shared brain-body lib + the answer-brain governor + the scripted gate)

Follows `docs/posture-4/DESIGN-DOC-TEMPLATE.md`. Strategy is settled in `master-plan.md`; Moves 1–3
landed the native brain (`move-1-design.md`), the stateless `POST /decide` wire (`move-2-design.md`),
and the life-agent capability bridge (`move-3-design.md`, in the life-agent repo). Those three gave
the body its two backends — the **daemon** (`/decide`, the single reasoner) and the **bridge**
(`/route`/`/retrieve`/`/extract`/`/probe`/`/utility`, the evidence). This move builds the **body**
that drives them in a govern+steer loop, and — per the author's de-dup decision (§5 Q1) — extracts the
transport-agnostic machinery into a **shared library** so the body is thin and credence-pi can later
stand on the same lib. It closes the loop end-to-end and **certifies it with a gate**.

## 0. Final-state alignment

`master-plan.md` §"Architecture" is three processes: a pi-mono answering agent (TS) proposing
`retrieve`/`extract`/`probe`/`answer`; the **answer-brain daemon** (Julia) holding the candidate
posterior and governing answer/gather/ask/abstain by EU; and **life-agent capabilities** (Python) the
agent's tools call. Moves 1–3 stood up the daemon and the bridge and froze the parity boundary
(`/extract`'s output is exactly `/decide`'s input). This move builds the **first process** — and
factors its reusable core out, because it is the second consumer of a pattern credence-pi already has.

**The de-dup cut (the author's decision, §5 Q1 — `+ shared TS governor lib`).** credence-pi is one
*application* of credence; the **library is `src/`**, and it already owns the **one BDSL parser**
(`src/parse.jl`, `src/eval.jl`) — credence-pi's Julia daemon parses its manifest *through the library*.
The duplication is credence-pi's TS `extension/src/manifest.ts` (159 lines), a **second parser in a
second language**. So "use credence as a library" is not "share a TS parser" — it is **own the manifest
in the library and have the body fetch it**: the answer-brain daemon loads its `*.bdsl` via the library
parser and serves the vocabulary (`GET /manifest`); the body verifies against it and **has no parser**.
The transport-agnostic governor (effector dispatch + fail-open/timeout + a pluggable "ask-the-brain"
seam) is extracted into a **shared TS lib** both bodies can stand on. The two bodies' *transports*
differ and are **not** shared: credence-pi reaches its daemon over an async SSE `/sensor`+`/signals` bus
with a correlation table; answer-brain's daemon is the stateless synchronous `POST /decide`
(`move-2-design.md`; `daemon/server.jl`), so its body just `await fetch`es. The lib abstracts what is
common (dispatch, fail-open, manifest-verify, wire types) and injects what differs (the transport).

Transient state, named so it is not drift:
- **credence-pi is not refactored in this move** (§5 Q1 resolution: "migrates later"). The shared lib is
  written for answer-brain (its first consumer) but with a transport seam credence-pi fits; credence-pi
  adopts it — and drops its TS parser — in a later move. The lib's API is the thing that freezes here.
- **The gather decision is an operator-set *feature-policy*, not yet full `voi_gather`** (§5 Q2). The
  per-probe VOI primitive is built and tested in the brain (`brain/answer_brain.jl:204`, `voi_gather`),
  but pricing a probe *without running it* needs a predictive kernel per probe (the date distribution
  for recency, the hit count for corroborate) — a modelling task deferred to a named successor. v0's
  brain emits `gather` from the **posterior-shape features** the bdsl already declares; honest about
  being a calibrated policy, not yet a kernel-priced VOI.
- **The gate is a scripted deterministic driver; the LLM app is a demo** (§5 Q3). The eval runs over the
  owner's real PII corpus, so no cloud model may drive it. A scripted Python driver (reusing
  `run_eval`) certifies the closed loop confound-free; the pi-mono TS body + a minimal local-Ollama app
  is the qualitative demonstration, **off the gate's critical path**. The loop stays thin in both (the
  daemon owns every decision), so the two drivers are not a meaningful re-implementation of policy.
- **The daemon stays stateless; learning is Stage 3.** Serving `GET /manifest` is a read; the gather
  policy is operator-set (cold). The observation-log replay + owner-verdict folding (`master-plan.md`
  Stage 3) is *not* in this move.

## 1. Purpose

Build the **answer-brain body** as a pi-mono extension (TS) that runs the govern+steer loop over the
two HTTP backends, factoring its transport-agnostic core into a **shared brain-body library**; add the
daemon's **`gather` branch** (feature-policy) and **`GET /manifest`**; and stand up the **scripted gate**
that certifies the closed loop. Concretely, the body: registers `retrieve`/`extract`/`probe_*`/`answer`
as tools that proxy to the bridge; **observes** their results to accumulate the evidence vector
(candidates + abstract observations + covariates); and **governs the `answer` tool** — on each `answer`
attempt it posts the accumulated evidence to `/decide` and, by the returned effector, **allows** the
answer (rewriting it to the decided value + citations), **blocks-and-steers** to the brain-chosen probe
(`gather`), **blocks** to an owner question (`ask-user`), or **blocks** to a withheld-leader abstention
(`abstain`). The brain is the only reasoner; the body transports its inputs and enacts its outputs.

**Why a shared lib (not a copy, not credence-pi-wholesale):** the manifest-verify, the effector
dispatch, the fail-open/timeout discipline, and the wire types are identical needs across brains; the
*transport* and the *effector/feature impls* are not. Extracting the former and injecting the latter is
the only cut that de-duplicates without forcing credence-pi's async bus onto answer-brain's sync daemon.

**Proof obligation:** (a) the shared lib is unit-tested transport-free (a fake `askBrain`, a fake
manifest) — dispatch, fail-open on timeout/unreachable, manifest-verify rejects a missing effector;
(b) the body's loop is tested against a **fake daemon + fake bridge** (scripted effector replies) —
the mobile-class case gathers-then-reports, an abstain withholds its leader, statelessness across
questions; (c) the daemon's `gather` branch + `/manifest` are tested in Julia (the feature-policy emits
`gather(recency)` on the era-split case, `report` after; `/manifest` round-trips the bdsl vocabulary);
(d) the **gate** — the scripted driver over the eval through the real bridge+daemon — runs and is
reported (zero new confident-wrong; `P(Δ>δ)` stated); (e) one opt-in **local-Ollama** end-to-end demo
is run and its result stated, not assumed.

## 2. Files touched

### A. The shared brain-body library (NEW) — `packages/brain-body/` (name/placement: §5 Q4)
- `src/manifest.ts` — `fetchManifest(daemonUrl) → {effectors, features}` (`GET /manifest`) + `verify(reg)`
  that every declared effector/feature has a registered impl. **No BDSL parsing** — the daemon (via the
  library `src/parse.jl`) is the one parser. Replaces credence-pi's `manifest.ts`.
- `src/governor.ts` — the transport-agnostic skeleton: `govern(event, {extractFeatures, askBrain,
  dispatch, timeoutMs})` → assemble features/evidence → `await askBrain(request)` (injected) → dispatch
  the returned effector → **fail-open** on timeout/throw. The pluggable seam is `askBrain`.
- `src/dispatch.ts` — the effector-registry + feature-extractor harness, parametric over the impl types
  (`EffectorRegistry<TImpl>`, `assembleFeatures(extractors, ctx)`).
- `src/wire.ts` — shared wire types: the effector signal, the manifest shape, the fail-open result.
- `package.json`, `tsconfig.json`, `tests/*` (transport-free unit tests).

### B. The answer-brain body (NEW) — `apps/answer-brain/extension/`
- `src/transport.ts` — the injected synchronous `askBrain`: compose the accumulated evidence + `u_bar`
  (bridge `/utility`) → `POST /decide` → the effector signal. Plus the bridge proxies
  (`/route`/`/retrieve`/`/extract`/`/probe/*`).
- `src/effectors.ts` — `answer` / `ask-user` / `abstain` / `gather` impls, each enacting a pi-mono
  `tool_call` decision (`{block, reason}` per `ToolCallEventResult`, pi-mono `…/extensions/types.ts:1020`;
  `answer` may rewrite `event.input` in place to the decided value+citations).
- `src/features.ts` — the posterior-shape extractors (dispersion / leader-band / era-split /
  owner-scoped) reading the accumulated evidence; their outputs are sent to `/decide` so the daemon's
  feature-policy can read them (the body computes the *projections*; the daemon *decides*).
- `src/tools.ts` — `pi.registerTool` for `retrieve`/`extract`/`probe_*`/`answer` (bridge proxies) + the
  per-session evidence accumulator (observe `tool_result`, grow candidates+observations+covariates).
- `src/index.ts` — the `ExtensionFactory` (`(pi) => …`, pi-mono `…/types.ts:1416`): wire the lib's
  `govern` + the injected transport + effectors + features + tools.
- `package.json`, `tsconfig.json`, `README.md`, `tests/*` (fake daemon + fake bridge).

### C. The answer-brain daemon (MODIFY) — `apps/answer-brain/daemon/`, `brain/answer_brain.jl`
- `GET /manifest` — load `bdsl/capabilities.bdsl` + `bdsl/features.bdsl` through the library parser and
  serve `{effectors, features}`. Read-only; the daemon stays stateless.
- The **`gather` branch**: `decide_full` (or a wrapping `decide`) gains the feature-policy. `/decide`'s
  request grows (additively) to carry the body-projected features the daemon cannot see — `era-split`,
  `owner-scoped` (the body holds the covariate/profile info); the daemon computes the posterior-derived
  features itself — `dispersion`, `leader-band` (from the `weights` it builds). v0's class-valid probe
  is **`recency` on `era-split`**, and because it is a *cheap re-weight* it is applied BEFORE the
  terminal report, not gated behind it: whenever `era-split` holds and `recency` is unapplied the daemon
  returns `gather(recency, target)`; the body re-weights and re-decides; `applied_probes` (resent) makes
  it fire at most once → termination. A confident report must first rule out the staleness `era-split`
  signals — a count-led STALE value can sit ABOVE the bar, so reporting it un-checked is a
  confident-wrong (`gather.py` applies recency pre-decision; the daemon ports that). The **below-bar
  EU-gate belongs to the EXPENSIVE gather** — `corroborate` (fetch new docs), the deferred §5 Q2
  successor; `subject` (mis-fires on contact facts) stays deselected. The policy prices no kernel (§5
  Q2): for the cheap recency re-weight, "apply when class-valid" is the VOI-sound move. **(Build
  correction:** an earlier below-bar gate on recency was removed when TDD/parity surfaced the
  confident-stale-report it would emit — the daemon must recency-check a count-led stale leader even
  above the bar.**)**
- `daemon/server.jl` — extend the `/decide` response with the optional `gather` fields; add `/manifest`
  dispatch. (`/decide`'s existing terminal shape is unchanged — additive.)
- The operator-set policy params (§5 Q5) live in `bdsl/utility.bdsl` (or a sibling `policy.bdsl`):
  calibration priors, committed before results, folded from outcomes later, **never fitted to the gate**.

### D. The scripted gate driver (NEW) — life-agent `scripts/`
- A deterministic Python driver: per eval question, run the loop (`/route` → `/retrieve` → `/extract`
  → `/decide` → [`gather` → `/probe/*` → `/extract` → `/decide`]\* → terminal) against the live
  bridge+daemon, collect the outcome, feed `run_eval --gate`. Reuses the Move-3 bridge clients and the
  existing gate. **This is the certification.** No model drives it; extraction is the bridge's local
  model (already how `/extract` works).

### E. The minimal pi-mono app (NEW) — `apps/answer-brain/app/`
- `createAgentSession({ model, resourceLoader })` (pi-mono `examples/sdk/01-minimal.ts`,
  `06-extensions.ts:30` inline factory) loading the answer-brain extension; a **local-Ollama** `Model`
  (`{provider:"ollama", baseUrl:"http://localhost:11434/v1"}`); a system prompt instructing the agent to
  retrieve→extract→probe→answer. The **demo**, not the gate.

## 3. Behaviour preserved

- **The parity boundary is untouched.** `/extract`'s output is still exactly `/decide`'s input
  (`bridge/observations.py` / `to_abstract_observations`, frozen Move 2–3); the body composes them, it
  does not reshape them. The brain's terminal decision (`decide_full`) is byte-for-byte as Stage 1/2a —
  the `gather` branch is additive; with the feature-policy disabled, the daemon reduces to Move-2a.
- **The single reasoner holds.** No posterior or EU is computed in the body or the lib; both transport
  the brain's outputs. `gather` selection is the *daemon's* (the feature-policy is Julia, beside
  `decide_full`), not the body's and not the model's.
- **Statelessness holds.** The daemon rebuilds the posterior per `/decide`; the body holds the
  per-question evidence and resends it (uniform with the bridge); per-question isolation by construction.
- **pkm is untouched; life-agent's live ask path is untouched.** The gate driver is a new script beside
  `ask.py`; it reuses the bridge, rebuilding nothing.

## 4. Worked end-to-end example (the mobile-number class — synthetic, no PII)

The canonical failure (`master-plan.md` §Context): the *current* value is retrieved and cleanly
extracted but loses on corroboration count to a *stale 2015-era* value; recency would flip it, subject
would wrongly drop the truth-bearing admin doc.

1. **retrieve / extract (baseline).** Body → `/retrieve` → hits; → `/extract` (no covariates) →
   candidates `{V_current, V_stale(2015), …}`, abstract observations, `rho`. Accumulated.
2. **decide.** Body composes evidence + `u_bar` (`/utility`) → `POST /decide` with `era_split = yes`.
   The daemon builds the posterior — `V_stale` leads by corroboration count (the documented failure),
   whether that leaves it below the bar (it would abstain) or, when it out-documents enough, ABOVE it (it
   would confidently report the *stale* value). Either way the **feature-policy** fires on `era-split`:
   recency is the class-valid discriminator and a cheap re-weight, so it precedes the report ⇒ `effector
   = gather`, `probe = recency`, `target = V_stale`. (Subject is **not** selected: it mis-fires on
   owner-scoped contact facts.)
3. **gather (steer).** Body receives `gather(recency)` → `/probe/recency` → `doc_date` covariate →
   `/extract` **with the recency covariate** → observations whose `time_factor` decays `V_stale`.
   Accumulated (the union only grows).
4. **decide again.** `POST /decide` with the augmented evidence. The posterior re-weights; `V_current`
   now leads **above the report bar** ⇒ `effector = report`, `value = V_current`. Body **allows** the
   `answer` tool, rewriting its input to `V_current` + the supporting citations.
5. **contrast — the abstain path.** Had the leader stayed dispersed (no discriminating covariate, or
   below bar after gathering), `/decide` returns `abstain`; the body **blocks** `answer` and emits the
   withheld leader (the abstain-show-withheld contract), never a confident wrong.

The loop closed because the **brain** selected recency (not subject) from the posterior's shape — the
gather.py policy, now brain-resident, EU-gated, and replaceable by `voi_gather`/learning.

## 5. Open design questions

**Q1 — the de-dup boundary (RESOLVED, author, in conversation: `+ shared TS governor lib`).** The one
BDSL parser stays in the credence library (`src/parse.jl`/`eval.jl`); the daemon serves the manifest
(`GET /manifest`); the body fetches+verifies and has **no** TS parser. The transport-agnostic governor
(dispatch + fail-open + the `askBrain` seam + wire types) is extracted into `packages/brain-body/`,
which answer-brain consumes now and credence-pi migrates onto later. The two transports (credence-pi
async SSE; answer-brain sync `/decide`) are injected, not shared. *Rejected:* copy credence-pi's
extension (drags the SSE bus); lift it wholesale into a lib (forces a false transport unification).

**Q2 — the gather steer (RESOLVED: feature-policy in the daemon).** The daemon emits `gather(probe,
target)` from the posterior-shape features (`features.bdsl`) via an operator-set policy that encodes
which covariate is class-valid (recency for era-split contact facts; subject for the partner-ID class;
**not** subject for contact facts). It is honest as a *calibrated policy approximating VOI* — not yet a
kernel-priced VOI. The cheap class-valid re-weight (recency on era-split) is applied **before** the
report, not gated by it — a count-led stale leader can sit above the bar, so it must be recency-checked
first (TDD/parity caught a below-bar gate that would have emitted that confident-wrong); the below-bar
EU-gate is reserved for the **expensive** `corroborate` (successor). Generalises the `gather` effector
param `(target)` → `(probe, target)` — a "deployment event" per `capabilities.bdsl`.
*Named successor:* wire the existing `voi_gather` (`answer_brain.jl:204`) with per-probe predictive
kernels (and a learned per-cell policy from the Stage-3 observation log). *Rejected:* full `voi_gather`
now (kernel-modelling risk on the gate); model-selects-probes (covariate selection leaves the brain —
the subject-hurts-contact-facts regression).

**Q3 — the gate driver (RESOLVED: scripted Python gate + local-Ollama demo).** A deterministic scripted
driver (reusing `run_eval`) over the bridge+daemon certifies the closed loop, PII-safe and
confound-free; the pi-mono TS body + minimal local-Ollama app is the qualitative demo, off the gate's
critical path. The daemon owns every decision, so the two thin drivers do not re-implement policy.
*Rejected:* the LLM app as the gate (tool-use nondeterminism confounds the decision-math measurement).

**Q4 — the shared lib's name + placement + workspace (OPEN, minor).** Proposed: a new top-level
`packages/brain-body/` (credence has no TS workspace today — only `apps/credence-pi/extension/` — so this
adds a root npm/pnpm workspace, or answer-brain depends on it by relative path). `brain-body` is a
placeholder name (alternatives: `pi-governor`, `credence-body`). **I recommend** `packages/brain-body/`
+ a minimal root workspace; push back on the name or if you'd rather it live under `apps/_shared/`.

**Q5 — the operator-set gather-policy params (OPEN, minor).** The feature→probe policy thresholds are
calibration priors. **I recommend** declaring them in `bdsl/utility.bdsl` (or a sibling `policy.bdsl`),
committed before the gate runs and folded from the outcomes stream later — **never fitted to the gate**
(the frozen-blind discipline). Push back if you want them as Julia constants instead of bdsl.

## 6. Risk + mitigation

- **The shared lib over-abstracts the transport.** *Mitigation:* the seam is a single injected
  `askBrain(request) → Promise<EffectorSignal>`; the lib knows nothing of SSE vs fetch. Proven by
  answer-brain's sync impl now; credence-pi's async impl (later) is the second consumer that validates
  it — until then the lib is shaped by one consumer and that is stated.
- **The body re-implements the gather *policy* (the thing we moved to the brain).** *Mitigation:* the
  body's `gather` effector only *enacts* a steer the daemon chose (`probe`, `target` come over the wire);
  the feature *extractors* compute projections, not decisions; the policy is Julia, beside `decide_full`.
- **A confident-wrong slips in via the gather loop.** *Mitigation (hard gate):* a covariate that mis-
  fires (subject on a contact fact) is excluded by the feature-policy's class-validity; the gate asserts
  **zero new confident-wrong**; any increment that adds one is reverted (the loop can only gather or
  abstain below bar, never report below bar).
- **The TS body's nondeterminism contaminates the gate.** *Voided by design (Q3):* the gate is the
  scripted driver; the LLM body is a separate demo whose result is reported, not gated.
- **`GET /manifest` drift from the bdsl.** *Mitigation:* the daemon serves it *through the library
  parser* (single source); the body's `verify` fails closed if an effector/feature lacks an impl.
- **pkm-freeze / PII egress.** *Mitigation:* no pkm change; the bridge stays loopback (Move 3); the gate
  driver runs fully on-machine; no real value appears in any credence/life-agent doc or fixture.

## 7. Verification cadence

End-of-move (all green): `packages/brain-body` unit tests (dispatch, fail-open on timeout/unreachable,
manifest-verify) — transport-free, deterministic; `apps/answer-brain/extension` tests against a fake
daemon + fake bridge (gather-then-report on the mobile class, abstain withholds its leader,
statelessness) — deterministic; the Julia daemon tests (`/manifest` round-trips the bdsl; the
feature-policy emits `gather(recency)` on the era-split fixture and `report` after; Stage-1/2a
regression unchanged with the policy off); `ruff` + `mypy --strict` on the gate driver (life-agent);
the **gate** (`run_eval --gate` through the scripted driver) **run and reported** — zero new
confident-wrong, `P(Δ>δ)` stated, the answer-rate + disagreement region published; **one** opt-in
local-Ollama end-to-end demo run and its result stated (its nondeterminism disclosed, not gated).

## 8. de Finettian discipline self-audit

1. **Every numerical query through `expect`?** The body + lib compute **no** posterior — they transport
   the daemon's `/decide` (built on `candidate_posterior` + `optimise`/`value`). The gather *selection*
   is the daemon's feature-policy; it is a *policy over the posterior's shape*, and where it stands in
   for VOI it says so (§0, §5 Q2) — it does not fabricate an EU it did not compute.
2. **Prevision inside a Measure or vice-versa?** No new wrapping. The lib + body hold no Credence
   object; the daemon's additions reuse `decide_full`'s posterior + the declared features.
3. **Opaque closure where declared structure fits?** The `askBrain` seam is an injected function (the
   transport genuinely differs per app); the effectors/features are a *declared registry* verified
   against the library-served manifest, not opaque dispatch.
4. **`getproperty` override on a Prevision subtype?** No — no methods added to any Prevision/Measure
   type, in either repo.

## Reviewer checklist

- [ ] §0 names transient state explicitly (credence-pi not refactored now; gather = feature-policy not
      full VOI; gate = scripted not LLM; daemon stays stateless, learning is Stage 3).
- [ ] §5 records the three author-approved resolutions (Q1 shared-lib boundary; Q2 feature-policy; Q3
      scripted gate) + the two minor open questions (Q4 lib placement, Q5 policy params).
- [ ] §2 keeps the single reasoner: no posterior/EU in the lib or body; `gather` selection is the
      daemon's; the body enacts and the features project.
- [ ] file:line / SHA citations for current-state references (pi-mono hooks @ae89286d
      `…/types.ts:1020`/`:1416`, `examples/sdk/01,06`; the library BDSL parser `src/parse.jl`/`eval.jl`;
      `daemon/server.jl` stateless; `answer_brain.jl:204` `voi_gather`; `capabilities.bdsl`/`features.bdsl`).
- [ ] The move needs no later move to retract it (the lib API + `/decide`'s additive `gather` fields +
      `/manifest` are additive; credence-pi's migration is purely additive onto the frozen lib).
