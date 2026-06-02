# credence-pi Pass 2 — master plan

> Modelled on `docs/posture-3/master-plan.md` and `docs/paper1/master-plan.md` per
> CLAUDE.md "Multi-move branches: design-doc before code". Read `PASS-2-BRIEF.md`,
> `apps/credence-pi/SPEC.md`, and `apps/credence-pi/PASS-2-NOTES.md` before touching code.
> Status: **DRAFT — Move 0**, awaiting author sign-off. No code lands until this is approved.

## 1. Context

Pass 1 (closed 2026-05-05, PR #102) shipped the credence-pi body-brain governance loop as an
**integration** deliverable: opaque-brain discipline end-to-end, three-effector EU-maximisation
(`ask`/`refuse`/`allow`), a single global `Beta(2,2)` over tool-approval, five declared kebab-case
features, a JSONL observation log with replay-to-tolerance, an SSE+POST wire, fail-open semantics,
and **zero production-side lint pragmas**. Pass 1's success criteria were integration, not
capability; its brain does not condition on features at decision time and does not learn from
outcomes.

**Pass 2's goal is a product.** A real, installable governance product that helps pi.dev / OpenClaw
users **save money and time** by halting wasted agentic token spend in real time, with a "dollars
saved" surface — credible enough to attract VCs as well as users. This is market-timed: WSJ
(2026-05-28) documents enterprises rationing AI as token costs skyrocket (only ~18% of agentic
*coding* token spend reaches shipped products per EntelligenceAI; Uber exhausted its annual agentic
budget by March), reacting with *crude* rationing (cutting tool access). credence-pi is the
*intelligent* alternative: an expected-utility calculation in front of each agent decision —
"is this token spend worth its expected contribution?" — halting waste by computation, not by a
hard access cap.

### 1.1 The trajectory of framings (honest epistemic history)

routing-as-product (credence-proxy; collapsed to always-Haiku, arXiv:2602.03478) → governance
sidecar v0.1 (`apps/credence-governance-sidecar/`; host-side EU arithmetic, hand-rolled regex
partitions, 29 lint pragmas; demonstrated on fixtures, never deployed) → **credence-pi** (the clean
rebuild: opaque brain, BDSL, zero pragmas) → **cost-aware deploy-first governance** (this plan). The
parked sidecar is deliberately *not* the model; its detectors are constitution-violating hard
thresholds (§7 Move 6).

### 1.2 Verified starting state (research run `wf_460c6158-00d`, 2026-06-02)

- **Dataset: EMPTY.** No `~/.credence-pi/` exists — zero accumulated events. This *forces*
  deploy-first (PASS-2-BRIEF.md lines 209–215: "a real outcome of the dataset check, not a
  fallback").
- **Extension is bound to no SDK.** `extension/package.json` has no pi dependency and builds
  `tsc --noEmit`; it defines abstract `PiAPI` interfaces. Never wired to a real harness.
- **Binding is a thin adapter.** pi `pi.on("tool_call", h) -> {block?, reason?}` (undefined =
  proceed) is an *exact* structural match to credence-pi's assumed `HookReturn`
  (`@earendil-works/pi-coding-agent` `src/core/extensions/types.ts:993-997,1132`). Deltas are
  mechanical (§7 Move 1).
- **Cost signal exists at per-TURN granularity** (`event.message.usage.cost.total`,
  `packages/ai/types.ts:264-277`), not per-tool. Custom/OAuth providers may report `cost.total=0`
  → token×price fallback needed.
- **Learning-brain primitives already exist in `src/`** — Beta-Bernoulli + WeightedBernoulli
  soft-evidence, per-cell posteriors (`Vector{BetaPrevision}`+`MixturePrevision`, the paper1 B2c
  pattern in `apps/julia/qa_benchmark/category_update.jl`), the Functional hierarchy, and
  `voi`/`net-voi`/`optimise`/`value`/`eu`. A dollar-denominated halt is *already* EU-optimal by
  computation. ~Zero new Tier-1 primitives.

## 2. Thesis (dual — but product-first)

**Priority (author, 2026-06-02): ship user/VC value as fast as possible; the academic/paper track
is secondary and must never gate shipping.** The two strands below are deliberately separated so the
gameable one never masquerades as the falsifiable one — but the product strand is the critical path
and the empirical strand is opportunistic (it rides on data the product collects anyway).

- **PRIMARY — Product (the demo, the user/VC surface, the thing that ships):** *dollars of wasted
  agentic token spend prevented per session/week, at a low false-intervention rate*, plus a
  reproducible "here's a runaway loop → credence-pi halts it → $X saved" demonstration. This is a
  single-user (n=1) operational metric — legitimate as a KPI and a demo, and it is what defines the
  MVP being shippable. It is **not** dressed up as a general empirical claim.
- **SECONDARY — Empirical (opportunistic, paper-feeding, never a ship gate):** *"On my own
  accumulated credence-pi observation log, feature-conditioned / structural-CEG learning produces
  better-calibrated per-partition posteriors (Brier/ECE) than Pass 1's single global Beta on the
  same observation stream."* Calibration is the honest headline *for a paper* because accuracy is
  gameable by always-`ask` and dollars is a single-user figure; "no detectable improvement at n=1"
  is a valid falsification. This strand is pursued only after the product MVP ships, and only if the
  accumulated data warrants it — it is a bonus output of the deployment, not a prerequisite for it.

## 3. Settled decisions (do not re-derive)

From PASS-2-BRIEF.md and the constitution: the five opaque-brain commitments; deploy-first when the
log is thin; calibration is the empirical headline; n=1 honesty; **belief updates only via
`condition()`**; **EU-max is the only decision mechanism** (waste-halting must be EU-optimal *by
computation*, never a hard threshold); **zero production-side lint pragmas**; pi/Claude-Code-class
harness only.

Settled this session (2026-06-02, with author):

| # | Decision | Resolution |
|---|----------|------------|
| S1 | Headline framing | **Dual** — dollars-saved product KPI + calibration empirical thesis (§2). |
| S2 | MVP-0 runtime + import scope | **`@earendil-works/pi-coding-agent@0.78.0`** — current pi-mono AND current upstream OpenClaw `v2026.5.28` (0 `@mariozechner` refs). The old-scope risk only existed against the author's ~2-month-stale `gfrmin/openclaw` fork. **Refresh to current OpenClaw / run pi @0.78.0 as the deploy target.** |
| S3 | Cost granularity | **Per-turn** (pi exposes no per-tool usage); the dollars-saved surface attributes a turn's cost to the tools it emitted and is honest that the unit is per-turn. |
| S4 | Dollar-utility form | **Fixed dollar constants** as a static `LinearCombination`/`Tabular` Functional (zero new measure) for the MVP; a *learned* cost magnitude (Gamma-on-PositiveReals + conjugate path) is deferred. |
| S5 | Per-cell brain authoring | **Brain-side Julia app** following `qa_benchmark/category_update.jl` (`Vector{BetaPrevision}` + WeightedBernoulli + `MixturePrevision` marginalisation), **zero `src/` change** — not new DSL forms. |
| S6 | Partition kind | **Hard partition** (Pass-1 feature extractors are deterministic → each observation lands in exactly one cell, plain BetaBernoulli). `:weighted-bernoulli` DSL keyword added only if soft buckets are later needed. |
| S7 | Sidecar detectors | **Mine the math, re-express as EU-optimal; do not revive/port.** T3 (detection) is a separate pass. |
| S8 | dollars-of-waste paper home | **Product KPI outside Papers 1–3** (maps to neither Paper 1's positional cost-Pareto thesis nor Paper 3's drift thesis). The *calibration* result feeds the paper track. |

## 4. Methodology (the empirical gate)

- **Dataset:** the deployed credence-pi log (Move 1 onward). Report counts of `tool-proposed`,
  `user-responded`, `tool-completed`, and per-turn cost events. Saturation of the discrete partition
  is the **precondition** for any continuous-feature follow-up (not a preference).
- **"Useful" =** per-partition calibration (Brier / ECE) as headline; ask-rate floor if accuracy is
  reported at all; optional secondaries (regret vs an oracle stopper; latency/cost-corrected
  utility).
- **Falsification:** what observation says "the CEG move-set is wrong"; "no detectable improvement
  at n=1" is explicitly a valid outcome to report.
- **Paper 3 connection:** Paper 3's thesis is score-delta-under-drift (non-stationarity tracking),
  Paper 1's is the positional cost-performance Pareto. The Pass-2 *calibration* work can feed Paper
  3's forgetting-quality story **iff** a drift scenario is involved; dollars-saved feeds neither and
  stays a product KPI (S8). The plan surfaces, rather than silently leaves, this routing.

## 5. Per-move design-doc template (mandatory)

Each move = a **design-doc PR** then a **code PR**. Design doc sections: Purpose / Files touched /
Behaviour preserved / Worked end-to-end example / Open design questions / Risk + mitigation /
Verification cadence. (Template: `docs/posture-3/DESIGN-DOC-TEMPLATE.md`.)

## 6. Hard prerequisites before any code lands

1. This master-plan PR approved (Move 0).
2. The Move-0 PR carries a **populated** methodology/paper stub, not a lightweight prelude
   (populated-artifact-scoping discipline).
3. The OpenClaw extension-load path + import scope resolved (OQ1; gates Move 1).

## 7. Move sequence (deploy-first; ordered by dependency)

**Two phases, product-first (per §2 priority):**
- **MVP — ship to users/VCs (Moves 0–2 + demo).** The critical path: deployed, installable,
  cost-aware governance that demonstrably halts wasted spend, with a dollars-saved surface and a
  reproducible demo. This is what we ship and show. The MVP brain stays simple (Pass-1 global Beta +
  a cost-denominated `net-voi` halt gate) — deliberately, to ship fast.
- **Phase 2 — make it smarter (Moves 3–6, post-MVP, opportunistic).** Per-context learning, outcome
  conditioning, CEG, and the optional calibration/paper output. Pursued only after the MVP ships and
  only as accumulated data warrants. **None of Phase 2 gates the MVP.**

> Methodology note: the populated Move-0 paper/methodology stub (per the brief's
> populated-artifact-scoping discipline) is kept lightweight here — a one-page stub naming the
> eventual calibration claim — precisely because the paper is now secondary. It exists so the
> deployment collects the right data, not because the paper is on the critical path.

### ═══ MVP (ship to users/VCs) ═══

- **Move 0 — Master plan + lightweight methodology stub + dataset-empty confirmation.** *No code.*
  This file + a one-page methodology stub (the calibration claim, named for later) + Paper-3
  routing note. Confirms `~/.credence-pi/` empty.

- **Move 1 (FORCED DEPLOY-FIRST) — Installable, bound, cost-ingesting credence-pi, deployed to
  accumulate.** Make credence-pi a real loadable pi extension and run it on the author's real
  sessions. Concretely: add `export default function(pi: ExtensionAPI)` wrapping the existing
  tested `installCredencePiExtension`; add `@earendil-works/pi-coding-agent@0.78.0` (S2) to deps;
  add the `"pi":{extensions:[...]}` manifest + config plumbing (daemonUrl, bdsl paths); adapt
  `confirm(text)→confirm(title,message)`; **switch correlation from `toolName` to the stable
  `toolCallId`** (pi runs tools in parallel; current `lastEventIdByTool` mis-correlates);
  **self-compute duration** by keying `toolCallId` across `tool_call`→`tool_execution_end` (no
  `durationMs` on the event); map `sessionId`/messages off `ctx.sessionManager`; ingest per-turn
  USD (`turn_end` → `usage.cost.total`) into the observation log. Run the Pass-1 brain **plus** a
  minimal cost-aware `net-voi` gate using the existing stdlib (zero new `src/`). Verify it receives
  `tool_call` inside current OpenClaw (OQ1); pi-direct is the guaranteed dev harness.
  *Depends on: Move 0; OQ1.*

- **Move 2 — Dollars-saved product surface + cost-denominated utility.** Replace `decide.bdsl`'s
  symmetric ±1 approval-unit preference with a dollar-denominated utility (payoffs + subtracted call
  cost) expressed as a typed `LinearCombination`/`Tabular` Functional (S4) fed to the existing
  `optimise`/`value`/`voi` — no host-side EU arithmetic, no opaque closure (Invariant 1+2). Daemon
  aggregates cumulative spend and "spend halted" itself; token×price fallback for `cost.total=0`
  (S3). This is the VC/user "here are the dollars saved" surface, offline-replayable from the log.
  *Depends on: Move 1.*

- **MVP gate — reproducible demo + ship.** A scripted, reproducible demonstration (the credence-pi
  analogue of the sidecar's 97.5%-loop / $4.35-dreaming-loop demos, but done the constitutional way):
  inject/replay a runaway-loop or redundant-call session → credence-pi halts it by cost-EU → the
  dollars-saved surface attributes the prevented spend. This + a short install/landing writeup is the
  user/VC artifact. **This is the shippable MVP.**
  *Depends on: Move 2.*

### ═══ Phase 2 — make it smarter (post-MVP, opportunistic; does NOT gate the MVP) ═══

- **Move 3 — Promote opaque fields + DSL on-ramp.** Promote extra tool names, `parent_tool_call_name`,
  and `project_id` from opaque strings to growing-categorical declared spaces; add the
  `:weighted-bernoulli` DSL `:family` keyword only if S6 turns up soft buckets. Precondition for
  per-partition conditioning. Small, additive, no behaviour change.
  *Depends on: Move 2; enough accumulated events to make partitioning meaningful.*

- **Move 4 — Outcome-based posterior updating.** Add a secondary-signal observation kernel that
  conditions on `tool-completed` outcomes; add the `tool-completed` JSON round-trip oracle **before**
  wiring (PASS-2-NOTES). Author per-(tool, cell) reliability as a brain-side Julia app (S5).
  Requires settling the `tool-completed` outcome schema first (OQ2).
  *Depends on: Move 3; Move 1 (outcome+cost events flowing).*

- **Move 5 (T1 headline) — CEG-with-meta-actions + calibration evaluation.** CEG over declared
  features (proposal distribution + triggering policy + complexity prior as **one** design doc, not
  four), outcome-conditioned, evaluated for per-partition calibration (Brier/ECE) against the
  Pass-1 single-Beta baseline on the *same* accumulated stream. Saturation of the discrete partition
  gates any continuous-feature follow-up. Report "no detectable improvement at n=1" as valid.
  *Depends on: Move 4; a saturating volume of deployed data.*

- **Move 6 — Sidecar deprecate-or-keep + EU-optimal re-expression of "stuck".** Own the sidecar
  decision (don't implicitly defer). Re-express its loop/waste signals as EU-optimal behaviour, NOT
  ported thresholds: **(a)** put call cost + VOI into the utility so `net-voi<0 ⇒ EU(halt)` wins
  argmax once the posterior concentrates (this *is* the stationarity detector's "repetition no longer
  informative", since its threshold `log(1+1/(α+β))` is an information quantity); **(b)** for genuine
  regime-shift, encode **drift-rate in the hypothesis space** so `condition()` learns when a tool's
  reliability has changed — per the no-forget doctrine, *not* a forget/decay mechanism (OQ3). T3
  detector machinery proper is a separate pass.
  *Depends on: Move 2 (cost utility) + Move 5 (structure).*

## 8. Open design questions

- **OQ1 — OpenClaw extension-load path + scope.** Does current OpenClaw forward `tool_call` to a
  native pi extension (drop into `.pi/extensions` / `--extension` / `loadExtensionFromFactory`), or
  must we register via OpenClaw's `registerCodexAppServerExtensionFactory` plugin registry?
  *(a)* native pi extension (preferred — guaranteed in pi-direct, verify in OpenClaw); *(b)* OpenClaw
  plugin registry; *(c)* both. **Gates Move 1.** Resolvable by reading current OpenClaw's session
  construction; not yet settled.
- **OQ2 — `tool-completed` outcome schema.** What the secondary-signal kernel consumes: the realised
  per-turn cost attributed to the tool, plus the "was-proceeding-right" signal. Note `duration_ms`
  and `error` are NOT on pi's event — self-derive (S-research). *Gates Move 4.*
- **OQ3 — "stuck" re-expression.** Move 6(a) cost+VOI vs Move 6(b) drift-rate-in-hypothesis-space vs
  both. The no-forget doctrine leans toward (b) for genuine regime-shift; (a) for pure repetition.
  Likely **both**, applied to different signals. *Resolve at Move 5/6.*
- **OQ4 — false-intervention budget.** What false-halt/false-ask rate is acceptable for the product
  KPI, and how is it surfaced? *Resolve at Move 2.*

(S1–S8 in §3 are the resolved former open questions, stamped 2026-06-02.)

## 9. Test strategy — three strata, in order

1. **Julia brain unit** — conjugate updates (exact α/β), EU-max selection, calibration computation,
   tolerances; `precedent:test-oracle` where a closed-form oracle is asserted.
2. **TypeScript body** — hook adapter, fail-open, `toolCallId` correlation, per-turn cost ingestion;
   plus a real `pi -e` smoke test (jiti resolution of `.js`-suffixed TS imports is unverified).
3. **End-to-end** — two-process (body ↔ daemon at `127.0.0.1:8787`) replay of a synthetic loop
   session → assert `halt` emitted and the savings surface attributes the prevented spend; calibration
   regression vs the Pass-1 baseline.

Capture canonical numerical state **pre-refactor** and assert `==` for any schema / posterior /
seeded-RNG change (`precedent:test-oracle`, `:capture-mechanism-invariants`). Never regenerate
fixtures to fix a load bug — fix the load code.

## 10. Documentation deliverables

Per-move design docs at `docs/credence-pi-pass-2/move-N-design.md`; the methodology section; updated
`apps/credence-pi/README.md` + `SPEC.md` (wire schema for the cost event); a precedent entry in
`docs/precedents.md` for any genuinely novel lint case (same PR).

## 11. PR cadence

Per move: design-doc PR, then code PR. **Rebase-merge** for linear history (`gh pr merge <N>
--rebase`, never squash — Pass 1's per-step commits are the precedent). Zero production-side lint
pragmas held throughout. CI green + author approval in conversation = merge.

## 12. Verification

`python3 tools/credence-lint/credence_lint.py --repo-root . check apps/credence-pi/` (zero
violations) after every move; Julia `test/test_*.jl` + `apps/credence-pi/tests/julia/*.jl`; the TS
suite + `pi -e` smoke; daemon health at `127.0.0.1:8787`. The MVP gate: installed into a current
pi/OpenClaw runtime, real sessions accumulate, and the "$ saved" surface shows a credible
prevented-waste figure at a low false-intervention rate.

## 13. Out of scope (routed to siblings)

DSL substrate changes (four frozen types, axiom-constrained fns — settled Postures 3+4); Paper 1
Phase B (independent, `paper1/*` branches); sidecar *improvements* (parked — may deprecate per Move
6, not improve); multi-harness expansion beyond pi-class; the `substitute` effector (T2 — separate
pass); detectors as hard thresholds (T3 — separate pass; only re-expressed per Move 6); continuous
features / embeddings (deferred until the discrete partition demonstrably saturates).

## 14. Critical files

`apps/credence-pi/extension/src/index.ts` (default-export gap; `toolCallId` correlation),
`extension/package.json` (pi dep + manifest + build emit), `daemon/server.jl` (cost ingest + outcome
conditioning), `bdsl/decide.bdsl` (dollar utility), `bdsl/features.bdsl`/`kernel.bdsl`/`prior.bdsl`,
`src/eval.jl` (`:weighted-bernoulli` keyword, if S6 needs it),
`apps/julia/qa_benchmark/category_update.jl` (per-cell precedent to mirror).

## 15. Done criteria

**MVP done (the ship gate — primary):** installable + deployed + accumulating on a current pi/OpenClaw
runtime; the cost / dollars-saved surface live and honest (per-turn); a reproducible waste-halting
demo with an attributed $-saved figure at a low false-intervention rate; a short install/landing
writeup for users/VCs; lint clean (zero production pragmas). **This is what "done" means for
shipping.**

**Phase 2 done (secondary, post-MVP, opportunistic):** per-context learning demonstrably improves the
product; the calibration claim demonstrated **or** honestly falsified at n=1; the sidecar
deprecate-or-keep decision made; the methodology resolves the Paper-3 connection. Willingness to
report a negative result is part of this (secondary) bar. None of this blocks the MVP ship.

## 16. Commit-cadence guardrails

Per-step commits preserved (Pass-1's eight-commit precedent). `gh pr merge <N> --rebase` explicit
(never squash). CI green + author approval = merge.
