# credence-pi Pass 2 — Move 2 design: the dollars-saved surface + demo

> Per `docs/posture-3/DESIGN-DOC-TEMPLATE.md`. Completes the MVP (Moves 0–2 + demo) — the
> user/VC-facing "return you can see." Product-first; the cost-denominated *decision* utility and
> feature-conditioned learning are Phase 2 (data-gated — see §"Deferred").

## Purpose

Turn the data Move 1 accumulates into a human-facing **governance + spend report** ("$ saved"), and
ship a **reproducible demo** that shows the product mechanic end-to-end on the real brain — without
needing a live OpenClaw or real data.

## What this lands

1. **Decision logging (daemon).** `emit_signal!` now writes a `decision` record
   (`{event_type:"decision", in_response_to, action}`) to the observation log alongside every
   effector signal. **Why:** the log previously held only inbound sensor events, so *silent
   auto-blocks/auto-proceeds* (no `user-responded`) were invisible — the savings surface undercounted
   prevented spend ~10× (the demo proved it: 1 counted vs 10 actual). Decision records are **derived**
   (replay reconstructs them from the posterior), so `replay_user_responses` ignores them and replay
   correctness is unaffected. Documented in SPEC.md.
2. **The dollars-saved surface** (`apps/credence-pi/savings.jl`). Reads the observation log and
   reports: total per-turn spend + tokens, brain decisions (proceed/block/ask), user
   approvals/denials, and an **explicitly-bounded** prevented-spend estimate
   (`blocked_calls × avg_turn_cost`). **Display-only / non-causal** — its arithmetic never feeds a
   decision or belief, so it is outside Invariant 1's scope (the "diagnostic telemetry" carve-out);
   it calls neither `condition` nor `expect`. Lint-clean.
3. **Reproducible demo** (`apps/credence-pi/demo/governance_demo.jl`). Drives the real daemon
   dispatcher on a synthetic session: the agent repeats a wasteful `bash` call → the brain asks → the
   user denies → the global approval posterior concentrates → the brain **auto-blocks by EU
   maximisation** (no hard-coded rule) → the savings report shows the caught spend (~$1.20 across 10
   turns). Output is the user/VC artifact.

## Honesty (baked into the report + demo framing)

- **Cost is per-turn, not per-tool** (pi exposes no per-tool usage) → prevented spend is an
  ESTIMATE (`blocked × avg-turn-cost`), labelled as such, never a guaranteed figure.
- **Pass-1 brain is a single GLOBAL approval posterior.** The demo learns this user's tool-approval
  rate; it does **not** detect loops per-context. The honest claim is "governs, learns from you,
  auto-blocks when confident, reports caught spend" — not "loop detection" (that is Phase 2).
- The demo prints these caveats inline.

## Files

`apps/credence-pi/daemon/server.jl` (decision logging in `emit_signal!`); `apps/credence-pi/savings.jl`
(new, display-only); `apps/credence-pi/demo/governance_demo.jl` (new); `apps/credence-pi/tests/julia/test_savings.jl`
(new); `apps/credence-pi/tests/julia/test_server.jl` (decision-logging assertion);
`apps/credence-pi/SPEC.md` (`decision` record + earlier `turn-cost`).

## Verification

`julia apps/credence-pi/tests/julia/test_server.jl` (22/22, incl. decision logging);
`test_savings.jl` (3/3, incl. silent auto-block counting); `governance_demo.jl` (shows ask→auto-block,
~$1.20 caught); credence-lint clean (9 files, 0 violations).

## Deferred to Phase 2 (data-gated — NOT forced on hypothetical data)

- **Cost-denominated *decision* utility** (replace decide.bdsl's symmetric ±1 with a dollar utility).
  Genuine cost-aware halting needs a per-call cost estimate at decision time + a value model; the cost
  signal is per-turn and arrives after the call, so this wants accumulated data to estimate per-context
  cost. The demo already shows EU-optimal blocking from learned approval; dollar-denomination refines
  *when* it blocks.
- **Feature-conditioned learning / CEG** (loop-*detection*), outcome conditioning, calibration eval —
  all need the real n=1 observation log the deployed plugin produces. Per the brief, designing these
  on hypothetical data is the trap to avoid. **The unlock is the user deploying Move 1** (runbook in
  the plugin README) and accumulating sessions.
