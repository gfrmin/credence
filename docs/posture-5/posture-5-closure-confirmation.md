# Posture 5 closure confirmation

## Summary

Posture 5 opened on 2026-04-27 targeting credence-proxy v0.1 as a model-tier router. Move 0's cache-discipline audit (PR #78) established that Anthropic's OAI-compatible endpoint does not support prompt caching — the single largest cost lever for a gateway proxy is structurally inaccessible without native Messages API migration. Move 2's routing benchmark then demonstrated that Bayesian model-tier routing collapses to always-Haiku under standard Anthropic pricing: across 20 workloads (N=3, 309 total turns), the router selected Haiku 100% of the time because Haiku's quality-per-dollar ratio is too high for the EU calculation to explore costlier models (since published as arXiv:2602.03478). The strategic shift to Bayesian governance sidecar followed via master plan amendment (PR #84), restructuring Posture 5 around in-loop governance of agentic harnesses with OpenClaw as the first integration target. Three moves to MVP. v0.1 ships as the governance product.

## The architectural claim

Bayesian decision theory operationalised for in-loop governance of agentic harnesses, OpenClaw integration first, with empirical demonstration on five scenarios drawn from documented OpenClaw failure modes.

Evidence: `docs/posture-5/move-3-demonstration-evidence.md` (PR #96). Five scenarios, all three detectors fire, both halt and escalate interventions exercised, cross-detector independence verified.

## What v0.1 contains

A working OpenClaw plugin (`apps/openclaw-plugin/`) with the four-intervention vocabulary (veto-and-halt, veto-and-downgrade, route-to-cheaper-model, escalate-to-user-confirmation). The plugin registers `before_tool_call`, `after_tool_call`, `before_compaction`, and `agent_end` hooks.

A Julia sidecar (`apps/credence-governance-sidecar/`) with:

- Continuously-updated Beta posteriors over (tool × category), where category is inferred from tool name and argument patterns (code, delete, deploy, version-control, dependency, generic).
- Per-user persistence surviving sidecar restarts, using Posture 3's schema-versioned serialisation format.
- Compaction-survival via instruction registration: seven regex patterns (Move 2 design §5.4) matching user instructions across four action classes (delete, deploy, privileged-exec, dependency), with self-tuning decay that retires registered instructions when accumulated evidence (approval rate exceeding denial rate with sufficient precision) warrants it.
- Fail-open behaviour: when the sidecar is unreachable, the plugin proceeds without governance and warns once.
- Three failure-mode detectors:
  - **#34574** (stationarity): halts when per-(tool, argument-hash) outcome variance falls below a posterior-derived threshold, catching exec-repetition loops.
  - **#1084** (compaction-survival): escalates post-compaction actions matching registered instruction classes, catching the Yue/Meta inbox-deletion failure pattern.
  - **#65550** (no-confidence): halts when the coefficient of variation of EU(proceed) over a sliding window exceeds a posterior-concentration-derived threshold for five consecutive evaluations, catching dreaming loops.

Demonstration evidence on five scenarios (PR #96): inbox-deletion class, exec-repetition, no-confidence dreaming, deploy-class compaction-survival, and mixed-mode cross-detector independence.

## Architectural properties verified during Posture 5

Several properties surfaced during the three-move build that are worth recording as durable architectural facts:

**Carrier-dependent dispatch invariant (§5.1a from Posture 4) carried through cleanly.** The brain operates at Measure level throughout (BetaMeasure posteriors for per-category beliefs). The Posture 4 invariant — that Prevision-level dispatch requires non-carrier-dependent leaves — didn't bite in any unanticipated way because the governance sidecar's belief objects are uniformly Beta, which is carrier-independent.

**Latency budget (100ms hook contract) is essentially unconstrained.** The brain runs at 0.1ms p99 (Move 1 prototype, PR #85). The 100ms hook budget from OpenClaw's `before_tool_call` contract leaves 99.9% headroom. Any computation Credence-shaped — posterior updates, EU calculations, detector checks — fits comfortably. This means future detector additions face no latency pressure.

**Cross-detector independence holds.** Scenario 5 (mixed-mode) demonstrated that two detectors (#34574 stationarity and #1084 compaction-survival) can fire on the same posterior trajectory in sequence without suppressing each other. The stationarity detector halted a retry loop at step 5; the instruction detector escalated a destructive action at step 11. Posteriors are per-(tool, category), so different detectors operating on different categories are structurally independent.

**Persistence survives sidecar restarts cleanly.** The schema-versioned format from Posture 3 supports the v0.1 schema without extension. User ID, posteriors, registered instructions, and observation counts round-trip through JSON serialisation at the `~/.credence/` state directory.

**All four interventions map natively to OpenClaw's `before_tool_call` return values.** Halt maps to `{ block: true, blockReason }`. Escalate maps to `{ requireApproval: { title, description, severity, timeoutMs, timeoutBehavior } }`. Downgrade and route map to the same vocabulary. No workarounds or protocol extensions required.

## The strategic shift

Posture 5 looks different from Posture 4. Posture 4 was substrate-shaped: eight moves of de Finettian migration, predictable and sequential, each move's scope derivable from the previous move's completion state. Posture 5 was methodology-shaped: the original five-move plan (cache audit → benchmark methodology → benchmark execution → release engineering → distribution) executed its first two moves and produced findings that invalidated moves 3–5. The master plan amendment (PR #84) restructured around governance, producing a three-move MVP path that was more ambitious than the original plan but better grounded in empirical evidence.

The lesson is that empirical evidence reshapes plans more than plans reshape evidence. The audit-driven cadence — design doc first, halt-on-surprise, constants tracked not absorbed — is what enabled the pivot. Without Move 0's cache finding and Move 2's routing-collapse benchmark, the routing product would have shipped into a structural dead end. The cadence surfaced the dead end before the product shipped.

## Cadence note

Posture 5's design-doc-first discipline transferred cleanly from Posture 4 despite the work shape being methodology-shaped rather than substrate-shaped. Three design docs landed (Move 1, Move 2, Move 3), each scrutinised as an actual document rather than a summary, each with an "Open design questions" section that invited pushback. The cadence is now established practice across two postures with different work shapes: design doc lands first, scrutiny of the actual document, halt-on-surprise rather than work-around, constants and deferred items tracked rather than absorbed silently. This is a durable claim about how this project operates.

## Merged PRs

| Phase | PR | Title |
|---|---|---|
| Master plan amendment | #84 | posture-5: master plan amendment for governance-sidecar MVP |
| Move 1 design + prototype | #85 | posture-5/move-1: design doc — OpenClaw integration prototype |
| Move 2 design doc | #86 | posture-5/move-2: design doc — plugin v0.1 |
| Move 2 design amendments | #87 | posture-5/move-2: design doc amendments — threshold derivation, race-window clarity, decay self-tuning |
| Move 2 implementation design | #88 | posture-5/move-2: implementation design — sub-PR phasing |
| Move 2 sub-PR 1 | #89 | posture-5/move-2 sub-PR 1: sidecar brain |
| Move 2 sub-PR 2 | #90 | posture-5/move-2 sub-PR 2: intervention vocabulary |
| Move 2 sub-PR 3 | #91 | posture-5/move-2 sub-PR 3: persistence machinery |
| Move 2 sub-PR 4 | #92 | posture-5/move-2 sub-PR 4: compaction-survival pattern matching |
| Move 2 sub-PR 5 | #93 | posture-5/move-2 sub-PR 5: fail-open behaviour |
| Move 2 sub-PR 6 | #94 | posture-5/move-2 sub-PR 6: three failure-mode detectors |
| Move 3 design | #95 | posture-5/move-3: design doc — targeted demonstration evaluation |
| Move 3 execution | #96 | posture-5/move-3: targeted demonstration evaluation — 5 scenarios |

## Open items

- **Constants-cleanup follow-up PR.** Three items queued: the 10× retirement ratio in compaction-survival decay, the 5-consecutive-evaluations span in #65550, the `p*(1-p)*0.1` proxy threshold in #34574. See `docs/posture-5/master-plan.md` §Queued.
- **pomdp_agent status.** Dormancy continues from Posture 4; no activity during Posture 5.
