# Master plan — Bayesian governance sidecar for agentic harnesses

Branch family: `posture-5/`

## Strategic context

Posture 5 opened targeting credence-proxy v0.1 as a model-tier router. Two findings invalidated that positioning. Move 0's cache-discipline audit (PR #78) established that Anthropic's OAI-compatible endpoint does not support prompt caching — the single largest cost lever available to a gateway proxy is structurally inaccessible without migrating to the native Messages API. Move 2's routing benchmark then demonstrated that Bayesian model-tier routing collapses to always-Haiku under standard Anthropic pricing: across 20 workloads (N=3, 309 total turns), the router selected Haiku 100% of the time because Haiku's quality-per-dollar ratio is too high for the EU calculation to explore costlier models. The "When Routing Collapses" paper (arXiv:2602.03478, Feb 2026) confirms this is a structural property of scalar-score-trained routers, not a Credence-specific bug — when the cheap model is "good enough" across categories, no amount of Bayesian learning shifts the posterior far enough to overcome the cost gap.

The competitive landscape clarifies the opportunity. Gateway-level routers (TensorZero, Not Diamond, Martian, Unify) optimise model selection at the request boundary. Harness-level guardrails (Helicone, Portkey, Bifrost) provide observability and policy enforcement after the fact. Neither operates *inside* the agent loop with a continuously-updated posterior over task outcomes, intervening on individual tool-call decisions in real time. This is the empty quadrant: in-loop Bayesian governance. The MVP is a plugin for an existing agentic harness that intercepts tool-call decisions, computes expected utility under a posterior that persists across compaction boundaries, and intervenes (veto, downgrade, re-route, halt, escalate) before runaway loops, redundant calls, and known failure modes materialise. Routing remains a feature inside governance; it is no longer the headline.

OpenClaw is the first integration target: permissive license, active development, documented plugin hook surface (`before_tool_call`, `after_tool_call`, `before_compaction`, `after_compaction`), well-catalogued failure modes in the issue tracker, and a persistence layer (MEMORY.md, SQLite + sqlite-vec) that provides a natural home for posterior state across compaction boundaries.

## MVP definition

The MVP is **a working OpenClaw plugin that demonstrates Bayesian governance of the agent loop, evaluated credibly enough that users can trust it solves the problem it claims to solve.**

Three properties, each load-bearing:

- **Viable.** The plugin does the thing it claims: intercepts tool-call decisions, computes EU under a persistent posterior, and intervenes on the specific failure modes it targets.
- **Minimum.** No scope beyond what's required for user trust. No multi-harness adapters, no publication-grade evaluation, no cache-aware governance, no personal-agent integration.
- **MVP.** A product real users can install and benefit from. Not a proof-of-concept, not a benchmark artifact, not a paper appendix.

The previous MVP definition (cost-savings model-tier router, credence-proxy v0.1) is superseded as of 2026-04-28.

## Moves

### Move 1 — OpenClaw integration prototype

**Scope.** Read the OpenClaw codebase deeply enough to understand the plugin hook surface in detail. Identify the precise integration mechanism for `before_tool_call`, `after_tool_call`, `before_compaction`, `after_compaction`, and the persistence layer. Prototype a minimal hook that calls Credence's Julia substrate via sidecar IPC and demonstrates one end-to-end intervention on a toy task.

**Deliverable.** An architectural design doc specifying the integration approach (IPC mechanism, posterior persistence strategy, hook registration), plus a working prototype demonstrating one complete intervention cycle (observe tool-call intent → compute EU → intervene → observe outcome → update posterior) on a synthesised toy task.

**Halting conditions.** If any of the following surface, Move 1 absorbs the finding and the design conversation resolves it before Move 2 commits:
- Hook surface doesn't expose the information needed for EU calculation (e.g., tool-call arguments not available in `before_tool_call`).
- Sidecar IPC latency makes synchronous `before_tool_call` intervention impractical (>500ms round-trip on representative hardware).
- Posterior persistence across compaction boundaries is architecturally blocked (MEMORY.md and SQLite both inadequate for the state shape).
- OpenClaw's TypeScript runtime has incompatibilities with Julia-side IPC that require a non-trivial bridge layer.

**Cadence.** Design doc precedes code, per Posture 4 convention. Estimated duration: 1–2 weeks.

### Move 2 — Plugin v0.1

**Scope.** Production-grade OpenClaw plugin implementing `before_tool_call` and `after_tool_call` hooks with persistent posterior across compactions (via the mechanism Move 1's design conversation determines). Intervention vocabulary covering the four interventions that map to known OpenClaw failure modes:

| Intervention | Trigger | Action |
|---|---|---|
| **Veto-and-downgrade** | EU of planned tool call < EU of a cheaper/less-risky alternative | Replace the planned call with the alternative |
| **Veto-and-halt** | EU of continuing < EU of idle (belief that "next call adds value" has degraded) | Refuse the tool call, halt the loop |
| **Route-to-cheaper-model** | Posterior over model-quality-on-this-task-category supports downgrade | Downgrade model selection without affecting tool semantics |
| **Escalate-to-user-confirmation** | EU dominated by preference uncertainty (high posterior variance, asymmetric error costs) | Surface the decision to the user via OpenClaw's confirmation surface |

The plugin specifically targets documented runaway-loop failure modes:
- **Issue #34574** (`exec` repetition): loop detection that catches non-`read` tool repetition and triggers veto-and-halt when the posterior belief in loop utility degrades below the idle threshold.
- **Issue #1084** (compaction wipes confirm instruction — the Yue/Meta inbox-deletion class): preserves confirm-instruction priors across compaction boundaries so that the governance posterior survives context truncation.
- **Issue #65550** (dreaming loops): no-confidence intervention thresholds with halt-and-escalate when posterior variance over outcome utility exceeds a configurable threshold.

**Deliverable.** An installable OpenClaw plugin. After Move 2, there is something a user can install and use.

**Halting conditions.** Findings from Move 1 that constrain the design are resolved in Move 1, not deferred to Move 2. Move 2's halting conditions are implementation-level: if any intervention is architecturally infeasible with the integration approach Move 1 selected, halt and redesign the affected intervention (not the whole plugin).

**Cadence.** Design doc precedes code. Estimated duration: 4–8 weeks, depending on what Move 1 surfaces about integration complexity.

### Move 3 — Targeted demonstration evaluation

**Scope.** The cheapest credible evaluation that converts "the plugin works on toy tasks" into "users can trust the plugin to solve the problem it claims to solve." This is *not* the publication-grade evaluation — that's post-MVP roadmap work.

The evaluation is a targeted demonstration on the specific failure modes the plugin claims to mitigate:

- 5–10 representative scenarios spanning the canonical OpenClaw failure modes (issue-tracker-documented or community-reported).
- For each scenario, run OpenClaw with and without the Credence plugin.
- Show that the plugin catches and intervenes on the failure the scenario exercises.
- Document outcome counts (success / failure / cost / token usage / wall-clock) per scenario per condition.
- Where source incidents are publicly documented (Yue/Meta inbox-deletion, Bitcoin-trading agent $1k–$5k/day burn, the 121×-same-shell-command incident from Issue #34574), the demonstration scenarios should be faithful reproductions or close synthetic analogues.

**Deliverable.** A short demonstration document plus a documented test suite. Possibly a video walkthrough for the canonical scenarios. The point isn't statistical power; it's demonstrating that the plugin does what it claims on the failures it claims to mitigate.

**Halting conditions.** If faithful reproduction of a source incident requires setting up harness states that are impractical to synthesise (e.g., a live email account for the Yue/Meta class), substitute a close synthetic analogue and document the substitution. If the plugin fails to intervene on a targeted failure mode, that is a Move 2 bug, not a Move 3 finding — halt and fix upstream.

**Cadence.** Estimated duration: 1–2 weeks if scenarios can be synthesised from issue tracker descriptions; longer if faithful reproductions require setting up specific harness states.

## Closure (2026-04-30)

v0.1 is shippable. All three moves complete. See `posture-5-closure-confirmation.md` for the full closure record.

## Queued for next conversation

**Constants-cleanup follow-up PR.** Three items:

1. **10× retirement ratio** (`brain.jl:292`, `precision > 10.0 * prior_strength`): the multiplier in compaction-survival decay that determines when a registered instruction is retired based on accumulated approval evidence.
2. **5-consecutive-evaluations span** (`detectors.jl:85`, `NO_CONFIDENCE_SPAN = 5`): how many consecutive high-CV evaluations the #65550 no-confidence detector requires before firing.
3. **`p*(1-p)*0.1` proxy threshold** (`detectors.jl:38`): the stationarity detector's outcome-variance threshold, currently a fixed 10% fraction of the posterior predictive variance.

All three are configurable defaults in code. The cleanup either moves them to config files with documented defaults, derives them from posterior structure (Amendment 1 form), or accepts them as deployment-tunable parameters with explicit naming. Decision deferred to that PR's design doc.

## v0.1 deployment

The next piece of work after the constants-cleanup PR is getting users running the plugin against real OpenClaw — collecting the deployment evidence the post-MVP roadmap items depend on. This is operational rather than design work and probably wants a Posture 6 conversation.

## Post-MVP roadmap

Each item is real, tracked, and deferred — not forgotten and not in scope for MVP. Roadmap re-affirmed at closure (2026-04-30) with additional items surfaced during the three-move build.

### Publication track

Two papers, sequenced.

**Paper One — workshop reframing** (weeks not months): *"When Bayesian Routing Collapses: Cost-Quality Trade-offs in Agentic Harness Routing."* Cites arXiv:2602.03478 explicitly; reframes Move 2's routing-collapse benchmark as motivating evidence for governance-over-routing. The superseded benchmark report (`docs/posture-5/superseded/move-2-routing-benchmark-results.md`) is the primary data source. Establishes credibility trail and prior art for Paper Two. Time-sensitive given the Feb 2026 arXiv paper's freshness.

**Paper Two — substantive contribution** (conference target AISTATS 2027 or NeurIPS 2026): *"Bayesian Governance of Agentic Harnesses: In-Loop Decision-Theoretic Intervention for Cost-Aware, Reliability-Aware Tool Use."* Full architectural pitch with HAL-submitted SWE-bench Pro / τ-bench / RoTBench Pareto frontiers, Bayesian Multi-LLM Orchestration-style ablations, IRD-style utility-function sensitivity validation.

### Multi-harness expansion

Two adapters, sequenced.

**Anthropic Agent SDK adapter.** Single integration covering Claude Code and NanoClaw (container-isolated harness); reuses Move 2's governance machinery; widens addressable surface.

**OpenHands V1 adapter.** Strongest published-numbers comparator (77% on SWE-bench Verified with Sonnet 4.5); academic-grade event-stream architecture with `SecurityAnalyzer` and `ConfirmationPolicy` interfaces built for Bayesian intervention; useful as a benchmarking platform for the publication track.

### Personal-agent direction

The previously-shelved Move 9 personal-agent work (Maildir email via mbsync, Telegram trainer with feedback, persistence, server loop) remains deferred to a later posture (provisionally Posture 6). The OpenClaw plugin gives the brain/body interface its first empirical test; by the time the personal-agent direction is queued, there's evidence about what bodies actually need from the brain. Now informed by v0.1 deployment evidence.

### Cache-aware governance v2

v1 ships scope-out for cache savings (Move 0 finding: prompt caching unsupported on OAI-compatible endpoint). v2 may absorb cache-aware governance via native Messages API migration if deployment evidence shows cache effects are load-bearing for users. Long-context coding tasks are the most likely place this matters.

### LLM-based instruction extraction

v0.1 uses regex pattern-match (7 patterns, Move 2 design §5.4). False negatives on unusual phrasings are the expected failure mode. v0.2 direction: LLM-based extraction replaces the regex approach.

### Cross-machine state sync

v0.1's per-machine persistence gives each sidecar instance its own posterior trajectory. Multi-machine users experience approximately-the-same-agent (Move 2 design §5.3). Beta-distribution sufficient statistics make future merge well-defined; implementing cross-machine sync makes the approximately-the-same-agent property exact.

### Semantic task categorisation

v0.1 infers tool category from name and argument patterns (code, delete, deploy, version-control, dependency, generic). v0.2 direction: semantic categorisation beyond feature-driven pattern matching.

### Detectors for additional failure modes

v0.1 targets Issues #34574, #1084, #65550. Additional documented OpenClaw failure modes for future detectors: Issues #14729, #41291, #28576. The detector architecture supports adding without rearchitecture — each detector reads the same posterior and triggers the same intervention vocabulary.

## Superseded material

The original five-move plan (cache-discipline audit → benchmark methodology → benchmark execution → release engineering → distribution) is superseded. Moves 0–2 of the original plan executed and produced the findings that motivated this amendment:

- **Move 0** (cache-discipline audit, PR #78): finding absorbed into strategic context.
- **Move 1** (benchmark methodology): methodology was sound; the benchmark it designed confirmed routing collapse.
- **Move 2** (benchmark execution): results preserved at `docs/posture-5/superseded/move-2-routing-benchmark-results.md`.
- **Moves 3–4** (release engineering, distribution): superseded by the governance-sidecar direction.
