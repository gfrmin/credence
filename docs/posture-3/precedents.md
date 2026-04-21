# Posture 3 precedents — working reference

A single-file index of patterns that have earned their keep across Moves 0–5 of the Posture 3 reconstruction. Consult before drafting §5 Open design questions and §6 Risk + mitigation in each move's design doc — most precedents apply to one of those two sections.

This file is a working reference, not a constitution. It records patterns that are invoked across multiple moves; first-instance patterns are candidates for inclusion; cross-move patterns belong in. When a precedent overlaps with a CLAUDE.md slug (the author-facing constitution), cross-reference rather than duplicate.

---

## 1. Grep-and-disposition before design-doc drafting

**Where established.** Move 3 design doc §6 R2 formalised the procedure with 351 hits across 29 files; Moves 4 and 5 repeated it with 80 hits (Move 4 R2) and 20 hits (Move 5 R2). The master plan §Verification defines the go/no-go thresholds.

**The rule.** Before drafting §2 Files touched, run a targeted `grep -rn '<pattern>'` across `src/`, `test/`, `apps/`, `docs/` for the type, field, or method the move relocates or reshapes. Pin the total hit count and the per-hit disposition under §6 R2 in three categories: **(a)** covered unchanged (e.g. by a `getproperty` shield, by an alias, by method-table invariance); **(b)** needs minor adaptation (import change, one-line edit); **(c)** is a mutation, a plan-amending finding, or a consumer site that can't accommodate the relocation. Go/no-go gate: **≥90% (a), <15% (b), 0 (c)**.

**When it applies.** Any move that refactors a consumer-visible type, field, or method signature (Moves 2–5 all did). The DESIGN-DOC-TEMPLATE.md §6 names the practice as suggested-not-mandatory — but the bar for omitting the grep is "the refactor touches no consumer-reachable surface" (Move 8-style cosmetic adaptation) and that's rare.

**Failure mode.** Without the grep, a relocation ships and surfaces a (c)-category consumer only at integration time — potentially after the PR has merged. The grep at design-doc-drafting time is cheap; discovering a scope-amending finding mid-code-PR is expensive and can compound across the moves that follow.

---

## 2. Shared-reference contract for `getproperty` shields

**Where established.** Move 3 design doc §3 and §6 R4 named this as a durable contract. The contract test lives at `test/test_prevision_unit.jl :: test_shared_reference_contract`.

**The rule.** When a Measure wraps a Prevision behind a `Base.getproperty` shield, the shield MUST return references to underlying fields — not copies. `push!` on a returned `Vector` must mutate the backing storage. The `getproperty` definitions carry a comment naming the contract test; the contract test asserts the invariant by constructing, reading through the shield, mutating in place, and re-reading to confirm the mutation is visible.

**When it applies.** Any move that adds a new `Base.getproperty` shield or extends an existing one — Move 5's MixtureMeasure shield inherited the contract (components + log_weights by reference); future Moves 6/7/8 that add Prevision subtypes with Measure wrappers will too.

**Failure mode.** A future session, seeing the shield return a mutable internal Vector, "hardens encapsulation" by returning `copy(m.prevision.components)`. `push!` on the copy succeeds silently; the original state is unchanged; no error fires; corruption surfaces later when a read through the shield returns stale components. The invariant-comment-names-test pattern guards against this class — see precedent #3.

---

## 3. Executable documentation (invariant comments name their tests)

**Where established.** Move 3 generalised this pattern from the ad-hoc comment-to-test references in earlier Moves. `CLAUDE.md`'s `test-oracle` precedent slug names a specific case (test-side manual computation); the broader precedent here — **every invariant the code depends on carries a comment that names the test, and the test references the invariant back** — is the Move 3 lift.

**The rule.** If code has a comment that states an invariant load-bearing to correctness (e.g. "shield returns by reference per contract X," "weights are normalised at construction point Y"), the comment ends with a reference to the test file and test name asserting the invariant. The test, in turn, carries a comment citing the code site whose invariant it pins. Breaking either breaks both — the comment drift + test drift becomes symmetric.

**When it applies.** Any invariant that a future session might "helpfully clean up" without understanding. `getproperty` shields (precedent #2), schema-version markers in persistence, seeding disciplines in particle filtering (Move 6), weight-normalisation points.

**Failure mode.** A comment states an invariant but names no test; a test asserts an invariant but references no code site. Six months later a refactor breaks the invariant; neither the comment nor the test catches the regression because the linkage was implicit. The explicit mutual-reference pattern makes the linkage load-bearing and maintenance-visible.

**Cross-reference.** `CLAUDE.md` Precedents section, `test-oracle` slug — this precedent generalises the narrower test-oracle case to any invariant-comment pairing.

---

## 4. Strata tolerances

**Where established.** The master plan §Verification pins the tolerances; Moves 2, 3, 4, 5 all repeated them in §3 Behaviour preserved with no drift.

**The rule.**
- **Stratum 1 (unit equivalence):** `isapprox(atol=1e-14)` for derived scalars; `==` where the arithmetic is closed-form on integer-accumulated values (α/β, log-weights normalised inside a seeded constructor).
- **Stratum 2 conjugate paths:** `==` for integer-accumulated updates (Beta-Bernoulli α+1); `rtol=1e-12` for floating-point arithmetic that legitimately reassociates (Gaussian posterior μ via precision-weighted averaging).
- **Stratum 2 particle under deterministic seeding:** `==`. The only legitimate drift is floating-point reassociation from constructor changes, bounded by ~1e-12 but tighter in practice. Looser tolerance silently masks sample-order changes that *are* posterior-changing.
- **Stratum 2 quadrature paths:** `atol=1e-14`. Pairwise-reduction-legal arithmetic budgets.
- **Stratum 3 end-to-end:** `isapprox(rtol=1e-10)` — halt-the-line at greater drift.

**When it applies.** Every design doc §3 Behaviour preserved; every code PR's halt-the-line conditions in §7 Verification cadence.

**Failure mode.** Relaxing a tolerance to make a failing test pass is a silent bug-hider, not a fix. The strata discipline names the arithmetic reason for each tolerance; a test failure at a tighter tolerance than the arithmetic predicts is a signal that an invariant has broken, not that the tolerance is wrong. Move 6's seeded-MC `==` discipline is the one most tempting to relax; doing so converts "we can't reproduce posteriors" into "close enough," which is the exact posture that invalidates bit-for-bit verification claims.

---

## 5. `_dispatch_path` observability convention

**Where established.** Move 4 design doc §5.2 committed the hook: underscore-prefixed, state-free, query-only, returns `Symbol`. Move 5 §5.4 extended for composite types (rollup).

**The rule.** `_dispatch_path(p, k)` is a test-only observability hook. At the Prevision level (Move 4): returns `:conjugate` if `maybe_conjugate(p, k)` matches, `:particle` otherwise. At the MixturePrevision level (Move 5): returns `:conjugate` iff every component routes conjugate under its resolved LikelihoodFamily, else `:mixed`. Never mutates state. Never appears in production code paths — only in test assertions that pin dispatch-path decisions explicitly. Tests assert `_dispatch_path(p, k) == :expected` **before** the value assertion in every conjugate-path test, because a silent registry miss would still produce the correct value via particle convergence — the dispatch-path check is the tripwire.

**When it applies.** Every new conjugate pair (Move 4-style) or routing primitive (Move 5) ships with `_dispatch_path` coverage in the Stratum-2 corpus. Move 6 will extend for particle/quadrature; Move 7 for event-primary `condition` paths.

**Failure mode.** A silent registry miss (typo in a type key, missing method for a (Prior, Likelihood) pair) falls through to particle. The particle path converges to the correct value at enough samples, so value assertions pass. Without `_dispatch_path`, the test suite gives a false-positive green; the regression surfaces only in production when a performance-sensitive code path hits the slow particle fallback instead of the fast conjugate update. `_dispatch_path` assertions make the miss loud and immediate.

---

## 6. Vocabulary pins across design doc and code

**Where established.** Move 5 established the spot-check discipline after review surfaced the risk of `Symbol` / type-name drift between §5.4's committed vocabulary (`:conjugate`, `:mixed`) and the code implementation. Spot-check performed at code PR review time.

**The rule.** Symbols, type names, and method names committed explicitly in a design doc's §5 (Open design questions) or §2 (Files touched) must appear **identically** in the code. When a design doc commits "returns `:mixed` as the honest partial-coverage label," the code returns the Symbol `:mixed` — not `:partial`, `:mixed_dispatch`, or similar near-synonyms. Spot-check is a review step on every code PR whose corresponding design doc committed specific vocabulary: `grep -n '<symbol>\|<type>' src/ test/ docs/posture-3/move-N-design.md` and confirm consistency.

**When it applies.** Every code PR that lands after a design-doc PR that committed specific names. The drift risk is highest when the design doc and code PRs are separated by days or weeks — the mental model that produced the design-doc name has cooled by the time the code is written, and a slightly different name can feel natural.

**Failure mode.** The design doc pins `:mixed`; the code returns `:partial`. Six months later a test that pattern-matches on `:mixed` (written against the design doc's vocabulary) fails mysteriously. Worse: a reviewer reading the design doc expects to find `:mixed` in the code, grep-misses, and can't find the implementation — the apparent absence looks like an unimplemented feature. Cheap to catch at code PR time; compounding cost across moves if not caught.

---

## 7. First-draft §5 reasoning strength

**Where established.** Review of the first drafts of Moves 1 §5.1, 2 §5.1, 2 §5.3, and 5 §5.1 surfaced a consistent pattern: first-draft Socratic recommendations reach for plausible-but-weak reasoning (appeal-to-authority, plan-adherence, process convention) before the strongest available technical reason. Review catches the pattern; the drafting instruction below internalises it.

**The rule.** When drafting §5 recommendations, list the reasons that support the chosen position. For each, ask: "is this technical or meta?" *Technical* means what the code gains, what invariant is preserved, what downstream work is enabled or prevented. *Meta* means appeals to authority (master plan says so), plan-adherence (committed scope is easier to defend), process convention (reopening signals the plan is negotiable), or procedural defensibility. **Drop the meta reasons.** Keep only the technical ones. If dropping leaves fewer than two technical reasons, the recommendation is weaker than it looked — either strengthen or soften to tentative.

**When it applies.** §5 Open design questions in every move design doc. Recommendations with three-plus reasons often have meta-padding; two strong technical reasons is sufficient and preferred.

**Failure mode.** A recommendation ships with meta-reasoning; review drops the meta; the recommendation stands firmer on the surviving technical reasons. Not catching it in the first draft costs a revision cycle. Over 8 moves, the compounding revision cost is non-trivial; the drafting-time discipline is cheaper than the review-time correction.

**Cross-reference.** Memory entry `feedback_design_doc_reasoning_strength.md` in the author's session memory; this precedent lifts the same pattern into the in-repo working reference.

---

## 8. Checkpoint-per-phase commits

**Where established.** Moves 3 (12 commits), 4 (8 commits), and 5 (4 commits) all followed the pattern. CLAUDE.md §Repo conventions → Rebase-merge for linear master history preserves the individual commits on master; the master-plan §Commit cadence guardrails mandates "every session ends with a green test suite, even mid-refactor."

**The rule.** Multi-phase moves commit per-phase, not per-PR. Each commit leaves the branch with a green test suite; each commit is individually bisectable; each commit is small enough to reviewfast. Rebase-merge (not squash-merge) preserves the commit history on master so bisect is useful across the merge boundary.

**When it applies.** Every multi-phase code PR. The phase boundary is "a coherent subset of the move's work that can be tested to green on its own," not an arbitrary line. Moves with genuinely atomic scope (small refactors, docs-only PRs) commit once; moves with sequenced work (Moves 3–5) commit per phase.

**Failure mode.** A single squash-merged commit hides regressions inside the squash: a test failing at phase 2 was green at phase 1, but the squash loses that signal, so bisect can only narrow the bug to "somewhere in this 400-line diff." The per-phase commit pattern makes each regression reproducible at a specific commit boundary.

**Cross-reference.** `CLAUDE.md` §Repo conventions — this precedent is the Posture 3 application of the "rebase-merge for linear master history" convention.

---

## 9. Fixture provenance

**Where established.** Move 3 established `test/fixtures/README.md` as the provenance protocol file. Move 3 design doc §2 and §6 R3 named fixtures as mandatory for schema-version migrations.

**The rule.** Every test fixture is commit-pinned: captured from a named SHA, the SHA recorded in `test/fixtures/README.md`, the fixture itself never regenerated to fix a loading bug. If a future-discovered bug affects how the fixture is loaded, the fix goes in the load code; the fixture stays as-is. New schema versions get new fixtures captured at the SHA that introduced the schema; migration tests load the pre-migration fixture and assert the post-migration behaviour.

**When it applies.** Every schema-version change (Move 3 introduced v2), persistence format change, or serialisation-boundary change. Move 6's ParticlePrevision may introduce a new schema if particle-state serialisation changes.

**Failure mode.** Regenerating a fixture to pass a failing migration test silently invalidates the migration test — it's no longer testing migration, it's testing round-trip of the post-migration shape. The provenance-as-immutable rule is what keeps migration tests honest.

**Cross-reference.** `test/fixtures/README.md` is the authoritative protocol document; this precedent entry is the design-doc-reference pointer. Do not duplicate the capture protocol here.

---

## 10. Pre-emptive skin RPC audit for skin-invariant claims

**Where established.** Move 0 established the audit practice with `docs/posture-3/move-0-skin-surface-audit.md`. Moves 3 and 4 applied it — each scoped its skin smoke test extensions at design-doc time, not at code PR time. Master plan §Verification names the discipline.

**The rule.** Any move claiming "JSON-RPC API surface preserved" or "skin-invariant" must verify the claim at design-doc time: audit `apps/skin/test_skin.py` for coverage against the specific wire paths the move touches. If a smoke test doesn't exist for an affected path, extend `test_skin.py` in the same PR as the code change — not as a follow-up. Move 0's audit is the baseline reference; subsequent moves extend it.

**When it applies.** Moves 3, 4, 6, 7 (master plan flags these as mandatory-skin-smoke); Moves 1, 2, 5, 8 are skin-invariant and smoke is optional. The audit at design-doc time forces the reviewer to confirm: (a) which wire paths change, (b) whether existing smoke covers them, (c) what new tests are needed.

**Failure mode.** A move ships claiming skin-invariance; a JSON-RPC shape silently changes; downstream Python consumers (`apps/python/*`) break in integration but not in the Julia core test suite. Discovering the break at production integration time is expensive; the pre-emptive audit catches it at design-doc time when the change is still local.

---

## Precedent lifecycle

Precedents enter this file when invoked across multiple moves. A precedent invoked once is a candidate for inclusion; a precedent invoked across Moves 2–5 belongs in.

Precedents exit this file when the pattern they describe has been either (a) lifted to `CLAUDE.md` as a constitutional slug (the durable author-facing form), or (b) rendered obsolete by a later reconstruction. Until lift or obsolescence, the precedent stays here as the working reference for active moves.

New precedents for Moves 6, 7, 8 should be drafted as each move surfaces them — not retroactively. The discipline is: notice the pattern the second time it's invoked; draft the entry the third time. Drafting from a single instance risks pattern-matching on noise.
