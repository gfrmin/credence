# Move-N design doc template

Every `docs/posture-3/move-N-design.md` PR uses this structure. Reviewers reject design docs that omit the "Open design questions" section, or that fill it with questions whose answer is obvious from this branch's master plan or from the code.

The template exists because design-doc-before-code earns its keep only if the docs surface real arguments rather than restating the implementation prose. If a section feels like sugar around an already-decided implementation, treat that as a signal that the move is either smaller than it looked or already done — not a signal to ship the design doc.

---

## 1. Purpose

One paragraph. What this move accomplishes; what foundational need it serves; what unblocks once it's in.

Avoid restating the master plan's framing. Reference it instead: "Move N as scoped in the master plan, with the following additions/refinements …"

## 2. Files touched

Exhaustive list. For each file, give:
- Path and line range (e.g. `src/ontology.jl:561-654`).
- One-line description of what changes.
- Whether this is a new file, a modification, or a deletion.

If the move adds a new test file, list it here too.

## 3. Behaviour preserved

What the strata-1, strata-2, and strata-3 tests will assert about pre-/post-refactor equivalence. State the tolerance for each assertion class explicitly:

- Strata-1 (unit equivalence): `isapprox(atol=1e-14)`.
- Strata-2 conjugate paths: `isapprox(rtol=1e-12)`.
- Strata-2 particle paths under deterministic seeding: `isapprox(rtol=1e-12)`.
- Strata-3 end-to-end: `isapprox(rtol=1e-10)` — halt-the-line at greater drift.

If this move legitimately needs looser tolerance for a particular test (e.g. parallelism that didn't exist before reorders particle samples), name the test, the relaxation, and the justification.

## 4. Worked end-to-end example

Trace one representative call through the new dispatch, naming which module owns each step. Pick a call that exercises the move's centrepiece — e.g. a `condition` through the new conjugate registry, an `expect` through the new TestFunction dispatch.

**This section is mandatory** for any move that introduces or extends dual residency (a type, value, or hook present in both an old home and a new home — e.g. Move 5's `FiringByTag` relocation, where the value lives on the kernel side for construction but the routing semantics live on the prevision side). Without the worked example, the design doc fails review.

The worked example should:
- Pick concrete inputs (e.g. `BetaPrevision(2.0, 3.0)`, `obs = 1`).
- Trace the full call chain step by step, showing which module/function owns each step.
- State the result, with the arithmetic that produces it.
- If dual residency is present, name which home is authoritative at each step.

## 5. Open design questions

1-3 specific points where Claude Code is expected to argue back during review of this design doc. Empty or boilerplate is grounds for revision.

Legitimate question shapes:
- **Sibling vs derived:** "should X be a sibling primitive or a derived form?"
- **Option choice:** "we pick (a) cleaner separation or (b) more uniform dispatch?"
- **Category check:** "is this category-different from the type it would inherit from?"
- **Scope question:** "should we fold Y into this move or defer to Move N+1?"
- **Invariant tradeoff:** "preserving X breaks Y; which do we keep?"

Questions whose answer is obvious from the master plan or the code are not real open questions. Surface them as resolved decisions in §1 or §3 instead.

If genuinely no design questions remain — e.g. the move is mechanical type-aliasing — say so explicitly: "No open design questions for this move. The implementation is determined by the master plan; reviewers are invited to challenge that claim."

## 6. Risk + mitigation

What could go wrong; what test catches it.

For each risk, give:
- The failure mode (e.g. "particle filtering silently reorders samples after the constructor refactor").
- The blast radius (which test files / which downstream moves break).
- The mitigation (e.g. "Strata-2 particle test pinned to seeded RNG with 1e-12 tolerance; pre-refactor sample sequence captured in `test/fixtures/particle_seed_42_v1.jls` and asserted byte-equal post-refactor").

Risks the master plan named for this move are repeated here with current-state additions.

**Suggested practice for moves that refactor consumer-visible types:** include a pre-emptive grep step under the relevant risk entry. Run `grep -rn '<pattern>'` across `src/`, `test/`, `apps/`, `docs/`; list each hit and its disposition (mechanical replacement / no edit needed / needs attention). Example: Move 2's grep for `<: Functional`, `isa Functional`, `::Functional` when aliasing the `Functional` hierarchy onto `TestFunction`. Not mandatory — not every risk benefits from a grep — but the default is: if you're renaming or aliasing a type that appears in method signatures or runtime checks, grep for it before the PR opens. Moves that don't touch consumer-visible types (e.g. cosmetic adaptations) don't need the grep.

## 7. Verification cadence

Which test files run at end of the corresponding code PR. Cite the bash invocations explicitly so reviewers can reproduce.

For Moves 3, 4, 6, 7 this section must include the skin smoke test (`python -m skin.test_skin`) — those moves change what crosses the JSON-RPC boundary. For Moves 1, 2, 5, 8 the skin smoke test is optional.

Halt-the-line conditions: any test failure at end-of-PR is a halt. No "will fix in next commit" states; the branch never sleeps with a red suite.
