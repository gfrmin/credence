# Posture 4 — Design doc template

Every `docs/posture-4/move-N-design.md` PR must include the following sections. Reviewers reject design docs that omit "Open design questions" or that fill it with questions whose answer is obvious from the master plan.

Inherited from `docs/posture-3/DESIGN-DOC-TEMPLATE.md` with two Posture-4-specific additions: §0 (final-state alignment check) and §8 (de Finettian discipline self-audit).

## 0. Final-state alignment

One paragraph confirming that the move as proposed converges the current tip toward `master-plan.md` §"Final-state architecture", with explicit callouts where the move leaves transient state not yet aligned (e.g. Move 2 leaves Measure subtypes alive while changing their internal fields; this is explicit, not drift). Empty or hand-waved is grounds for revision.

## 1. Purpose

What this move accomplishes; one paragraph.

## 2. Files touched

Exhaustive list with line ranges. Files created, files deleted, files renamed. For moves that split or delete files, the destination or deletion target is named explicitly.

## 3. Behaviour preserved

What Stratum-1/2/3 tests assert about pre-/post-refactor equivalence. The reference is the Move 0 capture at `test/fixtures/posture-3-capture/`. Every assertion whose value at the Posture 4 tip diverges from the Move 0 capture must be justified: either the divergence is mathematically expected (new seed consumption, known-benign reassociation) or the divergence is a bug.

## 4. Worked end-to-end example

Trace one representative call through the new dispatch, naming which module owns each step. Mandatory for any move that introduces or extends dual residency (a type, value, or hook present in both an old home and a new home) or that changes the wire shape of an external interface.

Worked examples must be concrete enough to copy-paste into a REPL; no pseudocode stubs.

## 5. Open design questions

1–3 specific points where Claude Code is expected to argue back. Example shapes:

- "Should X be a sibling primitive or a derived form?"
- "Do we pick option (a) or (b)?"
- "Is this category-different from the type it would inherit from?"
- "Does the most de Finettian option here conflict with Julia ergonomics, and if so, which yields?"

Empty or boilerplate is grounds for revision. Answered-by-reference-to-master-plan questions are boilerplate; the master plan settles the strategy, the design doc wrestles with the tactics.

## 6. Risk + mitigation

What could go wrong; what test or invariance check catches it. For high-risk moves (Moves 2, 5, 7, 9 per `master-plan.md`), the risk section covers both technical failure modes and review-process failure modes.

## 7. Verification cadence

Which test files run at end-of-PR. The default is the full suite; moves that require smoke tests (Move 7: `apps/skin/test_skin.py`; Move 8: Python bindings integration) name them explicitly. Moves that touch persistence (Move 3, Move 9) run the round-trip fixture test.

## 8. de Finettian discipline self-audit

Four checks, all of which must return "yes" or carry an explicit justification. This section exists because Posture 4's central discipline is the de Finettian commitment and a design doc that does not self-audit against it is a design doc that has not thought about the discipline.

1. **Is every numerical query in this move routed through `expect`?** Or equivalently: does this move introduce any named function that returns a `Float64` describing a probabilistic property of a prevision without calling `expect`? If so, justify on performance grounds and explain why the performance gain is worth the architectural concession.

2. **Does this move hold a Prevision inside a Measure, or a Measure inside a Prevision, for any reason?** The first direction is being retired; the second was the Posture 3 concession being retired now. A "yes" answer without a dated deprecation note pointing at the move that removes it is grounds for revision.

3. **Does this move introduce an opaque closure where a declared structure would fit?** Closures are a fallback for cases the declared structure doesn't cover; they are not a convenience for avoiding type-system work. If the move adds a closure-typed field, the justification names the declared structure it preferred and explains why the declared structure does not cover the case.

4. **Does this move add a `getproperty` override on any Prevision subtype?** The Posture 3 shields retired with Measure; introducing a new one on a Prevision regresses the discipline. A "yes" answer without a matching "and this is temporary, scheduled for removal in Move N" note is grounds for revision.

---

## Reviewer checklist

- [ ] §0 Final-state alignment is a paragraph, not a sentence, and names any transient state explicitly.
- [ ] §5 Open design questions contains non-trivial questions.
- [ ] §8 Self-audit returns "yes" on all four or carries explicit justification.
- [ ] File-path:line citations present where the design doc references current-state code.
- [ ] The move as described does not require a subsequent move to retract or rework it. If Move N requires Move N+1 to clean up after it, the two moves are collapsed or the sequencing is reconsidered.
