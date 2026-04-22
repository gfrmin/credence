# Move 8 design — Program-space adaptation + SPEC.md §1 targeted update + paper completion

Status: design doc (docs-only PR 8a). Corresponding code PR is 8b.

Template reference: `docs/posture-3/DESIGN-DOC-TEMPLATE.md`.

Master plan reference: `docs/posture-3/master-plan.md` § "Move 8 — Grammar and program-space adaptation".

Working reference: `docs/posture-3/precedents.md`.

## 1. Purpose

Move 8 is the final substantive move of Posture 3. Three scope pieces, each with concrete deliverables. The paper draft becoming submittable is Move 8's acceptance criterion — not a fifth item on a scope list, but the gate the other three pieces exist to support.

**Acceptance criterion:** `docs/posture-3/paper-draft.md` is submittable at Move 8's merge. Every section has prose (not outline markers); §2 operational consequences has the three worked examples prose-complete with before/after code snippets; §3 comparison has prose entries for Staton / Jacobs / Hakaru / MonadBayes; §4 implementation has the reference-implementation section populated with branch SHA, test count, and skin-layer claim; §5 conclusion is written; references list is complete, alphabetised, arXiv identifiers verified. No TODO markers remain, no "to be completed" subsections.

### 1.a SPEC.md §1 audit (input to scope decision)

The scoping decision for SPEC.md §1 follows from the audit result rather than the other way around. Audit performed 2026-04-22 against the post-Move-7 §1.0 framing and against `docs/posture-3/paper-draft.md` §1.1–§1.6:

| SPEC.md section | Classification | Note |
|-----------------|----------------|------|
| §1.0 Foundations: coherence, not measure | Reference (post-Move-7) | Peer-primary framing landed at Phase 6/Phase 4. No change. |
| §1.1 Bayes' rule (`condition`) | Orthogonal post-update | Phase 6 added the event-form peer-primary addendum. No further change. |
| §1.2 Expected utility maximisation (`optimise`) | Orthogonal | Savage's theorem; does not touch prevision-primacy, peer-primary conditioning, or Measure-as-view. |
| §1.3 The complexity prior | Orthogonal; updatable-trivial | Solomonoff's `P(program) = 2^{-|program|}`. Compatible with "prevision over programs" framing per paper §1.5; the phrasing is measure-idiom but not incompatible. Minor tweak available (one-sentence bridge); not required. |
| §1.4 The alignment commitment | Orthogonal | CIRL formulation; does not touch prevision-vs-measure. |
| §1.5 Prediction and integration (`expect`) | Orthogonal; updatable-trivial | Measure-theoretic integration formula `E_b[f] = ∫ f b dθ`. This is a σ-continuous realisation of the general prevision operator `expect(p, f) = p(f)`; compatible at the level of abstraction. Minor tweak available; not required. |
| §1.6 Why these axioms are necessary | Orthogonal | Wald's complete class theorem; does not touch prevision-vs-measure. |

**Audit result:** no conflicts. Two sections (§1.3, §1.5) are trivially updatable but do not require rewrite. The scoping decision: **Scoping 2, targeted** — add a bridge paragraph at the top of §1.1 framing §1.1–§1.6 as mechanical-details-under-Posture-3; apply the trivially-updatable phrasing tweaks to §1.3 and §1.5 if they remain one-sentence diffs. Remove the "Restructure note (2026-04-22)" added in Move 7 Phase 6, since the audit now supersedes it.

### 1.b Three scope pieces

1. **Grammar and program-space cosmetic adaptation.** Per master plan:
   - `src/program_space/types.jl`: add docstrings for `Grammar` and `Program` framing them as components of a prevision over program ASTs (the complexity prior is a prevision over programs per paper §1.5; Grammar and Program are its carriers). No type-signature changes — the structs are unchanged under Posture 3 because the Measure-as-view pattern (Move 3) preserves the existing API.
   - `src/program_space/agent_state.jl`, `enumeration.jl`, `perturbation.jl`: confirmation that `add_programs_to_state!`, `analyse_posterior_subtrees`, `perturb_grammar` work unchanged via the Move 3 `getproperty` shield. Expected diff: zero algorithmic changes; docstring polish only where the phrasing reads as "Measure-primary" rather than "Measure-as-view."
   - **Expected diff size:** ~30–60 lines of docstring updates across 4 files. No algorithmic change; no test-expected-value change.

2. **SPEC.md §1 targeted update.** Per the §1.a audit:
   - Add a bridge paragraph at the top of §1.1 framing §1.1–§1.6 as mechanical-details-under-Posture-3. Phrasing per §5.1 Open design question.
   - Remove the "Restructure note (2026-04-22)" at the top of §1 (added in Move 7 Phase 6 as a known-gap marker); the audit-result framing now replaces it.
   - Optional one-sentence tweaks to §1.3 and §1.5 if they remain trivial diffs. If the phrasing pushes past one sentence each, defer to a post-Move-8 follow-up; the bridge paragraph covers the framing otherwise.
   - **Expected diff size:** ~15–30 lines of SPEC.md edits.

3. **Paper draft completion.** The gating artifact. Per §1 acceptance criterion above. Deliverables broken out in §2 Files touched; §5 Open design questions raise the shape-of-submission questions (§3 proportionality; §4 SHA pinning).
   - **Expected diff size:** substantial. §2 worked examples, §3 comparisons, §4 implementation, §5 conclusion all move from "outline" or "stub" to prose-complete. References-list polish: verify all arXiv identifiers, alphabetise, remove "(partial — to be completed in Move 8)" disclaimer.

## 2. Files touched

**Modified:**

- `src/program_space/types.jl:114-129` — Grammar struct gains a docstring framing it as "a prevision over program ASTs" per paper §1.5; Program struct gains a docstring framing it as "a carrier of the complexity prior." No field changes. No constructor changes.
- `src/program_space/types.jl:135-139` — Program docstring. Same framing as Grammar.
- `src/program_space/agent_state.jl:4-10` — `AgentState` module docstring bridge paragraph: "MixtureMeasure belief" framing updated to acknowledge the underlying MixturePrevision per Move 3 wrapping; consumer code reads through the `getproperty` shield unchanged.
- `src/program_space/perturbation.jl:12,143` — `analyse_posterior_subtrees` and `perturb_grammar` docstrings: minor phrasing updates if current wording reads "Measure-first." No signature changes.
- `src/program_space/enumeration.jl:49` — `enumerate_programs` docstring: note the `enumerate_programs_as_prevision` wrapper (Move 6 Phase 5) as the typed-carrier alternative for consumers that want EnumerationPrevision directly.
- `SPEC.md`:
  - Remove the Move 7 Phase 6 "Restructure note (2026-04-22)" at the top of §1 (now superseded by Move 8's audit).
  - Add bridge paragraph at the top of §1.1 per §5.1.
  - Optional one-sentence tweaks to §1.3 and §1.5 if trivial; otherwise defer.
- `docs/posture-3/paper-draft.md` — comprehensive completion per §1 acceptance criterion. Six sections:
  - **§2 Operational consequences** — three worked examples (§2.1 Conjugate dispatch as type-structural registry; §2.2 Mixture conditioning through exchangeability representation; §2.3 Particle methods as previsions). Each needs before/after code snippets and a closed-form derivation of its operational claim.
  - **§3 Comparison** — four entries (§3.1 Staton: measure-categorical PPL semantics; §3.2 Jacobs: effectus theory; §3.3 Hakaru and disintegration-first treatments; §3.4 MonadBayes and score-primitive monads). Proportionality per §5.2.
  - **§4 Implementation** — branch SHA, test count (target ~400+ tests across the 13 Julia test files plus POMDP 55), skin JSON-RPC layer summary, uv-workspace summary. SHA pinning per §5.3.
  - **§5 Conclusion** — paragraph-length summary of what Posture 3 delivered, with the paper's central thesis restated: coherent linear functionals are the primitive, not measures.
  - **§6 References** — alphabetised, arXiv identifiers verified (DLRS 2502.03477 confirmed Move 7; remaining arXiv entries audited).

**Not touched:**

- `src/ontology.jl`, `src/prevision.jl` — no algorithmic changes in Move 8. Move 7's Option B implementation stands.
- `test/*.jl` — all test files must continue to pass unchanged. No new test fixtures, no new test files, no assertion-value changes.
- `CLAUDE.md` — no further edits. Move 7 Phase 7 landed the three Option B precedent slugs; Move 8 does not extend them.

## 3. Behaviour preserved

Docs-only changes at the SPEC.md and paper level; cosmetic docstring changes at the code level. Per precedents.md §4 tolerances:

- All Julia tests continue to pass at their current tolerances (`==`, `atol=1e-14`, `rtol=1e-12`, `rtol=1e-10`). Move 8 introduces no arithmetic.
- POMDP agent 55/55 unchanged.
- Skin smoke unchanged — Move 8 does not touch the JSON-RPC surface.

Verification invariant: the precedent `capture-before-refactor` (Move 6 Phase 0 pattern) does not apply to Move 8 — Move 8's changes are docstring rewrites + paper prose, not seeded-RNG refactoring. Confirmed per `project_posture_3_precedent_11_candidate.md`'s "When it does NOT apply" list: "Cosmetic documentation adaptations. Move 8's program-space docstring adaptation."

## 4. Worked end-to-end example

Move 8 is cosmetic + paper-completion; there is no new dispatch path to trace. The worked example for this design doc is the **paper §2.1 worked example** — tracing a conjugate-dispatch `condition` call through the registry, which Move 4 shipped and Move 8 prose-completes. The before/after code snippet the paper needs:

```julia
# Pre-Posture-3 (master before Move 4):
function condition(m::BetaMeasure, k::Kernel, obs)
    if k.likelihood_family isa BetaBernoulli
        # inline conjugate update
        obs == 1 ? BetaMeasure(m.alpha+1, m.beta) : BetaMeasure(m.alpha, m.beta+1)
    else
        # fall through to grid / particle
        _condition_by_grid(m, k, obs)
    end
end
# Adding a new conjugate pair requires editing this method body.
# N conjugate pairs → N case branches across M Measure subtypes =
# N×M edits to cover.
```

```julia
# Post-Move-4 (Move 8 paper-ready):
function condition(m::BetaMeasure, k::Kernel, obs)
    cp = maybe_conjugate(m.prevision, k)
    if cp !== nothing
        updated = update(cp, obs).prior
        return BetaMeasure(m.space, updated.alpha, updated.beta)
    end
    _condition_by_grid(m, k, obs)
end
# Adding a new conjugate pair registers one (Prior, Likelihood) type pair
# in the Prevision registry; dispatch fires automatically across all
# Measure subtypes that wrap the Prior. N×M edits → N edits.
```

The paper §2.1 wraps this snippet with the operational claim (case-analytic → type-structural; dispatch complexity reduction from quadratic to linear in (N, M)) and cites PR #28 as the landing SHA. Similar structural before/after snippets land in §2.2 (mixture conditioning) and §2.3 (particle posteriors).

## 5. Open design questions

### 5.1 SPEC.md §1 bridge paragraph phrasing

The audit returns mostly-orthogonal (§1.2/§1.3/§1.4/§1.5/§1.6) with §1.1 already updated in Move 7 Phase 6. The bridge paragraph at the top of §1.1 needs to frame §1.1–§1.6 without reading as "we didn't rewrite these yet." Three candidate phrasings:

- **Option A (mechanical-details framing):** "§1.0 establishes the prevision-first foundation; §1.1–§1.6 below describe operational mechanics of specific measure subtypes and axiom-derived properties. Under Posture 3's Measure-as-view framing, each subsection's content describes a well-defined primitive; the reframing is architectural, not derivative-replacing."
- **Option B (layered-abstraction framing):** "§1.0 names the constitutional layer — Prevision is primitive, Measure is a view. §1.1–§1.6 describe the operational layer — how specific measure subtypes realise the primitives. Both layers are load-bearing; neither supersedes the other."
- **Option C (no bridge paragraph, just remove the 'Restructure note'):** the Move 7 Phase 6 note explicitly said "retains pre-Posture-3 operational shape." If the audit confirms no conflicts, removing the note and adding no replacement is an honest signal that the sections work under the new framing without rewrite.

**Recommendation: B (layered-abstraction framing).**

Two technical reasons:

1. **Option A's "mechanical details" phrasing implicitly diminishes §1.1–§1.6.** Readers coming from the operational-mechanics level could read "mechanical details" as "less important than §1.0." That's not what the audit finds; the audit finds them orthogonal, which means independently load-bearing. Option B's "constitutional layer vs operational layer" framing preserves that they are both load-bearing, with different scopes.

2. **Option C leaves future readers without a map.** A reader who reads §1.0's Option B peer-primary framing and then immediately encounters §1.1's pre-Posture-3-era Bayes-rule formulation (even with the Phase 6 addendum) may reconcile the two as alternative framings rather than as layered ones. The bridge paragraph names the layering explicitly. Cost: ~3 sentences. Benefit: readers never have to infer that §1.1–§1.6 are orthogonal.

**Invitation to argue.** Option C becomes correct if the bridge paragraph reads as hedging — a reader who thinks "why does the doc need to defend the coexistence of §1.0 and §1.1?" might conclude the coexistence is problematic when the audit found it isn't. If PR #36 review surfaces this reading, collapse to Option C (remove the Restructure note, add nothing). Option A becomes correct if operational-mechanics is the right frame for §1.1–§1.6; the doc's own phrasing ("The DSL primitive `optimise` implements this") is operational, so A is defensible. Commit B; collapse to C on review feedback; lift to A if §1.3 and §1.5 phrasing tweaks land and reshape the sections more substantially.

### 5.2 Paper §3 comparison entries — proportionality

The four comparisons are:
- §3.1 Staton: measure-categorical PPL semantics
- §3.2 Jacobs: effectus theory
- §3.3 Hakaru and disintegration-first treatments
- §3.4 MonadBayes and score-primitive monads

Staton and MonadBayes are closer to our framework operationally (PPL semantics and Haskell-idiom monad both deal with probability-computation-as-programming). Jacobs and Hakaru are farther (effectus theory is more abstract; Hakaru is disintegration-primitive which Posture 3 explicitly scopes out).

Two candidate shapes:

- **Option A (one paragraph each):** equal-length entries. Easier to draft; reads as fair-handed.
- **Option B (proportional length):** longer entries for closer frameworks (Staton, MonadBayes) — maybe 2–3 paragraphs each. Shorter entries for farther frameworks (Jacobs, Hakaru) — one paragraph each. Reads as submission-ready where equal-weight-across-the-board would read as padding on the farther ones or under-weighting on the closer ones.

**Recommendation: B (proportional length).**

Two technical reasons:

1. **The reader needs proportional engagement, not proportional airtime.** A paper's comparison section is read for "how does your framework relate to each of these others?" The answer depth varies: for Staton, the answer is "shared commitment to PPL semantics but different primitive (coherence vs measure)." That's 2–3 paragraphs of substantive engagement. For Jacobs, the answer is "categorical abstraction with shared concerns but much greater scope." That's 1 paragraph of honest framing. Equal-length entries for unequal-depth questions waste the reader's engagement on the shorter ones and dilute the longer ones.

2. **The paper is being submitted, not journal-equally-weighted.** Submission-ready papers manage reader attention; reviewer attention is finite. The three deep comparisons (Staton, MonadBayes, plus Hakaru on disintegration specifically) are where Posture 3's claim earns its weight. Jacobs's effectus framework is a legitimate reference but the relationship is "shared concerns at greater categorical abstraction" — one honest paragraph.

**Invitation to argue.** Option A becomes correct if one-paragraph-each is enough depth for all four to be properly engaged. The test: can the Jacobs entry be written in one paragraph that doesn't feel like it's skipping over substance? If yes, A's equal-weight shape is defensible. If the one-paragraph Jacobs entry reads as incomplete, we're in Option B territory already. Draft one paragraph for each, then see which expand naturally on the second pass; if three expand and one doesn't, commit B; if all four expand equally, reconsider A.

### 5.3 Paper §4 reference-implementation — SHA pinning strategy

Move 8 completes Posture 3; the branch merges to master at Move 8's merge. Subsequent body-work (Gmail, Telegram connections) lands on a different branch. The paper's §4 pins a specific commit SHA as the "reference implementation." Three candidate strategies:

- **Option A (pin the Move-8-merge SHA):** Stable artifact; captured at reconstruction completion; pre-body-work. The SHA that the paper cites is the SHA where every test file passed under Posture 3's framing. Easy to reproduce.
- **Option B (pin a later SHA at paper-submission time):** Reflects actual submission state. If body-work has accrued between Move 8 merge and paper submission, §4 cites the later state. May have more tests, more artifacts, but the relationship between paper claims and cited SHA is less clean.
- **Option C (describe a range):** cite Move-8-merge SHA as "reconstruction-complete" and any post-submission state as "paper-submission." Two SHAs named; reader sees both.

**Recommendation: A (pin the Move-8-merge SHA).**

Two technical reasons:

1. **The paper's claims about the reconstruction are made against the reconstruction-complete state.** §2 worked examples reference Move 4's registry, Move 5's MixturePrevision, Move 6's ParticlePrevision, Move 7's event-form primary. Every claim in the paper is validated at the Move-8-merge SHA. Post-Move-8 body-work may add features but doesn't change reconstruction-level claims. Pinning the Move-8-merge SHA makes the paper-to-artifact relationship tight.

2. **Reproducibility posture.** A reader wanting to reproduce the paper's operational claims checks out the pinned SHA and runs the test suite. At the Move-8-merge SHA, the Julia suite is 13 files, POMDP is 55/55, skin smoke covers 22 wire tests — all validated in the reconstruction's own discipline. A later SHA adds body-work that may fail for reasons orthogonal to the paper's claims (body-work has its own failure modes; the reconstruction's discipline was `==` bit-exactness on canonical fixtures). Pinning later forces the reader to disentangle.

**Invitation to argue.** Option B becomes correct if a paper reviewer explicitly asks "does the implementation still work in its current state?" and the answer at submission differs from the answer at Move-8-merge. Option C is bureaucratically correct but reads as hedging. Commit A; if reviewer feedback specifically asks for current-state, add the later SHA as an auxiliary reference in §4, not as the primary pin.

## 6. Risk + mitigation

**Risk R1 (low): paper §2 worked examples have insufficient concrete detail.** The paper's operational-consequences claims rest on the three worked examples reading as real demonstrations, not as vague architectural gestures. For §2.1 (conjugate registry), the before/after snippet needs N×M → N reduction made explicit. For §2.2 (mixture conditioning), the TaggedBetaMeasure routing-to-MixturePrevision trace needs to cite the Phase 7 `condition(::MixtureMeasure, ::TagSet)` code. For §2.3 (particle posteriors), the Move 6 typed-carrier shape (CategoricalMeasure wrapping ParticlePrevision by reference) must be the distinguishing claim. *Mitigation:* §4 worked example in this design doc shows the §2.1 shape; §2.2 and §2.3 follow the same template (before/after code + operational claim + SHA citation). Review posture at paper PR: if a worked example reads abstractly, push back.

**Risk R2 (low): pre-emptive grep for program-space surfaces.** Pattern search 2026-04-22 across `src/program_space/` and adjacent:

| Target | Hits | Category (a) | (b) | (c) |
|--------|-----|--------------|-----|-----|
| `struct Grammar`, `struct Program` definitions in types.jl | 2 hits (lines 114, 135) | Both docstring-adaptation targets; no field changes | 0 | 0 |
| `add_programs_to_state!` call sites | 3 (src/Credence.jl, test/test_email_agent.jl, apps/julia/rss/host.jl) | All covered by the Move 3 `getproperty` shield; no change needed | 0 | 0 |
| `analyse_posterior_subtrees`, `perturb_grammar` call sites | 6 in src, 2 in test | All covered by shield; docstring-adaptation target in perturbation.jl | 0 | 0 |
| `Ontology.Measure` references in `src/program_space/agent_state.jl` | 8 hits | All construct-TaggedBetaMeasure patterns; Measure-as-view preserves the API. Minor docstring update at module-docstring level to acknowledge MixturePrevision substrate. | 0 | 0 |

Go/no-go gate: **GO.** 100% (a); no consumer-site changes. Move 8's program-space adaptation is genuinely cosmetic.

**Risk R3 (low): Reference-list arXiv identifier drift.** Paper references include arXiv identifiers (DLRS 2502.03477 verified at Move 7; others to be audited at Move 8). *Mitigation:* before paper-complete sign-off, fetch each arXiv identifier in the references list and verify title + authors match. For non-arXiv references (books, peer-reviewed journal papers with DOIs), visual inspection is sufficient. This is a manual step in the Move 8 code PR; flag as a halt-the-line item if any citation is wrong.

**Risk R4 (low): §3 comparison entries misrepresent the referenced frameworks.** Paper §3 makes specific claims about Staton / Jacobs / Hakaru / MonadBayes. Misrepresenting the referenced framework's position is an embarrassment at submission. *Mitigation:* each §3 entry cites at least one specific paper or book chapter from that framework; the claim is scoped to "our reading" rather than "the framework's claim." Review posture: if an entry reads as "they do X, we do Y, we're better," rewrite to "they address P, we address Q, both P and Q are legitimate but orthogonal concerns."

**Risk R5 (low, procedural): Move 8's merge triggers body-work on a new branch; the paper-submission-ready artifact is frozen at Move 8 SHA.** Per §5.3 Option A, the paper cites the Move-8-merge SHA. After Move 8 merges, the branch proceeds into body-work (connections PR, credence-proxy iteration); paper references stay pinned to Move 8. *Mitigation:* at paper-submission time, if the pinned SHA's tests have regressed on master (unlikely — body-work doesn't touch the reconstruction), re-validate by checking out the SHA. No action needed in Move 8 itself; this is operational discipline for the submission event.

## 7. Verification cadence

At end of Move 8's code PR (8b):

```bash
# All Julia tests must pass unchanged — no new test files.
julia test/test_core.jl
julia test/test_events.jl
julia test/test_flat_mixture.jl
julia test/test_program_space.jl      # exercise docstring updates
julia test/test_host.jl
julia test/test_grid_world.jl
julia test/test_email_agent.jl
julia test/test_rss.jl
julia test/test_persistence.jl
julia test/test_prevision_unit.jl
julia test/test_prevision_conjugate.jl
julia test/test_prevision_mixture.jl
julia test/test_prevision_particle.jl

# POMDP agent.
cd apps/julia/pomdp_agent && julia --project=. -e 'using Pkg; Pkg.test()'

# Skin smoke — optional for Move 8 per master plan §Verification.
# Move 8 does not touch the JSON-RPC surface. Recommended as sanity check.
python -m skin.test_skin
```

**Halt-the-line conditions:**

- Any Julia test regression. Move 8 is cosmetic at the code level; a regression signals the docstring-adaptation accidentally touched arithmetic.
- Any paper-draft.md section reads as incomplete (has TODO markers, "to be completed" notes, outline bullets without prose).
- Any arXiv identifier in the references list fails audit.
- Any §3 comparison entry misrepresents its referenced framework.
- SPEC.md bridge paragraph reading as hedging or as contradictory to §1.0 (per §5.1 Option B → Option C collapse if that reading surfaces at review).

Paper-submission-ready is the Move 8 completion gate. If the paper has any of the above issues at end-of-PR, the PR is not ready to merge regardless of CI status. Per `precedents.md` §8 (checkpoint-per-phase): paper completion likely lands across multiple commits (§2, §3, §4, §5, references polish), each green-tested at the suite level; final commit is "paper submission-ready" and the PR merges on that commit plus CI green plus author approval.
