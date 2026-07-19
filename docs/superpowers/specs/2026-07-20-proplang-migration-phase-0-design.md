# Phase 0 — the proplang migration decision gate

**Status:** design, awaiting review
**Date:** 2026-07-20
**Scope:** measurement only. This phase migrates nothing.

---

## 1. Purpose

Decide, on evidence, whether migrating any Credence consumer onto the proplang
engine is viable — before spending effort on adapters, cutover, or increments.

The prior attempt (increment H) was built, frozen under `govhost-freeze`,
deployed, and gated. It failed: agreement **21/29 = 0.724** against a declared
**0.95** bar, with all eight disagreements identical in shape
(`julia=proceed, membrane=ask`) — `HOSTS_H_REPORT.md:29`.

Two things have since changed, and they point in opposite directions:

- **The stated mechanism of failure is now fixable.** The emission grid caps
  `p1` at 0.9, but since step 1 that grid is an ordinary argument to an exported
  function (`Enumerate.hs:344`), fixed only by the host's own call site
  (`Host.hs:261`). It is no longer "a frozen alphabet-data change" as
  `HOSTS_H_REPORT.md:67` framed it.
- **The engine that failed no longer exists.** The govhost layer was demolished
  at the step-3 sentence freeze; steps 1–10 re-derived the ontology. Every
  measurement in the H record is bound to a build that is gone.

So the failure verdict is stale, and so is the diagnosis. Phase 0 re-establishes
the facts.

## 2. What this phase is not

- Not a migration, cutover, or adapter change.
- Not a proplang increment. W1, W3 and W4 touch no frozen surface at all; only
  W2 requires an author boundary, and it is explicitly conditional.
- Not a commitment to Phase 1. The exit criteria may say stop.

## 3. The decision this phase produces

A go/no-go on **Phase 1: re-run the H differential gate on the waste-only
decision.** Nothing further. Phases 2+ remain unplanned by design — they are
conditional on Phase 1, which is conditional on this.

---

## 4. Workstreams

### W1 — Does `p1` discriminate on context? *(the gate)*

**This is the whole phase. The other three are supporting work.**

The H post-mortem's own conclusion was that the ceiling was the lesser problem:

> "The flat p1 is the graver finding, not the ceiling … p1 ≈ 0.898 across
> benign, attack, and empty contexts alike means the engine is (on this
> evidence) a near-constant function of context" — `HOSTS_H_REPORT.md:167`

That has **never been re-measured on the re-derived engine.** The current oracle
contains exactly one `p1` assertion, an approximate golden
(`test-membrane/Membrane.hs:282`); nothing in the suite would fail if the engine
were again near-constant in its feature stream.

**Why this is measurable today, with no author involvement.** Discrimination is
about whether `p1` *varies* with context, not whether it *reaches* 0.96. The
0.9 ceiling bounds the range but does not flatten it. So W1 runs against the
current `proplang-host` binary, unmodified, over the wire — **no src change, no
oracle change, no freeze boundary.** This is why it sequences first.

**Method.** Drive `proplang-host` over JSON-lines stdio: one `said@1` handshake,
then tick lines carrying feature streams drawn from the governor's existing
captured corpus (the RED_TEAM / BENIGN / captured-benign split named at
`HOSTS_PLAN.md:376-384`). Record `p1` and `entropy_bits` per tick from the
decision reply (`Host.hs:311`, `:316`). Three context classes: benign, attack,
and empty.

**Success criteria.**

| criterion | bar |
|---|---|
| non-degeneracy | interquartile spread of `p1` across the corpus **> 0.05** |
| directional correctness | median `p1`(benign) > median `p1`(attack) |
| reporting | full per-class distributions, not a summary statistic |

The 0.05 figure is **a proposal, not a derived quantity** — it is set an order of
magnitude above the flatness the old engine exhibited, and wants author
ratification before the run rather than after, so the gate is not set to fit its
own result.

**Interpretation.** A near-constant `p1` means the grid fix (W2) is cosmetic and
no host migration can succeed regardless of adapter work — Phase 1 is
unjustified and the phase ends here. A discriminating `p1` means the ceiling is
the only thing between the engine and a re-runnable gate.

### W2 — Thread the emission grid through the handshake *(conditional on W1)*

Lift the structural cap so a terminal action above 0.9 is reachable at all.

```haskell
-- Enumerate.hs:118-119 — the grid
thetaPoints = 0.1 :| [0.2 .. 0.9]
-- Enumerate.hs:344 — parameterized and exported
enumerateSentencesGrid :: NonEmpty Double -> …
-- Host.hs:261 — the call site that fixes it
pop = enumerateSentencesIn nsN gs fragFull
```

**Design constraint: the new hello field must be optional, defaulting to
`thetaPoints`.** This is what keeps the change small. The default path preserves
the pinned populations — 1169 (`Enumerate.hs:329`), 1241/1529
(`Sentence.hs:287-288`), 1601 (`Unify.hs:126-127`) — so **no existing frozen
oracle row moves.** New behaviour occurs only when a world declares a grid.

**This still requires an author boundary.** `Host.hs` is under `src/`, and the
two-phase discipline binds src changes to a frozen oracle: a new row pinning the
declared-grid path must exist, runtime-red, before implementation. The optional
default minimizes that boundary to one new row; it does not eliminate it. The
exact protocol path (new `test-*` stanza vs. an addition to an existing suite)
is an author call, not a builder call.

Tracked as proplang issue #4.

### W3 — Establish a performance baseline

No benchmark, timing harness, or wall-clock figure exists anywhere in the step
1–10 packs. The cut list's position — *"build only if something is actually
slow"* (`WRITEUP.md:414`) — has become unfalsifiable, because no instrument
exists to trip the gate.

This matters because the most tractable consumer is also the most
latency-sensitive: `credence-governor` decides per tool call on an interactive
hook and is engineered around client timeouts (`daemon.py:720-726`). A cutover
cannot be committed to without a number.

Known shapes, all asymptotic and unmeasured: `predictive` is `O(|pop|)` with a
`!!` list-index inside a per-hypothesis map (`Enumerate.hs:586`, `:593`,
`:604`); the option space is exponential in menu names with exhaustive argmax as
the declared general route (`Membrane.hs:90-93`).

**Deliverable:** ms/tick at each pinned population size, via a script under
`tools/`. Measurement only — this touches no frozen surface. Prior-engine
numbers (8.26 ms/tick at 1241 models; ~36 ms/tick at governor scale) are **not
transferable**: different build, different populations, different decision path.

**Proposed bar: ≤ 50 ms/tick at governor-scale population.** Like W1's, this
number wants ratification before the run. Its basis: the governor decides
synchronously inside a PreToolUse hook, and already carries a fast-503
`proceed` path specifically to avoid waiting out a client timeout
(`daemon.py:720-726`) — so the budget is set by human-perceptible latency on
every tool call, not by a throughput target.

Tracked as proplang issue #6.

### W4 — Resolve the govhost binary's provenance *(housekeeping)*

`~/.local/bin/proplang-govhost` (sha256 `96ec3de7…`, built 2026-07-10) is what
`credence-governor`'s live shadow runs, pinned in `deploy/membrane.conf`. It was
compiled from proplang tag `d-close`, whose source is **not on master** — the
host layer was deleted at step 3.

**Correction to an earlier assessment:** the source is *not* lost. Tag `d-close`
resolves to `6afa24f` and carries the full tree — `host-governor/{Main,Wire,WireU}.hs`,
`test-govhost/`. The binary is rebuildable via `git checkout d-close` with that
commit's `cabal.project.freeze`. This is undocumented provenance and silent
drift, not unrecoverable loss, and it is correspondingly lower priority.

Two facts nonetheless need recording somewhere a reader will find them:

1. The running shadow's engine is two ontologies behind current proplang.
2. The wire it speaks (`table@1`/`latent@1`) is dead — current `proplang-host`
   accepts only `said@1` (`Host.hs:248`), and `membrane-wire.md:175-183`
   scope-brackets the old sections as binding on nothing current.

**Deliverable:** a provenance note in `credence-governor/docs/membrane-shadow.md`
stating the tag, the sha256, the rebuild command, and the wire divergence. Plus
an explicit decision — keep the shadow running as a historical baseline, or
retire it. Leaving it undecided is the one option that should be closed off,
because its output is currently being read as if it described current proplang.

---

## 5. Sequencing

```
W1 (gate) ──────────────┐
                        ├──▶ exit decision
W3 (baseline) ──────────┤
W4 (housekeeping) ──────┘
                        
W2 ── only if W1 passes; requires author boundary
```

W1, W3 and W4 are independent and parallelizable; none touches a frozen surface.
W2 is gated on W1 because an author boundary spent on a cosmetic fix is waste.

## 6. Exit criteria

**Proceed to Phase 1** if W1 shows `p1` discriminating on both criteria, W2 has
lifted the ceiling, and W3 comes in at or under the ratified ms/tick bar.

**Stop** if W1 shows `p1` near-constant in context. In that case the migration
question is closed on evidence until the engine changes again, and the
consumer-side conclusion is that Credence stays primary everywhere.

**Partial** — `p1` discriminates but W3 shows latency outside the governor's
budget: Phase 1 is still justified, but the first target should be a
latency-tolerant consumer. `answer-brain` is the candidate (300s client timeout,
~3–10 calls/question) — though it carries its own blockers, tracked as proplang
issues #9 and #10.

## 7. Explicitly out of scope

- Every consumer other than the governor's waste-only decision. `rssfeed`
  (proplang #13), `life-agent`'s coupled latents (#14), routing (#12), the harm
  overlay (#11), K-ary (#9), mid-episode K growth (#10).
- Any adapter or cutover work.
- The documentation debt in proplang (#1, #2, #3, #7, #8) — real, filed, and
  independent of this decision.

## 8. Risks and open questions

**The W1 bar is unratified.** 0.05 IQR is a proposal. It should be fixed before
the run, by the author, or the gate is set by whoever sees the result first.

**W1 measures the shipped decision path, which is myopic.** `Host.tick` /
`Host.choose` (`Host.hs:288-352`) is single-shot EU argmax with no think/act
ladder; step 10's deliberation composition is pinned in `test-reflexive/`, not
shipped (proplang #8). So W1 measures what a host can actually get today, which
is the right thing to measure for this decision — but a discriminating `p1`
under a myopic rule does not imply the full agent behaves well.

**`said@1` utility is unpriced** (proplang #1), so the W1 world's declared
utility is a point mass regardless of its complexity. This does not affect
whether `p1` discriminates — `p1` is the predictive, upstream of utility — but
it means the W1 harness should not be read as exercising the pricing story.

**The corpus is the governor's, and the governor is one host.** A discriminating
`p1` on tool-call features is evidence about tool-call governance, not about
answer-brain's question-answering. Generalizing beyond the measured domain is
not licensed by this phase.
