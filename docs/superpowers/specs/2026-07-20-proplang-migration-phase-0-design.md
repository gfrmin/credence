# Phase 0 — the proplang migration decision gate

**Status:** revision 2, awaiting review
**Date:** 2026-07-20, revised 2026-07-21
**Scope:** measurement only. This phase migrates nothing.

> **Revision note (2026-07-21).** Between revision 1 and this one, proplang
> executed four boundaries — the wire opening, W3 (arity), W4 (priced grammar),
> and boundary R (R0 + R1, the joint purchase increment). Three of this spec's
> four workstreams were overtaken: the performance baseline was executed by
> proplang itself, the emission-grid change was **refused permanently by
> ruling**, and the `p1` gate was executed **in half**. The surviving half is the
> half proplang cannot do without a host, and it is still ours. Revision 1 is
> preserved in git history at `9c74790`.

---

## 1. Purpose

Decide, on evidence, whether migrating any Credence consumer onto the proplang
engine is viable — before spending effort on adapters, cutover, or increments.

The prior attempt (increment H) failed its gate at **21/29 = 0.724** against a
declared **0.95** bar, with all eight disagreements identical in shape
(`julia=proceed, membrane=ask`) — `HOSTS_H_REPORT.md:29`. Revision 1 held that
both the verdict and its diagnosis were stale, because the engine that failed had
been demolished and re-derived.

That reading is now confirmed by measurement rather than argued from the record.
**The two defects the H post-mortem named have both been answered — one
discharged, one refused-and-replaced.** What remains blocking is neither of them.

## 2. What this phase is not

- Not a migration, cutover, or adapter change.
- Not a proplang increment. No workstream here touches a frozen surface.
- Not a commitment to Phase 1. The exit criteria may say stop, or wait.

## 3. The decision this phase produces

A go/no-go on **Phase 1: re-run the H differential gate on the waste-only
decision.** Nothing further.

---

## 4. What proplang settled (workstreams retired)

### Retired — performance baseline → **discharged**

`OB-3` discharged at `wire-open`. The instrument is `test-measure/ g2`: ms/tick
at the four pinned populations (1169, 1241, 1529, 1601), run at each freeze so
regressions stay visible. Opening figures **9.4–14.6 ms/tick** (prototype, `-O2`;
the suite's own green run is the figure of record).

That is comfortably inside revision 1's proposed ≤50 ms/tick bar, and inside the
governor's interactive budget. Two details worth carrying into any host-side
harness:

- **The timing rows are report, not gate.** The gate half of the same row is the
  *population pin* — a freeze run that cannot reproduce the pinned population
  fails before it reports a time. Timing figures are machine-relative; the
  population is not.
- **Setup and steady-state are separated** (the mandate-6b repair): laziness
  defers agent realization out of the hello reply, so a naive hello window
  undercounts and the first tick overpays. Measure the same way or mis-attribute
  the cost. `SETUP` = hello + first observed tick; `STEADY-STATE` = the next 100.

Issue #6 is answered.

### Retired — emission grid at the handshake → **refused, permanently**

Revision 1 proposed threading the emission grid through the handshake as an
optional field. **This is now ruled out forever, from anyone.** `OB-4` is
discharged *by ruling, not by landing* — R-W1 ruled outcome (i)+(iii) on
2026-07-20: condemnation held, **no emission grid key ever**.

The ground is an asymmetry worth internalising, because it is the principle that
will decide every future request we make of this wire:

> a grid for *belief about the world's law* is the agent's representational
> choice and never crosses the wire; a grid for *the world's own declared
> preferences* is the declaration itself — the principal is the authority on what
> it values and at what resolution it cares to say so. Economics, not epistemics.
> — `wire-author-pack.md:1068-1083`

W4 shipped `cgrid`, an optional constant-grid key in the utility block, and it is
lawful for exactly the reason the θ-grid is not: `cgrid` declares *preferences*.
Ours declared *epistemics*.

**The design constraint revision 1 was most pleased with — optional field,
default-identical, no frozen row moves — was never the objection.** Backward
compatibility was not the problem. The direction of the declaration was. That is
the lesson, and it is cheaper to learn once than twice (see §9).

**The replacement is better than what we asked for.** Boundary R landed
`R-vocab`: the agent purchases its own vocabulary refinement in-language, under
prices, with `thetaPoints` retired from constant to *initial* vocabulary. The
acceptance anchor is our own use case, pinned as an oracle row:

```haskell
-- test-refine/Refine.hs:417-421
testCase "the governor anchor: 0.96 threshold cleared with no host declaration (p* = 0.96)"
```

with the recurring-stakes falsifier beside it (`g9`) and the myopic one-tick case
pinned as *the chosen rung*, not a branch. So the ceiling that killed H is solved
— and solved without the host declaring anything. See §5's W2′ for the catch.

Issue #4's answer is boundary R. `OB-9` (issue #8, the deliberation ladder) was
discharged in the same increment: depth is now a rung the same argmax chooses
under prices, so revision 1's "the shipped path is myopic" risk is retired *in
the library* — and only there.

### Retired in half — does `p1` discriminate on context?

`OB-2` discharged at `wire-open`, `test-measure/ g1`, and **the flat-p1 defect
does not reproduce**: measured `S = p1_attack − p1_benign = 0.794` against a gate
of `0.4`, pre-stated before the evidence program ran and derived rather than
plucked (half the shipped grid's maximum achievable separation, since θ ∈
[0.1, 0.9] bounds `S` at 0.8). It clears at 2×. The attribution row is a live
seeded defect: drop `FGuardHead` from the fragment and `S` collapses to **exactly
0.0**, so discrimination provably comes from the guard families and nowhere else.

This is a better instrument than revision 1 specified, on every axis that spec
cared about — gates pre-stated rather than post-hoc, red-reachability
demonstrated rather than asserted, attribution partitioned rather than argued.

**But it measures the structural half only, and proplang says so itself:**

> the "attack"/"benign"/"empty" names here are SYNTHETIC risk-bit contexts, NOT
> the govhost corpora those words named in the H era … Whether the engine
> discriminates on REAL governance features **stays unmeasured until a
> host-corpus differential re-run; no such re-run is claimed.**
> — `test-measure/Measure.hs:122-133`

The synthetic stream is the most favourable possible: 60 ticks, evidence
perfectly correlated with an alternating risk bit. Failure there would have been
failure everywhere — which is why it was the right thing to measure first, and
why passing it does not settle the question.

**So the gate survives, narrowed to exactly the half that requires a host.** That
half is ours, it duplicates nothing proplang has done, and proplang has declared
it outside its own scope. It is the one measurement in this spec that still needs
to happen.

---

## 5. Workstreams

### W1′ — Does `p1` discriminate on *real governance features*? *(the gate)*

**The narrowed survivor. This is the whole phase; the rest is hours of work.**

**Method.** Drive `proplang-host` over JSON-lines stdio: one `said@1` handshake,
then tick lines carrying feature streams from the governor's captured corpus (the
RED_TEAM / BENIGN / captured-benign split at `HOSTS_PLAN.md:376-384`). Report the
same statistic proplang reports — `S = median p1(attack) − median p1(benign)` —
plus full per-class distributions, not a summary alone.

**Reuse their gate discipline, not their gate value.** `S ≥ 0.4` was derived as
half the achievable separation *on a perfectly-informative synthetic stream*.
Real governance features are not perfectly informative, so that number does not
transfer and reusing it would be a category error. **The bar for W1′ must be
derived from the real corpus's own achievable separation and pre-stated before
the run.** The pre-statement is the transferable part, and it is the part
revision 1 got right.

**Two harness constraints, both learned from another host's scars:**

1. **`proplang-host` block-buffers stdout under pipes** (proplang issue #18). A
   host that spawns it with ordinary `stdin=PIPE, stdout=PIPE` writes a
   well-formed hello, flushes, and *never receives the reply* — the buffer only
   flushes at process exit, so the first exchange deadlocks silently. Either
   allocate a PTY (nethack-beater's forced workaround, which pushes `ECHO`/
   `ICANON` and `EIO`-on-exit handling onto the host) or run in-process against
   `serveLine`, as `test-measure` does. Budget for this before it eats an
   afternoon.
2. **Evidence replay is O(ticks)**, one round-trip per tick — `observe_batch` /
   `observe_counts` are specified in `membrane-wire.md §6.3` but not implemented
   (issue #17). At governor-corpus scale this is probably tolerable; that is a
   thing to measure, not to assume.

**Interpretation.** A discriminating `p1` on real features means the engine reads
governance context and the H post-mortem's graver finding is fully dead. A
near-constant `p1` on real features *despite* `S = 0.794` synthetically would
mean the governor's features carry no signal the guard families can find — a
**feature-encoding verdict, not an engine verdict**, pointing at the adapter
rather than at proplang. Revision 1 could not have drawn that distinction; the
synthetic result is what makes it available.

### W2′ — The fragment route: purchase is in the library, not on the wire

**This is the new blocker. It has no issue and no ledger row of its own.**

`PropLang.Purchase` and `PropLang.Lattice` are real shipped modules with real
bodies — 24/24 green, every frozen anchor byte-stable, zero alphabet productions.
**`Host.hs` references none of them.** The wire still runs the myopic `choose`
over hard-wired `thetaPoints` (`Host.hs:261`), so the highest `p1` a wire client
can reach is still 0.9, and `test-measure/ g1` pins that ceiling as a row.

proplang records the gap precisely and does not hide it:

> the genuinely distinct fragment route (purchases flowing through
> `enumerateSentencesGrid` into wire-level agents) is **not an R1 frozen row**;
> the rider's re-measurement travels to the boundary that lands that integration
> — `r-author-pack.md` Part V, Rider 2

So the ceiling is **solved in the language and unsolved on the wire.** For a
migration that reaches the engine *only* over the wire, that distinction is the
entire difference between unblocked and blocked. R_SCOPE's own interim wording
agrees: high-threshold hosts stay out of scope until R — and the governor at 0.96
is the canonical high-threshold host.

This is the same shape as issue #16 (the step-10 deliberation composition is
unreachable through `said@1` for want of belief-scoped heads): a capability
proven in-engine, not exposed at the membrane. Two instances now. Worth naming as
a class in the filing, because the class is what will keep recurring.

**Deliverable: file it.** Rider 2's travel note is the only thing carrying this,
and rider notes travel to boundaries that have not been scheduled. An issue with
a named consumer and a stated threshold is precisely what turned issue #4 into
boundary R. Cheap, and nothing else in this phase is blocked on it.

### W3′ — Resolve the govhost binary's provenance *(housekeeping, unchanged)*

`~/.local/bin/proplang-govhost` (sha256 `96ec3de7…`, built 2026-07-10) is what
`credence-governor`'s live shadow runs, pinned in `deploy/membrane.conf`, built
from tag `d-close` (`6afa24f`) whose source is not on master. The source is *not*
lost — the tag carries the full tree and the binary is rebuildable. Drift, not
loss, exactly as revision 1's correction established.

The case for acting has strengthened: the shadow is now **four boundaries**
behind (wire, W3, W4, R), and the wire it speaks (`table@1`/`latent@1`) is dead —
current `proplang-host` accepts only `said@1` (`Host.hs:248`).

**Deliverable:** a provenance note in `credence-governor/docs/membrane-shadow.md`
— tag, sha256, rebuild command, wire divergence — plus an explicit keep-or-retire
decision. Leaving it undecided is the one option to close off, because its output
is currently being read as though it described current proplang. It does not, by
four boundaries.

### W4′ — Reconcile the issue tracker with the ledger *(housekeeping)*

proplang's `OBLIGATIONS.md` records eight of our fourteen issues as answered, but
**all fourteen are still open on GitHub.** The ledger says "closes"; nothing
closed them.

| issue | obligation | disposition |
|---|---|---|
| #1 | OB-7 / OB-8 | pricing half shipped; parameter-latent half **refused**, narrowed to the degenerate latent |
| #2 | OB-1 | VoI non-negativity landed, `test-law/ g1` |
| #3 | OB-6 | full priced grammar shipped (`/`, `log`, `exp`, `neg`; `<` refused as a bit-identical argument swap) |
| #4 | OB-4 | **refused by ruling**; answered by boundary R |
| #5 | OB-2 | defect does not reproduce — structural half only |
| #6 | OB-3 | instrument landed |
| #8 | OB-9 | depth purchased in-language |
| #9 | OB-5 | arity half shipped (W3); grid half ruled out |

Close them citing the discharge event, or comment the disposition where it is
partial (#1, #5, #9 are all partial). Issues #10–#14 remain `RULING-PENDING` and
stay open.

**Where a disposition is a refusal (#1 half 2, #4), the close must record the
ground, not just the state.** Those grounds are the wire's doctrine, and we will
be reasoning against them again.

---

## 6. Sequencing

```
W1′ (the gate) ─────────┐
W2′ (file the issue) ───┤
W3′ (provenance) ───────┼──▶ exit decision
W4′ (tracker) ──────────┘
```

All four are independent and parallelizable. None touches a frozen surface. W2′,
W3′ and W4′ are hours; W1′ is the phase.

## 7. Exit criteria

**Proceed to Phase 1** if W1′ shows `p1` discriminating on real governance
features **and** the fragment route has landed on the wire. Both are required:
discrimination without the fragment route means the engine reads context but
still cannot clear 0.96 through `said@1`, so an H re-run would fail the same
way for a different reason — and would wrongly look like a repeat verdict.

**Wait** — W1′ passes, fragment route not landed. **This is the expected
outcome, and it is not a failure.** The migration is unblocked in principle and
blocked on an acknowledged-but-unscheduled integration. Phase 1 waits; W2′'s
issue is the lever that schedules it.

**Stop** if W1′ shows `p1` near-constant on real features. The migration question
closes on evidence until the feature encoding changes, and Credence stays primary
everywhere. Note this is now a verdict about *our* encoding, not about the
engine — the engine's structural capacity is established.

## 8. Explicitly out of scope

- Every consumer other than the governor's waste-only decision — `rssfeed`
  (#13), `life-agent`'s coupled latents (#14), routing (#12), the harm overlay
  (#11), K-ary's open half (#9/#10).
- Any adapter or cutover work.
- `nethack-beater`. It is a **native proplang host, not a Credence consumer**, so
  it is not a migration target — but it is now the project's leading indicator of
  wire defects (#15–#18 all came from it, and #18 would have cost us an afternoon
  blind). Worth watching, not worth scoping.

## 9. Risks and open questions

**The one-way ratchet on wire requests.** The economics/epistemics asymmetry is
now enforced four times and stated as doctrine, with the author noting that a
future boundary audit would flag `cgrid` itself if the asymmetry were not on the
record. Any future ask of this wire should be pre-classified before filing: if it
asks the world to declare something about *how the agent should reason*, it will
be refused, and the lawful form is in-language and purchased. Our emission-grid
ask cost one round trip to learn this. It should not cost a second.

**Issue #15 partly subsumes routing (#12).** The governor's routing needs learned
per-arm `P(correct | features)` — action-conditional outcome learning, which is
exactly what #15 reports the wire has no mechanism for. #12's disposition may
therefore be decided by #15's, and the two should be read together rather than
ruled separately.

**W1′ measures the shipped decision path, which is still myopic on the wire.**
`Host.tick` / `Host.choose` remains single-shot EU argmax. R1 made depth a
purchased rung *in the library*; until the fragment route lands, the wire cannot
choose it. So W1′ measures what a host can actually get today — the right thing
for this decision — but a discriminating `p1` under a myopic rule does not imply
the full agent behaves well.

**The corpus is the governor's, and the governor is one host.** A discriminating
`p1` on tool-call features is evidence about tool-call governance, not about
answer-brain's question-answering. Generalizing beyond the measured domain is not
licensed by this phase.
