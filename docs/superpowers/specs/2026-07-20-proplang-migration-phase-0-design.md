# Phase 0 — the proplang migration decision gate

**Status:** CLOSED 2026-07-22 — decision **WAIT** (§7.1). All four workstreams reported.
**Date:** 2026-07-20, revised 2026-07-21, executed and closed 2026-07-22
**Scope:** measurement only. This phase migrates nothing.
**Results:** `2026-07-22-w1-prime-prestatement.md` (+ Amendments 1–2) · `2026-07-22-w1-prime-results.md`

> **[SUPERSEDED IN PART 2026-08-09 — read `2026-08-09-proplang-replan-post-19.md`
> first.]** proplang closed the `#19` sitting on 2026-08-08 and **both** of this
> spec's §7 exit blockers are discharged — the θ ceiling by dissolution, the
> namespace projection by loss of authority (`HOSTS_PLAN` is archived and binds
> on nothing current). Neither discharge unblocks the migration: Stage A of the
> replan measured every evidence channel the shadow engine was fed at **0.99+**,
> so the constant `p1` was a correct convergence on a degenerate channel, not a
> ceiling artifact — and lifting the ceiling now makes the reading *worse*. Two
> diagnoses inside this document are corrected in place below (the W3′ section),
> with the falsified words quoted. The **decision** (WAIT) stands; its stated
> **reasons** do not.

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

> **EXECUTED 2026-07-22 — and the method below was SUPERSEDED before it ran.**
> The straight attack-vs-benign contrast specified here would have produced a
> **confounded pass**: the red-team fixtures are constant on five of the six
> features the wire can see, at values field traffic does not share, so a large
> `S` would have meant *"synthetic fixture vs real session"* and read as success.
> What actually ran is a two-arm matched design with a pre-stated,
> permutation-derived, sample-aware bar. Read
> `2026-07-22-w1-prime-prestatement.md` (+ Amendments 1–2) and
> `2026-07-22-w1-prime-results.md` **instead of** this method paragraph; it is
> kept for provenance, not as instructions.

**Method (as originally specified — superseded, see above).** Drive
`proplang-host` over JSON-lines stdio: one `said@1` handshake, then tick lines
carrying feature streams from the governor's captured corpus. Report the same
statistic proplang reports — `S = median p1(attack) − median p1(benign)` — plus
full per-class distributions, not a summary alone.

*Citation repaired:* revisions 1 and 2 cited "the RED_TEAM / BENIGN /
captured-benign split at `HOSTS_PLAN.md:376-384`" without naming the repo, which
made it unfindable from here — the file is **proplang's** `HOSTS_PLAN.md`, and
the split is described at **lines 400-401**, not 376-384. (The `HOSTS_PLAN 2.1`
citations elsewhere in these docs, for the epoch-1 waste-only scope, are correct:
`### 2.1 Scope`, line 236.)

That plan describes running the governor's `posterior_eval` harness twice over
those corpora. The executed run did not use it. Its concrete sources were the 20
authored cases in `credence-governor`'s `training/red_team.py`
(`RED_TEAM_CASES`) for the attack side, and captured `tool-proposed` records
from `~/.credence-governor/observations.jsonl` — pinned to a frozen snapshot —
for the benign side.

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

**Interpretation — now a three-way, not a two-way.** A discriminating `p1` on
real features means the engine reads governance context and the H post-mortem's
graver finding is fully dead. A near-constant `p1` *despite* `S = 0.794`
synthetically would have meant the governor's features carry no signal the guard
families can find — a **feature-encoding verdict, not an engine verdict**,
pointing at the adapter rather than at proplang. Revision 1 could not have drawn
that distinction; the synthetic result is what makes it available.

**W3′ added a third arm, and it is the one that must be ruled out first.** A
near-constant `p1` can also be **the ceiling** — the posterior saturating at
`thetaPoints`' top rung and sitting there, which is precisely what the shadow did
on 98.7% of 190k real governance decisions. That is neither a feature verdict nor
an engine verdict; it is the wire's grid, and it would say nothing about the
governor's features at all. **So W1′ must report the full `p1` distribution and
explicitly check for rail-at-0.9 before interpreting any flat result** — a
summary statistic alone cannot tell the three arms apart, and `S ≈ 0` is what all
three produce. Concretely: if `p1` takes few distinct values and the modal one is
`0.8999999999999999`, the run has measured the ceiling and the feature question
remains open behind it. This is also the strongest argument that W1′'s bar must
be derived from the corpus's *observed achievable* separation rather than
assumed — the achievable separation may be capped by the grid before the features
ever get a say.

### W2′ — The fragment route: purchase is in the library, not on the wire

> **DONE 2026-07-22 — filed as [proplang#19](https://github.com/gfrmin/proplang/issues/19).**
> Named consumer `credence-governor`, stated threshold `p* = 0.96`, classed with
> #16. Filed with the exit-shaped ask below rather than a bare demand.

**This was the new blocker. It had no issue and no ledger row of its own.**

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

**Filed. The lever the filing actually used** was sharper than "please schedule
this," and is worth recording for reuse. `R_SCOPE.md:190-196` brackets the
ceiling with *"Until R closes: … hosts binding at thresholds above the ceiling
(0.95/0.96/0.9942 registered) are OUT OF SCOPE."* **R has closed** —
`r1-freeze-r1`, OB-9 and OB-17 both discharged. The bracket's stated condition is
satisfied and the bracket does not lift, because what would lift it for a wire
host is not what R1 landed. So the bracket now has no exit condition at all.

The ask was therefore put as a fork, both arms usable: schedule the route (ledger
row, target boundary, Rider 2's re-measurement homed to it), **or** re-state the
bracket as *permanent for wire hosts* — a declared limitation with a named heir,
in the `R-D23` shape proplang already uses. The unusable state is the current one,
where the bracket reads as interim and nothing is scheduled to end it. Asking a
project to close its own open-ended interim is a stronger move than asking it to
build something.

### W3′ — Resolve the govhost binary's provenance *(no longer housekeeping)*

> **DONE 2026-07-22 — `credence-governor/docs/membrane-shadow.md` §6** (items
> 18–29), plus a dated pointer in §0 so the falsified half of the stated
> prediction cannot be read as standing. This stopped being housekeeping the
> moment the shadow's own output was counted.

`~/.local/bin/proplang-govhost` (sha256 `96ec3de7…`, built 2026-07-10) is what
`credence-governor`'s live shadow runs, pinned in `deploy/membrane.conf`. Two
corrections to what this spec previously claimed, both from re-deriving against
proplang's current tree:

- **The source IS on master.** `d-close` = `6afa24f`, and
  `git merge-base --is-ancestor d-close master` returns true. Rebuild is
  `git checkout d-close && cabal build proplang-govhost`.
- **This is not drift — it is a deliberate upstream retirement.** Commit
  `3eb291a` (step-3 Phase B, D3 retirements) removed the `proplang-govhost`
  executable, `host-governor/`, and `test-govhost/` together, stating "THE
  HOST-LESS WINDOW … in their place"; `proplang-host` arrived later (`c4aa9bb`)
  as a *different* executable on a different wire. There is no refresh path,
  because there is nothing to refresh it to. Wire divergence is total:
  `Host.hs:252` accepts `said@1` and nothing else.

**The field reading is the actual result, and it was not anticipated.** Over
189,858 shadow records (94,988 `latent@1` + 94,936 `table@1`):

| form | action | `p1` | `entropy_bits` | `sensitivity` |
|---|---|---|---|---|
| `latent@1` | `block` × **100%** | {0.9, 0.4975} | one value | `false` × 100% |
| `table@1` | `ask` × **100%** | {0.9, 0.4975} | one value | — |

`latent@1`'s stated prediction held exactly. But `table@1` — deployed and
documented as *"the non-degenerate challenger policy the Phase-1 outcome bench
scores"* — **is also degenerate**, because `p1` rails at `thetaPoints`' top rung
of 0.9 and stays there. Across 190k live governance decisions the wire produced
two `p1` values and one value of every other readout. That falsifies the premise
of Phase 2's exit criterion: the comparison window is 95k decisions long and has
nothing in it to compare.

> **[DIAGNOSIS CORRECTED 2026-08-09 — Stage A, `2026-08-09-stage-a-results.md`.]**
> The operational conclusion above stands: a challenger that asks on 100% of
> ticks cannot serve the Phase-2 comparison, and the window is empty. The
> **diagnosis** is wrong. The falsified words are *"`table@1` … **is also
> degenerate**, because `p1` rails at `thetaPoints`' top rung"* — attributing
> the constant to the policy and to proplang's grid. Measured: every evidence
> channel the engine was fed runs at 0.99+ (warm prior 39,314 @ 0.990945;
> outcome good-bit 71,867 @ 0.999471, after discarding 49.3% of outcome records
> as ambiguous; human verdict 8 @ 1.0). Against a grid capped at 0.9, 0.9 is the
> **KL-nearest rung** — so `p1` was correct, and the policy was faithfully
> reporting a **degenerate channel**, not being a degenerate policy. The gap
> `|p1 − rate|` equals `rate − 0.9` at every checkpoint. `table@1` additionally
> received **eight** live evidence ticks across 95,031 decides, so its `p1` is
> the warm prior's projection and nothing more.

**Why this transfers.** `thetaPoints = 0.1 :| [0.2 … 0.9]` is byte-identical at
`d-close` (`Enumerate.hs:95-96`) and at master (`Enumerate.hs:127`) — unchanged
across all four boundaries. Everything else here measures a retired binary, but
the mechanism producing the rail is the one current `proplang-host` still uses.
**These 190k decisions are proplang issue #19 observed rather than argued**: the
issue says the 0.9 ceiling blocks a 0.96-threshold host in principle; this is
that block, in the field.

> **[SUPERSEDED 2026-08-09 on both halves.]** *(a)* The transfer argument has
> expired: `thetaPoints` has **zero occurrences** in proplang `src/` since
> `c2ca82c` (2026-07-25). The emission grid is required consumer-declared hello
> data (`Host.hs:260`), so the quoted line no longer exists at master and the
> ceiling changed owner rather than dissolving. *(b)* The claim *"These 190k
> decisions are proplang issue #19 observed rather than argued"* is **wrong as
> written**. #19 was about a hard-wired grid and an unrouted fragment; what
> these records observe is our **own** evidence channel sitting above the
> ceiling. Stage A prices the counterfactual: extending the grid upward moves
> `p1` from 0.9 to 0.999 on this data — a more confident report of the same
> degenerate channel, at ~3× the per-tick cost. The ceiling bound here, but it
> was not the binding constraint.

**Decision: retire the shadow, keep the binary and the reading** (§6.3, flagged
for the author per the register's convention). Both forms have returned their
readings and both readings are constants; further ticks re-observe a constant
while costing worker time, log volume, and the standing respawn-drop posture.
Retire is a drop-in deletion + restart. The successor is not another `table@1`
window — it is a `said@1` shadow against current `proplang-host`, which is
Phase 1, gated on W1′ and on #19.

### W4′ — Reconcile the issue tracker with the ledger *(housekeeping)*

> **DONE 2026-07-22.** All eight closed with dispositions written. Open tracker is
> now #7, #10–#14 (`RULING-PENDING` + stale-docs), #15–#18 (nethack-beater wire
> defects), #19 (the fragment route). Every open issue is now genuinely open.

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

**What the closes recorded, now that they are written.** Both refusals are the
same asymmetry in different clothing, and the pair is more instructive than
either alone: `cgrid` — a constant grid over *declared preferences* — shipped,
while both a grid over θ (#4) and a world-declared `P(evidence | utility-param)`
(#1 half 2) were refused. Same mechanism, opposite direction. The line is not how
much structure the world may declare; it is whether the declaration is about the
world's values or about the agent's inference.

#4's close states plainly that the compatibility-shaped design revision 1 was
proudest of answered an objection nobody raised. #5's close carries its residue
rather than closing clean: what is discharged is the structural half, and the
real-corpus half is named as a consumer-side measurement that cannot be a
proplang obligation — which is exactly W1′'s warrant, now on the record in
proplang's own tracker. #8's close names #19 and #16 as its wire residue, so a
host blocked by the myopic wire is not misled by a green ledger row.

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

> **DECIDED 2026-07-22 — WAIT.** All four workstreams have reported. W1′ passed
> its pre-stated bar; the fragment route has not landed. This is the outcome the
> spec anticipated below, reached for the anticipated reason. **Phase 1 does not
> start.** Full results:
> `2026-07-22-w1-prime-prestatement.md` (+ Amendments 1–2) and
> `2026-07-22-w1-prime-results.md`.

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

### 7.1 The decision as taken

**WAIT**, on both required conditions:

| condition | status |
|---|---|
| `p1` discriminates on real governance features | **met** — decision-rule outcome 3, PASS on the pre-stated bar |
| fragment route landed on the wire | **not met** — proplang #19 open, no disposition |

**What W1′ actually established.** The engine discriminates when threat features
are *declared to it*: p5 of 20 pinned draws = +0.001482 against a null q95 of
+0.000421, all 20 draws positive (p ≈ 1.8e-4 against the empirical null sign rate
of 0.65). The result rests on **direction, not magnitude** — `S` spans 0.0015 to
0.702 and 9 of 20 draws fall below the null's own maximum, so no point estimate
from it should be quoted.

**And it established something the spec did not anticipate at all**, which is the
more consequential half. The control arm returned `S_ctl = 0.000000` on every
draw, with a single `p1` value across 1,320 probes — because through the current
wire, with the six waste features held, **an attack context and a benign context
encode to a byte-identical vector.** Not "hard to separate." The same point. All
nine threat-bearing features are projected out at the handshake by the epoch-1
waste scope. So the third exit branch ("Stop — a verdict about our encoding") was
nearly the right diagnosis and would have been reached for nearly the right
reason; what the two-arm design added was the ability to say *which* of the
engine and the encoding was at fault, instead of guessing.

**The blocker list therefore grew by one, and it is ours, not proplang's.**
Phase 1 now requires:

1. **proplang #19** — the fragment route, so 0.96 is reachable over the wire.
   Filed, open, no disposition. Not ours to close.
2. **The namespace projection** — the threat features must cross the handshake or
   a `said@1` shadow measures nothing. This is an epoch-1 scope decision
   (HOSTS_PLAN 2.1) and the author's. **New: this was not in the spec.**

**One live doubt carried forward.** W1′'s pass is measured at 100 training ticks.
The post-hoc saturation sweep does not reproduce the collapse round 2 wrongly
reported, but it does show concentration setting in — distinct `p1` falling
44 → 42 → 12 and max `p1` climbing 0.746 → 0.776 → 0.798 toward the 0.9 rung
between 100 and 1000 ticks — while the retired live shadow, at ~95k ticks, sat at
**two** distinct values with 98.7% at 0.9. Same direction, not yet arrived, and
the deployment regime is unmeasured. **Any Phase-1 shadow must re-measure `S` at
deployment tick counts rather than inherit this number.**

### 7.2 Process record

Worth keeping, because the phase's methodology failed twice before it worked and
both failures were caught by the design rather than by luck:

- **Round 1** fired the confound gate (`S_ctl = +0.370` vs a +0.000181 bar) and
  both arms were discarded per the committed rule, against my stated expectation.
  Cause: a fifth fixture artifact (`parent-tool-call-name = none`, 0.75 attack vs
  0.006 field).
- **Round 2** was retracted entirely: the corpus was never pinned. `corpus.field()`
  re-read the live, growing log, so a fixed seed drew different data each run —
  the same configuration produced 0.000311, 0.228573, and 0.566410. That bug also
  exposed a wrong bar: a permutation null shuffles labels with the sample held
  fixed, so it is structurally blind to sample-noise, which dominates here by
  three orders of magnitude.
- **Round 3** pinned the corpus (sha256 `0833f9b9…`, n = 130,799) and replaced the
  bar with a sample-aware one requiring the 5th percentile of 20 draws to clear
  the null.

The transferable lesson is the cheap one: **pinning the seed is not pinning the
data.** This repo's own convention already said so (`CLAUDE.md`, commit-pinned
fixtures); it simply was not applied to a live log.

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
