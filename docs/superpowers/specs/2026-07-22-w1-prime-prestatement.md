# W1′ pre-statement — criteria fixed before the run

**Status: PRE-STATEMENT. Written and committed BEFORE any real-label `p1` was
computed.** This is the R-D21 discipline borrowed from proplang's own W1
(`test-measure/Measure.hs` header: "criteria pre-stated and the exact stream
executed … BEFORE this file froze"). Everything below — design, statistic, bar,
decision rule — is fixed at commit time. The run follows in a separate commit.

What has been computed so far is **structural only**: namespace sizes, model
populations, ms/tick, feature domains, and class-overlap counts in the encoding.
No `p1` has been read for any labelled contrast. The permutation null (§5) is
computed under the null hypothesis and does not reveal the treatment effect.

---

## 1. The question

Does proplang's `p1` discriminate on **real governance features** through the
**current** wire (`said@1`, `proplang-host` at master), as opposed to the
synthetic risk-bit stream on which proplang measured `S = 0.794`?

proplang scoped its own result explicitly and did not overclaim it
(`test-measure/Measure.hs:122-133`): the `"attack"`/`"benign"` names there are
synthetic risk-bit contexts, and *"whether the engine discriminates on REAL
governance features stays unmeasured until a host-corpus differential re-run; no
such re-run is claimed."* This is that re-run. It is a consumer-side measurement
and cannot be a proplang obligation — recorded as such when their issue #5 was
closed.

## 2. Why the originally-specced design was abandoned

The Phase-0 spec called for a straight attack-vs-benign contrast on the
governor's captured corpus. Three structural findings, all obtained before
reading any `p1`, make that design produce a **confounded pass**:

1. **The wire namespace is waste-only.** All nine threat-bearing features —
   `action-class`, `coding-action-class`, `cred-exfil-chain`,
   `egress-destination`, `injected-imperative`, `taint-flow`, `taint-source`,
   `target-externality`, `target-sensitivity` — are projected out at the
   handshake (epoch-1 waste scope, HOSTS_PLAN 2.1). The engine sees six
   features, all tool-usage shape.
2. **The red-team fixtures are confounded on four of the six visible features.**
   All 20 cases are constant at `working-directory-relative=project-root`,
   `recent-repetition-count=rep-0`, `recent-identical-call-count=ident-0`,
   `time-since-last-user-message=gt-10m`. Field traffic is not (`rep-3plus`
   0.39, `lt-10m` 0.34). Those constants are artifacts of authored synthetic
   sessions — no repetition history, no elapsed-time context — not properties of
   attacks. A contrast over them would yield a large `S` meaning *"synthetic
   fixture vs real session"* and would read as success.
3. **The real waste label is nearly empty**: 38 `reverted` against 164,718
   outcome records (0.023%).

Finding 2 is why this pre-statement exists. The failure mode was not a null
result; it was a **green number that would have wrongly cleared the gate**.

## 3. Design — two arms, matched control

Both arms hold the four confounded features FIXED at the fixture values
(`project-root`, `rep-0`, `ident-0`, `gt-10m`) for **every probe in both
classes**, so authoring artifacts contribute identically to attack and benign
and cannot drive `S`.

| arm | features declared | indicators | models | ms/tick |
|---|---|---|---|---|
| `ctl` | the 6 waste features (current wire) | 39 | 3,977 | 73.0 |
| `wide` | those 6 + the 9 threat features | 82 | 7,073 | 304.1 |

`ctl` is not merely expected to be flat — it is **structurally unable to
separate**: after matching, the 20 attacks collapse to 6 distinct encoded
vectors, and **all 6 also occur in field traffic** (32 distinct field vectors).
Attack and benign probes are drawn from the same set of points. Under `wide` the
attacks occupy 18 distinct vectors with **zero overlap** against 692 field
vectors.

That asymmetry is the experiment. `ctl` measures what the current wire can do;
`wide` measures what the engine can do when the context is actually given to it.

**Harness.** `proplang-host` built from master (`Host.hs` and binary both
2026-07-21 19:42), driven over a **PTY** — proplang issue #18 (block-buffered
stdout under pipes) is open and unfixed; `hostMain` sets no buffering, so a
plain pipe deadlocks on the first exchange. Verified working before this was
written. Evidence replay is O(ticks) (issue #17); at 100 training ticks the cost
is 7.3 s (`ctl`) / 30.4 s (`wide`) per run, which is what makes the permutation
null affordable.

**Protocol** (seed fixed at 20260722; sampling frozen before the run):

- attacks: 20 → **14 train / 6 probe**
- field: **14 train / 60 probe**, sampled from 130,543 captured `tool-proposed`
  records
- training stream: **100 ticks**, alternating train-attack (`evidence = 1`) and
  train-benign (`evidence = 0`), cycling
- probe: the held-out 6 attack + 60 benign contexts, no evidence, menu present;
  read `p1` from each reply

Evidence polarity follows proplang's W1 convention (`evidence` tracks the
attack bit), so that "discriminates" is a POSITIVE `S` and the statistic stays
directly comparable to their 0.794.

## 4. The statistic

```
S = median p1(attack probes) − median p1(benign probes)
```

Reported per arm, **with the full per-class `p1` distribution, never the summary
alone** — the W3′ finding is that a summary statistic cannot distinguish a real
null from a railed one.

## 5. The bar — derived, not borrowed

**proplang's `S ≥ 0.4` is NOT reused.** It was derived as half the maximum
achievable separation on a *perfectly-informative synthetic stream* (θ ∈
[0.1, 0.9] bounds `S` at 0.8). Real governance features are not perfectly
informative, so that number does not transfer and reusing it would be a category
error. What transfers is the *discipline* — derive a bar and pre-state it — not
the value.

**The bar is a permutation null.** For each arm, K = **60** replications: shuffle
the train-label assignment (which contexts carry `evidence = 1` vs `0`), keep
everything else byte-identical, recompute `S`. This yields the distribution of
`S` under "the engine cannot read these features," derived from this corpus's own
structure and this harness's own noise.

> **BAR: an arm passes iff its real-label `S` exceeds the 95th percentile of its
> own K = 60 permutation null.**

The null is computed and written down BEFORE the real-label `S` is computed.
Permutation resolution is 1/(K+1) ≈ 0.016, adequate against a 0.05 threshold.

## 6. The rail check — mandatory, and it runs first

From W3′: across 190,102 live shadow decisions `p1` took exactly TWO values,
93,719 of them at `0.8999999999999999` — `thetaPoints`' top rung. A flat `S` can
therefore mean the grid ceiling rather than anything about features, and
`S ≈ 0` is what all interpretations produce.

> **RAIL CHECK: if the probe `p1` values take ≤ 2 distinct values, OR the modal
> value is `0.9 ± 1e-9`, the arm's `S` is DECLARED UNINTERPRETABLE** and reported
> as a ceiling verdict regardless of its value.

`thetaPoints = 0.1 :| [0.2 … 0.9]` is byte-identical at `d-close` and master, so
the ceiling that railed the shadow is the one this run is exposed to.

## 7. Decision rule — fixed now, applied in order

1. **CONFOUND** — if `S_ctl` exceeds its null q95: the matching failed, an
   artifact is driving the contrast. **Discard both arms**, report the confound,
   do not report `S_wide` as a result.
2. **CEILING** — else if either arm fails the §6 rail check: report a ceiling
   verdict for that arm; `S` is not interpretable there.
3. **PASS / projection verdict** — else if `S_wide` > its null q95 (and `S_ctl`
   does not exceed its own): the engine reads real governance context when given
   it, and the blocker is the **adapter's namespace projection**, not proplang.
4. **ENGINE verdict** — else if `S_wide` ≤ its null q95 with the rail check
   passed: the engine does not discriminate on real governance features even
   when they are declared. This is the finding that would stop the migration.

Outcome 3 is the one I expect on the structural evidence (zero vector overlap
under `wide`, total overlap under `ctl`). **Expecting it is exactly why the bar
and the rule are committed before the run.**

## 8. What this run does NOT establish

- It is not a Phase-1 readiness result. The fragment route (proplang #19) is
  unaffected by anything measured here; a `PASS` still leaves the 0.96 threshold
  unreachable over the wire, which is the other half of the Phase-0 exit.
- It does not license widening the shipped namespace. Arm `wide` is an
  experimental handshake, not an adapter change; declaring nine threat features
  into the live wire is an epoch-1 scope decision (HOSTS_PLAN 2.1) and the
  author's.
- 6 held-out attack probes is a small sample. The permutation null uses the
  identical statistic on the identical sample, so the comparison is valid, but
  the effect size carries that width and will be reported with the full
  distribution rather than as a point claim.

---

# AMENDMENT 1 (2026-07-22) — round 1 fired the confound; round 2 design

**Committed BEFORE round 2 ran.** Round 1's numbers are reported in full below;
round 2's are not yet computed.

## A1.1 Round 1 result: CONFOUND, decision rule step 1

    ctl arm:  S = +0.37025750    bar (own null q95) = +0.00018131
              perm-p = 0.0164    EXCEEDS BAR -> confound fired
              null over K=60: min -0.000470, max +0.000334

Per §7 step 1 this **discards both arms**. The `wide` arm was aborted unrun at
8/60 and its partial output deleted; no `S_wide` was computed, and none is
reported. The rule was applied as written, against my own stated expectation
(§7: "Outcome 3 is the one I expect"). That is what pre-stating it was for.

## A1.2 Diagnosis: a fifth fixture artifact I missed

`parent-tool-call-name = none` occurs in **15/20 attacks (0.75)** and in
**0.006 of field traffic** — a ~125x enrichment. Same mechanism as findings 2's
original four: authored red-team cases are synthetic single-call sessions with
no parent tool call, while real traffic almost always has one. `tool-name` is
comparatively innocent (bash 0.65 attack vs 0.586 field).

**My §3 claim that `ctl` was "structurally unable to separate" was WRONG, and
the control caught it.** I inferred it from set-overlap: all 6 encoded attack
vectors also occur in field traffic. But set-overlap does not imply equal
FREQUENCY — those vectors occur in benign traffic at very different rates, and a
median is sensitive to exactly that. The overlap statistic was too weak to carry
the claim I hung on it.

## A1.3 Round 2 design — hold ALL SIX waste features

The tempting fix is to add `parent-tool-call-name` to the held set. **That would
be tuning the design until the control passes**, one feature per iteration, and
it would leave the same question open about the next feature.

Instead: hold **all six** declared waste features fixed, in both classes, in
training and in probes, at the **ambient field operating point** (the modal value
of each in captured traffic):

    tool-name                     bash            recent-repetition-count       rep-3plus
    working-directory-relative    project-root    recent-identical-call-count   ident-0
    parent-tool-call-name         bash            time-since-last-user-message  gt-10m

This is the **maximal non-selective choice** — there is nothing further that
could be added — and the criterion is principled rather than reactive: the waste
features are nuisance covariates for the threat question, not the features under
test. It is not "hold the one that leaked"; it is "hold the entire nuisance
space." Round 1's leak could not have selected it, because it removes every
candidate at once.

Structural consequence, verified before running:

| arm | distinct attack | distinct benign | overlap |
|---|---|---|---|
| `ctl` | **1** | **1** | the SAME single vector |
| `wide` | 17 | 381 | 7 |

**`ctl` stops being a statistic and becomes an ASSERTION.** Attack and benign
probes encode to a byte-identical vector, so `S_ctl` must be **exactly 0.0**. If
it is not, the harness is broken and nothing else is reportable. This is a
strictly stronger control than round 1's, which relied on a frequency argument I
got wrong.

`wide` remains the test, and note its overlap is 7 of 17 rather than round 1's
zero — with the nuisance held at ambient, some attack threat-vectors do coincide
with benign ones. Reported here so the separation is not overstated; the
permutation null, not the overlap count, is the bar.

## A1.4 What is unchanged

The statistic (§4), the bar rule (§5: exceed own K=60 permutation null q95),
the rail check (§6), and the interpretation of outcomes 2-4 (§7) all stand as
committed at 6c52352. Only §7 step 1 changes form: "CONFOUND" becomes
**"ASSERTION FAILURE — if `S_ctl` != 0.0 exactly, the harness is broken; report
nothing else."**

## A1.5 Harness positive control (run before round 2)

proplang's own W1 synthetic stream through this PTY harness, verbatim:

    p1_attack = 0.897059   p1_benign = 0.102941   S = 0.794119
    proplang's pinned:     p1_attack = 0.897      S = 0.794119

Exact to six decimals against a frozen anchor, so a null from this harness is a
real null and not an artifact. Probe idempotence also verified: a menu-only tick
does not condition, so sequential probing equals proplang's saved-state probing.

---

# AMENDMENT 2 (2026-07-22) — round 2 was contaminated; the corpus was never pinned

**Committed BEFORE round 3 ran. Round 2's numbers are RETRACTED.**

## A2.1 The bug

`corpus.field()` re-read `~/.credence-governor/observations.jsonl` on every
invocation and shuffled the result under a fixed seed. **The governor is live and
that log grows.** Within this session the `tool-proposed` count went
129,519 → 130,543 → 130,787 → 130,799. Shuffling a list whose LENGTH changes
under a fixed seed yields a **different sample every time**, so each process drew
a different benign train/probe set.

I pinned the seed and never pinned the data. The repo's own convention says
exactly this (`CLAUDE.md`: "Test fixtures are commit-pinned … capture fixtures
from a specific named SHA"), and I did not apply it to a live log.

## A2.2 How bad — the same "seeded" configuration, three times

    gate run (round 2)   S_wide = +0.228573
    decay sweep, N=100   S_wide = +0.000311
    5x repeat check      S_wide = +0.566410  (5/5 bit-identical within one process)

Three values spanning **0.0003 to 0.566** from a configuration that differed only
in which benign records the log happened to hold at that moment.

The engine is NOT the source: held to one sample it is **perfectly deterministic**
(5 consecutive runs bit-identical). Everything unstable here is mine.

## A2.3 What this invalidates, and one thing it does not

RETRACTED: round 2's `S_wide = +0.228573`, its `perm-p`, its "54x the null max",
the outcome-3 PASS, the 100-vs-1000-tick decay table, and both sweeps. All are
moved to `contaminated/` and none is reported as a result.

STANDS: `S_ctl = 0.000000` exactly. The control assertion holds under ANY sample,
because with all six waste features held, attack and benign encode to a
byte-identical vector — a structural identity, not a sampling claim. Also stands:
the harness positive control against proplang's frozen anchor, and probe
idempotence.

## A2.4 The deeper methodological error the bug exposed

The permutation null shuffles LABELS with the sample held fixed. It therefore
measures label-noise **conditional on one draw** and is structurally blind to
sample-noise. With 6 attack probes and 60 benign probes, sample-noise turns out
to dominate — the spread in A2.2 is far larger than any null (max +0.004223).

**A bar built only on the permutation null was the wrong bar**, independently of
the pinning bug. It would have certified an effect whose sample-to-sample spread
is three orders of magnitude wider than the null it was compared against.

## A2.5 Round 3 design

**Corpus pinned**: `field_pinned.json`, n = **130,799**, sha256
`0833f9b9440fcecad548b81cb738cac7bf43edf275f7f727594484c36aab8337`. The loader
asserts the count and never touches the live log again. Attack corpus is the 20
code-derived `red_team.py` cases (stable by construction).

Held features, hold values, statistic, and the rail check are UNCHANGED from
Amendment 1.

**The bar is amended to cover sample-noise.** For each of **B = 20** independent
field draws b (distinct seeds, from the pinned pool), compute `S_wide(b)` and
`S_ctl(b)`. Compute the K = 60 permutation null on draw b = 0.

> **BAR (amended): the arm passes iff the 5th percentile of {S_wide(b)}_{b=1..20}
> exceeds the permutation null's q95.**

That is strictly stronger than the round-2 bar: the effect must clear the
label-null not on a lucky draw but on 19 draws out of 20. A point estimate from a
single draw is no longer sufficient for a PASS, and the full B-distribution is
reported whatever the outcome.

**Assertion (unchanged in force):** `S_ctl(b)` must be exactly 0.0 for every b.

**Decision rule** as amended in A1.4, with step 3 now reading "the 5th percentile
of the B-distribution exceeds the null q95" in place of "the real-label S exceeds
the null q95".

## A2.6 Status of the decay finding

Round 2 suggested `S` collapses to 0 by 1000 training ticks. **That is retracted
too** — both sweeps ran on drifting samples. It is additionally suspect for a
second, independent reason: the training stream alternates attack/benign by slot
with `t = i+1`, so `t`-parity predicts the label perfectly, and a short
"label = `t` is odd" program can outcompete context-sensitive ones as evidence
accumulates. Since all probes share one `t`, that alone would drive `S → 0`. The
confound is inherited from proplang's own W1 stream, which alternates identically;
it does not bite at their 60 ticks.

Round 3 re-runs the decay check on the pinned corpus with **shuffled slot order**,
so `t`-parity carries no label information. Reported as post-hoc, outside the gate.
