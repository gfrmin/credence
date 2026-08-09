# Stage A pre-statement — how the empirical-frequency check will be read

**Committed before the numbers exist.** Same discipline as the W1′
pre-statement (`2026-07-22-w1-prime-prestatement.md`), and for the same reason:
by the time this is written I already hold a strong expectation about the
answer, which is exactly when fixing the reading in advance is worth something.

Corpus: `stageA.json.gz`, n = 143,034 events, sha256
`2814a2a5603cd8babe7e0a6f8d3c3325f6b8418986ce6bb538eb0a488b72d475`, built by
`stagea_extract.py` from the closed observation log. Local-only
(`~/git/research/w1-prime`) — the log is months of real tool-call telemetry and
`gfrmin/credence` is public. Only aggregates and the digest appear here.

---

## 1. What is already known at authoring time

Stating this because a pre-statement written *after* peeking is worthless, and
the honest position is that some of these numbers were computed while replanning
and while inspecting the corpus's structure in A1:

| known | value | where from |
|---|---|---|
| primary `proceed` rate | 129,224 / 143,035 = **0.9034** | live log, during replanning |
| retired shadow `p1` (modal) | **0.8999999999999999** | Phase 0 W3′ |
| `p1` distinct values across 190k | 2 (the above + ≈0.4978 post-boot) | Phase 0 W3′ |
| `entropy_bits`, `residual_mean` | one value each | Phase 0 W3′ |
| `user-responded` records in log | **8** | A1 |
| outcome records by (source, completed) | grounding T 71,861 / F 69,852; result-hook T 54,822 / F 12 | A1 |
| `reverted is True` records | 38 | A1 |

**Not** yet known, and what this document is fixing the reading of: `p1` split by
`utility_form`; the empirical rate of the stream the engine actually conditioned
on; the gap between them; and the gap's **trend** over the stream.

## 2. The mechanism, read from code (not from data)

Two evidence paths reach the engine, and only two:

- **`replay_contexts`** (`log.py:35-57`) — `user-responded` records, `yes`→1,
  `no`→0. Timeouts are not labels. This is `table@1`'s *only* evidence, and
  `latent@1`'s v1-continuity stream. **n = 8** in the entire log.
- **`replay_outcome_contexts`** (`log.py:99-135`) — `latent@1` only. One
  representative outcome record per event (source precedence
  grounding > openclaw > result-hook), mapped through `_outcome_good_bit`
  (`log.py:89-97`): **0 iff `reverted is True`, 1 iff `completed is True`,
  `None` otherwise — and `None` is skipped, not counted.**

The primary's own `proceed`/`block`/`ask` action is **not an evidence source**.
Nothing in `membrane.py`, `shadow.py`, or `log.py` feeds it across the wire.

### 2.1 A correction to this replan's own headline hypothesis

The approved plan proposed that the shadow's rail was the posterior converging
onto the rung KL-nearest the primary's 0.9034 approve rate, with the grid capped
0.0034 below it. **The code says that cannot be the mechanism**: the primary's
action never crosses the evidence channel, so 0.9034 is not a rate the engine
could have been estimating. The numerical near-coincidence with the 0.9 top rung
stands, but it is now a coincidence with no causal path behind it unless the
actual evidence stream's rate lands there too. Recorded here rather than quietly
dropped.

## 3. The candidate readings, fixed in advance

Let `r` be the empirical rate of the stream the engine conditioned on, and `g`
the gap `|p1 − r|`. Exactly one of these is the finding:

**R1 — GRID-CAPPED (leg D).** `r` lies above the grid's top rung 0.9, `p1` pins
to 0.9, and `g ≈ r − 0.9` is stable or shrinking toward that floor. The
constant is the declared grid being too coarse *at the top*, and the remedy is
rungs above 0.9. This is the reading the replan was built on.

**R2 — LABEL-DEGENERATE.** `r` is at or near 1.0 **because the label definition
discards nearly all negatives** — `_outcome_good_bit` maps `completed=False`
without revert evidence to `None`, and those are ~69,862 of ~141,717 records.
Then the engine converged correctly on an almost-constant-1 stream and the
constant `p1` is a faithful report of a nearly information-free channel. **The
remedy is upstream of the grid**, and a finer grid alone would make the answer
*more confident and no better* — the #19 false-clear shape, sourced in our
label rather than our codebook.

**R3 — EVIDENCE-STARVED.** The form conditioned on ~8 ticks (`table@1`), so
`p1` is essentially the prior and `g` is whatever the prior happens to be. Not
a statement about the world at all.

**R4 — NONE OF THESE.** `g` is small and stable with `r` comfortably inside the
grid's interior. Then the grid was adequate, and the constant `p1` needs a
different explanation than any of the above — reopen with no preferred answer.

### 3.1 The trend, which discriminates independently of `r`

Per the #19 record, a gap that **grows under data** is the leg-C false-clear
signature and cannot be cured by waiting. Gap trend is therefore computed and
reported **separately** from the level, over the arrival-order `seq`:

- **shrinking or flat** → the declared rung is at or near the true rate;
- **growing** → the KL-projection is landing on a rung the rate does not
  support, and more evidence sharpens the error.

A growing gap is a finding regardless of which of R1–R4 fires on the level.

## 4. Non-claims, fixed now

- Nothing here is a measurement of `said@1` on current `proplang-host`. The
  records come from the retired `proplang-govhost` at `d-close`. Stage A cannot
  and will not make a claim about the current engine.
- No claim about whether the migration should proceed. Stage A decides only
  whether the Phase-0 W3′ reading is right, and what the θ grid would have to be.
- `r` is a rate over **our own declarations**, not ground truth about waste. If
  R2 fires, that is a statement about the label we chose, not about the world.

## 5. What gets repaired if R1 or R2 fires

Phase 0's W3′ recorded the constant as the posterior "saturating at the wire's
grid ceiling", and recorded `table@1` as **falsified** in its documented role as
"the non-degenerate challenger policy". Under R1 the first sentence is
imprecise; under R2 or R3 **both are wrong as written** — the shadow would not
have been a degenerate policy but a faithful report of a degenerate channel.
Either way the repair is in place, with the falsified words quoted, in:

- `2026-07-20-proplang-migration-phase-0-design.md` (the W3′ section)
- `credence-governor/docs/membrane-shadow.md` §6
- `credence-governor/.../deploy/membrane.conf` (the RETIRED banner)
