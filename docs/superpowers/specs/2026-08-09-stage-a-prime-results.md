# Stage A′ results — the discarded half is instrument failure, not a label choice

**Verdict: S1 (INSTRUMENT-LIMITED) fires, decisively.** Reading fixed in advance
in `2026-08-09-stage-a-prime-prestatement.md` (commit `b015f31`, committed before
the reclassification ran). The pre-stated threshold for S1 was `d3 < 10%`;
measured `d3 = 4.31%`.

**Stage A's closing framing is falsified.** It said the next question was
*"a decision, not a measurement … a question about what waste **means**, and it
is the user's call."* It was a measurement, the data to make it was already on
disk, and it is now made.

---

## 1. What was measured

`outcomes.SessionIndex.classify` re-run over the raw capture — four zstd
archives, 185 GiB decompressed, 141,714 records, 644 sessions — reproducing the
grounding classifier exactly, but **keeping the three inputs that
`ground_capture.py:99` collapses into one bool**.

Harness: `~/git/research/w1-prime/stagea2_reclassify.py` (local-only; the capture
is months of real tool telemetry and `gfrmin/credence` is public).

### Validity first

| | agreement |
|---|---|
| `completed` reproduced | 141,448 / 141,712 = **99.81%** |
| `reverted` reproduced | 141,712 / 141,712 = **100%** |

The 264 disagreements are dominated by 235 in the **predicted** direction:
calls the original run could not locate but this one can, because the original
ran incrementally against a capture file cut by four rotations while this pass
sees the concatenated whole. That bias runs *against* the finding below — it can
only make D1 look smaller than it was.

## 2. The split

Of the 69,850 outcome records `_outcome_good_bit` discards:

| cause | n | share | what it is |
|---|---|---|---|
| **D1 — not located** | 66,117 | **94.66%** | the classifier could not find the call |
| D3 — errored | 3,014 | 4.31% | a genuine negative |
| D2 — no follow-on | 484 | 0.69% | session ended |
| (this pass located it) | 235 | 0.34% | rotation artefact |

And the subset that removes all doubt. Restricted to the **26,867 calls the live
result hook proves executed** — the hook fired at result time, so the call
demonstrably ran:

| cause | n | share |
|---|---|---|
| **D1 — not located** | 25,547 | **95.09%** |
| D3 — errored | 1,005 | 3.74% |
| D2 — no follow-on | 215 | 0.80% |

These calls ran. The structural classifier could not find them. That is not a
label convention being conservative; it is an instrument missing half its
targets.

**The shipped `_outcome_good_bit` is therefore correct.** Discarding those
records is the right thing to do with them — they carry no information about
whether the call was worth making. The engine was fed a thin but honest stream.

## 3. The sensitivity check resolved the other way

§3.1 pre-stated that if D3 collapsed under a strict error regex, S2 would be
withdrawn. It does shrink (3,014 → 1,140), but inspection of 400 sampled
matches says the **shipped regex is the better test, not the worse one**:

- the identified false-positive class — a match explained entirely by a
  zero-valued counter such as `n_error: 0` in a benchmark readout — accounts for
  **3%** of loose-only matches;
- the remainder is dominated by recognisable failures (compiler diagnostics,
  rejected pushes, build-tool compilation errors, authorisation failures), all
  of which use a lowercase `error:` and are therefore **missed** by the strict
  regex.

So `outcomes.py`'s docstring worry ("benign output legitimately contains the word
'error'") is real but small, and the strict alternative trades a 3% false-positive
rate for a much larger false-negative one. Either way D3 is a small minority and
the verdict is unchanged.

## 4. Why flipping the label would have been actively harmful

The obvious "fix" — treat `completed=False` as a negative — produces a global
rate of 0.5068 and a large per-context spread. That spread is not waste signal:

| variant | contexts | n | global | weighted MAD | mass >0.1 away | contexts w/ both classes |
|---|---|---|---|---|---|---|
| V0 shipped | 1,084 | 71,867 | 0.9995 | 0.0010 | 0.03% | 20 |
| V1 `False`→0 | 1,179 | 141,717 | 0.5068 | **0.3075** | 91.4% | 531 |
| V1, restricted to hook-proved-executed | 104 | 42,850 | 0.4656 | **0.3137** | 92.0% | — |
| **V2 corrected** (`errored`→0) | 1,089 | 74,879 | 0.9592 | 0.0303 | 1.5% | 196 |
| warm prior | 118 | 39,314 | 0.9909 | 0.0179 | 0.9% | **0** |

Row 3 is the tell: restricting V1 to calls **known to have executed** leaves its
spread untouched (0.3075 → 0.3137). The variation is driven almost entirely by
`parent-tool-call-name` — roughly 0.75 when the parent is `bash`, roughly 0.05
when it is `other` — which is a measure of whether a call sat inside a chain
with later tool calls to locate it against. It is an **observability gradient**.

Since 94.7% of V1's zeros are calls that ran fine and could not be found,
adopting V1 would have taught the engine that *unobservable* calls are wasteful.
That is worse than the degenerate channel it had: a confident, structured,
wrong signal in place of a flat one.

### 4.1 A correction to this stage's own pre-statement

§4 of the pre-statement said spread would mean *"the shipped label is destroying
per-context signal that exists, and that quantity is the expected gain."* That
inference does not hold and I withdraw it. Spread establishes that **something**
varies by context, not that the something is waste. The hook-restricted row is
what distinguishes them, and it was not in the pre-stated plan — it was added
when the V1 numbers came back with a suspicious shape.

## 5. The corrected label does not move the reading

Even taking the best available correction (V2: a located, continued call whose
result matched the error test is a 0), the rate is **0.9592**, and on the shipped
9-rung grid the KL-nearest rung to 0.9592 is still **0.9**:

| label | rate | KL-nearest rung on `[0.1 … 0.9]` |
|---|---|---|
| V0 shipped | 0.999471 | 0.9 |
| V2 corrected (shipped regex) | 0.959161 | **0.9** |
| V2 corrected (strict regex) | 0.983822 | **0.9** |
| V1 flip-the-label | 0.506848 | 0.5 |

So the retired shadow's `p1 = 0.9` was a faithful report under the shipped label
**and** under every defensible correction of it. Stage A's headline — the
posterior converged correctly on a near-constant channel — survives this stage
intact and is now better founded: the channel is near-constant because the
instrument cannot see failures, not because we chose to ignore them.

The underlying failure rate this traffic actually supports is ~4% (3,014 of
75,361 located calls). Reporting *that* faithfully would need a rung near 0.96 —
10 rungs, 3,520 models, ~403 ms/tick against the shipped 2,817 / ~266 ms — but
with a weighted MAD of 0.0303 it is a global rate, not a per-context signal, so
the extra rung buys a better-calibrated constant rather than a better decision.

## 6. Where the real defect is

The two instruments are mismatched in complementary ways:

- **`result-hook`** proves the call executed, but `hook.py:74` returns `True` for
  any plain-string `tool_response`. A Bash call that fails with a traceback
  returns a string body, so it is recorded `completed=True`. Hence 99.98% true,
  and `log.py:83`'s note that its `completed` is "universal".
- **`grounding`** applies a real error test, but can only locate ~50% of the
  calls it grounds, and its `error` field is `None` on all 141,713 records.

Neither alone yields a two-sided label with coverage. The same repository already
contains both halves: `outcomes._is_error` is exactly the test `hook.py`'s
string-body branch is missing, and the `/result` payload already carries an
`error` field (`daemon.py:447`) that nothing ever populates. Applying the
existing error test to the response body at result time would produce a
two-sided label over ~54,800 events with no structural replay at all.

**Flagged, not actioned** — that is a change to a live production hook and to the
meaning of a recorded field, and it is outside this stage's scope.

## 7. Consequences for the migration

1. **The waste-label question is answered, and the answer is "leave it alone."**
   A `completed=False` with no revert evidence is an absence 94.7% of the time,
   and the shipped convention treats it correctly.
2. **The binding constraint is instrument coverage, not the label and not the
   grid.** No θ declaration, no namespace projection, and no engine change
   affects it.
3. **Phase 1 still should not start**, but the reason has moved again: not
   "awaiting a ruling on what waste means" but "the outcome instrument resolves
   ~4% of a signal it can only observe half the time." A `said@1` shadow would
   reproduce the same rail at higher cost.
4. The cheapest thing that would change any of this is a hook fix (§6), and its
   effect is measurable in advance from this corpus.

## 8. Non-claims

- Nothing here measures `said@1`, current `proplang-host`, or any engine.
- D3 is "the classifier's error test fired", never "the call was waste". A failed
  call can be worth making; a successful one can be waste. This corpus contains
  no label for value, only for outcome.
- The ~4% failure rate is over **located** calls. The 66,117 unlocated ones have
  no error evidence either way, so it is an estimate on the observable half, not
  a population rate.
