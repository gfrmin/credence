# Stage A′ pre-statement — how the `completed=False` decomposition will be read

**Committed before the numbers exist**, same discipline and for the same reason
as `2026-08-09-stage-a-prestatement.md`: I already hold a strong expectation
about the answer, which is exactly when fixing the reading in advance is worth
something.

Stage A closed by saying the waste label was *"a decision, not a measurement …
a question about what waste **means**, and it is the user's call."* That was too
quick, and this document is the retraction of the framing rather than of the
numbers. `completed=False` is not one fact awaiting a ruling; it is **three
facts wearing one name**, and which of them dominates is measurable from data
we already hold.

---

## 1. What is already known at authoring time

A pre-statement written after peeking is worthless. Everything below was
computed before this document was written:

| known | value | where from |
|---|---|---|
| outcome records discarded by `_outcome_good_bit` | 69,850 of 141,717 (49.3%) | Stage A |
| grounding field tuples | `completed=T` 71,825 · `completed=F` 69,818 · `reverted=T` 38 | Stage A′ census |
| `error` / `retries` on grounding records | `error=None` universally; `retries≥1` on **32** records | Stage A′ census |
| events carrying **both** a result-hook and a grounding record | 54,573 | Stage A′ cross-tab |
| of those, result-hook `completed=True` | 54,561 (99.98%) | Stage A′ cross-tab |
| …**and** grounding `completed=False`| **26,868** (49.25%) | Stage A′ cross-tab |

The last row is the reason this stage exists. `log.py:83` states that a
shadow-mode result-hook's `completed=true` is *universal* — it asserts the call
**executed**, nothing more. So on the overlap we have independent confirmation
that the call ran, and the structural classifier still declines `completed` on
essentially half of them.

**Not** yet known, and what this document fixes the reading of: *why* it
declines.

## 2. The mechanism, read from code

`ground_capture.py:99`:

```python
"completed": bool(o.executed and o.continued and not o.errored),
```

with `outcomes.SessionIndex.classify` supplying the three inputs. A `False` is
therefore reachable three ways, and they are not the same kind of thing:

| | condition | in `classify` | what it is |
|---|---|---|---|
| **D1** | `not executed` | call not located in any later snapshot, or unobserved tail | an **absence** — the classifier failed to observe |
| **D2** | `executed, not continued` | located, but no follow-on action | an **absence** — the session ended |
| **D3** | `errored` | located, continued, `tool_result` matched `_ERR` | a **negative** — real evidence |

Only D3 is information about the world. D1 and D2 are information about the
instrument. The recorded records cannot tell them apart: `ground_capture` writes
the collapsed bool, and `o.label` (`accepted`/`reverted`/`ambiguous`) is
discarded. Hence a re-run of the classifier over the raw capture.

`retries` is the one surviving hint — `_retries_after` returns 0 unless
`errored` — so `retries≥1` ⇒ D3. That gives a **floor of 32**, and the
docstring says the count is deliberately strict and under-counts, so it is a
floor and nothing more.

## 3. The readings, fixed in advance

Let `d3` be D3's share of the 69,818 `completed=False, reverted=False` records.

**S1 — INSTRUMENT-LIMITED (`d3` small, under ~10%).** The discarded half is
overwhelmingly unobserved, not failed. Then the shipped label's conservatism is
**correct**, there is no ruling that recovers the channel, and Stage A's framing
was wrong in a specific way: the question was never what waste *means*. The
remedy is upstream instrumentation — result-hook coverage, session-tail capture
— and **no label convention, and no θ grid, rescues this corpus**. Phase 1 does
not start on it.

**S2 — LABEL-LOSSY (`d3` large, over ~25%).** Real failures are being thrown
away by a convention that cannot distinguish them from absences. Then the fix is
a **code change** — `errored → 0` — with a computable expected information gain,
not a philosophical ruling, and the corrected channel is two-sided.

**S3 — INTERMEDIATE.** Report both; `d3` sizes the gain, and the decision is
whether that gain is worth the cost law of §2.1 of the replan.

### 3.1 The sensitivity check that can flip S2

`_ERR` includes bare `Error:` and `error:`, which `outcomes.py`'s own docstring
concedes benign output contains (*"never flips accepted↔reverted, since benign
output legitimately contains the word 'error'"*) — it was written as an
annotation, and `ground_capture` then promoted it to a label input. So D3 is
computed **twice**: once with the shipped `_ERR`, once with a strict subset
dropping `Error:|error:|: Exception`. If `d3` is large under the shipped regex
and collapses under the strict one, **S2 is withdrawn in favour of S1**, and the
finding is that the label is not merely lossy but noisy in a way that would have
fed the engine false negatives.

## 4. The independent test: does a corrected channel carry anything?

Even a perfectly corrected label is worthless if its rate is the same
everywhere: the engine conditions on **per-context** θ pairs, so a global 0.5068
that is 0.5068 in every context is exactly as uninformative as a global 0.9995.
Computed from the pinned corpus, per declared feature context, under each label
variant — n-weighted mean absolute deviation from the global rate, and the share
of evidence mass in contexts more than 0.1 away.

- **Flat** (spread near zero) → the corrected channel is *also* degenerate, S1
  and S2 both terminate in the same place, and the corpus is spent.
- **Spread** → the shipped label is destroying per-context signal that exists,
  and that quantity is the expected gain a successor would be buying.

This test is pre-stated because it can contradict S2: `d3` can be large while
the corrected channel is still flat, and in that case the label fix buys
nothing.

## 5. Non-claims

- Re-running the classifier over the **concatenated** archives gives it strictly
  more session context than the original incremental runs had, which were cut by
  four rotations. This biases the pass toward locating **more** calls, i.e.
  toward under-counting D1. Agreement with the recorded `completed`/`reverted`
  is reported alongside, and the decomposition is trusted only that far.
- Nothing here measures `said@1`, current `proplang-host`, or any engine. It
  measures our own instrument against our own data.
- `_is_error` is a regex over tool output. D3 is "the classifier's error test
  fired", never "the call was waste".
