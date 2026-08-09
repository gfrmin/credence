# Stage A results — the rail was our label, not proplang's ceiling

Criteria fixed and committed before any number existed:
`2026-08-09-stage-a-prestatement.md` (`2277660`).
Corpus: `stageA.json.gz`, n = 143,034, sha256
`2814a2a5603cd8babe7e0a6f8d3c3325f6b8418986ce6bb538eb0a488b72d475`
(local-only, `~/git/research/w1-prime`; aggregates only appear here).

---

## 1. Verdict

**R2 fires, with R1 as its proximate mechanism, and R3 additionally for
`table@1`.**

Every evidence channel the shadow engine was ever fed runs at **0.99 or above**.
The declared grid's top rung was 0.9. The posterior therefore pinned to 0.9 —
which is the **correct** rung, KL-nearest a rate of 0.993–0.996 on a grid capped
at 0.9. The engine was not failing and was not uninformative about our features.
It was faithfully reporting a channel that carries almost no information.

The channel is that way because of **our own label definition**, not proplang's
ceiling. And so the ceiling's dissolution at the `#19` sitting — the event that
prompted this replan — **does not unblock this consumer**.

## 2. What the engine was actually fed

Three evidence paths reach it, not the two the pre-statement named. The third
was found while checking why `table@1` reached 0.9 on eight verdicts, and it
dominates by volume:

| stream | source | n | ones | rate |
|---|---|---|---|---|
| **warm prior** | `warm_brain.counts.json`, replayed at boot (`membrane.py:381-401`) | 39,314 | 38,958 | **0.990945** |
| **human verdict** | `user-responded`, yes→1 no→0 (`log.py:35-57`) | 8 | 8 | **1.000000** |
| **outcome good-bit** | `_outcome_good_bit` (`log.py:89-97`), `latent@1` only | 71,867 | 71,829 | **0.999471** |

The warm prior ships **inside the installed package**, not in the observation
log — it is invisible to any analysis that reads only the log, which is why the
pre-statement missed it. It is 99.1% approvals by construction: 100 of its 118
contexts have zero negatives.

The outcome stream reaches 0.9995 by **discarding 69,850 of 141,717 outcome
records — 49.3%**. `_outcome_good_bit` returns `1` iff `completed`, `0` iff
`reverted`, and `None` otherwise; `None` is skipped rather than counted. Every
`completed=False` record without explicit revert evidence is dropped. What
survives is 71,829 ones against **38** zeros.

## 3. The empirical-frequency check

`p1` across 190,104 shadow records, both forms, two values each:

| form | `p1` | n |
|---|---|---|
| `latent@1` | 0.8999999999999999 | 93,866 |
| `latent@1` | 0.4975137570219514 | 1,207 |
| `table@1` | 0.8999999999999999 | 93,824 |
| `table@1` | 0.4975137570219497 | 1,207 |

The 0.4975 value is the post-boot cold plateau — a respawned agent before
evidence, i.e. the prior.

> **Count reconciliation.** Phase 0 and `deploy/membrane.conf` record **190,102**
> shadow records (95,072 / 95,030); this document says **190,104**. The raw log
> holds **190,106** `membrane-shadow` lines (95,074 `latent@1` / 95,032
> `table@1`), of which one event id (`evt_11b088b4`) is emitted twice per form —
> so 190,104 is the distinct-(form, event) count and is what the pin carries.
> Phase 0's figure is low by two per form. The same duplication, plus a second
> id (`evt_12ed978c`), accounts for the joined event count of 143,034 against
> 143,036 `tool-proposed` lines. Nothing here turns on the difference; it is
> reconciled so two adjacent merged documents do not carry two counts.

Running gap against the stream each form saw, at decide checkpoints. `latent@1`
also receives the 8 verdict ticks (`replay_contexts` runs for **both** forms,
`membrane.py:410-411`); they are omitted from its column because 8 further ones
on a ~99,867-tick stream move the rate in the seventh decimal. Noted so the
column is not read as an architecture claim.

```
latent@1 (warm + outcome)          table@1 (warm + human)
decides   n_evid   rate     gap    decides  n_evid   rate     gap
      1    55197  0.99337  0.0934        1   39322  0.99095  0.0909
    100    55266  0.99338  0.0934      100   39322  0.99095  0.0909
   1000    55455  0.99340  0.0934     1000   39322  0.99095  0.0909
  10000    57279  0.99359  0.0936    10000   39322  0.99095  0.0909
  30000    67143  0.99446  0.0945    30000   39322  0.99095  0.0909
  95000    99867  0.99606  0.0961    95000   39322  0.99095  0.0909
```

- **`latent@1`: the gap GROWS**, 0.0934 → 0.0961 over 95,000 decides. Slowly,
  but monotonically, and for the reason the #19 record names — the stream is
  pulling *further* above the ceiling, so the projection error widens. This is
  the leg-C direction with a leg-D cause.
- **`table@1`: the gap is flat** at 0.0909, because after boot it received
  **eight** further evidence ticks in 95,031 decides. Its `p1` is the warm
  prior's projection and nothing else — **R3**.

In both cases `gap ≈ rate − 0.9`: the distance from the true rate to the grid's
ceiling, exactly.

## 4. The operating rate is a property of our label, not of the world

The same corpus, four conventions for the good-bit:

| variant | definition | n | rate |
|---|---|---|---|
| **V0 (shipped)** | `log.py:89-97` — ambiguous is skipped | 71,867 | **0.999471** |
| V1 | `completed=False` counts as a negative | 141,717 | **0.506848** |
| V2 | grounding-source records only, `completed` as the bit | 141,712 | **0.507085** |
| V3 | the primary's own `proceed` rate (not an evidence source) | 143,033 | **0.903442** |

**The convention moves the operating rate by 0.49.** Whether `completed=False`
without revert evidence is a negative or an absence is a substantive question
about what "waste" means — the shipped code's conservatism is defensible, and
this document does not settle it. What it does settle is that the choice, not
the engine and not the ceiling, is what determines the reading.

## 5. Pricing the grid — and why the obvious fix is the wrong one

Model count from `Enumerate.hs:115-186`,
`models = |θ| + Σ_guards |guard grid|·|θ|·(|θ|−1)`, at 39 singleton guards.
ms/tick from the log-log fit through proplang's own P0 baseline
(1601→92.6ms, 4803→635ms, 8005→1870ms) — **steady shallow depth, so a floor**,
since proplang measures a depth-300 tick at ~3.5× its depth-30 cost.

| grid | rungs | models | ms/tick | `p1` at V0 (0.9995) | `p1` at V1 (0.5068) |
|---|---|---|---|---|---|
| shipped | 9 | 2,817 | ~266 | 0.9 | **0.5** |
| naive top-extend | 12 | 5,160 | ~824 | **0.999** | 0.5 |
| placed | 10 | 3,520 | ~403 | **0.999** | 0.5 |

Read the V0 column. Extending the grid upward — the change the `#19` sitting
made available, and the one this replan was built to exploit — moves `p1` from
0.9 to **0.999**, at 1.8× the models and ~3× the per-tick cost. That is a more
confident report of the same degenerate channel: the false clear, arrived at
honestly. Against a corrected label the **shipped 9-rung grid is already
adequate** — 0.5068 lands on the interior rung 0.5, nowhere near an extreme.

**The grid was never this consumer's binding constraint.**

## 6. What this does not establish

- **Not a measurement of `said@1` on current `proplang-host`.** These records
  come from the retired `proplang-govhost` at `d-close`. Stage A makes no claim
  about the current engine, as the pre-statement fixed in advance.
- **Not a verdict on which label is correct.** V1/V2 are the range the rate
  moves over, not proposals. Choosing among them is a decision about what waste
  means and it is not a measurement's to make.
- **Not a claim that proplang's ceiling never mattered.** It bound here — but it
  bound *because* our stream sat above it, and the second consumer (life-agent)
  hit it on a genuinely different path.
- **Not a large-sample result about features.** No feature-conditional claim is
  made; the warm prior's 118 contexts and the field's schema drift are unaddressed.

## 7. Consequence for the migration

Phase 0 exited **WAIT** on two blockers, and the replan opened because both
moved. Stage A says the movement is beside the point for this consumer:

1. **The fragment route / θ ceiling** — dissolved by proplang, and it does not
   help. §5.
2. **The namespace projection** — its authority evaporated with `HOSTS_PLAN`,
   and it is untested here.
3. **NEW, and now the binding one: the evidence channel is degenerate.** Fixing
   the grid, the namespace, or the engine changes nothing while 49.3% of
   outcomes are discarded and the warm prior is 99.1% ones.

A `said@1` shadow built today would reproduce this rail at higher cost. The next
decision is not "which grid" but **"what is the waste label"** — and that is the
user's call, not a measurement's.
