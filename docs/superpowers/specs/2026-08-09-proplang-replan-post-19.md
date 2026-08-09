# proplang migration — replan after the #19 sitting

**Status: Stage A CLOSED 2026-08-09. Phase 1 remains not-started, for a
different reason than Phase 0 gave.**

Supersedes the exit blockers in `2026-07-20-proplang-migration-phase-0-design.md`
§7. Reading: `2026-08-09-stage-a-prestatement.md` (criteria, committed first),
`2026-08-09-stage-a-results.md` (the numbers).

---

## 1. Why this replan opened

Phase 0 closed 2026-07-22 with **WAIT** on two blockers. On 2026-08-08 proplang
ran and closed the `#19` sitting (HEAD `94fd4eb`, tags `doctrine-sitting-r0/r1`),
and both blockers moved. The standing rule for this work is to re-derive against
proplang's current state rather than its historical record, so the spec was
re-derived rather than amended.

## 2. What proplang changed

| | then | now |
|---|---|---|
| θ grid | hard-wired `thetaPoints = 0.1 :| [0.2 … 0.9]` | **zero `src/` occurrences**; required consumer-declared hello data (`Host.hs:260`, since `c2ca82c` 2026-07-25) |
| 0.96 threshold | unreachable over the wire | **reachable today**, zero proplang change (leg B: 0.96659 at 40 ticks with one extra rung) |
| fragment route (#19) | filed, open | **CLOSED**, ruled (ii) — doctrine canonised, door deferred and demand-gated behind prefreeze-lint L10 |
| `HOSTS_PLAN` 2.1 scope | the author's epoch-1 ruling | **archived, banner-marked HISTORICAL** — "binds on nothing current" |
| `table@1` / `latent@1` | shipped | **retired**, zero `src/` occurrences |
| stdout buffering (#18) | pipe deadlock, PTY required | **fixed** (`Host.hs:617`) |
| evidence replay (#17) | O(ticks) | **still open** |
| new declarable surface | — | `obs_arity`, `breadth`, `clock` (priced `think`), utility `cgrid`, #20 K-ary readout |

### 2.1 A cost law the Phase-0 spec never anticipated

Per-tick cost grows **worse than linearly in fold depth** (half-slopes 3.725 vs
5.095, ratio 1.37); cumulative cost is worse than quadratic in session length.
At K=6 / 8005 sentences the wire tick crosses 2s at depth ~30, 5s at ~209, and
lands ~6.6s at 300. With `#17` still open, replay is one round-trip per tick.

**The 190k-decision shadow Phase 1 assumed cannot be rebuilt.** That corpus came
from an engine that no longer exists (`proplang-govhost`, ~4.6ms/decide). Any
future shadow must be bounded-window, which is a semantic change — no
cross-window learning — that has to be declared rather than smuggled.

### 2.2 A consumer obligation that now has teeth

`p1` converges to the declared rung **KL-nearest the true rate**, and a threshold
`p*` clears *iff that rung exceeds `p*`* — possible with the true rate below
`p*`, and **the error grows under data** (proplang leg C: 0.0024@100 →
0.0200@1200). This is the one error mode in that repo evidence sharpens rather
than repairs. The prescribed diagnostic is free: compare reported `p1` against
the empirical frequency of one's own stream.

## 3. What Stage A found

Applying that diagnostic to the 190,104 retired shadow records:

**Every evidence channel the engine was ever fed runs at 0.99 or above.**

| stream | n | rate |
|---|---|---|
| warm prior (`warm_brain.counts.json`, replayed at boot) | 39,314 | 0.990945 |
| outcome good-bit (`latent@1` only) | 71,867 | 0.999471 |
| human verdict (`user-responded`) | 8 | 1.000000 |

On a grid capped at 0.9, **0.9 is the KL-nearest rung to all of them**. The
posterior did not saturate — it converged correctly. The measured gap
`|p1 − rate|` equals `rate − 0.9` at every checkpoint, growing 0.0934 → 0.0961
across 95,000 `latent@1` decides.

The outcome stream reaches 0.9995 only because `_outcome_good_bit`
(`log.py:89-97`) discards **69,850 of 141,717 outcome records — 49.3%**: every
`completed=False` without explicit revert evidence. What survives is 71,829 ones
against 38 zeros. Change that one convention and the rate moves to **0.5068**.

**The engine was reporting a degenerate channel faithfully. The channel is
degenerate because of our label.**

## 4. The consequence, stated plainly

Extending the θ grid — the capability the `#19` sitting handed us, and the thing
this replan opened to exploit — **makes the reading worse**:

| grid | rungs | models | ms/tick | `p1` at the shipped label |
|---|---|---|---|---|
| shipped | 9 | 2,817 | ~266 | 0.9 |
| naive top-extend | 12 | 5,160 | ~824 | **0.999** |

A more confident report of the same near-constant stream, at 1.8× the models and
~3× the per-tick cost. That is the #19 false clear arrived at honestly — sourced
in our label rather than our codebook. Against a corrected label the **shipped
9-rung grid is already adequate** (0.5068 → interior rung 0.5).

## 5. Phase-0 §7 blockers, superseded

1. ~~**proplang #19** — the fragment route, so 0.96 is reachable.~~
   **DISCHARGED** by dissolution (§2). Does not unblock this consumer (§4).
2. ~~**The namespace projection** — an epoch-1 scope decision and the author's.~~
   **DISCHARGED**: `HOSTS_PLAN` binds on nothing current, so this is our call
   now. Still untested — Stage A makes no feature-conditional claim.
3. **NEW, and the binding one: the evidence channel is degenerate.** No change
   to grid, namespace, or engine alters this while 49.3% of outcomes are
   discarded and the boot prior is 99.1% ones.

## 6. What happens next is a decision, not a measurement

The next question is **what the waste label should be** — whether a
`completed=False` outcome with no revert evidence is a negative or an absence.
The shipped code's conservatism is defensible; the cost of it is that the
engine learns from a stream carrying almost no information. Stage A deliberately
does not settle this: it is a question about what waste *means*, and it is the
user's call.

Until it is answered, a `said@1` shadow would reproduce this rail at
substantially higher cost, and Phase 1 should not start.

### Deferred, with the work already scoped

**B** rebuild `proplang-host` at `94fd4eb`, pin sha256, write the `said@1`
handshake (note: `membrane.py`'s current handshake omits `codebooks` and would
be **refused outright** by a current build), reproduce proplang's leg-A/leg-B
witness numbers as a positive control. **C** the empirical-frequency guard as an
always-on diagnostic. **D** a bounded-window measurement sized to §2.1, bar
pre-stated. **E** decide — where "the exact engine is not deployable at this
traffic volume" is a legitimate outcome of D, not a failure of it.

## 7. Flagged, not actioned

- **life-agent is live on a stale binary.** `~/.local/bin/proplang-host` is
  dated 2026-07-22, predating the 07-25 re-founding, and
  `src/life_agent/membrane/world.py:handshake_decl` emits no `codebooks` block —
  it cannot move to a current build without a handshake change. Its measured
  −0.58 EU/question regression is the railed `p1`. **Its channel should be
  checked against §3 before its grid is touched**: the same diagnostic applies,
  and a finer grid against a one-sided stream would make its regression worse,
  not better. Its `membrane/` package is also the only working `said@1` consumer
  in the tree.
- **`credence-governor/docs/governance-roadmap.md` Phase 2** — "the membrane
  shadows live traffic on `latent@1`" — is dead; `latent@1` has zero `src/`
  occurrences. Needs a dated supersession banner.
- **`membrane.py` targets two retired utility forms.** A `said@1` adapter is a
  rewrite, not an edit.
- **proplang is at its roadmap terminus** for wire scope — the wire docket's
  three scheduled items all stand closed. Filing further issues is not a plan.
