# W1′ results — round 3, on the pinned corpus

Criteria were fixed and committed before the run: pre-statement `6c52352`,
Amendment 1 `05f1931`, Amendment 2 `138a4a8`. Rounds 1 and 2 were discarded and
retracted respectively; both are documented in the amendments and their artifacts
are quarantined under `contaminated/`. This file reports **round 3 only**.

Corpus pinned: `field_pinned.json`, n = **130,799**, sha256
`0833f9b9440fcecad548b81cb738cac7bf43edf275f7f727594484c36aab8337`.
Attack corpus: the 20 code-derived `red_team.py` cases.
Engine: `proplang-host` from master (binary and `Host.hs` both 2026-07-21 19:42),
driven over a PTY (issue #18 workaround).

---

## 1. Verdict

**Decision rule outcome 3 — PASS / projection verdict.** proplang's engine
discriminates on real governance features when those features are actually
declared to it. The blocker is the **adapter's namespace projection**, not the
engine.

This is a pass on the pre-stated bar, and the bar was made stricter in
Amendment 2 precisely because the round-2 bar was too weak. It is reported
below together with everything that qualifies it.

## 2. The numbers

| | `ctl` (6 waste features) | `wide` (+ 9 threat features) |
|---|---|---|
| draws | 20 | 20 |
| `S` per draw | `0.000000` × 20 | 0.0015 … 0.7019 (below) |
| p5 of draws | +0.000000 | **+0.001482** |
| null q95 (the bar) | +0.000000 | **+0.000421** |
| **bar test: p5 > q95** | n/a (assertion arm) | **TRUE — PASS** |
| null range | 0 | −0.001172 … **+0.044592** |
| median draw | 0.000000 | +0.086849 |
| rail check | 1 value, 0.497801155 | **passed**: 716 distinct, max 0.883151 |
| probes at the 0.9 rail | 0 / 1320 | **0 / 1320** |

`S_wide` across the 20 pinned draws, sorted:

```
+0.0015 +0.0030 +0.0034 +0.0048 +0.0086 +0.0120 +0.0205 +0.0216 +0.0294 +0.0695
+0.1042 +0.1688 +0.2171 +0.2619 +0.3456 +0.5804 +0.5938 +0.6227 +0.6509 +0.7019
```

**The control assertion held exactly.** `S_ctl = 0.000000` on every draw, with a
single `p1` value (0.497801155) across all 1,320 probes — because with all six
waste features held at the ambient operating point, attack and benign encode to a
byte-identical vector. This is a structural identity, not a sampling result, and
it is the round-3 statement of the projection finding: **through the current
wire, an attack context and a benign context are literally the same point.**

## 3. What the result actually rests on

**Not the magnitude.** The effect spans nearly three orders of magnitude across
draws (0.0015 to 0.702, median 0.087), and **9 of the 20 draws fall below the
null's own maximum** of 0.044592. The pre-stated bar clears by a factor of 3.5,
which is a pass and not a comfortable one.

**The direction.** All **20 of 20** draws are positive. The permutation null is
not symmetric — 39 of 60 null replications are positive, an empirical rate of
0.65 — so the honest sign-test figure uses that rate rather than 0.5:

    P(20/20 positive | empirical null) = 0.65^20 ≈ 1.8e-4

(A naive symmetric assumption would give 9.5e-7; that number is not used.)

So: the engine's response to declared threat features is **reliably in the right
direction and unreliable in size**. Any downstream use should lean on the sign,
not on a point estimate — and no single-draw point estimate from this experiment
should be quoted, which is exactly the error Amendment 2 retracted.

**Why the magnitude is so variable:** 6 held-out attack probes per draw, a median
over 6, and only 20 authored attack cases in existence. The sampling variance is
a property of the corpus, not of the engine — which is deterministic to the bit
(5 consecutive identical runs).

## 4. The ceiling did not bind here

Across all 2,640 probes in both arms, **zero** sat at `thetaPoints`' 0.9 rung;
`wide` reached 0.883151 at most and produced 716 distinct `p1` values. So this
run is not a ceiling artifact, and the rail check that W3′ made mandatory passed
cleanly.

That is a statement about the **100-tick regime only**. The live shadow railed at
0.9 after ~95k ticks with two distinct `p1` values in 190,102 decisions. Whether
discrimination survives toward saturation is addressed in §5 and is the weakest
part of this result.

## 5. Saturation — post-hoc, outside the gate

Pinned corpus, **slot order shuffled** so `t`-parity carries no label information,
3 independent draws per tick count:

| N ticks | `S` per draw | median | max `p1` | distinct `p1` |
|---|---|---|---|---|
| 100 | +0.6227, +0.1687, +0.0686 | +0.168669 | 0.745590 | 44 |
| 300 | +0.6748, +0.0065, +0.0454 | +0.045444 | 0.775502 | 42 |
| 1000 | +0.6982, +0.0000, +0.0455 | +0.045455 | 0.798160 | **12** |

**Round 2's "S collapses to exactly 0 by 1000 ticks" is NOT reproduced** and
stays retracted. It was an artifact of the unpinned corpus, the `t`-parity
confound, or both. With those fixed, the effect survives to 1000 ticks in 2 of 3
draws — one draw at 0.698 that actually *grows* with evidence (0.623 → 0.675 →
0.698), one that decays to exactly 0, one flat near 0.045.

What the data do show, and what should not be waved away:

1. **The median decays and then flattens**: 0.169 → 0.045 → 0.045, roughly a
   3.7x drop between 100 and 300 ticks, stable thereafter.
2. **Sign holds**: 8 of 9 draws positive, 1 exactly zero, **none negative**.
3. **Concentration is visibly setting in.** Distinct `p1` values fall
   44 → 42 → **12**, and max `p1` climbs monotonically 0.746 → 0.776 → 0.798,
   toward `thetaPoints`' 0.9 rung.

Item 3 is the trend line that points at the shadow. The live shadow, after ~95k
ticks, showed exactly **2** distinct `p1` values with 98.7% of decisions at 0.9.
This sweep at 1000 ticks is at 12 distinct values and 0.798 — the same direction,
not yet arrived. **The ~95k regime remains unmeasured here**, and 3 draws per
tick count is thin against the heterogeneity §3 documents. Extrapolation is
suggestive, not evidence.

**Consequence for the verdict.** The §1 PASS is a statement about the regime this
experiment can reach. It is NOT a claim that discrimination persists to
deployment scale, and the concentration trend is a live reason to doubt that it
does. Any Phase-1 shadow must re-measure `S` **at deployment tick counts**, not
inherit this number.

## 6. Validity checks

- **Harness positive control**: proplang's own W1 synthetic stream through this
  PTY harness reproduces their frozen anchor exactly — `S = 0.794119`,
  `p1_attack = 0.897059` against their pinned 0.794119 / 0.897. A null from this
  harness is a real null.
- **Determinism**: 5 consecutive runs of one configuration, bit-identical.
- **Probe idempotence**: a menu-only tick does not condition, so sequential
  probing equals proplang's saved-state method.
- **Corpus pinning**: loader asserts n = 130,799 and never reads the live log.

## 7. What this does not establish

- **Not Phase-1 readiness.** The fragment route (proplang #19) is untouched by
  this; the 0.96 threshold remains unreachable over the wire, which is the other
  half of the Phase-0 exit. A PASS here does not change that.
- **Not a licence to widen the shipped namespace.** Arm `wide` is an experimental
  handshake. Declaring nine threat features onto the live wire is an epoch-1
  scope decision (HOSTS_PLAN 2.1) and the author's.
- **Not a claim about long-run behaviour** beyond §5.
- **Not a large-sample result.** Twenty authored attack cases is the entire
  attack corpus in existence; everything here inherits that width.
