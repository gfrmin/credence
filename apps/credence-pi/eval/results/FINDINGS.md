# credence-pi: measured results on real OpenClaw sessions

**One line:** on 3,729 real frontier-model OpenClaw sessions, with the
argument-repetition feature the learned governor now identifies repeated-identical-
call waste at **precision 1.0 / recall 1.0** on held-out sessions — blocking 0.7%
of calls — versus a static "block all repeats" rule at 72.5% block / 0.9%
precision. It does **not** over-block off-distribution (0% on ATBench's distinct
reads) and carries **no** content-safety signal (it is a process-waste governor by
design). Read the honest scope at the bottom: 1.0/1.0 is against the exact-repeat
*definition* of waste; the causal "saves money without hurting completion" still
needs live enforcement.

Corpus + method: `MANIFEST.md`, `../README.md`. Numbers: `clawsbench_openclaw.summary.json`.

## The arc (this is a real finding, not a straight line)

1. **Original features `(tool, parent, repetition-bucket)` → negative.** Correctly
   measured, the brain blocked *nothing* (recall 0): those features capture
   *tool-level* repetition, but real waste is *argument-level* — the SAME call
   repeated — which a rep-bucket cannot isolate (the cell mixes a re-run of one
   command with distinct commands of the same tool). An earlier "0.894" was a
   corpus artifact (ClawsBench strips `read` paths, so legit sequential reads
   looked like loops). The "prove it" gate correctly failed the original design.
2. **Add the missing signal → it works.** A `recent-identical-call-count` feature —
   how many prior calls this run were the SAME `(tool, args)` — counted
   session-wide, only for informative args, with a collision-free content hash.
   The structure-BMA absorbed it (6 features, no brain-code change).

## The result (held-out test split: 1,015 sessions, 11,306 calls)

Objective loops (exact repeated informative `(tool, command)` within a session): 76.

| governor | proceed | block | precision | recall |
|----------|--------:|------:|----------:|-------:|
| **WARM** (learned, frozen) | 11,230 | **76** | **1.00** | **1.00** |
| COLD (fresh prior) | 0 | 0 (asks all) | — | — |
| static rule "block all rep≥2" | — | 8,200 (72.5%) | 0.009 | 0.987 |

The learned brain blocks **exactly** the 76 loops (tp 76, fp 0, fn 0) and nothing
else, generalising across held-out sessions. A static repeat-rule would block
nearly three-quarters of all calls to achieve comparable recall — wrecking the
agent. The learned governor is precise because it conditions on the full context
and learns the per-context approval rate via `condition`.

## Cross-corpus + safety (independent check)

Trained on ClawsBench, run over **ATBench-Claw** (500 OpenClaw trajectories, distinct
sequential reads, independent `is_safe` labels): **0% block on both safe and unsafe**.
So the refined feature does **not** over-block legitimate distinct reads
off-distribution, and credence-pi shows **no content-safety discrimination** — it is
a process-waste governor, not a safety classifier (honest scope; risk-aware features
would be separate work).

## Cost / scale

Prevented spend: 76 calls × the daemon default \$0.50 = \$38 over 1,015 test sessions
(**labelled estimate**; ClawsBench has no per-call tokens). The whole replay runs in
~20s thanks to the sparse structure-BMA backend (was >10 min dense) — see
`src/sparse_structure.jl` and `test/test_sparse_structure_equivalence.jl`.

## Honest scope (what 1.0/1.0 does and does not mean)

- **Does:** the end-to-end machinery (feature extraction → Bayesian learning →
  sparse brain → EU-max decision) reliably and precisely detects *re-execution of
  identical calls* and generalises to held-out real-model sessions. That is a real,
  useful capability.
- **Does not:** prove "saves money without hurting task completion." The metric is
  against the *exact-repeat definition* of waste. Some re-execution is legitimate
  (re-running after a fix), so whether to *block* every repeat is a policy question;
  real-world precision-against-true-waste and net task outcome need **live
  enforcement telemetry** (the deferred causal phase — shadow mode + opt-in users).
- The feature now closely matches the waste definition, so high precision/recall
  reflects that the *right feature makes the task learnable* — which is the point of
  the exercise — more than a hard discrimination problem.

## Reproduce

`../README.md` → the pipeline. Pinned corpus revision in `MANIFEST.md`. Seed 0,
train-frac 0.7, call-cost 0.50.
