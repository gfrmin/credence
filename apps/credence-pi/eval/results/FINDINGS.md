# credence-pi: measured results on real OpenClaw sessions

**One line (honest, and it's a negative):** evaluated correctly on real
frontier-model OpenClaw sessions, the *current* credence-pi brain provides **no
measurable governance** — it proceeds essentially every tool call. Its features
(`tool`, `parent`, `repetition-bucket`) capture *tool-level* repetition, but real
waste is *argument-level* repetition, which those features cannot see. The "prove
it's better" gate **fails for the current design**; launch stays blocked until a
real capability fix (an argument-repetition feature) is added and re-proven.

Corpus + method: `MANIFEST.md`, `../README.md`. Numbers: `clawsbench_openclaw.summary.json`.

## The result

Held-out test split (ClawsBench openclaw harness): 1,015 sessions, 11,306 tool
calls. Objective loops (exact repeated `(tool, command)` within a session, counted
only where the command is genuinely distinguishable — see below): **76 (0.7%)**.

| governor | proceed | ask | block | catches loops? |
|----------|--------:|----:|------:|----------------|
| **WARM** (trained, frozen) | 11,306 | 0 | **0** | no — recall 0 |
| COLD (fresh prior) | 0 | 11,306 | 0 | n/a (asks all) |
| static rule "block all rep≥2" | — | — | 8,200 (72.5%) | only by blocking everything (precision 0.009) |

The warm brain **blocks nothing**. The reason is structural, not a tuning issue:
the cell `(exec, parent, rep-3plus)` aggregates loop *and* non-loop exec calls
(the brain never sees the command), so its approval stays above the 0.5 block
threshold and it never fires. No cost setting changes this — the feature
granularity cannot isolate loops.

## Why the earlier numbers looked good (the artifact)

An earlier pass reported "precision 0.894, blocks 3.3%." That was an **artifact**:
ClawsBench collapses every `read` into `command="read"` (the path is stripped), so
a legitimate sequence of reads of *different* files (`read a.md`, `read b.md`, …)
is indistinguishable from re-reading the *same* file. Counting all repeated reads
as "loops" both inflated the waste rate and let the brain's `rep` feature
"predict" them. Once loops are counted only where the command is genuinely
distinguishable (present and ≠ the tool name), the read pseudo-loops vanish, real
loops drop to 0.7%, and the brain's apparent skill disappears.

Two further bugs were found and fixed en route (recorded for honesty): an
unrealistic `--call-cost 0.01` that made an unseen context break the proceed/block
tie toward *block* (the daemon default \$0.50 correctly *asks*); and a
cross-session loop label that saturated and marked 39% of calls loops (correct
per-session rate ~0.7–4.5%). None of these rescue the result — corrected, the
brain governs nothing.

## Cross-corpus + safety (independent)

Trained on ClawsBench, run over **ATBench-Claw** (500 OpenClaw trajectories with
independent `is_safe` labels): **0% block on both safe and unsafe** trajectories.
So credence-pi is, as designed, a process-waste governor and **not** a
content-safety classifier — and on real data it currently catches neither.

## What this means

- **Honest outcome:** the gate the author insisted on did its job. On real data,
  with correct methodology, the shipped feature set cannot demonstrate that
  credence-pi saves money or time. The demo "surgical win" was a hand-constructed
  scenario with explicit per-loop user denials; it does not reproduce on real
  sessions where the loop signal is argument-level and the features are not.
- **The real fix (well-motivated by this evidence):** add an **argument-repetition
  feature** — "does this call repeat the arguments of a recent call?" — so a loop
  becomes visible to `condition`/`expect`. Then re-evaluate on a corpus that
  preserves tool arguments (exec is faithful; read is not in ClawsBench).
- **What ships now:** nothing new to users. The warm brain rubber-stamps, so it is
  **not** shipped enforcing. The eval harness (this directory) is the durable
  product of the exercise — it is what found the problem and what will re-prove a
  fixed design.

## Reproduce

`../README.md` → the pipeline. Pinned corpus revision in `MANIFEST.md`. Seed 0,
train-frac 0.7, call-cost 0.50.
