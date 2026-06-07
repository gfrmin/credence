# credence-pi: measured results on real OpenClaw sessions

**One line:** on 3,729 real frontier-model OpenClaw sessions, the learned
feature-conditioned governor auto-blocks **3.3%** of tool calls at **89%
precision** against repeated-call waste — a static "block all repeats" rule
blocks **72.5%** at **2.5%** precision — and its blocks concentrate on
safety/failed runs, not successful ones.

Corpus + method: `MANIFEST.md` and `../README.md`. Numbers: `clawsbench_openclaw.summary.json`.

## The headline contrast

Held-out test split: 1,015 sessions, 11,306 tool calls; objective loops
(exact repeated `(tool, command)`) = 504 (4.5%).

| governor | block rate | precision (blocked∧loop / blocked) | recall (blocked∧loop / loops) |
|----------|-----------:|-----------------------------------:|------------------------------:|
| **Learned WARM brain** (trained, frozen) | **3.3%** (369) | **0.894** | 0.655 |
| Static rule "block all rep ≥ 2" | 72.5% (8,200) | 0.025 | 0.407 |

The learned Bayesian governor is **~36× more precise** while blocking **22×
fewer calls**, and still recovers more loops. The static rule — the obvious thing
a rules engine does — would block nearly three-quarters of all tool calls,
wrecking the agent. The brain blocks surgically because it conditions on the full
context (tool × parent × repetition) via structure-BMA, not on repetition alone.

## Independent corroboration (not the training signal)

Joining each decision to ClawsBench's real per-run outcome (`passed`/`is_safety`,
which the brain never sees):

| run outcome | calls | block rate | loop rate |
|-------------|------:|-----------:|----------:|
| **safety** (unsafe-behaviour tasks) | 636 | **9.1%** | 11.9% |
| failed | 471 | 2.3% | 4.2% |
| passed | 128 | **1.6%** | 3.1% |

The governor blocks **most on safety tasks, least on passed tasks** — it
interferes least with success and most where waste/unsafe behaviour actually
occurred. This ordering comes from labels independent of the loop signal it was
trained on.

## Prevented spend

369 calls auto-blocked on the test split. ClawsBench carries no per-call tokens,
so dollars are a **labelled estimate**: at an assumed \$0.01/call, \$3.69 over
1,015 test sessions (~\$0.0036/session). The defensible result is the
**call-count and precision**, not the dollar figure — a measured-dollars number
needs live enforcement with real token accounting (Phase 4 telemetry).

## What this is and isn't

- **Is:** a measured detection/calibration result on real, recorded,
  frontier-model OpenClaw sessions, at zero spend — the learned governor is far
  more precise than a static rule and concentrates its blocks where outcomes were
  bad.
- **Isn't:** proof that enforcing it *improves net task completion*. That is
  causal and needs live enforcement (opt-in users running the warm brain in
  enforcing mode), not a replay. The replay assumes the trajectory is unchanged
  when a call is removed — valid for the loop-waste class measured here.

## Reproduce

`../README.md` → the four-command pipeline. Pinned corpus revision in
`MANIFEST.md`. Seed 0, train-frac 0.7.
