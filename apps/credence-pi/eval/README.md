# credence-pi eval — offline replay harness

Turn real recorded agent sessions into the brain's feature event stream and
replay them through the **actual daemon brain** to measure, at zero spend,
whether governance catches waste without blocking useful calls.

Two stages, one source of truth each:

1. **`extract.ts`** (Node/tsx) — adapt a transcript to `NormalizedEvent` JSONL.
   Feature bucketing is delegated to the body's `FeatureTracker`
   (`../openclaw-plugin/src/features.ts`) so replay sees exactly what a live body
   emits. Adapters: `native` (OpenClaw `~/.openclaw/.../sessions/*.jsonl` +
   `qa/scenarios` fixtures) and `clawsbench` (HF `benchflow/ClawsBench`).
2. **`replay.jl`** (Julia) — drive the brain's wired env closures
   (`make-prior` / `decide-action` / `observe-response`, the same ones the daemon
   uses) over the event stream: a WARM arm (trained on a split, frozen, decides
   the held-out split) and a COLD arm (fresh prior). Records every test call.

Then **`summarize.py`** / **`analyze_outcomes.py`** consolidate the per-call
records into committed summary JSON (calibration, static-rule baseline, and the
independent outcome correlation against ClawsBench's real `passed`/`is_safety`).

## Pipeline (ClawsBench, the primary corpus)

```bash
# 0. fetch the corpus (gitignored; see results/MANIFEST.md for the pinned SHA)
mkdir -p data/credence_pi_eval/corpora/clawsbench
curl -sL -o data/credence_pi_eval/corpora/clawsbench/train.jsonl \
  https://huggingface.co/datasets/benchflow/ClawsBench/resolve/main/data/train-00000-of-00001.jsonl
curl -sL -o data/credence_pi_eval/corpora/clawsbench/results.csv \
  https://huggingface.co/datasets/benchflow/ClawsBench/resolve/main/results/01-pilot-40tasks_master.csv

# 1. extract real tool calls -> normalized events (reuses the body's FeatureTracker)
cd apps/credence-pi/eval && npm install
node --import tsx extract.ts --format clawsbench --harness openclaw \
  ../../../data/credence_pi_eval/corpora/clawsbench/train.jsonl \
  --out ../../../data/credence_pi_eval/clawsbench_openclaw.events.jsonl

# 2. replay through the brain (WARM vs COLD), record every test call (~10 min)
cd ../../.. && julia --project=. apps/credence-pi/eval/replay.jl \
  --events data/credence_pi_eval/clawsbench_openclaw.events.jsonl \
  --out    data/credence_pi_eval/clawsbench_openclaw.replay.jsonl \
  --train-frac 0.7 --seed 0 --call-cost 0.01

# 3. consolidate -> committed summary JSON
python3 apps/credence-pi/eval/summarize.py \
  --replay data/credence_pi_eval/clawsbench_openclaw.replay.jsonl \
  --results data/credence_pi_eval/corpora/clawsbench/results.csv \
  --out apps/credence-pi/eval/results/clawsbench_openclaw.summary.json
```

Native sessions / loop fixtures use `--format native` with one file per session.

## What this measures (and what it can't)

- **Can:** detection + calibration (precision/recall of would-block vs an
  objective loop label and vs real task outcomes), prevented-call count,
  ask-rate — all on real frontier-model sessions, free.
- **Can't:** the full causal "enforcing it improves net task outcome." That needs
  live enforcement (the community-telemetry phase), not a replay.

See `results/FINDINGS.md` for the measured numbers and `results/MANIFEST.md` for
corpus provenance + the pinned revision.
