# Eval corpus provenance (SHA-pinned)

Raw corpora and per-call replay records live under `data/credence_pi_eval/`,
which is **gitignored** (`/data/` in the repo `.gitignore` — eval artefacts are
outputs, not inputs). This manifest pins the inputs so every committed number is
reproducible from the recipe in `../README.md`.

## ClawsBench (primary corpus)

| field | value |
|-------|-------|
| HF dataset | `benchflow/ClawsBench` |
| pinned revision (sha) | `e7c45cc9ff486502176267c1294ac5809cf0700a` |
| lastModified | 2026-04-08 |
| license | CC-BY-NC-SA-4.0 (non-commercial — research/eval use only) |
| file (traces) | `data/train-00000-of-00001.jsonl` (7,834 records, 142 MB) |
| file (outcomes) | `results/01-pilot-40tasks_master.csv` (2,731 rows) |
| harness filter | `openclaw` → 3,729 records |
| extracted tool calls | 39,314 (`exec` 33,816 / `read` 2,946 / `other` 2,245 / `process` 163 / `write` 126 / `edit` 18) |
| models present (via openclaw harness) | anthropic claude-opus-4-6 / sonnet-4-6, openai gpt-5.4, gemini-3.1-{pro,flash}, zai glm-5 |

Re-pull at the pinned revision:
```bash
curl -sL -o train.jsonl \
  "https://huggingface.co/datasets/benchflow/ClawsBench/resolve/e7c45cc9ff486502176267c1294ac5809cf0700a/data/train-00000-of-00001.jsonl"
```

## Replay run parameters (for the committed summary)

| param | value |
|-------|-------|
| `--train-frac` | 0.7 |
| `--seed` | 0 |
| `--call-cost` | 0.50 (the daemon's default fallback; labelled assumption — corpus has no per-call tokens. NB: c ≤ ~$0.2 spuriously blanket-blocks via the θ=0.5 tie — see FINDINGS "never block on no evidence") |
| sessions (with ≥1 tool call) | 3,384 (train 2,369 / test 1,015) |
| train observations | 28,008 |
| test calls | 11,306 |

## Known limitations (honest scope)

- **Feature subset.** ClawsBench carries no timestamps or workspace root, so the
  brain conditions on the 3 available features (tool-name, parent, repetition);
  `features_available` is logged per event.
- **Waste label = training signal.** The objective loop label (exact repeated
  `(tool, command)` in a session, counted only where the command is genuinely
  distinguishable — present and ≠ the tool name) is both the training feedback
  proxy and the calibration target. The independent checks are the outcome
  correlation (`passed`/`is_safety`) and the ATBench-Claw cross-corpus run.
- **Read-arg collapse (load-bearing).** ClawsBench stores every `read` as
  `command="read"` (path stripped), so legitimate sequential reads of *different*
  files are indistinguishable from re-reading one file. Counting those as loops
  inflated an earlier result to "0.894 precision"; excluding collapsed-arg repeats
  drops real loops to 0.7% and the brain's apparent skill to zero. **Corrected
  result is NEGATIVE — see FINDINGS.md.** ClawsBench can faithfully measure only
  `exec` waste.
- **Prevented-spend is a labelled estimate**, not measured dollars (no per-call
  tokens). The primary results are the call-count, precision/recall, and outcome
  correlation — not the dollar figure.
- **Replay ≠ enforcement.** Counterfactual decisions on recorded traces; the
  causal "improves net task outcome" claim needs live enforcement (Phase 4).
