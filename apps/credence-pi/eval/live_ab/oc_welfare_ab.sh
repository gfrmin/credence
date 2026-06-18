#!/usr/bin/env bash
# Role: eval
# oc_welfare_ab.sh — the MVP-D live welfare A/B, end to end:
#   1. build the LIVE matrix (easy tasks × {qwen,haiku,sonnet}) through REAL OpenClaw + the
#      credence-pi plugin + the live daemon (oc_welfare_matrix.py);
#   2. derive the eval latency belief (incl. the free qwen tier) from the measured matrix
#      (welfare_latency_from_matrix.py);
#   3. score per-profile net WELFARE vs every fixed router, with the daemon's routing decision
#      (oc_welfare_score.jl) — the time⟷money + quality⟷cost divergences + dominance.
#
# Prereqs (the de-risked plumbing): daemon up on :8787, ollama up with qwen pulled, the baked
# runner image built (Dockerfile.welf → credence-pi-welf:latest), the plugin built (dist/).
# Spends real API budget on the paid arms (qwen is free); ~$0.5 at reps=1.
#
#   REPS=2 bash oc_welfare_ab.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPS="${REPS:-1}"; MODELS="${MODELS:-qwen,haiku,sonnet}"
MATRIX="$DIR/results/welfare_matrix.jsonl"
REPO="$(cd "$DIR/../../../.." && pwd)"

echo "== 1/3 build the live matrix (real OpenClaw × ${MODELS}, reps=${REPS}) =="
python3 "$DIR/oc_welfare_matrix.py" --models "$MODELS" --reps "$REPS" --out "$MATRIX"

echo "== 2/3 eval latency belief from the matrix (measured per-tier wall-time) =="
python3 "$DIR/welfare_latency_from_matrix.py" "$MATRIX"

echo "== 3/3 score per-profile welfare vs fixed routers =="
julia --project="$REPO" "$DIR/oc_welfare_score.jl" "$MATRIX" | tee "$DIR/results/welfare_score.txt"
