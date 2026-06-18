#!/usr/bin/env bash
# Role: eval
# oc_tb_spotcheck.sh — run OpenClaw LIVE on a REAL Terminal-Bench task, in the task's OWN
# container (its Dockerfile env), graded by the task's OWN tests. Grounds the welfare/routing
# result (assembled from the full TB matrix) in the actual product on real TB tasks — the
# "spot-check + assemble" MVP path. No plugin/daemon here: the spot-check proves OpenClaw RUNS
# real TB tasks (the de-risk already proved plugin+daemon route+govern live); routing dominance
# is scored offline over the matrix.
#   TASK=fix-permissions MODEL=anthropic/claude-haiku-4-5-20251001 bash oc_tb_spotcheck.sh
set -euo pipefail
DS="$HOME/.cache/terminal-bench/terminal-bench-core/0.1.1"
OC=/home/g/git/openclaw
NODE_VER=v26.2.0
TASK="${TASK:?set TASK}"
MODEL="${MODEL:-anthropic/claude-haiku-4-5-20251001}"
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/1000/podman/podman.sock}"
TDIR="$DS/$TASK"; [ -d "$TDIR" ] || { echo "no task dir: $TDIR"; exit 2; }
INSTR=$(python3 -c "import yaml; print(yaml.safe_load(open('$TDIR/task.yaml'))['instruction'])")
IMG="tb-spot-${TASK}:latest"

echo "[build] $TASK image (its Dockerfile env)"
podman build -t "$IMG" "$TDIR" >/tmp/spot-build-$TASK.log 2>&1 || { echo "BUILD_FAIL"; tail -8 /tmp/spot-build-$TASK.log; exit 3; }

DOTOC=$(mktemp -d); LOGS=$(mktemp -d); trap 'rm -rf "$DOTOC"' EXIT
# Minimal config: register the one model, workspace=/app (the task's working dir). No plugin.
python3 - "$DOTOC" "$MODEL" <<'PY'
import json, sys
dotoc, model = sys.argv[1], sys.argv[2]
mid = model.split('/', 1)[1]
c = {"models": {"mode": "merge", "providers": {"anthropic": {"models": [
        {"id": mid, "name": mid, "reasoning": False, "input": ["text"],
         "cost": {"input": 1, "output": 5, "cacheRead": 0.1, "cacheWrite": 1.25},
         "contextWindow": 200000, "maxTokens": 8192,
         "compat": {"supportsTools": True, "supportsUsageInStreaming": True}}]}}},
     "agents": {"defaults": {"model": {"primary": model}, "workspace": "/app"},
                "list": [{"id": "t", "name": "t", "workspace": "/app"}]},
     "plugins": {}}
json.dump(c, open(dotoc + "/openclaw.json", "w"))
PY

echo "[run+grade] OpenClaw ($MODEL) on real TB task '$TASK' in its container"
ANTHROPIC_API_KEY=$(secret-tool lookup service env key ANTHROPIC_API_KEY) \
podman run --rm -v "$OC":/oc:ro -v "$DOTOC":/root/.openclaw -v "$TDIR/tests":/tests:ro -v "$LOGS":/logs \
  -e ANTHROPIC_API_KEY -e INSTR="$INSTR" -e MODEL="$MODEL" -e NODE_VER="$NODE_VER" \
  -w /app "$IMG" bash -c '
    set -e
    if ! command -v node >/dev/null 2>&1; then
      apt-get update -qq && apt-get install -y -qq curl xz-utils libatomic1 >/dev/null 2>&1
      curl -fsSL "https://nodejs.org/dist/${NODE_VER}/node-${NODE_VER}-linux-x64.tar.xz" -o /tmp/n.tar.xz
      mkdir -p /opt/node && tar -xJf /tmp/n.tar.xz -C /opt/node --strip-components=1
    fi
    export PATH=/opt/node/bin:$PATH
    echo "--- agent ---"
    node /oc/openclaw.mjs agent --agent t --session-id spot --local --message "$INSTR" \
      --model "$MODEL" --json --timeout 300 >/logs/result.json 2>/logs/err.log || echo "(agent exit $?)"
    echo "--- grade (task tests) ---"
    (apt-get update -qq && apt-get install -y -qq python3-pip >/dev/null 2>&1) || true
    python3 -m pip install --quiet --break-system-packages pytest >/dev/null 2>&1 \
      || pip3 install --quiet pytest >/dev/null 2>&1 || true
    cd /app && python3 -m pytest /tests/test_outputs.py -rA 2>&1 | tail -12
  '
echo "[meta]"
python3 -c "
import json
d=json.load(open('$LOGS/result.json')); m=d.get('meta',{}); a=m.get('agentMeta',{})
print('  model=',a.get('model'),'dur_ms=',m.get('durationMs'),'usage=',a.get('usage'))" 2>/dev/null || echo "  (no result.json — see $LOGS/err.log)"
