#!/usr/bin/env bash
# Role: eval
# oc_welfare_run.sh — ONE real OpenClaw run, in a container, WITH the credence-pi plugin
# loaded and routing/governing THROUGH the live host daemon. The unit the welfare A/B
# (oc_welfare_ab.sh) loops over; also the MVP-D plumbing de-risk.
#
# What this proves that oc_container_smoke.sh did NOT (its stated remainder):
#   - the credence-pi PLUGIN loads in the one-shot agent path (`--local` forces plugin load
#     even with `--json`; without `--local`, `--json` suppresses plugins — OpenClaw
#     command-catalog policy), and
#   - before_model_resolve ROUTES the turn via the daemon (the daemon's route decision becomes
#     OpenClaw's modelOverride), governing + cost-logging over the wire.
#
# Reachability: --network host so the container shares the host net namespace — the daemon
# (127.0.0.1:8787) and ollama (127.0.0.1:11434, the free qwen tier) are reachable as-is.
#
# Roster (real prices — the plugin's roster.ts reads config cost FIRST, so a 0 would make a
# paid model look free): anthropic haiku/sonnet at the verified $/Mtok, ollama qwen at $0.
# Each model carries cost so buildRoster yields ≥2 priceable models ⇒ routing is active.
#
# Params (env):
#   PROFILE   credence-pi profile preset            (default balanced)
#   SESSION   unique session id (per-run log key)   (default oc_<pid>)
#   MSG       the task instruction                  (default: create hello.txt)
#   TIMEOUT   agent timeout seconds                 (default 240)
#   WORK      host workspace dir (mounted /work)    (default: fresh mktemp)
#   LOGS      host logs dir (mounted /logs)         (default: fresh mktemp)
#   IMAGE     container image                       (default the TB python image)
#   NODE_VER  node to install if absent in image    (default v26.2.0)
#   KEEP      1 ⇒ don't rm WORK/LOGS on exit        (default unset)
# Echoes a final line:  RUN <session> work=<WORK> logs=<LOGS> exit=<code>
set -euo pipefail

OC=/home/g/git/openclaw
PLUGIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../openclaw-plugin" && pwd)"
PROFILE="${PROFILE:-balanced}"
SESSION="${SESSION:-oc_$$}"
MSG="${MSG:-Create a file named hello.txt in the current directory containing exactly: Hello}"
TIMEOUT="${TIMEOUT:-240}"
MODEL="${MODEL:-}"                 # provider/id to pin (harness-driven routing); empty ⇒ config default
WORK="${WORK:-$(mktemp -d)}"
LOGS="${LOGS:-$(mktemp -d)}"
IMAGE="${IMAGE:-ghcr.io/laude-institute/t-bench/python-3-13:latest}"
NODE_VER="${NODE_VER:-v26.2.0}"
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/1000/podman/podman.sock}"

[ -f "$PLUGIN/dist/index.js" ] || { echo "plugin not built: $PLUGIN/dist (run npm install && npm run build)"; exit 2; }

DOTOC=$(mktemp -d)
trap '[ "${KEEP:-}" = 1 ] || rm -rf "$DOTOC"' EXIT

# openclaw.json: real-priced anthropic ladder + free qwen + the credence-pi plugin (this
# PROFILE) + an agent whose allow-list is all three (so buildRoster sees the full roster).
python3 - "$DOTOC/openclaw.json" "$PROFILE" <<'PY'
import json, sys
out, profile = sys.argv[1], sys.argv[2]
def m(i, n, ci, co, cr, cw, ctx=200000, mx=8192, params=None):
    d = {"id": i, "name": n, "reasoning": False, "input": ["text"],
         "cost": {"input": ci, "output": co, "cacheRead": cr, "cacheWrite": cw},
         "contextWindow": ctx, "maxTokens": mx,
         "compat": {"supportsTools": True, "supportsUsageInStreaming": True}}
    if params: d["params"] = params
    return d
cfg = {
  "models": {"mode": "merge", "providers": {
    "anthropic": {"models": [
        m("claude-haiku-4-5-20251001", "Claude Haiku 4.5", 1.0, 5.0, 0.10, 1.25),
        m("claude-sonnet-4-6",        "Claude Sonnet 4.6", 3.0, 15.0, 0.30, 3.75),
    ]},
    "ollama": {"baseUrl": "http://127.0.0.1:11434", "api": "ollama", "apiKey": "OLLAMA_API_KEY",
               "models": [m("qwen2.5:7b-instruct", "qwen2.5:7b-instruct", 0, 0, 0, 0,
                            ctx=32768, params={"num_ctx": 32768})]},
  }},
  "plugins": {"enabled": True, "load": {"paths": ["/credence-plugin"]},
              "entries": {"credence-pi": {"enabled": True, "config": {
                  "daemonUrl": "http://127.0.0.1:8787", "profile": profile,
                  "routing": True, "silent": False}}}},
  "agents": {"defaults": {"model": {"primary": "ollama/qwen2.5:7b-instruct"},
                          "workspace": "/work",
                          "models": {"anthropic/claude-haiku-4-5-20251001": {},
                                     "anthropic/claude-sonnet-4-6": {},
                                     "ollama/qwen2.5:7b-instruct": {}}},
             "list": [{"id": "welf", "name": "welf", "workspace": "/work"}]},
}
json.dump(cfg, open(out, "w"), indent=2)
PY

# In-container runner (quoted heredoc ⇒ NODE_VER/MSG/SESSION/TIMEOUT expand IN the container,
# from -e). node persists nowhere across --rm runs, so install per run unless the image has it.
RUNNER="$DOTOC/run.sh"
cat > "$RUNNER" <<'INNER'
set -e
if ! command -v node >/dev/null 2>&1 && [ ! -x /opt/node/bin/node ]; then
  apt-get update -qq && apt-get install -y -qq curl xz-utils libatomic1 >/dev/null 2>&1
  curl -fsSL "https://nodejs.org/dist/${NODE_VER}/node-${NODE_VER}-linux-x64.tar.xz" -o /tmp/node.tar.xz
  mkdir -p /opt/node && tar -xJf /tmp/node.tar.xz -C /opt/node --strip-components=1
fi
export PATH=/opt/node/bin:$PATH
cd /work
MODEL_FLAG=""
[ -n "${MODEL:-}" ] && MODEL_FLAG="--model $MODEL"
node /oc/openclaw.mjs agent --agent welf --session-id "$SESSION" --local $MODEL_FLAG \
  --message "$MSG" --json --timeout "$TIMEOUT" > /logs/result.json 2> /logs/stderr.log
INNER

set +e
ANTHROPIC_API_KEY=$(secret-tool lookup service env key ANTHROPIC_API_KEY) \
podman run --rm --network host \
  -v "$OC":/oc:ro -v "$PLUGIN":/credence-plugin:ro \
  -v "$DOTOC":/root/.openclaw -v "$WORK":/work -v "$LOGS":/logs \
  -e ANTHROPIC_API_KEY -e NODE_VER="$NODE_VER" -e SESSION="$SESSION" \
  -e MSG="$MSG" -e TIMEOUT="$TIMEOUT" -e MODEL="$MODEL" \
  -w /work "$IMAGE" bash /root/.openclaw/run.sh
code=$?
set -e

echo "RUN $SESSION work=$WORK logs=$LOGS exit=$code"
