#!/usr/bin/env bash
# OpenClaw-in-Terminal-Bench-container smoke — PROVEN WORKING 2026-06-17.
#
# Proves the hard plumbing for the (c) OpenClaw grounding demo: OpenClaw runs INSIDE a
# Terminal-Bench task container and completes a real agentic task. Result: created
# /work/hello.txt = "Hello" via anthropic/claude-haiku-4-5 (transport=embedded).
#
# What this de-risks (the scary unknowns, all solved):
#   - The 2.3GB node_modules need NOT be copied: bind-mount the host openclaw repo (ro).
#   - node: the host binary is dynamically linked (libuv/libatomic) and won't run mounted;
#     install a MATCHING node (v26 — same ABI as the mounted node_modules) from the
#     official tarball into /opt (+ apt libatomic1, xz-utils, curl).
#   - config: OpenClaw needs an explicit models.providers["anthropic"] with the model id
#     REGISTERED (not built-in); agents.list defines the agent + workspace=/work.
#
# What this does NOT yet do (the full-build remainder, pending the go/no-go decision):
#   - Route through credence: needs the credence gateway (credence-router serve, :8377) OR
#     the credence-pi daemon (:8787) running on the host + reachable from the container
#     (--network host or host.containers.internal), with OpenClaw's `credence` provider.
#   - Escalation orchestration: run with cheap model → run tb tests → on fail re-run with
#     the escalated model (RoutingBrain.escalation_next gates each step) — the task-level
#     loop the per-call gateway can't do alone.
#
# Usage: bash oc_container_smoke.sh   (ANTHROPIC_API_KEY auto-loaded from keyring)
set -euo pipefail

OC=/home/g/git/openclaw
NODE_VER=v26.2.0   # match the host node that built openclaw/node_modules
WORK=$(mktemp -d)
DOTOC=$(mktemp -d)
export DOCKER_HOST=unix:///run/user/1000/podman/podman.sock

# Minimal in-container config: anthropic provider (haiku registered) + one 'smoke' agent.
python3 - "$DOTOC" <<'PY'
import json, sys
c = json.load(open('/home/g/.openclaw/openclaw.json'))
anthropic = c['models']['providers']['anthropic']
c['agents']['defaults']['model'] = {'primary': 'anthropic/claude-haiku-4-5-20251001'}
c['agents']['defaults']['workspace'] = '/work'
c['agents']['defaults'].pop('experimental', None)
c['agents']['defaults']['models'] = {'anthropic/claude-haiku-4-5-20251001': {}}
c['agents']['list'] = [{'id': 'smoke', 'name': 'smoke', 'workspace': '/work'}]
c['plugins'] = {}
c['models'] = {'mode': 'merge', 'providers': {'anthropic': anthropic}}
json.dump(c, open(sys.argv[1] + '/openclaw.json', 'w'), indent=2)
PY

ANTHROPIC_API_KEY=$(secret-tool lookup service env key ANTHROPIC_API_KEY) \
podman run --rm \
  -v "$OC":/oc:ro -v "$DOTOC":/root/.openclaw -v "$WORK":/work \
  -e ANTHROPIC_API_KEY -w /work \
  ghcr.io/laude-institute/t-bench/python-3-13:latest \
  bash -c "
    apt-get update -qq && apt-get install -y -qq curl xz-utils libatomic1 >/dev/null 2>&1
    curl -fsSL https://nodejs.org/dist/${NODE_VER}/node-${NODE_VER}-linux-x64.tar.xz -o /tmp/node.tar.xz
    mkdir -p /opt/node && tar -xJf /tmp/node.tar.xz -C /opt/node --strip-components=1
    export PATH=/opt/node/bin:\$PATH
    node /oc/openclaw.mjs agent --agent smoke --session-id s1 \
      --message 'Create a file named hello.txt in the current directory containing exactly: Hello' \
      --json --timeout 90 --model anthropic/claude-haiku-4-5-20251001 2>&1 | tail -6
  "
echo "RESULT: $(cat "$WORK/hello.txt" 2>/dev/null || echo MISSING)"
rm -rf "$WORK" "$DOTOC"
