#!/usr/bin/env bash
# Deterministic oracle (Exercism change, MIT). Exit 0 = tests pass.
set -u
WS="${1:?usage: verify.sh <workspace_dir>}"
cd "$WS" || exit 2
python3 -m pytest -q change_test.py >/dev/null 2>&1
