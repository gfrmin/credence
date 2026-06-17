#!/usr/bin/env bash
# Deterministic oracle for task e01_sum_bug. Exit 0 = success (tests pass).
# Usage: verify.sh <workspace_dir>
set -u
WS="${1:?usage: verify.sh <workspace_dir>}"
cd "$WS" || exit 2
python3 -m pytest -q test_calc.py >/dev/null 2>&1
