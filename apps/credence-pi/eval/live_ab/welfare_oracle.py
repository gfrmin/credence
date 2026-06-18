#!/usr/bin/env python3
# Role: eval
# welfare_oracle.py <task_id> <work_dir>  ->  exit 0 if solved, 1 if not (prints SOLVED/FAILED).
#
# Deterministic, formatting-tolerant oracles for the MVP-D easy-task suite. Easy by design:
# the free qwen tier can usually pass them, so QUALITY is ~equal across models and the welfare
# A/B isolates the money<->time exchange. Script tasks (fizzbuzz/sum) execute the AGENT's file
# in a throwaway sandbox (python:3.13-slim, --network none, work mounted read-only) — never on
# the host; only the deterministic stdout is compared.
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

DOCKER_HOST = os.environ.get("DOCKER_HOST", "unix:///run/user/1000/podman/podman.sock")
SANDBOX_IMAGE = os.environ.get("ORACLE_SANDBOX_IMAGE", "docker.io/library/python:3.13-slim")

def norm(s: str) -> str:
    return s.strip().replace("\r\n", "\n")

def read(work: str, name: str):
    p = Path(work) / name
    return p.read_text() if p.exists() else None

def run_py_sandboxed(work: str, script: str, timeout: int = 30):
    env = dict(os.environ); env["DOCKER_HOST"] = DOCKER_HOST
    try:
        r = subprocess.run(
            ["podman", "run", "--rm", "--network", "none", "-v", f"{work}:/w:ro",
             "-w", "/w", SANDBOX_IMAGE, "python3", script],
            capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"

def check_hello(work):
    t = read(work, "hello.txt");  return t is not None and norm(t) == "Hello"

def check_config_json(work):
    t = read(work, "config.json")
    if t is None: return False
    try: d = json.loads(t)
    except Exception: return False
    return isinstance(d, dict) and d.get("name") == "credence" and d.get("version") == 1

def check_notes_md(work):
    t = read(work, "notes.md")
    if t is None: return False
    lines = [l.strip() for l in norm(t).split("\n") if l.strip()]
    return len(lines) >= 2 and lines[0] == "# Notes" and any(l == "- first item" for l in lines)

def check_grocery(work):
    t = read(work, "grocery.txt")
    if t is None: return False
    lines = [l.strip().lower() for l in norm(t).split("\n") if l.strip()]
    return lines == ["apples", "bananas", "cherries"]

def check_fizzbuzz(work):
    if not (Path(work) / "fizzbuzz.py").exists(): return False
    rc, out, _ = run_py_sandboxed(work, "fizzbuzz.py")
    if rc != 0: return False
    exp = []
    for n in range(1, 16):
        exp.append("FizzBuzz" if n % 15 == 0 else "Fizz" if n % 3 == 0 else "Buzz" if n % 5 == 0 else str(n))
    got = [l.strip() for l in norm(out).split("\n") if l.strip()]
    return got == exp

def check_sum100(work):
    if not (Path(work) / "sum.py").exists(): return False
    rc, out, _ = run_py_sandboxed(work, "sum.py")
    return rc == 0 and "5050" in norm(out).split()

ORACLES = {"hello": check_hello, "config_json": check_config_json, "notes_md": check_notes_md,
           "grocery": check_grocery, "fizzbuzz": check_fizzbuzz, "sum100": check_sum100}

def main():
    if len(sys.argv) != 3:
        print("usage: welfare_oracle.py <task_id> <work_dir>", file=sys.stderr); sys.exit(2)
    tid, work = sys.argv[1], sys.argv[2]
    fn = ORACLES.get(tid)
    if fn is None:
        print(f"unknown task {tid}", file=sys.stderr); sys.exit(2)
    try:
        ok = bool(fn(work))
    except Exception as e:
        print(f"FAILED (oracle error: {e})"); sys.exit(1)
    print("SOLVED" if ok else "FAILED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
