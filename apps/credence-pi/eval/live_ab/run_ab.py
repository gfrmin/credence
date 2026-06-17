#!/usr/bin/env python3
"""Live A/B runner for the credence-pi routing experiment.

For each (arm x task x rep): reset the live-ab workspace to the task seed, run a
real OpenClaw agent turn (fixed model, or routing if the arm has no model), grade
with the task's deterministic verify.sh, and harvest cost from the session log —
priced PER TURN by the model actually used (correct for the routing arm, where
the daemon picks the model and it is recorded in the session).

Arms:
  - fixed:  {"name": "always-haiku", "model": "anthropic/claude-haiku-4-5-20251001"}
  - routing:{"name": "routing", "model": None}  # daemon decides; needs daemon up + routing:true

Honest scope: success is the verify.sh exit code (deterministic). Cost uses the
real published Anthropic prices below on the real measured token counts. Local
ollama arms DO run the agentic loop (fix chain: params.num_ctx=32768 so the
native provider doesn't silently truncate to 4096; a non-coder instruct model
whose template structures tool_calls; agents.defaults.experimental.localModelLean
to fit the prompt under the reserve budget). Local marginal cost = $0 (own GPU);
the open question this runner answers is its *capability ceiling* vs haiku.

Usage:
    ANTHROPIC_API_KEY auto-loaded from the keyring.
    python3 run_ab.py --reps 1 --tasks e01_sum_bug,h01_... --arms haiku,sonnet,opus,routing
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

OPENCLAW = ["node", "/home/g/git/openclaw/openclaw.mjs"]
OPENCLAW_CWD = "/home/g/git/openclaw"
WS = Path("/tmp/credence-live-ab/ws")
ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"
SESSIONS = Path.home() / ".openclaw/agents/live-ab/sessions"

# Published prices, USD per 1M tokens (input, output, cacheRead, 5m-cacheWrite),
# verified 2026-06-17 against platform.claude.com/docs/.../pricing. cw is the
# 5-minute cache write (1.25x base input), the tier Claude Code's ephemeral_5m
# caching uses. NB: these are the CURRENT 4.x prices — Opus 4.8 is $5/$25, NOT
# the old Opus-4.1 $15/$75; Haiku 4.5 is $1/$5, NOT the old Haiku-3.5 $0.8/$4.
# These reproduce claude's own total_cost_usd exactly.
PRICES = {
    "local":  dict(i=0.0,  o=0.0,  cr=0.0,  cw=0.0),
    "haiku":  dict(i=1.0,  o=5.0,  cr=0.10, cw=1.25),
    "sonnet": dict(i=3.0,  o=15.0, cr=0.30, cw=3.75),
    "opus":   dict(i=5.0,  o=25.0, cr=0.50, cw=6.25),
}
# Named arms -> full model ref (None = routing arm: omit --model, let the daemon route).
# The fleet is a cost/capability ladder: local ($0, but can't do agentic tool-use) ->
# haiku -> sonnet -> opus. always-local is the naive "use my free GPU" trap baseline.
ARMS = {
    "local-9b": "ollama/qwen3.5:9b",
    "local-7b": "ollama/qwen2.5:7b-instruct",
    "local-3b": "ollama/qwen2.5:3b-instruct",
    "haiku":    "anthropic/claude-haiku-4-5-20251001",
    "sonnet":   "anthropic/claude-sonnet-4-6",
    "opus":     "anthropic/claude-opus-4-8",
    "routing":  None,
}


def tier_of(model: str | None) -> str | None:
    """Map a recorded model id to a price tier (local models are free)."""
    if not model:
        return None
    m = model.lower()
    if "haiku" in m:
        return "haiku"
    if "sonnet" in m:
        return "sonnet"
    if "opus" in m:
        return "opus"
    if any(k in m for k in ("qwen", "llama", "gemma", "ollama")):
        return "local"
    return None


def load_key() -> str:
    out = subprocess.run(
        ["secret-tool", "lookup", "service", "env", "key", "ANTHROPIC_API_KEY"],
        capture_output=True, text=True)
    return out.stdout.strip()


def reset_ws(task_dir: Path) -> None:
    """Clear prior task working files (keep scaffold .md + dirs), copy this task's seed."""
    WS.mkdir(parents=True, exist_ok=True)
    for f in WS.iterdir():
        if f.is_file() and f.suffix != ".md":
            f.unlink()
    for sub in ("__pycache__", ".pytest_cache"):
        shutil.rmtree(WS / sub, ignore_errors=True)
    for f in (task_dir / "seed").iterdir():
        if f.is_file():
            shutil.copy(f, WS / f.name)


def harvest_cost(session_id: str) -> dict:
    """Sum per-turn usage from the session log, pricing EACH turn by the model it used."""
    f = SESSIONS / f"{session_id}.jsonl"
    usd = 0.0
    tot = dict(turns=0, input=0, output=0, cacheRead=0, cacheWrite=0)
    models_used: dict[str, int] = {}
    if f.exists():
        for ln in f.read_text().splitlines():
            try:
                r = json.loads(ln)
            except Exception:
                continue
            msg = r.get("message") or {}
            u = msg.get("usage") or r.get("usage")
            if not isinstance(u, dict):
                continue
            model = msg.get("model") or r.get("model") or ""
            tier = tier_of(model)
            ti = u.get("input", 0) or 0
            to = u.get("output", 0) or 0
            cr = u.get("cacheRead", 0) or 0
            cw = u.get("cacheWrite", 0) or 0
            tot["turns"] += 1
            tot["input"] += ti; tot["output"] += to
            tot["cacheRead"] += cr; tot["cacheWrite"] += cw
            if tier:
                p = PRICES[tier]
                usd += (ti * p["i"] + to * p["o"] + cr * p["cr"] + cw * p["cw"]) / 1e6
                models_used[model] = models_used.get(model, 0) + 1
    tot["usd"] = round(usd, 5)
    tot["models_used"] = models_used
    return tot


def run_one(task_dir: Path, arm: str, rep: int, env: dict, timeout: int) -> dict:
    reset_ws(task_dir)
    prompt = (task_dir / "prompt.txt").read_text()
    sid = f"{task_dir.name}__{arm}__r{rep}"
    model = ARMS[arm]
    cmd = OPENCLAW + ["agent", "--agent", "live-ab", "--session-id", sid,
                      "--message", prompt, "--json", "--timeout", str(timeout - 20)]
    if model:
        cmd += ["--model", model]
    t0 = time.time()
    try:
        subprocess.run(cmd, cwd=OPENCLAW_CWD, env=env, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass
    dur = round(time.time() - t0, 1)
    vr = subprocess.run(["bash", str(task_dir / "verify.sh"), str(WS)])
    success = vr.returncode == 0
    cost = harvest_cost(sid)
    row = dict(task=task_dir.name, arm=arm, rep=rep, success=success,
               duration_s=dur, **cost)
    print(f"  [{arm:>8} r{rep}] {task_dir.name:<18} "
          f"{'PASS' if success else 'FAIL'}  ${cost['usd']:<8} "
          f"{cost['turns']}t {dur}s  {list(cost['models_used'])}")
    return row


def summarize(rows: list[dict]) -> dict:
    arms = sorted({r["arm"] for r in rows})
    summ = {}
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        n = len(rs)
        passes = sum(1 for r in rs if r["success"])
        total_usd = sum(r["usd"] for r in rs)
        cost_per_success = (total_usd / passes) if passes else None
        summ[a] = dict(
            n=n, success_rate=round(passes / n, 3) if n else None,
            total_usd=round(total_usd, 4),
            mean_usd=round(total_usd / n, 5) if n else None,
            cost_per_success=round(cost_per_success, 5) if cost_per_success else None,
        )
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="", help="comma list of task dir names; default=all")
    ap.add_argument("--arms", default="haiku,sonnet,opus", help="comma list from: " + ",".join(ARMS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--out", default=str(RESULTS_DIR / "live_ab.jsonl"))
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # stream per-task lines when redirected

    env = dict(os.environ)
    key = load_key()
    if not key:
        sys.exit("ANTHROPIC_API_KEY not found in keyring")
    env["ANTHROPIC_API_KEY"] = key

    task_names = [t for t in args.tasks.split(",") if t] or \
        sorted(p.name for p in TASKS_DIR.iterdir() if (p / "verify.sh").exists())
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            sys.exit(f"unknown arm {a}; valid: {list(ARMS)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    print(f"tasks={task_names} arms={arms} reps={args.reps}")
    for task in task_names:
        td = TASKS_DIR / task
        for arm in arms:
            for rep in range(1, args.reps + 1):
                rows.append(run_one(td, arm, rep, env, args.timeout))

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summ = summarize(rows)
    print("\n=== SUMMARY (per arm) ===")
    print(json.dumps(summ, indent=2))
    print(f"\nrows -> {args.out}")


if __name__ == "__main__":
    main()
