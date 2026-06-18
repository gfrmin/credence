#!/usr/bin/env python3
# Role: eval
# oc_welfare_matrix.py — build the LIVE welfare matrix for MVP-D: run every easy task on every
# model through REAL OpenClaw (oc_welfare_run.sh, plugin loaded, governance live via the daemon),
# capturing per (task, model): solved (oracle), money (usage x verified prices; 0 for the free
# qwen), time (wall-clock the user waits + model-only durationMs when present), and governance
# asks (the daemon-log slice for the run). oc_welfare_score.jl then scores each user profile's
# realized welfare over this matrix against the daemon's live routing decision and the fixed
# routers — the escalation_live.jl methodology, on real usage.
#
# Model choice during the matrix run is PINNED (--model) so each cell is that model's real
# behaviour; the routing DECISION (which model a profile gets) is the daemon's, queried by the
# scorer. Governance runs with a neutral profile here (asks reported as a secondary signal; the
# headline money<->time<->quality coordinates are model x task properties, profile-independent
# in execution).
#
# Usage:
#   python3 oc_welfare_matrix.py --reps 1 --out results/welfare_matrix.jsonl
#   python3 oc_welfare_matrix.py --tasks hello,fizzbuzz --models qwen,haiku
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DAEMON_LOG = os.environ.get("CREDENCE_PI_LOG_FILE", "/tmp/credence-pi-derisk.jsonl")

# tier -> {exec id for OpenClaw --model, belief id (warm routing/latency key), free?}
MODELS = {
    "qwen":   {"exec": "ollama/qwen2.5:7b-instruct",         "belief": "qwen2.5:7b-instruct",  "free": True},
    "haiku":  {"exec": "anthropic/claude-haiku-4-5-20251001", "belief": "claude-haiku-4-5",    "free": False},
    "sonnet": {"exec": "anthropic/claude-sonnet-4-6",         "belief": "claude-sonnet-4-6",   "free": False},
    "opus":   {"exec": "anthropic/claude-opus-4-8",           "belief": "claude-opus-4-8",     "free": False},
}
# Verified $/Mtok (input, output, cacheRead, cacheWrite) — tb_matrix.py, platform pricing 2026-06.
PRICES = {
    "haiku":  dict(i=1.0, o=5.0,  cr=0.10, cw=1.25),
    "sonnet": dict(i=3.0, o=15.0, cr=0.30, cw=3.75),
    "opus":   dict(i=5.0, o=25.0, cr=0.50, cw=6.25),
}

def cost_from_usage(tier: str, u: dict) -> float:
    if MODELS[tier]["free"] or not u:
        return 0.0
    p = PRICES[tier]
    return (u.get("input", 0) * p["i"] + u.get("output", 0) * p["o"]
            + u.get("cacheRead", 0) * p["cr"] + u.get("cacheWrite", 0) * p["cw"]) / 1e6

def parse_result(logs_dir: Path):
    """Pull model/duration/usage from OpenClaw's result.json (meta.agentMeta). Missing/partial
    (e.g. qwen compaction abort) -> (None, None, {}); the oracle + wall-clock still score it."""
    f = logs_dir / "result.json"
    if not f.exists() or f.stat().st_size == 0:
        return None, None, {}
    try:
        d = json.loads(f.read_text())
        am = (d.get("meta") or {}).get("agentMeta") or {}
        return am.get("model"), (d.get("meta") or {}).get("durationMs"), (am.get("usage") or {})
    except Exception:
        return None, None, {}

def count_governance(log_path: str, base_lines: int):
    """Decisions in the daemon-log slice produced by THIS (sequential) run."""
    asks = proceeds = blocks = 0
    try:
        with open(log_path) as fh:
            for i, ln in enumerate(fh):
                if i < base_lines or not ln.strip():
                    continue
                e = json.loads(ln).get("event", {})
                if e.get("event_type") == "decision":
                    a = e.get("action")
                    asks += a == "ask"; proceeds += a == "proceed"; blocks += a == "block"
    except FileNotFoundError:
        pass
    return asks, proceeds, blocks

def log_lines(log_path: str) -> int:
    try:
        with open(log_path) as fh:
            return sum(1 for _ in fh)
    except FileNotFoundError:
        return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="", help="comma task ids (default: all in welfare_tasks.json)")
    ap.add_argument("--models", default="qwen,haiku,sonnet")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--profile", default="balanced", help="governance profile for the matrix runs")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--out", default=str(RESULTS / "welfare_matrix.jsonl"))
    args = ap.parse_args()

    suite = json.loads((ROOT / "welfare_tasks.json").read_text())
    if args.tasks:
        want = set(args.tasks.split(","))
        suite = [t for t in suite if t["id"] in want]
    tiers = [m for m in args.models.split(",") if m]
    for t in tiers:
        if t not in MODELS:
            sys.exit(f"unknown model tier {t}; valid: {list(MODELS)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    run_sh = str(ROOT / "oc_welfare_run.sh")
    env = dict(os.environ)
    env.setdefault("DOCKER_HOST", "unix:///run/user/1000/podman/podman.sock")
    env["IMAGE"] = env.get("IMAGE", "credence-pi-welf:latest")

    for rep in range(1, args.reps + 1):
        for task in suite:
            for tier in tiers:
                sess = f"{task['id']}_{tier}_{rep}"
                work = f"/tmp/welf/{sess}/work"; logs = f"/tmp/welf/{sess}/logs"
                Path(work).mkdir(parents=True, exist_ok=True)
                Path(logs).mkdir(parents=True, exist_ok=True)
                base = log_lines(DAEMON_LOG)
                renv = dict(env, PROFILE=args.profile, MODEL=MODELS[tier]["exec"],
                            SESSION=sess, MSG=task["message"], WORK=work, LOGS=logs,
                            TIMEOUT=str(args.timeout), KEEP="1")
                t0 = time.monotonic()
                proc = subprocess.run(["bash", run_sh], env=renv,
                                      capture_output=True, text=True)
                wall_s = round(time.monotonic() - t0, 1)
                # Grade by ORACLE (robust to OpenClaw's noisy exit code).
                orc = subprocess.run([sys.executable, str(ROOT / "welfare_oracle.py"),
                                      task["id"], work], env=env, capture_output=True, text=True)
                solved = orc.returncode == 0
                model, dur_ms, usage = parse_result(Path(logs))
                cost = cost_from_usage(tier, usage)
                asks, proceeds, blocks = count_governance(DAEMON_LOG, base)
                row = dict(task=task["id"], tier=tier, rep=rep, solved=solved,
                           cost_usd=cost, duration_ms=dur_ms, wall_s=wall_s,
                           asks=asks, proceeds=proceeds, blocks=blocks,
                           model=model or MODELS[tier]["exec"], run_exit=proc.returncode)
                rows.append(row)
                print(f"  [{tier:>6}] {task['id']:<12} "
                      f"{'OK ' if solved else 'FAIL'} ${cost:.4f} "
                      f"wall={wall_s}s dur={dur_ms}ms asks={asks}", flush=True)

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    # Compact per-tier summary.
    print("\n=== matrix summary (per tier) ===")
    for t in tiers:
        rs = [r for r in rows if r["tier"] == t]
        n = len(rs); s = sum(r["solved"] for r in rs)
        c = sum(r["cost_usd"] for r in rs)
        w = sum(r["wall_s"] for r in rs) / n if n else 0
        print(f"  {t:>6}: {s}/{n} solved | ${c:.4f} total | {w:.1f}s mean wall")
    print(f"\nrows -> {args.out}")

if __name__ == "__main__":
    main()
