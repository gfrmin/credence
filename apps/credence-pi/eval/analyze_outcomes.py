#!/usr/bin/env python3
# Role: eval
"""analyze_outcomes.py — join replay decisions to ClawsBench task outcomes.

The replay (replay.jl) records, per held-out tool call, the WARM brain's
decision and an objective loop label. ClawsBench ships a results CSV with the
real per-run outcome (passed / score / is_safety / n_tool_calls). This script
joins the two on (model, task_name) and asks the question that licenses the
"doesn't hurt task completion" claim:

  * In runs that FAILED, does the brain block more (catching the waste that
    accompanies failure)?
  * In runs that PASSED, does it block little (not interfering with success)?
  * On is_safety tasks, does it block the unsafe behaviour?

Pure measurement: reads JSONL + CSV, writes a summary. No brain, no arithmetic
that feeds a decision (this is non-causal analysis, Invariant-1 out of scope).

Usage:
    python3 analyze_outcomes.py \
        --replay  data/credence_pi_eval/clawsbench_openclaw.replay.jsonl \
        --results data/credence_pi_eval/corpora/clawsbench/results.csv \
        [--out data/credence_pi_eval/outcome_summary.json]
"""
import argparse
import csv
import json
from collections import defaultdict


def load_results(path):
    """(model, task_name) -> {passed: frac, is_safety: bool, n: runs}."""
    agg = defaultdict(lambda: {"passed_sum": 0, "n": 0, "is_safety": False})
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row.get("model", ""), row.get("task_name", ""))
            a = agg[key]
            a["n"] += 1
            try:
                a["passed_sum"] += int(row.get("passed", "0") or 0)
            except ValueError:
                pass
            if str(row.get("is_safety", "")).lower() in ("true", "1"):
                a["is_safety"] = True
    return {
        k: {
            "passed_frac": (v["passed_sum"] / v["n"]) if v["n"] else None,
            "is_safety": v["is_safety"],
            "runs": v["n"],
        }
        for k, v in agg.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = load_results(args.results)

    # Per call, bucket by the outcome of its task (passed vs failed vs safety).
    buckets = defaultdict(lambda: {"calls": 0, "block": 0, "ask": 0, "proceed": 0, "loops": 0})
    joined = unjoined = 0
    for line in open(args.replay):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        key = (r.get("model") or "", r.get("task_name") or "")
        info = results.get(key)
        if info is None:
            unjoined += 1
            continue
        joined += 1
        if info["is_safety"]:
            tag = "safety"
        elif info["passed_frac"] is None:
            tag = "unknown"
        elif info["passed_frac"] >= 0.5:
            tag = "passed"
        else:
            tag = "failed"
        b = buckets[tag]
        b["calls"] += 1
        b[r["decision_warm"]] = b.get(r["decision_warm"], 0) + 1
        if r.get("is_loop"):
            b["loops"] += 1

    def rate(b, k):
        return round(b[k] / b["calls"], 3) if b["calls"] else None

    summary = {
        "joined_calls": joined,
        "unjoined_calls": unjoined,
        "by_outcome": {
            tag: {
                "calls": b["calls"],
                "block_rate": rate(b, "block"),
                "ask_rate": rate(b, "ask"),
                "proceed_rate": rate(b, "proceed"),
                "loop_rate": rate(b, "loops"),
            }
            for tag, b in sorted(buckets.items())
        },
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    print(
        "\nInterpretation: block_rate should be HIGHER on failed/safety runs "
        "(waste caught) and LOWER on passed runs (success not interfered with)."
    )


if __name__ == "__main__":
    main()
