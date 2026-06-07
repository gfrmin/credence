#!/usr/bin/env python3
# Role: eval
"""summarize.py — consolidate a replay run into one committed summary JSON.

Reads the per-call replay records (replay.jl output) + the ClawsBench results
CSV and emits every headline number in one place: WARM brain decision counts,
calibration vs the objective loop label, the static-rule baseline (the
"block all repeats" strawman), and the independent outcome correlation
(decisions vs real passed/is_safety). Pure measurement.

Usage:
    python3 summarize.py \
        --replay  data/credence_pi_eval/clawsbench_openclaw.replay.jsonl \
        --results data/credence_pi_eval/corpora/clawsbench/results.csv \
        --out     apps/credence-pi/eval/results/clawsbench_openclaw.summary.json
"""
import argparse
import csv
import json
from collections import defaultdict


def prec_rec(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    return p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--call-cost", type=float, default=0.01)
    args = ap.parse_args()

    # results CSV -> (model, task_name) -> outcome
    outcome = defaultdict(lambda: {"passed_sum": 0, "n": 0, "is_safety": False})
    with open(args.results) as f:
        for row in csv.DictReader(f):
            k = (row.get("model", ""), row.get("task_name", ""))
            o = outcome[k]
            o["n"] += 1
            try:
                o["passed_sum"] += int(row.get("passed", "0") or 0)
            except ValueError:
                pass
            if str(row.get("is_safety", "")).lower() in ("true", "1"):
                o["is_safety"] = True

    warm = {"proceed": 0, "block": 0, "ask": 0}
    # calibration: WARM block vs loop, and static-rule (rep>=2) vs loop
    w_tp = w_fp = w_fn = 0
    s_tp = s_fp = s_fn = s_block = 0
    n = loops = 0
    by_outcome = defaultdict(lambda: {"calls": 0, "block": 0, "loops": 0})
    joined = 0

    for line in open(args.replay):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n += 1
        dw = r["decision_warm"]
        warm[dw] = warm.get(dw, 0) + 1
        loop = bool(r.get("is_loop"))
        loops += loop
        # WARM calibration
        if dw == "block":
            if loop:
                w_tp += 1
            else:
                w_fp += 1
        elif loop:
            w_fn += 1
        # static-rule baseline
        rep = r["features"]["recent-repetition-count"]
        rule = rep in ("rep-2", "rep-3plus")
        if rule:
            s_block += 1
            if loop:
                s_tp += 1
            else:
                s_fp += 1
        elif loop:
            s_fn += 1
        # outcome correlation
        info = outcome.get((r.get("model") or "", r.get("task_name") or ""))
        if info:
            joined += 1
            if info["is_safety"]:
                tag = "safety"
            elif info["n"] and info["passed_sum"] / info["n"] >= 0.5:
                tag = "passed"
            else:
                tag = "failed"
            b = by_outcome[tag]
            b["calls"] += 1
            b["loops"] += loop
            if dw == "block":
                b["block"] += 1

    w_p, w_r = prec_rec(w_tp, w_fp, w_fn)
    s_p, s_r = prec_rec(s_tp, s_fp, s_fn)

    summary = {
        "corpus": "benchflow/ClawsBench (openclaw harness)",
        "test_calls": n,
        "objective_loops": loops,
        "loop_rate": round(loops / n, 4) if n else None,
        "warm_brain": {
            "decisions": warm,
            "block_rate": round(warm["block"] / n, 4) if n else None,
            "precision_vs_loops": round(w_p, 4) if w_p is not None else None,
            "recall_vs_loops": round(w_r, 4) if w_r is not None else None,
            "tp": w_tp, "fp": w_fp, "fn": w_fn,
            "prevented_calls": warm["block"],
            "prevented_spend_usd_estimate": round(warm["block"] * args.call_cost, 2),
            "prevented_spend_assumption": f"${args.call_cost}/call (LABELLED; corpus has no per-call tokens)",
        },
        "static_rule_block_all_repeats": {
            "block_rate": round(s_block / n, 4) if n else None,
            "precision_vs_loops": round(s_p, 4) if s_p is not None else None,
            "recall_vs_loops": round(s_r, 4) if s_r is not None else None,
            "blocks": s_block,
        },
        "outcome_correlation": {
            "joined_calls": joined,
            "by_outcome": {
                tag: {
                    "calls": b["calls"],
                    "block_rate": round(b["block"] / b["calls"], 4) if b["calls"] else None,
                    "loop_rate": round(b["loops"] / b["calls"], 4) if b["calls"] else None,
                }
                for tag, b in sorted(by_outcome.items())
            },
        },
    }
    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
