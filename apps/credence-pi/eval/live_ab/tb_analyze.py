#!/usr/bin/env python3
"""Descriptive analysis of the Terminal-Bench capability×cost matrix.

DISPLAY / DIAGNOSTICS ONLY. This summarises the measured matrix and computes the
realized cost of the *fixed* baseline policies and a per-task oracle — research
baselines for empirical contrast (precedent: baseline-comparison). It does NOT
implement the agent's decision mechanism: the LEARNED Bayesian router (the
credence-pi claim) lives in the Julia harness (tb_dominance.jl), where every
belief update is `condition` and the routing argmax is the single `optimise`.
Nothing here feeds a live agent decision; it is the descriptive backbone of the
report.

Shows, per tier and per difficulty cell:
  - resolve rate (solved / n)
  - mean cost per run, cost per solve
And across tasks measured on all tiers:
  - fixed-policy totals: always-haiku / always-sonnet / always-opus
  - per-task oracle (cheapest tier that solves) — the unreachable upper bound the
    learned router approximates from features alone.

Usage: python3 tb_analyze.py results/tb_matrix_pilot.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

TIERS = ["haiku", "sonnet", "opus"]
TIER_RANK = {"haiku": 0, "sonnet": 1, "opus": 2}  # cheap -> dear


def load(path: str) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    # keep only complete rows with a cost (drop install-fail/no-result rows but report them)
    return rows


def by_task(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """task -> tier -> row."""
    d: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        d[r["task"]][r["tier"]] = r
    return d


def cell_table(rows: list[dict]) -> None:
    print("\n=== resolve rate & cost per tier × difficulty ===")
    print(f"{'tier':>7} {'difficulty':>10} {'n':>3} {'solved':>6} "
          f"{'rate':>5} {'$mean':>8} {'$/solve':>8} {'turns̄':>6}")
    cells = defaultdict(list)
    for r in rows:
        cells[(r["tier"], r.get("difficulty"))].append(r)
    for tier in TIERS:
        for diff in ("easy", "medium", "hard", None):
            rs = cells.get((tier, diff))
            if not rs:
                continue
            n = len(rs)
            solved = sum(1 for r in rs if r["resolved"])
            costs = [r["cost_usd"] for r in rs if r["cost_usd"] is not None]
            turns = [r["num_turns"] for r in rs if r["num_turns"] is not None]
            tot = sum(costs)
            mean = tot / n if n else 0
            psolve = tot / solved if solved else float("nan")
            tbar = sum(turns) / len(turns) if turns else 0
            print(f"{tier:>7} {str(diff):>10} {n:>3} {solved:>6} "
                  f"{solved/n:>5.2f} {mean:>8.4f} {psolve:>8.4f} {tbar:>6.1f}")


def per_task(rows: list[dict]) -> None:
    bt = by_task(rows)
    full = {t: m for t, m in bt.items() if all(k in m for k in TIERS)}
    print(f"\n=== per-task tier comparison ({len(full)} tasks measured on all tiers) ===")
    print(f"{'task':>34} {'diff':>6} | " +
          " | ".join(f"{t:>6}" for t in TIERS) + " | best")
    for task, m in sorted(full.items()):
        diff = (m["haiku"].get("difficulty") or "?")[:6]
        cells = []
        solving = []
        for t in TIERS:
            r = m[t]
            ok = "OK" if r["resolved"] else "XX"
            c = r["cost_usd"] or 0
            cells.append(f"{ok}{c:>4.2f}")
            if r["resolved"]:
                solving.append((TIER_RANK[t], t, c))
        best = min(solving)[1] if solving else "none"
        print(f"{task:>34} {diff:>6} | " + " | ".join(f"{c:>6}" for c in cells) +
              f" | {best}")
    return full


def policies(full: dict) -> None:
    print("\n=== realized policy totals (tasks measured on all tiers) ===")
    print(f"{'policy':>22} {'solved':>7} {'$total':>9} {'$/solve':>9}")

    def tally(pick):
        solved = tot = 0
        for task, m in full.items():
            t = pick(m)
            r = m[t]
            tot += r["cost_usd"] or 0
            solved += 1 if r["resolved"] else 0
        return solved, tot

    n = len(full)
    for t in TIERS:
        s, c = tally(lambda m, t=t: t)
        print(f"{'always-' + t:>22} {s:>3}/{n} {c:>9.4f} "
              f"{c/s if s else float('nan'):>9.4f}")
    # Oracle: cheapest tier that solves; if none solve, cheapest (sunk cost).
    def oracle_pick(m):
        solving = [(TIER_RANK[t], t) for t in TIERS if m[t]["resolved"]]
        return (min(solving)[1] if solving
                else min((TIER_RANK[t], t) for t in TIERS)[1])
    s, c = tally(oracle_pick)
    print(f"{'ORACLE(cheapest-solve)':>22} {s:>3}/{n} {c:>9.4f} "
          f"{c/s if s else float('nan'):>9.4f}")


def incomplete(rows: list[dict]) -> None:
    bad = [r for r in rows if not r.get("stream_complete", True) or r["cost_usd"] is None]
    if bad:
        print(f"\n!! {len(bad)} incomplete runs (no result event — timeout/install-fail):")
        for r in bad:
            print(f"   {r['tier']:>7} {r['task']:<30} fail={r.get('failure_mode')} "
                  f"stop={r.get('stop_reason')}")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "results/tb_matrix.jsonl"
    rows = load(path)
    print(f"loaded {len(rows)} rows from {path}")
    cell_table(rows)
    full = per_task(rows)
    if full:
        policies(full)
    incomplete(rows)


if __name__ == "__main__":
    main()
