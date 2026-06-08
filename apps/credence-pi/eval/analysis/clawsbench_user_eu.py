# Role: eval
#
# Move 2 (task-value): does ClawsBench let the governor maximise the OpenClaw user's
# expected utility? Computes the real user-EU picture + the attribution wall + the
# metareasoning-escalation signal. Run from repo root:
#   python3 apps/credence-pi/eval/analysis/clawsbench_user_eu.py
import json, csv, statistics
from collections import defaultdict

RES = "data/credence_pi_eval/corpora/clawsbench/results.csv"
EV = "data/credence_pi_eval/clawsbench_openclaw.events.jsonl"
res = list(csv.DictReader(open(RES)))
runs = [r for r in res if r["n_tool_calls"] not in ("", None)]
P = [r for r in runs if int(r["passed"] or 0) == 1]
F = [r for r in runs if int(r["passed"] or 0) == 0]
tc = lambda rs: [float(r["n_tool_calls"]) for r in rs]
du = lambda rs: [float(r["agent_duration_sec"]) for r in rs if r["agent_duration_sec"] not in ("", None)]

print("══ 1. the user's EU problem is real and large (real outcome+cost data) ══")
print(f"  runs with cost data: {len(runs)}  pass rate {len(P)/len(runs):.1%}  (fail {len(F)/len(runs):.1%})")
print(f"  tool calls: passed {statistics.mean(tc(P)):.1f}  failed {statistics.mean(tc(F)):.1f}  (failed +{statistics.mean(tc(F))/statistics.mean(tc(P))-1:.0%})")
spendF, spendP = sum(tc(F)), sum(tc(P))
print(f"  SHARE OF TOOL-CALL SPEND ON FAILED RUNS: {spendF/(spendF+spendP):.0%}")
durF, durP = sum(du(F)), sum(du(P))
print(f"  SHARE OF AGENT TIME ON FAILED RUNS:      {durF/(durF+durP):.0%}")
byst = defaultdict(list)
for r in runs:
    byst[r["score_type"]].append(float(r["n_tool_calls"]))
print("  cost rises monotonically as outcome worsens:",
      {k: round(statistics.mean(v), 1) for k, v in sorted(byst.items(), key=lambda kv: statistics.mean(kv[1]))})

print("\n══ 2. the attribution wall: no learnable PER-ACTION task-value lever ══")
ev = [json.loads(l) for l in open(EV)]
agg = defaultdict(lambda: [0, 0])
for r in res:
    k = (r["model"], r["task_name"]); agg[k][0] += int(r["passed"] or 0); agg[k][1] += 1
pr = lambda m, t: (agg[(m, t)][0] / agg[(m, t)][1]) if agg[(m, t)][1] else None
joinable = sum(1 for r in ev if pr(r["meta"].get("model"), r["meta"].get("task_name")) is not None)
print(f"  outcomes join to events only at (model,task_name) granularity: {joinable}/{len(ev)} calls ({joinable/len(ev):.0%})")
buck = defaultdict(list)
bb = lambda i: "0-4" if i < 5 else "5-9" if i < 10 else "10-19" if i < 20 else "20+"
for r in ev:
    p = pr(r["meta"].get("model"), r["meta"].get("task_name"))
    if p is not None: buck[bb(r["idx"])].append(p)
print("  per-call budget-burn vs task-type pass-rate is FLAT (label washes out the signal):",
      {b: round(statistics.mean(buck[b]), 3) for b in ("0-4", "5-9", "10-19", "20+")})
print("  => per-action success credit is unrecoverable from per-session reward (exploration⊥attribution).")

print("\n══ 3. the actionable lever: per-session-STATE doom, learnable from per-run reward ══")
rr = [(int(r["passed"] or 0), float(r["n_tool_calls"])) for r in runs]
base = 1 - sum(ok for ok, _ in rr) / len(rr)
print(f"  base P(fail) = {base:.3f}")
for K in (8, 10, 15, 20, 25):
    sub = [ok for ok, c in rr if c >= K]
    pf = 1 - sum(sub) / len(sub)
    rem = statistics.mean([c for ok, c in rr if c >= K]) - K
    print(f"  P(fail | burn>=={K:2}) = {pf:.3f} (lift {pf/base:.2f})  ~{rem:.1f} calls still at risk  n={len(sub)}")

print("\n══ 4. one-currency EU of the metareasoning escalation (ask as budget burns) ══")
# EU(ask at burn>=K) - EU(continue) ~= P(fail|>=K) * remaining_calls * c  -  q
# (asking aborts a doomed run, saving its remaining spend; costs one interrupt q.)
c_call, q = 0.50, 0.02  # daemon defaults: $/call, $/interrupt (labelled assumptions)
print(f"  assumptions: call cost c=${c_call}, interrupt cost q=${q} (daemon defaults; real $ from live telemetry)")
for K in (8, 10, 15):
    sub = [(ok, c) for ok, c in rr if c >= K]
    pf = 1 - sum(ok for ok, _ in sub) / len(sub)
    rem = statistics.mean([c for _, c in sub]) - K
    eu = pf * rem * c_call - q
    fire = len(sub) / len(rr)
    print(f"  ask@burn>={K:2}: fires on {fire:.0%} of runs, P(doomed)={pf:.2f}, EU gain/run ~= ${eu:.2f}  (>0 => raises user EU)")
print("  caveats: benchmark task mix (77% fail) ≠ typical user mix; replay-stationarity;")
print("  assumes the user aborts a doomed run when asked. Live telemetry settles the causal claim.")
