#!/usr/bin/env python3
# Role: eval
# welfare_latency_from_matrix.py — build the eval routing-latency belief E[time|model,short] from
# the live welfare matrix, so the routing decision knows each tier's MEASURED wall-time — crucially
# the free local qwen tier, which the shipped TB-derived belief (paid tiers only) has no entry for
# (and without it the daemon would read qwen as 0s and a speed-first user would wrongly pick it).
#
# Schema = RoutingBrain.reconstruct_latency's: per_model[].{model_id, rate_s, contexts[].{ctx,
# sum_turns, n_obs}}, E[time]=(α0+Σt)/(β0+n)·rate_s. The eval measures end-to-end wall-time (the
# time the user waits, consistent across tiers — qwen has no model-only durationMs), so the turns
# decomposition is collapsed: choose sum_turns=n−0.9 ⇒ E[turns]=(1+n−0.9)/(0.1+n)=1 exactly, and
# rate_s=mean wall-time ⇒ E[time]=mean wall-time. Honest + exact.
#
#   python3 welfare_latency_from_matrix.py [matrix.jsonl] [out.json]
import collections, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MATRIX = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "welfare_matrix.jsonl")
OUT = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "results" / "welfare_latency.counts.json")
BELIEF = {"qwen": "qwen2.5:7b-instruct", "haiku": "claude-haiku-4-5",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}

rows = [json.loads(l) for l in open(MATRIX) if l.strip()]
times = collections.defaultdict(list)
for r in rows:
    times[r["tier"]].append(float(r["wall_s"]))   # wall-clock the user waits (consistent across tiers)

per_model = []
for tier, ts in times.items():
    n = len(ts); mean_t = sum(ts) / n
    per_model.append({"model_id": BELIEF[tier], "rate_s": mean_t,
                      "contexts": [{"ctx": ["short"], "sum_turns": n - 0.9, "n_obs": n}]})

out = {"artifact": "MVP-D eval routing LATENCY belief — measured mean wall-time per tier (short bucket)",
       "source": f"welfare matrix {Path(MATRIX).name}; end-to-end wall-time, turns collapsed (E[turns]=1, rate_s=mean wall-time)",
       "note": "E[time|model,short]=expect(Gamma(1+sum_turns,0.1+n),Identity)*rate_s = 1.0*mean wall-time",
       "turns_prior": [1, 0.1], "per_model": per_model}
json.dump(out, open(OUT, "w"), indent=2)
print("wrote", OUT)
for pm in per_model:
    print(f"  {pm['model_id']:<22} E[time]={pm['rate_s']:.1f}s  (n={pm['contexts'][0]['n_obs']})")
