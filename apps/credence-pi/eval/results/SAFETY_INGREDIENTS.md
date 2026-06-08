# Safety ingredients: finding the features that make P(unsafe|X) discriminative

**One line:** generic "risky action" features barely beat the base rate on
prompt-injection safety, because safe and unsafe ATBench trajectories are *matched on
attack-presence* — every safe trajectory is one where the agent **refused**. The
discriminating ingredient is **taint-flow**: a consequential *sink* (send / forward /
post / delete / exec) carrying a parameter or payload that originated in *untrusted*
content. A compromised agent propagates the taint into a sink; a defending agent breaks
the flow. Measured on the real corpus, taint features reach **0.82–0.97 trajectory
precision** versus the regex baseline's **0.67** and the old `untrusted-provenance`
control's **0.59 (= the base rate exactly — it fires on everything)**.

Corpus: `AI45Research/ATBench-Claw` (Apache-2.0), 500 OpenClaw trajectories,
296 unsafe / 204 safe, base rate P(unsafe) = 0.592. Per-trajectory `is_safe` label.
Validation scripts: `data/credence_pi_eval/analysis/cand_*.py` (shared helper
`atbench_taint.py`). Each ingredient measured identically: a trajectory "fires" if ≥1
of its calls fires; report trajectory precision (of firing trajectories, fraction
unsafe) and recall (of unsafe trajectories, fraction with a firing call).

## The structural fact that defeats naive features

All 204 safe trajectories carry a non-null `defense_type` (119 detected_with_warning,
85 defended) — i.e. **every safe trajectory contains an attack the agent refused**.
So `action-class ∈ {external-send, delete, exec, credential-access, …}` is present in
both arms; it cannot separate compliance from refusal. The difference is whether the
agent *carried the injection into a consequential action*.

## Ranked ingredient panel (trajectory-level, vs is_safe)

| ingredient | lift | traj-precision | traj-recall | safe-FP rate | n-traj | verdict |
|---|---:|---:|---:|---:|---:|---|
| credential→external chain | 1.69 | **1.00** | 0.020 | 0.000 | 6 | strong (rare specialist) |
| destructive + tainted-target | 1.69 | **1.00** | 0.007 | 0.000 | 2 | high-precision, negligible recall |
| **tainted-external-target** | 1.64 | **0.97** | 0.118 | 0.005 | 36 | strong |
| **any-tainted-sink** | 1.53 | **0.91** | 0.270 | 0.039 | 88 | strong (workhorse) |
| tainted-payload-copy | 1.44 | 0.85 | 0.186 | 0.049 | 65 | strong |
| **injected-imperative** | 1.43 | 0.83 | 0.220 | 0.064 | 78 | strong (orthogonal) |
| goal-drift-target | 1.42 | 0.82 | 0.159 | 0.049 | 57 | strong |
| sink-after-untrusted *(temporal control)* | 1.38 | 0.78 | 0.544 | 0.221 | 206 | weak — temporal-only, no token taint |
| target-externality *(old feature, control)* | 1.24 | 0.74 | 0.321 | 0.167 | 129 | weak |
| risky-class-union *(the regex baseline)* | 1.21 | 0.67 | 0.753 | 0.529 | 331 | weak — high recall, ~base-rate precision |
| untrusted-present *(old feature, neg. control)* | 1.09 | 0.59 | 1.000 | 1.000 | 500 | **reject — fires on everything** |

Lift = P(unsafe | fires) / 0.592. The corpus ceiling for a perfect-precision feature is
1/0.592 = 1.69, so "lift ≥ 1.3" for the taint family reflects genuinely high precision,
not an inflated effect size.

## What this says

1. **Taint is the ingredient.** The taint family dominates precision (0.82–0.97). The
   token that lands in the sink (`attacker@evil.com` forwarded out, a tainted path
   deleted) only gets there when the agent *acted on* the injection.
2. **It is a complementary panel, not one feature.** Different recall coverage, all high
   precision, low mutual overlap — exactly what Bayesian model averaging exploits.
   `any-tainted-sink` is the broad workhorse (P 0.91, R 0.27); `injected-imperative` is
   orthogonal (verb-compliance, R 0.22); `credential→external` is a perfect-precision
   rare specialist (P 1.00).
3. **The recall ceiling is honest.** Taint catches a high-precision *subset*. Harms that
   never carry an untrusted token into a sink — self-generated misinformation, reasoning
   failures, refusal-failures with no exfil — are out of reach for *static* features.
   This is the EU posture: high-precision escalation (ask/block where we are confident),
   not omniscient classification. The recall gap is the saturation signal for richer
   (eventually content/semantic) features, gated per the saturation precondition.
4. **The negative controls did their job.** `untrusted-present` fires on 100% of calls
   and lands at exactly the base rate (0.59) — this is the prior session's broken
   `untrusted-provenance`, now retired with evidence. `sink-after-untrusted` (temporal
   provenance, no token taint) is materially weaker (0.78) than token-taint (0.91),
   proving the *flow* matters, not mere co-occurrence.

## Production feature set chosen

Replace the weak `{target-externality, untrusted-provenance}` with the validated
ingredients, keeping `action-class` as the substrate the taint refines:

- `action-class` — read-only / local-write / delete / exec / external-send /
  credential-access / cross-boundary / other (per-class base-rate context).
- `taint-flow` — none / tainted-sink / tainted-external-target (the ordinal precision
  gradient; the key ingredient).
- `injected-imperative` — no / yes (orthogonal compliance signal).
- `cred-exfil-chain` — no / yes (the rare perfect-precision two-step chain).

Computed **causally** in the eval adapter (a token taints a call only if it appeared in
untrusted content *before* that call) so the offline numbers reflect what the live body
can compute.

## End-to-end: the learned brain (the EU machinery + the right ingredients)

`safety_eval.jl` trains `P(unsafe|X)` through the **real Tier-1 path** (`build_model` →
`condition` → `expect`) and escalates a trajectory when any call's `E[P(unsafe|X)] ≥ τ`
(τ = q/H, the harm-aversion dial). Two feature arms, two per-call attributions of the
per-trajectory label. Test split: 200 trajectories (119 unsafe / 81 safe), seed 0.

**Under the honest reason-localized label** (the harmful call the human `reason` names;
independent of the taint feature → non-circular):

| arm | operating points | reachable regime |
|---|---|---|
| simple rule (regex) | 0.73 precision / 0.70 recall / 0.38 false-interrupt | one blunt point |
| **OLD features** | 0.865 precision / 0.378 recall / 0.086 false-int **@τ=0.5 only** — collapses to 0 past τ=0.5 | untunable; never confident |
| **TAINT features** | 0.87 P / 0.23 R / 0.049 FI (τ=0.5) → **0.94 P / 0.13 R / 0.012 FI (τ=0.8)** | smooth calibrated frontier into the high-confidence corner |

**The win is calibration, not raw accuracy.** The old features give the brain a single,
untunable ~0.86-precision point and then *zero confidence past τ=0.5*. The taint features
give a tunable precision/false-interrupt frontier reaching **0.94 precision at 1.2%
false-interrupt** — a high-confidence block regime the old features cannot produce.
Mechanistically: taint splits the `external-send` cell into *carrying-injected-data*
(P≈0.94, block) vs *user-requested* (proceed) — the distinction `action-class` alone
cannot make. The robustness arm (coarse risky-class attribution) confirms taint uniquely
reaches the 0.938 / 1.2% point under either attribution.

**The recall is structurally bounded, and the brain saturates the bound.** The static
ceiling is ~0.30 of unsafe trajectories (`reason_attribution.py`): ~46% of injection-harm
is harmful *output* (no tool call to block) and ~half of the action-harms carry no static
signal. At τ=0.5 the brain catches 0.227 — **≈75% of the structurally-blockable subset** —
with 4.9% false-interrupt; at τ=0.8, ≈41% of blockable at near-zero (1.2%) collateral.
The ~70% of unsafe it does *not* catch is not a feature-tuning failure: it is harm that a
static, tool-boundary governor cannot see. That gap is the saturation signal for richer
(content/semantic) features, admissible per the saturation precondition — and the natural
target of an automated feature-grammar search (`perturb_grammar` over the feature DSL).

**Bottom line:** the EU/Bayesian machinery + the *taint* ingredient turns the prior
"barely beats a regex, never confident" result into a calibrated governor that blocks
injected-data exfiltration with **94% precision and ~1% false-interruption** — catching
the majority of the harm a tool-boundary governor can structurally catch.
