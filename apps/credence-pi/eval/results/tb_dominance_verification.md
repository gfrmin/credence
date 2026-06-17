# tb_dominance — adversarial verification record (2026-06-17)

A 5-skeptic + synthesis workflow attacked the escalation-eu routing-dominance result
(each skeptic re-ran `tb_dominance.jl` with perturbations). Verdict:
**SURVIVES-WITH-CAVEATS.** One MAJOR mislabel was caught and fixed before this shipped.

| Dimension | Flaw? | Severity | Finding |
|---|---|---|---|
| Leakage | No | none | Split disjoint+exhaustive; beliefs train-only; gate reads test *covariates* before the outcome. A leak-variant (gate on `t.resolved`) gives *different* numbers ⇒ no peek. Clean. |
| **Foil fairness** | **Yes** | **MAJOR** | The cumulative cascade was mislabeled "clairvoyant upper bound — knows success before paying" but pays for failed rungs. A genuine per-task oracle necessarily beats any deployable policy, so "dominates the clairvoyant cascade" was impossible/false. Deployable foils (RouteLLM, best-fixed, argmax) verified FAIR. |
| Metric | Yes | minor | escalation-eu wins 0/3 profiles outright (the win is the minimax aggregate). Headline 0.035 was the optimistic point; true ~0.04–0.14 across splits. Robust to profile set (cherry-pick refuted by reward sweeps). |
| Data-noise | Yes | minor | 3 all-fail tasks (not 4); cheapest-solver haiku 9 / sonnet 2 / opus 3; 4 sonnet-timeout lower-bound costs (under-count flatters always-sonnet, not escalation-eu); single-rep path-tracing inversion. Ranking robust to perturbations. |
| Mechanism | Yes | minor | Gate is a MYOPIC single-step EU rule, not exact sequential EU-max (exact DP worst-regret ~0.073; myopic edges it here via conservatism). Learning all `condition` (correct). Live-brain escalation not yet wired. |

## Fixes applied (this commit)

1. **[MAJOR]** Renamed arm `oracle-cascade` → `frugalgpt-cascade` (a deployable
   pay-as-you-go peer = escalation-eu minus the gate). Added a separate NON-deployable
   `clairvoyant-oracle` reference (per-task cheapest-solver / abstain). Struck all
   "clairvoyant/upper-bound/beats-the-bound" framing from the harness + EXPERIMENT.md.
   Corrected claim: **best deployable policy; dominates the gateless cascade; recovers
   ~95% of the clairvoyant ceiling; trails the true oracle** (as it must).
2. **[minor]** Minimax regret now computed over DEPLOYABLE arms only; reported as a range;
   "scale-fair" comment corrected to "regret-vs-best robustness statement."
3. **[minor]** "EU-max/EU-rational" → "myopic single-step EU gate"; exact-DP figure noted.
4. **[minor]** Data description fixed (3 all-fail, haiku 9/sonnet 2/opus 3); 1-rep noise,
   timeout lower-bounds, and the all-fail dependence flagged.

Net: the core thesis holds (escalation-eu is the best deployable router, beats always-opus
on every profile and the plain cascade, lowest minimax regret), with the clairvoyant
overclaim removed. Up-front feature routing still loses (honest negative). Next de-risk:
≥3 reps/cell; live-wire the gate through `optimise`.
