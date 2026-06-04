# Paper 1 — credit-rule de-confound: the fair loss is bulletproof. Lock A.

You found the real hole: the credit-assignment rule (B2c soft-credit) was the
untested upstream axis under the fair result, with all four prior de-confounds
downstream of it. Tested. The fair loss survives every credit rule, including the
proper one.

## The test (cost-blind submit + horizon-probing, inferred, 20 seeds)

| credit rule | greedy | horizon-VOI | gap | note |
|---|---:|---:|---:|---|
| soft (B2c, deployed)    | 149.8 | 116.7 | −33.1 | fractional leak to every category |
| hard (argmax-π)         | 142.5 | 130.4 | **−12.1** | zero leak; best-case for exploration |
| post (π_c·likelihood)   | 151.4 | 126.7 | −24.6 | the de Finettian rule B2c approximates |

Greedy wins fair under all three. Your mechanism call was exactly right — better
attribution helps exploration (soft→hard: horizon +13.7) and hurts minimal-query
greedy (−7.3), narrowing the gap from −33 to −12. But even at exploration's best
case (hard-credit, zero-leakage) it loses by 12, and the *proper* rule (post) lands
at −24.6 because it also lifts greedy. Oracle (perfect attribution) is the only
regime where horizon wins (+27). **The 78% classifier under no credit rule is
enough.** The fair loss is a property of imperfect attribution, not of soft-credit.

## Two findings that fall out

1. **Optimal credit rule depends on query strategy.** Zero-leakage hard-credit is
   best for exploration-heavy horizon-VOI, worst for minimal-query greedy; soft/post
   reverse it. The credit rule and the query strategy interact — a second interaction
   law beneath the first.
2. **B2c is a measurably suboptimal silent approximation.** Post-credit beats B2c
   even for the *deployed* greedy (151.4 vs 149.8). Per your last caveat, this is a
   latent constitution issue: B2c soft-credit was adopted in move B2c as a tractable
   approximation to the joint latent-category update without sweeping alternatives —
   the kind the constitution forbids going silent on. **Separate follow-up** (the
   deployed agents arguably should use post-credit); does not change the paper.

## Direction — locked

Per your pre-commitment (hard-credit loses fair ⇒ A as the interaction law), now
bulletproofed against the proper rule: **lock A.** The headline is the interaction,
both conditions are data:

> Exploration's value in category-conditioned tool selection is contingent on
> attribution quality. With given categories, horizon-aware VOI beats optimism-
> greedy (+27) — the under-exploration of myopic VOI is fixable. With inferred
> (fair) categories, attribution noise denies exploration the clean per-category
> signal it needs, under every credit rule, and minimal-query optimism wins. +27
> oracle and −12…−33 fair are two points on one curve.

Substrate result folded in as support, not headline: the belief substrate is
vindicated both conditions; VOI is the frugal frontier point (dominates the free
Llama); greedy is a Bayesian-family member, so the family owns the cheap frontier.
"VOI doesn't beat greedy under fair conditions" stated plainly, framed as evidence
for the contingency thesis (the duality reframe), not a wound.

## Executing A now

Lifting the freeze. Plan:
1. **RESULTS.md** → the interaction-law framing above (both conditions, the credit-
   rule robustness table, the decomposition, frugal-frontier support).
2. **agent.bdsl** — replace the falsified "horizon-VOI is future work" comment with
   the built mechanism + the contingency finding (executable documentation).
3. Wire `run_horizon_seed` (oracle) into the host so the +27 result is reproducible
   in-benchmark; commit the gate scripts (`paper1-horizon-gate.jl`, the credit-rule
   prototype) — they're the ceilings the central claim rests on.
4. master-plan / NOTES → the executed reality (B built; oracle win; fair loss;
   contingency law; credit-rule de-confound).
5. **Constitution follow-up (separate issue, not this PR):** B2c-vs-post-credit —
   file it; the deployed agents may want post-credit.
6. credence.tex reframe is the subsequent phase, against the locked RESULTS.md.

Unless you flag the post-credit constitution nuance as blocking, I'm writing A.
