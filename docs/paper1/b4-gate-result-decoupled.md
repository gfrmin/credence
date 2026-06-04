# Paper 1 — the decoupled gate flips it. Verdict: B ALIVE. (expert loop 2)

You were right, and I was wrong. The single-query gate measured the wrong MDP —
it coupled probe and submit, which expresses greedy's move exactly (the 189.4
match) but **structurally cannot express VOI's defining move, probe-without-
submit**. I tested a handicapped protagonist and called it dead. The decoupled
gate — the deployable action space — flips the verdict.

Same exact backward induction, no truncation (so no myopic-tail bootstrap),
one change: per question the agent submits tool *s* **and may probe one tool
*r≠s*** — paying *r*'s fee, learning *r*'s Beta from ground-truth feedback, but
submitting *s*'s answer. Exploration now costs the probe fee, not the −5 answer
risk.

## The two numbers you asked for

**Benchmark (20 seeds, oracle categories):**

| probe budget K | score | acc | probe-Q/seed |
|---:|---:|---:|---:|
| 0 (submit-only) | 180.8 | 66.8% | 0.0 |
| 1 | 210.0 | 71.9% | 5.0 |
| 3 | **211.8** | 72.4% | 6.1 |
| 6 | 211.2 | 72.4% | 6.2 |

Plateau past K=3 (budget non-binding ⇒ effectively the unbounded decoupled
ceiling). **Decoupled optimum 211.8 vs greedy 189.4 — +22.4. > 189 ⇒ B alive.**

**Matched-prior control (θ ~ Beta(1,1)):** decoupled-optimal − argmax-greedy =
**+15.1**, *larger* than the single-query +9.2 — exactly your prediction (cheap
probing helps more when the prior is right).

## Why I now believe it (three checks, since I over-claimed once already)

1. **Solver cross-check.** K=0 (submit-only) reproduces the single-query exact
   optimum (180.8) **to the decimal** — two independently-written solvers agree.
   The new machinery's submit logic is correct; the jump to 211.8 is the probe
   term, not a bug.
2. **Faithfulness.** `host.jl:233–248` updates reliability for *every* queried
   tool (`for (t,resp) in tool_responses`), each scored against ground truth,
   regardless of which was submitted. Probe-without-submit-but-learn is **literally
   the deployed mechanism** — the agent already queries tools, updates beliefs,
   and may submit a different answer. The +22 is deployable, not a modelling gift.
3. **Mechanism reads right.** The optimum probes ~6 questions/seed and lifts
   accuracy 66.8% → 72.4%; pulls shift toward the expensive-good KB (19.9) and
   the bimodal calc (16.6) — it pays small fees to learn reliabilities cheaply,
   then submits well-targeted answers. That's the learner-becomes-explorer story
   made concrete.

## What this settles, and what it doesn't

**Settles:** the +59 headroom was *not* a pure hindsight mirage — the hindsight
schedule's structure (decouple probe from submit) is real and a *closed-loop*
policy captures most of it. The deployable mechanism's exact ceiling clears greedy
by +22 (oracle). Your diagnosis was exactly right: online vs hindsight didn't
dominate — but the *coupling*, not the closed loop, was what killed it.

**Doesn't settle (the honest residual):** 211.8 is the exact ceiling (d=∞). The
DSL version will be a **depth-d lookahead** approximating it, and the *deployed*
number is the *inferred*-category one. Projection: 211.8 oracle − the ~40–53
inference penalty ≈ ~160–170 inferred vs greedy-inferred 149.6 — a win, likely
narrow. Per your reframe that's still a method paper (~210→narrow deployed win),
but it must be **built and measured**, not projected.

## The build (Direction B, greenlit by your stopping rule)

Following your §5 spec, now anchored to a real ceiling:

1. **Horizon-aware VOI in the DSL.** The decoupling already exists in the host
   (query-then-submit-other); what's missing is the *horizon term* — today's `voi`
   values information only for the current question. Extend it to value info over
   the remaining-category horizon: a depth-d lookahead, exact within d, pure EU-max
   (exploration emergent, no ε-greedy/bonus — constitution-clean).
2. **Tail.** Your prime suspect (myopic-tail bootstrap) is now testable against the
   exact ceiling: sweep d, report capture-fraction of 211.8. The K=3 plateau says a
   shallow d should get most of it; if performance keeps climbing with d, the tail
   is biting and I'll swap in a less-biased continuation before concluding.
3. **Measure** in oracle *and* inferred, vs greedy and myopic VOI; report honestly
   how much of the 211.8 ceiling the deployable depth-d version captures, and
   whether it clears greedy in both conditions.

The gate (`/tmp/horizon_voi_gate_decoupled.jl`) becomes a committed script: it's
the exact ceiling the paper's central claim rests on.

## What I'm doing

Per your pre-committed rule (211.8 > 189 ⇒ build) and "take the branch, don't
reopen": **proceeding to build horizon-aware VOI.** RESULTS.md, credence.tex, and
#108 stay frozen — but the framing they'll take is now inverted from the NO-GO
draft: not "optimism is near-Bayes-optimal, VOI doesn't pay," but "myopic VOI is a
learner not an explorer; horizon-aware VOI — valuing information over the question
horizon, decoupling probe from submit — is the principled fix, and it beats
greedy." I'll bring back the depth-d capture number; that's the one that decides
whether the deployed win is clean or narrow, and either way the paper is B.
