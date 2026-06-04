# Paper 1 — horizon-VOI built. Oracle: wins (+27). Fair: loses (−16). (expert loop 3)

I built the constitution-clean DSL horizon-VOI per the greenlight, validated it,
and ran it in both conditions. The verdict splits on the condition, and the
*fair* (deployed, headline) condition is the one that decides the paper:

- **Oracle (categories given): horizon-VOI beats greedy by +27.** The mechanism works.
- **Inferred (fair): horizon-VOI LOSES to greedy by ~16**, robustly across every
  degree of freedom I tested. The exploration advantage is **oracle-only**.

I was wrong with the single-query NO-GO (handicapped mechanism). I do not think
I'm wrong here — this time I tested the *right* mechanism (decoupled probing, the
one that wins oracle) and de-confounded it four ways. But it's the third turn, so
read the de-confounding and tell me if there's a hole.

## What got built (and works)

Constitution-clean DSL `horizon-step` (`apps/julia/qa_benchmark/agent.bdsl`):
depth-1 horizon-VOI — value a probe by its expected improvement to the best
single-submit over the remaining (h−1) same-category questions, minus the fee.
Pure EU-max over the horizon, exploration emergent, no ε-greedy/bonus. The
probe-without-submit move is the deployed host mechanism (every queried tool is
learned from ground-truth feedback). Validated: DSL 216.8 ≈ Julia depth-1
reference 217.6, vs greedy 189.4 (oracle). The build was not wasted — it produces
the oracle result and the mechanism.

## The decomposition (20 seeds)

| condition | myopic VOI | greedy (optimism) | horizon-VOI | winner |
|---|---:|---:|---:|---|
| **oracle** (categories given) | 163.7 | 189.4 | **216.8** | horizon-VOI **+27** |
| **inferred** (fair) | 110.4 | **149.8** | 134.2 | **greedy +16** |

Inference penalty by policy: greedy −39.6, myopic VOI −53.3, **horizon-VOI −82.6**.
Horizon-VOI is the *most* inference-sensitive policy — its whole advantage
(precise exploration) is the most fragile to category noise.

## Why fair kills it — and the four-way de-confounding

Mechanism: under inferred categories the reliability update is B2c — a tool's
outcome is credited to *every* category weighted by the (noisy) posterior π. So
**every query injects category-misattribution noise** into the per-category Betas
(a probe on a question the classifier gets wrong corrupts the wrong category's
reliability). Greedy queries one tool/question → least corruption → wins. Probing
adds queries → more corruption.

I worried this was another handicapped mechanism, so I isolated it:

1. **Submit-policy confound found and removed.** My horizon-VOI uses cost-aware
   (EU-max) submit; greedy is cost-blind (argmax mean reliability). Cost-aware
   *mis-trades* under noise (dodges a reliable-but-expensive tool to save a unit,
   eats −5 wrongs): cost-aware no-probe = 119.6 vs cost-blind greedy 149.8. So I
   gave horizon-VOI greedy's *own* cost-blind submit + probing → **116.7**, still
   −33. Probing hurts *both* submit policies.
2. **Probe-intensity sweep** (λ scales the horizon term, λ=0 = no probe): cost-
   aware 119.6→136.6→126.4→134.2; cost-blind monotonically worse with probing.
   No setting clears greedy.
3. **Confidence-gating** (only probe when max π ≥ τ): τ=0/0.5/0.7/0.9 →
   116.7/116.7/120.0/120.0. Barely moves — the LOO classifier is *confidently
   wrong* often (high max-π even when misclassified), so the gate rarely fires.
4. **Anchor reproduces**: greedy_inferred 149.8 ≈ host 149.6 to the decimal.

Every exploration variant loses to greedy under fair conditions. The ceiling
under inferred is ~greedy, reached by *not* exploring.

## The honest thesis this forces

The original "Bayesian VOI tool selection wins under fair conditions" is **false**
— greedy (optimism) wins fair. But the complete decomposition is a real, novel
result:

> Cost-efficient tool selection decomposes into a belief substrate (reliability
> learning — necessary; non-learners collapse), an exploration policy, and
> category inference. Horizon-aware exploration fixes myopic VOI's
> under-exploration and **beats greedy when categories are given (+27)**. But the
> exploration advantage is **contingent on category-attribution quality**: under
> inferred (fair) categories, every exploratory query injects misattribution noise
> into per-category reliability learning, so **minimal-query optimism is the
> cost-efficient winner**. The three levers interact — exploration's value is
> destroyed by the very inference noise it depends on.

What survives for the Bayesian framing: (1) the *substrate* is vindicated in both
conditions; (2) **VOI is still the frugal frontier point** (dominates the free
Llama on cost and score — last round's fix); (3) greedy is itself a Bayesian-
family member (optimism-under-uncertainty), so the *family* owns the cheap
frontier. What dies: "VOI specifically beats greedy under fair conditions."

## Direction — and I want your read before I write a word of RESULTS.md

This is honest and the contingency/interaction is genuinely novel (exploration ⟂
attribution-quality is not, to my knowledge, a standard result). But the
protagonist loses the headline condition. Three ways to frame:

- **(A) The contingency decomposition** — headline = the three-lever decomposition
  + the interaction finding; horizon-VOI wins oracle, greedy wins fair, here's the
  mechanism. Honest, complete, novel interaction; protagonist loses fair. *My lean.*
- **(B) Noise-robust exploration** — build an agent that models π's *own*
  uncertainty and probes only when attribution is reliable. I expect it ceilings
  at ~greedy (the exploration that would beat greedy is exactly what's corrupted),
  so likely a tie, not a win — but I haven't built it. Higher effort, low
  expected payoff.
- **(C) Reframe to the substrate** — headline = the Bayesian *substrate* + frugal-
  frontier result (VOI dominates Llama, family owns the cheap frontier); the
  exploration-vs-inference interaction as a secondary finding. Sidesteps "which
  policy wins" by elevating the substrate claim.

My read: **A** (or A folded into C). The interaction is the contribution; pretending
horizon-VOI wins fair would be the dishonest reach we've spent three loops avoiding.

Specific questions:
1. Is the **contingency finding** (exploration's value ⟂ category-attribution
   quality, with the B2c-noise mechanism) enough of a contribution to carry the
   paper, with greedy winning the headline?
2. Is **(B)** worth one prototype afternoon, or is the "ceilings at greedy"
   argument strong enough to skip it?
3. Hole-check: I de-confounded submit-policy, probe-intensity, and confidence-
   gating. Is there a fifth degree of freedom I'm missing that could let
   exploration clear greedy under fair conditions?

RESULTS.md, credence.tex, #108 stay frozen until you weigh in. The oracle DSL
mechanism is built and committed-ready; the fair-condition result is the finding.
