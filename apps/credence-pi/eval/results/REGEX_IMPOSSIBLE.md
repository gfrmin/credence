# What the governor does that no fixed rule can (and the honest boundary)

The recurring challenge to credence-pi is: *"you didn't need Bayesian decision theory for
this — a regex would do."* This is the rigorous, adversarially-stress-tested answer. We do
**not** fight a strawman regex; we steelman the strongest non-Bayesian alternative an
engineer would actually write, concede exactly what it matches, and show the two things it
cannot — and why matching them re-derives the brain.

Runnable: `julia --project=. apps/credence-pi/eval/regex_impossible.jl`. Every number is
reproduced against the real `decide()` (`apps/credence-pi/brain/feature_brain.jl` +
`src/stdlib.jl`), constants **λ=1, c=$0.50, q=$0.02** printed alongside each decision (every
decision flips under different constants, so a quoted decision without its dial is meaningless).

## The headline (bulletproof two ways)

> At a byte-identical input the governor returns two different actions, and the difference is
> carried entirely by the **second moment** of its belief. To reproduce its full action map
> you must compute value-of-information and maximise expected utility — the minimal correct
> implementation **is** Bayesian decision theory.

- **Impossibility:** at the same feature context with posterior mean θ=0.5 *to the last bit*,
  a wide belief `Beta(2,2)` → **ask** and a narrow `Beta(10,10)` → **proceed**. No stateless
  map (regex) and no point-estimate (mean-only) classifier can emit two outputs for one input.
- **Reconstruction:** separating those cases requires the joint of (distance-to-boundary,
  concentration, c, q, λ) — exactly EVPI + EU-max. There is no cheaper program.

## What a stateful counter DOES match (conceded, out loud)

The strongest non-Bayesian rule is a per-context counter with add-2 smoothing,
`rate = (n1+2)/(n1+n0+4)`, plus a threshold. It reproduces:

- **Calibration, bit-for-bit.** Brain θ vs counter rate on `(18,2)→0.833`, `(10,10)→0.5`,
  `(2,18)→0.167`, `(1,0)→0.6`, `(0,0)→0.5` — identical to every digit. *This is the point,
  not a defeat:* the counts **are** the Beta sufficient statistics, the +2/+2 **is** the prior,
  the smoothed rate **is** the conjugate posterior mean. The engineer re-derived one cell of
  `condition`. (Raw `n1/(n1+n0)` gives 0.9/1.0/NaN — only the +2-prior variant matches.)
- **"Different decision per user"** (a per-(user,context) majority counter) and the **dial
  flip** at a confident belief (a tunable threshold `θ < 1/(1+λ)`).

So we do **not** claim those are beyond a heuristic. Honesty here is load-bearing: the bare
"a calibrated number / a different decision per user" claims are matchable — by a heuristic
that has thereby re-derived a Beta cell.

## What it CANNOT match — break #1: the ask surface is EVPI, not a threshold

All the counter sees is its rate and count `n`. The brain's decision cannot be sorted by any
threshold on them:

| belief | counter-rate | count n | variance | brain decision |
|---|---|---|---|---|
| Beta(2,2) | 0.5 | 0 | 0.050 | **ask** |
| Beta(4,4) | 0.5 | 4 | 0.028 | **ask** |
| Beta(10,10) | 0.5 | 16 | 0.012 | **proceed** |
| Beta(4,2) | 0.667 | 2 | 0.032 | **proceed** |

- **Variance inverts:** Beta(4,4) (var 0.028) asks, but *higher*-variance Beta(4,2)
  (var 0.032) proceeds → `ask iff var>τ` is unsatisfiable (τ ∈ [0.032, 0.028) = ∅).
- **Count contradicts:** Beta(4,2) n=2 proceeds while Beta(4,4) n=4 asks → `ask iff n<N`
  cannot hold.

The gate is `EVPI = E_o[max EU after seeing o] − max EU now`, weighed against q — the joint of
(distance-to-boundary, concentration, c, q, λ). Beta(4,2)'s mean 0.667 is far from the 0.5
block/proceed boundary, so information cannot change the call ⇒ low VOI ⇒ proceed *despite*
high variance. Matching this reconstructs `voi` + `optimise`.

## What it CANNOT match — break #2: novel-context backoff (model averaging)

Train **only** on `build/ident-1` (×20 approve), then query contexts never seen:

| queried context | brain θ | flat per-context counter |
|---|---|---|
| build/ident-1 (trained) | 0.917 | 0.5 |
| build/ident-2plus (UNSEEN) | **0.708** | 0.5 (no entry → prior) |
| build/ident-0 (UNSEEN) | **0.708** | 0.5 |
| other/ident-0 (UNSEEN) | **0.604** | 0.5 |

The brain keeps counts at *every* feature-subset granularity (`{tool,ident}`, `{tool}`,
`{ident}`, `{}`), scores each by its Beta-Binomial marginal likelihood, and posterior-weights
them — so evidence from one context informs unseen siblings. A flat counter has no entry for an
unseen context and returns the prior 0.5. Matching the 0.708 reconstructs Bayesian model
averaging (the structure-BMA).

## Conclusion

A stateless regex can do none of this. A stateful counter matches the calibrated number —
*because that number is the Beta posterior mean*. But the **ask surface** (EVPI; no threshold
sorts it) and **novel-context backoff** (model averaging) defeat any counts+threshold
heuristic, and matching them re-derives `condition` + `voi` + `optimise`. The minimal correct
implementation **is** Bayesian decision theory — a re-derivation, not a trick.

This is the conceptual companion to the empirical results: waste detection at precision/recall
1.0 (`FINDINGS.md`), injection-harm blocking at 0.94 precision / 1.2% false-interrupt
(`SAFETY_INGREDIENTS.md`), and the user-EU accounting (`USER_EU.md`). Those show the governor
*helps*; this shows *why a fixed rule cannot be it*.

*Method note: the claims above were hardened by an adversarial red-team that steelmanned the
counter and verified every break against the real `decide()` (variance inversion, count
contradiction, bit-exact add-2 calibration, pooling 0.708/0.604). Framing scoped per its
findings: "regex cannot" is asserted only for stateless rules; the heuristic-resistant weight
is carried by EVPI + model averaging + the reconstruction argument.*
