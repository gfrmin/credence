# Reflective closure: the survivor dissolves by derivation

> Durable, in-repo record of how the "reflective / embedded-agency closure" question — the one
> survivor of a foundational gap-analysis of Credence's five axioms — is adjudicated. **It is not a
> branch arc and opens no moves.** Its whole content is: the clean cases dissolve *by derivation*
> (below), with **no axiom and no frozen-type change**; the rest is one conditional stdlib primitive,
> one empirical demo, and one named open problem that is deferrable indefinitely.
>
> **STATUS: settled note (clean cases) + one named open problem (deferred).** Written after four
> rounds of author review collapsed an over-built research plan to its real size. The session plan is
> at `~/.claude/plans/` (session-external); this note is the durable capture.
>
> **One-line disposition:** write nothing more here today; the engine's attention returns to the
> `exploration-budget` arc (Move 3 pending). Re-open only on a concrete use-case (§5) or an attempt on
> the open problem (§4).

## 1. What the gap analysis found, and what is rejected

A criterion-vs-content audit of the five axioms (`SPEC.md §1`) found that of ~10 candidate
foundational omissions, the dissolutions are sound and already match the project's posture —
causality (Richens–Everitt, robust agents learn causal models), corrigibility-from-CIRL (the
off-switch game is a *theorem* of A3+A4, not a new axiom), ergodicity-as-content, empowerment-as-VOI
— **no action**. Exactly one candidate survived: a **reflective / embedded-agency closure** of the
hypothesis space ("closed under inclusion of agents of equal power, including the agent's own
policy"). It is genuinely absent from the repo: CIRL is asymmetric-cooperative; meta-actions
(`SPEC.md §9`) are self-reference *about computation*, not self-*representation*; there is no
opponent-model, no grain-of-truth / reflective-oracle machinery.

The report's prescription — **"replace argmax-EU with a Thompson-sampling-like rule" — is rejected.**
It violates **Invariant 1** (`CLAUDE.md` — "Epsilon-greedy, UCB, Thompson sampling … cannot be
hard-coded as mechanisms outside EU-max"), and it runs directly against the project's most recent
foundational direction: `collapse-towers` retired `rand` from action selection, and the active
`exploration-budget` thesis is *"EU-max — not random, not capped."* The report called randomization
"the active ingredient"; it is not.

**Why the prescription was wrong, not just inconvenient.** Leike–Taylor–Fallenstein Example 27 (two
reflective-Bayes-optimal agents fail to converge on matching pennies under dogmatic priors) indicts
the **dogmatic prior**, not the absence of randomization. Matching pennies' equilibrium *is* a
distribution; a hypothesis space that cannot represent the agent's own mixed policy has assigned zero
mass where the answer lives — it cannot converge because it cannot represent the answer. **Thompson
sampling is a VOI estimator, not a primitive**: LTF reach for it as a *proof device* (a
known-convergent rule one can write a theorem about), not a claim about what exploration *is*. The
survivor is the **closure property**; the mechanism that achieves convergence is **policy-level
evaluation** — `max over my policies of min over the peer's responses` — under which the pure
policies are *dominated*. That is the **act-EU → policy-EU** upgrade: a change in *what object the
argmax ranges over*, not a new decision rule and not an axiom.

## 2. Three jobs, three instruments (the methodological precedent this note establishes)

The trap, recorded here so future sessions inherit the corrected instinct rather than rediscover it:
**do not put a proof and an experiment in the same gate slot.**

| Instrument | Decides | Status |
|---|---|---|
| **Derivation** | the *constitution* — does closure follow from the axioms? | clean cases: **settled, verdict = dissolves, no axiom** (§3) |
| **Simulation** | the *implementation* — does the built `optimise` compute the proof's value? | regression test on **code**, never the constitution (§5–6) |
| **Proof attempt** | the *general case* — exact policy-level VOI vs arbitrary computable peers | **named open problem**, deferrable (§4) |

The category error to avoid: reading a **green simulation as constitutional vindication.** A green
matching-pennies run establishes only "we coded the maximin correctly." If the general case (§4)
later turns out to need genuinely irreducible randomization, a repo full of green experiments would
*mask* the open hole the experiments were never able to detect. Routing the constitutional question
to the derivation keeps that hole **visible** — the point of the stall-honestly culture.

## 3. The derivations (the constitutional gate; clean cases settled)

### 3.1 Clean zero-sum — matching pennies: minimax → dissolves, no axiom
Row commits to a mixed policy `p = P(Heads)`; the zero-sum payoff is `EU(p,q) = (2p−1)(2q−1)`. With
the peer best-responding to the *committed* policy, `min_q EU(p,q) = −|2p−1|`, maximised at `p = ½`
with value `0`; every pure policy (`p ∈ {0,1}`) scores `−1`. So the maximin policy is the mixed
`(½,½)` and the pure policies are **strictly dominated**. The mixed play is **selected by argmax over
the policy space**, not injected. ∎

**Verdict: dissolves into the act→policy upgrade; no axiom.**

### 3.2 Clean cooperative — twin-PD: SEE → dissolves, no axiom (the diagonal reduction must be *earned*)
The tempting non-proof: "policy-level EU compares `C ↦ (C,C)=R` against `D ↦ (D,D)=P`, `R>P`, so
cooperate." That **smuggles its conclusion into its setup** — it *assumes* the reduction to the
diagonal `{(C,C),(D,D)}`, where cooperation trivially wins, and the entire difficulty is that `T>R`
tempts defection when the peer's move is held fixed. The content is *when the diagonal reduction is
licensed*:

- **Identical source code.** The peer provably plays your action, so your choice *is* a single
  decision over the diagonal; on the diagonal `R > P` selects **cooperate**. The reduction is
  licensed because it is literally one decision. ∎

- **Approximately shared source code.** The reduction is *not* automatically licensed. Under the
  symmetric-correlation model — the peer matches your action with probability `ρ`, plays the opposite
  with `1−ρ` — `EU(C) = ρR + (1−ρ)S` and `EU(D) = ρP + (1−ρ)T`. Then `EU(C) > EU(D)` reduces to
  `ρ(R−P) > (1−ρ)(T−S)`, i.e. cooperation is policy-EU-optimal **iff**

  > **`ρ > ρ* = (T−S) / [(R−P) + (T−S)]`.**

  Below `ρ*`, defection is policy-EU-optimal — **the threshold *is* the game.** (Sanity check with
  the canonical `(T,R,P,S) = (5,3,1,0)`: `ρ* = 5/7 ≈ 0.71`; at `ρ = 0.4`, `EU(D) = 3.4 > EU(C) = 1.2`,
  defection strictly wins — so any threshold must sit above `0.4`, and `5/7` does. The naïve
  `(T−R)/(T−S) = 0.4` would license cooperation exactly where defection wins; it is the wrong number.)

**Verdict: dissolves (Subjective-Embedded-Equilibrium); no axiom — unconditional under identity,
conditional on `ρ > ρ*` under approximation.** This is Credence's real product geometry (a human is a
peer who models the agent back), which is exactly why the note must carry the **threshold**, not an
unconditional "cooperate".

## 4. The named open problem (deferred, honestly)

Does **exact** policy-level VOI argmax attain a grain of truth against **arbitrary computable
peers**, where minimax/SEE no longer rescue it? LTF prove convergence for **Thompson sampling** over
a reflective-oracle-closed class — **not** for exact policy-level argmax. The honest trichotomy:

1. **exact works in general** → dissolves, no axiom; or
2. **exact intractable** → Thompson re-enters as the bounded agent's *EU-justified, compute-priced*
   estimator (chosen because net_value-optimal, not hard-coded) → **still no axiom** (Invariant 1
   preserved); or
3. **randomization is irreducibly primitive** — required for convergence in a way EU-max provably
   cannot select as either an emergent maximin or a compute-priced estimator → **only here does the
   constitution move.** The prior is strongly against this, but it is genuinely open.

This question is gated by **attempting the proof**, not by running a simulation. Deferrable
indefinitely (open in the literature for a decade).

**Deferred ≠ resolved.** Disjunct 3 is the *single live path* by which this whole question could
still reach the axioms. It must stay visible — "deferred indefinitely" is a scheduling call, not a
verdict — precisely because everything around it is now so satisfyingly closed that disjunct 3 is the
one most likely to be silently re-remembered as "settled". It is not.

## 5. The implementation finding ((d) governs the build, not the constitution)

What it would take to *build* the policy-level `optimise` whose output §3 already characterizes:

| Property | Verdict | Evidence |
|---|---|---|
| (a) mixed policy as an **object** (Measure over an action Space) | **free** | `CategoricalMeasure(Finite(actions), logw)` (`src/ontology.jl:129`); `apps/julia/email_agent/host.jl:147` already builds a distribution over the *user's* actions |
| (b) **grammar-reachable** / (c) **self-bindable to the agent's own policy** | **absent** | programs evaluate to `Symbol`s; closure is `{Predicates → Actions}` (`src/program_space/types.jl:58`, `enumeration.jl:51-143`, `compilation.jl:103`); no type represents an agent or a policy-as-content |
| (d) **`optimise` evaluable at the *policy* level, not the act level** | **UNKNOWN** | `optimise`/`value`/`eu` (`src/stdlib.bdsl:12-20`) argmax over *actions given a fixed belief*; the maximin needs an *outer* argmax over the agent's own (mixed-closed) policy-space against a best-responding peer — a nested `min`-inside-`max` act-level stdlib may not express |

- (a) is frozen-legal → hosting a mixed policy implies **no frozen-type/axiom change**.
- **Build on demand only.** If a real use-case needs policy-level evaluation: first verify whether it
  composes from existing stdlib; else add a **mutable-layer** `optimise`-over-policies (still not an
  axiom/frozen-type change). Its correctness is a **one-line unit assertion**
  (`optimise_policy(matching_pennies) ≈ (0.5, 0.5)`, and the SEE assertion), *not* a repeated-play
  harness.
- (b)/(c) are **downstream** of (d): they govern only whether the agent could *autonomously discover*
  a policy-level hypothesis; pursue only if that is a goal.

## 6. What is deliberately *not* built

- **A repeated-play matching-pennies convergence harness — not built.** A 2×2 maximin is closed-form;
  a correct primitive returns `(½,½)` by construction, and a wrong one is caught by the first unit
  test of the primitive. A convergence harness would be a *regression test for a theorem* — ceremony.
  Collapsed to the one-line assertion in §5.
- **Test B (twin-PD) regression half — collapsed** to one SEE assertion. Its *demo* half is the only
  genuinely empirical work in the whole thread, and is built **only when twin-PD is product-relevant**
  (`apps/julia/experiments/grain_of_truth/`, when needed): the SEE *theorem* is settled (§3.2), but
  **where the real human-peer falls on the `ρ` axis relative to `ρ* = (T−S)/[(R−P)+(T−S)]`** is not a
  theorem — it depends on how much genuine action-correlation a messy, partially-closed deployment
  carries. The demo *locates the deployment's `ρ` against the threshold*; it does not watch a
  guaranteed cooperation "survive".
- **Löbian qualification to the §9 "reflectively stable" claim — out of scope.** A different axis
  (proof-based vs probabilistic self-trust). Not appended here; left for its own thread if it earns
  one.

## 7. Opportunity-cost verdict

After §3, this thread is a *settled note plus a deferred open problem* — not a research project. The
clean-case constitutional question is closed (dissolves, no axiom); the implementation is a single
conditional stdlib primitive built only on demand; the one empirical task waits for product
relevance; the open problem is deferrable indefinitely. **The engine's attention returns to the
`exploration-budget` arc (Move 3 pending).** Re-open this note only on a concrete use-case (§5) or an
attempt on §4.
