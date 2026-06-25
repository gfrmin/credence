# Decouple Move 1 — life-agent pilot (repoint + close the arithmetic leak)

Design doc per `docs/posture-4/DESIGN-DOC-TEMPLATE.md` (adapted for `decouple`, as
Move 2 did). Reference master plan: `docs/decouple/master-plan.md`.

## 0. Final-state alignment

Converges the tip toward `docs/decouple/master-plan.md` §"Final-state architecture":
life-agent becomes *data + a thin body* that pins a versioned `credence-skin` image
and drives every belief update and decision over the wire, carrying **no probability
arithmetic**. Today the seam is already clean in *shape* — `brain.py` spawns the skin
over stdio JSON-RPC with opaque state handles (`SubprocessTransport`) — but it leaks
in *substance*: the body shapes beliefs in Python and ships pre-computed log-density /
EU vectors (`lookup.py:562-580`, `utility.py`). This move (a) repoints `brain.py:35`
off the source checkout onto the pinned image, and (b) closes the leak per the
**governing principle** (consumer picks a *correct, exact* model; credence does all
calculation and provides the kernels). Transient state left explicit (not drift): the
move adds new engine kernel families (below) — this is sanctioned stdlib accretion
(the consumer needs them), not a violation of "thin brain"; the *thinness* is in the
body, which ends with zero `math.log`/`exp` on beliefs.

## 1. Purpose

Make life-agent a thin wire client of a pinned `credence-skin` image, and — the
load-bearing part — replace its host-side *approximations* (the §4.2 tempering; the
host-computed reaction likelihood; the host utility-grid fold) with the **correct
exact models**, supplying the engine kernels/verbs those models require. The body
ends declaring spaces/kernels/priors/utilities as data and calling Tier-1 ops; the
arithmetic lives in the engine.

## 2. The modelling pass (life-agent's call — the substance of this move)

Per the governing principle, life-agent owns *which* models; credence supplies the
computation. Each current leak is an *incorrect or host-resident* model; the fix names
the correct one and the engine primitive it needs.

### 2.1 The lookup posterior — remove tempering (load-bearing)

**Current (incorrect) model** (`lookup.py:544-608`). A categorical over `{cand_1…cand_K,
NONE}` is conditioned once per observation by a noisy-channel `tabular_log_density`
with reliability `r_i = ρ·authority_i·subject_i·time_i`, `match = r+(1−r)/A`,
`miss = (1−r)/A` — **but each observation's log-likelihood is raised to a power**
`scale_i = s_anc·s_mod` (`temper_scales`, `:544-559`) to fake the fact that the
observations are not independent. This is an approximation; it must go.

**The real correlation, exactly:**
- **`s_anc` — shared ancestry.** `m` observations sharing an `artifact_cache_key` are
  `m` chunk-extractions of *one document*; they are correlated because they share the
  document's content/reliability. Exact model: a per-document latent `D_d`; the `m`
  chunks are conditionally independent given `(V, D_d)`. Because documents are
  independent given `V`, `D_d` **marginalises per-document** into a closed-form
  group-likelihood `P(extractions of doc d | V) = Σ_{D_d} P(D_d) Π_{c∈d} P(o_c | V, D_d)`
  — O(L·m) per document, **no hypothesis-space growth**.
- **`s_mod` — shared instrument.** Every observation shares one extractor whose
  reliability `ρ` is *uncertain and shared*; that shared uncertainty is precisely the
  cross-document correlation `s_mod` discounts. Exact model: carry `ρ` as a latent
  (a Beta / a small grid), shared across all observations, and **marginalise it**:
  `P(V | obs) ∝ Σ_ρ P(ρ) Π_d P(doc d | V, ρ)`. A single shared latent ⇒ one top-level
  mixture, no blowup. This *also* replaces the separate approximation "`ρ` is a point
  estimate moved by audit outcomes" (`lookup.py` docstring §2/§14) with proper
  conditioning — `ρ` becomes a belief the audit outcomes condition.

**Net exact model:** hypothesis `V × ρ × {D_d}`, chunks conditionally independent
given `(V, ρ, D_d)`; read `V` by marginalising `{D_d}` (per-document, closed form) and
`ρ` (one top mixture). Tractable: O(L_ρ · D · L_D · m). Tempering disappears — it was
the rank-1 approximation to this marginalisation.

**Engine primitive needed:** a declared **group-noisy-channel kernel** that computes
the per-document group-likelihood `P(extractions | V, ρ)` (marginalising `D_d`
analytically) — an exact categorical-correlation kernel (Dirichlet-multinomial / Pólya
shape), the categorical-leaf analogue of the linear-Gaussian full-covariance conjugate
landed this session. The `ρ`-mixture is an existing `MixturePrevision` + marginalise
(`push`). See §5 Q1 for the carry-vs-integrate choice.

### 2.2 The reaction model — `LogisticReaction` kernel

`utility.py:155-166` computes `P(react=1 | x) = Σ_τ w_τ · sigmoid((sign·x − threshold)/τ)`
host-side over a declared `τ`-grid, then ships it as a `tabular_log_density`. This is
an *exact* model (the grid is the declared `τ`-prior, not an approximation of a
continuum); it is only mis-*located*. Fix: a declared **`LogisticReaction` kernel
family** doing the exact `τ`-grid quadrature engine-side; the body ships
`(sign, threshold, τ-grid, τ-weights)` as declared data. Density:
`logpdf(react | x) = log Σ_τ w_τ·σ(sign·(x−threshold)/τ)` for `react=1`, `log(1−…)` for 0.

### 2.3 The utility-posterior fold — quadrature belief-spec + marginalise

`utility.py` builds a Gaussian-on-grid prior host-side (`exp(-½…)/Z`,
`gaussian_weights`), conditions a categorical grid, then re-sums the joint to
marginalise (`:446`). Exact as a *grid* model; the host construction is the leak.
Fixes (both small, both exact): a **discretised-Gaussian / quadrature belief-spec** so
`create_state` builds the grid prior engine-side; and the **`push`/`marginalise` verb**
(marginalisation is `push` through a `Projection` — an existing axiom op with no wire
verb today) so the joint→per-latent marginals happen engine-side, not by host re-sum.

### 2.4 The EU rows + narrative inclusion — already / nearly clean

- Action-utility EU (`decide.py:60-64`, `lookup.py:611-655`): already a
  `functional_per_action`/`tabular` preference to `optimise`; the expectation is the
  engine's. **No work.**
- Narrative inclusion EU (`narrative.py:276-316`): `p²·u_c + p(1−p)·u_w − κ` is
  arithmetic on the credence `p`. Express as a typed **quadratic functional** over the
  claim Bernoulli (an existing-or-small `LinearCombination`-of-`Projection`-products
  extension); the host stops touching `p`.
- Realised-utility scoring (`gate.py:120-137`): non-causal eval diagnostic — out of
  invariant scope, stays host-side.

## 3. Files touched

### Engine (credence repo)
- `src/kernels.jl` / `src/conjugate.jl` — the group-noisy-channel kernel family (§2.1)
  + conjugate/predictive; the `LogisticReaction` family (§2.2); self-register in
  `FAMILY_REGISTRY` so BDSL `:family` reaches them.
- `src/stdlib.jl` — the quadratic claim functional (§2.4) if not expressible by the
  existing `LinearCombination`.
- `apps/skin/server.jl` — `build_kernel` specs for the two kernels; a
  discretised-Gaussian/quadrature belief-spec in `build_prevision` (§2.3); a
  `marginalise` (push-through-`Projection`) verb in `SKIN_METHODS` (588) + dispatch.
  `PROTOCOL_VERSION` bump.
- `apps/skin/protocol.md` — the new verb + kernel specs + version/changelog.
- `test/test_group_noisy_channel.jl`, `test/test_logistic_reaction.jl`,
  `test/test_marginalise.jl` — exact-oracle, rtol ~1e-12, vs hand-computed values; the
  group kernel's `m=1` degenerate case ≡ the current single-obs noisy channel.

### Consumer (life-agent repo)
- `src/life_agent/core/brain.py:35,147-151` — repoint: `docker run -i
  credence-skin@<digest>` argv reusing `SubprocessTransport`; `$CREDENCE_SKIN_IMAGE`
  env, source-tree spawn kept only as a dev fallback.
- `src/life_agent/core/lookup.py` — delete `temper_scales`; `lookup_posterior` declares
  the group-noisy-channel kernel per document (+ the `ρ` latent) and conditions once
  per document; `observation_densities` becomes the declared kernel's data, not a
  host-computed log-density.
- `src/life_agent/core/utility.py` — declare the `τ`-grid `LogisticReaction` kernel and
  the quadrature prior; replace the host fold/marginalise with `condition` +
  `marginalise` verbs.
- `src/life_agent/core/narrative.py` — ship the quadratic claim functional to `expect`.

## 4. Worked end-to-end example

One document `d` with `m=2` corroborating chunks reporting candidate `j`, extractor
reliability prior `ρ ~ Beta`, under the exact model:

```
-> create_state {"type":"mixture","components":[ <one categorical over V per ρ-grid point> ],"log_weights":[ <P(ρ)> ]}
<- {"state_id":"s_v_rho"}
; one document = ONE group observation (its 2 chunks), not two independent conditions
-> condition {"state_id":"s_v_rho","kernel":{"type":"group_noisy_channel","authority":…,"subject":…,"time":…,"counts":{"j":2}},"observation":<doc d>}
<- {"state_id":"s_v_rho","log_marginal":…}
; read V by marginalising ρ
-> marginalise {"state_id":"s_v_rho","keep":[0]}        ; project to the V coordinate
<- {"state_id":"s_v"}
-> optimise {"state_id":"s_v","actions":…,"preference":{"type":"functional_per_action",…}}
<- {"action":"report","eu":…}
```
The body ships document-level extraction *counts* + covariates + the ρ prior; the
engine computes the correlated group-likelihood and the marginalisation. No
`temper_scales`, no host `math.log`.

## 5. Open design questions

1. **Carry vs. integrate the latents (load-bearing).** For the *within-document* `D_d`:
   integrate analytically (the group-noisy-channel kernel; query-local, no hypothesis
   growth) — recommended, since `D_d` is not reused across queries. For the *extractor*
   `ρ`: carry it as an explicit shared latent and marginalise (recommended — it is
   reused across queries and the audit outcomes are exactly its evidence, so carrying
   it *replaces* the "ρ moved by audits" approximation with conditioning). The fork is
   whether `ρ` is elevated **now** (the exact `s_mod` removal requires it) or whether
   Move 1 ships the `D_d` group-kernel first and `ρ`-as-latent is a fast-follow — i.e.
   is "remove `s_anc` exactly, keep `ρ` a conditioned point for now" an acceptable
   *interim* that is still strictly more correct than tempering, or does "do it all"
   require both terms gone in this move? *Lean: both — but sequence the group-kernel
   commit before the ρ-latent commit so each lands with its own exact-oracle test.*
2. **Group-kernel form.** The categorical correlated-group likelihood (marginalising
   `D_d`) — Dirichlet-multinomial vs. an explicit small-`L` mixture over a discrete
   `D_d`. *Lean: explicit discrete `D_d` mixture* (transparent, matches the noisy-channel
   semantics, exact), with Dirichlet-multinomial noted as the continuous limit.
3. **`marginalise` verb shape.** A general `push`-through-`Projection`/`NestedProjection`
   verb (reusable; also closes `utility.py:446`) vs. a narrow lookup-only helper.
   *Lean: general `marginalise` verb* — it is an existing axiom op (`push`) merely lacking
   a wire surface, and three sites want it.

## 6. Risk + mitigation

- **The exact model changes the posterior** (it should — tempering was wrong).
  Mitigation: characterise the *intended* divergence (the group kernel at `m=1` and `ρ`
  at its point value reproduces the current single-obs path bit-for-bit; divergence
  appears only with `m>1` / uncertain `ρ`, and is the correction). Pin the `m=1`/point-`ρ`
  reduction as an exact-equality test; assert the `m>1` direction (corroborating chunks
  move the posterior *less* than `m` independent docs would) against a hand-computed
  oracle.
- **New kernel families are exact-but-untested.** Mitigation: rtol ~1e-12 exact-oracle
  tests per family (`feedback_tightest_invariant_in_tests`), incl. the degenerate
  reductions.
- **Repoint breaks on image/protocol skew.** Mitigation: `brain.py` pins the image by
  digest + asserts the `initialize` protocol major; life-agent's suite spawns the image
  and reproduces golden `decide`/`lookup` outputs.
- **Scope creep via `ρ`-as-latent.** Mitigation: §5 Q1 sequences it as its own commit;
  if review prefers, it splits to Move 1b without leaving the body doing arithmetic
  (the interim conditions a point-`ρ`, still tempering-free for `s_anc`).

## 7. Verification cadence

`julia test/test_group_noisy_channel.jl` + `test_logistic_reaction.jl` +
`test_marginalise.jl` + `test_core.jl`; `credence-lint check apps/` clean;
`python -m apps.skin.test_skin` (new kernels/verb + protocol bump + backward compat);
life-agent: `uv run pytest` with `brain.py` spawning the pinned image; golden-value
parity on `decide`/`lookup`/`utility`; a grep gate asserting no `math.log`/`math.exp`
on beliefs remains in `src/life_agent/core/` (eval-only `gate.py` excepted).

## 8. de Finettian discipline self-audit

1. **Every numerical query through `expect`?** Yes — the lookup posterior is built by
   `condition` + `marginalise` (`push`); EU through `optimise`; the reaction likelihood
   is a declared kernel `condition`. No host `Float64` belief query remains.
2. **Prevision-in-Measure / Measure-in-Prevision?** No new nesting; the `ρ` mixture is a
   `MixturePrevision` of categoricals (Prevision-in-Prevision via the existing mixture).
3. **Opaque closure where declared structure fits?** No — the correlation is a declared
   group-kernel family + a declared `ρ` latent; the reaction is a declared
   `LogisticReaction`; tempering (an undeclared host exponent) is *removed*.
4. **`getproperty` override on a Prevision subtype?** No.

---

## Reviewer checklist
- [ ] §2.1 states the exact correlated-evidence model, not a re-hosted tempering.
- [ ] §5 Q1 wrestles with carry-vs-integrate and the `ρ`-now-or-1b sequencing.
- [ ] New kernels carry exact-oracle tests incl. degenerate reductions.
- [ ] §8 yes-or-justified on all four.
- [ ] The body ends with zero belief arithmetic (the grep gate in §7).

---

## 9. Landed — the deferred commit-3 finish (3c/3d, protocol 1.7)

The §2.3/§2.4 pieces deferred at first pass landed together once the body rewire pinned
their shapes, with two refinements over the original sketch:

- **§2.4 claim functional → reuse `CenteredPower{n}`, not a new type.** The integrated
  claim-inclusion EU is `E_θ[θ·u_assert(θ)] − κ = (u_c−u_w)·E[θ²] + u_w·E[θ] − κ`, a
  `LinearCombination([(u_c−u_w, centered_power n=2), (u_w, identity)], offset=−κ)`. The
  existing `CenteredPower{n}` (the central/raw moment test function) gains an exact
  closed-form `expect(::Beta{Measure,Prevision}, ::CenteredPower)` (raw moment via rising
  factorials), so the claim decision is `optimise{include,withhold}` over the cell Beta —
  integrated **exactly**, keeping the `Var(θ)·(u_c−u_w)` term the body's point-estimate `p̄²`
  dropped. No new functional type; composes with the existing `linear_combination` spec.
  This is the [[feedback_decouple_express_proper_model_not_port]] principle in practice — the
  decoupled model is the *proper* integral, not a port of the host's plug-at-the-mean.
- **§2.3 `marginalise` verb shape → `{state_id, shape, axis}`, not `keep:[...]`.** The body
  builds one flat row-major (C-order) product-grid categorical; `shape` (per-axis sizes) +
  `axis` names the coordinate projection unambiguously and the engine sums out the rest —
  the pushforward through π_axis, server-side. One axis per call (the fold reads each latent's
  marginal in turn). `marginalise` lives in `src/stdlib.jl` (the canalised `push` path); the
  skin verb only marshals (lint-clean, no apps/ arithmetic).

Verification: `test/test_centered_moment.jl` (exact moments + integrated claim-EU Wald
flip + marginalise row-major folds + guards); `apps/skin/test_skin.py`
`test_centered_power_claim_eu` + `test_marginalise_joint_grid` (both over the wire,
zero client arithmetic); engine regression (routing/structure-BMA/typed-decision/core)
green; `check apps/` clean (180 files); header==const(1.7).
