# Winner's-curse pricing — the lookahead as a virtual injection (the machinery applied to itself)

> Exploration-budget arc, follow-up to removal consumption (#192/#193) and the §8 measure–utility
> alignment (#194). Design-doc-before-code; ratify before any code lands. Constitutional
> grounding: `CONSTITUTION.md` (Tractatus Credentiae, 2nd ed.), cited inline as T-x.y. The
> acceptance criteria were **fixed in advance** by dominance-design §8: worst-seed (minimax)
> mean-rate gap ≥ 0 and the q10 gap vs `random_p005` and `fixed_k50`, with mean-parity CIs and
> the passing headline preserved. Authored 2026-07-04; **revision 3**. The revision history is
> part of the design's argument and is kept in §0.

---

## 0. Revision history (the argument, compressed)

- **Draft 1 (rejected):** a bespoke spike-and-slab over candidate rates with uniform π, in a new
  file — a second inference mechanism, stipulated rather than derived. The constitutional review
  (author + pixel6) demolished it: the posterior it reached for is the marginal likelihood under
  the complexity prior; the BMA **sum** prices selection/multiplicity (Scott & Berger — a proper
  prior over model space *is* the multiplicity correction); `2^{−Δ|G|}` *is* the spike (uniform π
  discards the informative prior odds); and the reflective-boundary cover belongs to the
  **un-entertained** ops only (escape — learned returns are the honest floor there) — growth
  candidates are enumerated and fit, so exact model comparison is available and mandatory
  (T-2.32). Bellwether stakes: admit one bespoke corrector and a bespoke drift detector follows;
  the framework dies of a thousand locally-reasonable patches.
- **Revision 2 (superseded):** derived the right *formulas* — (A) evidence as the
  marginal-likelihood ratio `log(π̃ + Σ_c π̃_c e^{fit_c})`, (B) flow as the posterior-weighted
  per-event gain — but implemented them as hand arithmetic beside the engine: a hand-rolled
  logsumexp over hand-normalised grammar priors, feeding the old valuation seam. The author's
  follow-up challenge ("can't we use the Bayesian machinery of the credence engine on itself?")
  exposes the residue: π̃ over the entertained grammar set is still a stipulated normalisation,
  and the formulas duplicate — outside the engine — computations the engine already performs.
- **Revision 3 (this document):** the formulas dissolve into **engine queries**. The lookahead
  becomes a *virtual coherent injection* on a scratch state — the same Tier-1 code path as the
  transition — and every score component is an existing canalised read (`log_predictive`,
  `probability(·, TagSet)`, `expect`, `injection_yield_nats`). No new inference mechanism, no
  grammar-prior normalisation (programs are the atoms; the two-part complexity prior is already
  total over them), and score/transition unity holds **by construction** because the score is
  computed by running the transition virtually.

- **Revision 4 (the ratified amendment — see §8):** TDD deleted the flow. The §1b flow story
  is unsatisfiable by theorem: any forward per-event flow taken in the union's own posterior
  expectation equals `KL(P_union ‖ P_inc) ≥ 0` per context (Gibbs) — a Bayesian mixture cannot
  expect its own enlargement to score worse by its own lights, so no derivation delivers
  "zero-or-negative on chance windows", and `× H` amplifies whatever positive remains (measured:
  a clean 4-event chance fit fires at +0.55 nats with P_newcomers ≈ 0.11 — failing this
  design's own mechanism claim (i)). The resolution is a further deletion: the ratified yield
  observable **is** the union-over-incumbent window Bayes factor
  (`log((1−m₀)/(1−m)) = log[(1−m₀) + m₀·BF]`), so the score is `net_value(yield, op_cost)` —
  fire when the realised evidence clears the declared price. No flow, no `× H`, no plateau at
  the growth seam. Rev 3's one genuinely new computation is retired; every score read now
  pre-existed this design.

## 1. Purpose

Make the growth ops (`:gw_explore`, `:gw_add_feature`) score and transition through **one pass of
the machinery the agent already trusts**, eliminating the point-max-extrapolation score whose
three implicit inferences drive the round-4 gate's remaining tail failures (worst-seed mean-rate
gap −0.5 / −0.381, q10 −0.476 / −0.333; the instrumented shape: `add_feature` fires at steps 4–8
on 2–6-event windows).

**The bug (diagnosis unchanged since draft 1).** The current score
`plateau · (max_c fit_c / n_buf) · H + prior_term − cost` performs, with unearned certainty:
**(1) selection** — the scored fit is an order statistic over K chance-fitting candidates (a max
where Bayes says sum); **(2) extrapolation** — a realised log Bayes factor over 2–6 events treated
as the known future per-event rate and multiplied by H ≈ 200 (independent of selection: a K = 1
chance fit of 1.1 nats scores ~26 nats and fires); **(3) collapse** — the op installs the argmax
candidate scored at face value ("pick the winning grammar" — the `average-not-collapse` error at
the meta seam).

**The fix (rev 3): the lookahead IS a virtual injection.** The agent already possesses the exact
machinery for "what would I believe if I entertained these candidates?" — it is the coherent
injection (#187): build the candidates' programs, seed them at their **two-part complexity
prior** (`2^{−|G|−|p|}` — the general prior, total over programs, no grammar-level normalisation
to stipulate), condition them on the evidence window through Tier-1 `condition` (the two-ledger
replay), and hold the union posterior that the counterfactual union-from-start agent would hold.
The growth score is then **queries against that virtual state**, and the transition — if the op
fires — is **keeping it**:

    1. SCRATCH:  st′ = copy(state);  add_programs_to_state!(st′, candidates…, observations = buffer)
                 — the union belief, coherently conditioned; the SAME code path as the transition.
    2. SCORE:    evidence = injection_yield_nats(st′, n_added)          [the ratified observable]
                 flow     = expect-based per-event predictive gain of st′.belief over state.belief
                            on the window's feature records              [the ×H license, §1b]
                 score    = plateau · flow · H − compute_cost           [growth_value, unchanged]
    3. FIRE:     state ← st′  (adopt the scratch — zero recompute; score ≡ transition, T-3.55
                 by construction, not by a shared candidate function)

**Why each pathology dies, and by whose hand:**

- **Selection** is priced by the **program-level complexity prior + likelihood**, with no π̃: all
  K candidates' programs enter *one* mixture; duplicates dedup (the existing `expr_equal`
  discipline — threshold refinements share most of their language); each genuinely-new program
  arrives at `2^{−|G|−|p|}` and earns only the mass the window likelihood grants it. A lucky
  program among thousands holds a lucky-program-sized posterior share — the multiplicity
  correction is the mixture's normalisation doing its ordinary job. The max-over-K *disappears
  as a mechanism*: there is no per-candidate argmax to curse, because the candidates are not
  compared — they are **carried** (`average-not-collapse`, applied to the transition itself).
- **Extrapolation** is licensed by the posterior, not the window rate. The flow is the
  posterior-weighted expected per-event predictive gain of the union belief over the incumbent
  belief — an `expect` query against declared functionals on the window's feature records (the
  declared event measure, §5 Q2). Its incumbent-side term is **negative** (carrying junk hedges
  the predictions — the real cost the tail seeds pay), so on a chance window, where newcomers
  hold ≈ prior mass, the flow sits near zero-or-negative and no ×H amplifies it. The conjugate
  smoothing bounds what any tiny window can prove (a "perfect" 4-event program's Bayes factor is
  ≈ e^{1.6}, not e^{26}).
- **Collapse** dies because nothing is collapsed: the transition adopts the union. "Install the
  argmax candidate" (the current transition, and rev 2's Q2 recommendation) is revealed as an
  `argmax_m P(m|D)` collapse the constitution already names illegal for decisions; the engine-
  native transition lets `condition` arbitrate and the **existing** hygiene (`sync_prune!`,
  `sync_truncate!`, and #193's replacement consumption for features that go dead) clean up —
  no new mechanism there either.

**The GrowthReturns symmetry (T-3.53, now exact).** Escape ops (`enumerate_more`, `deepen`)
cannot enumerate what they would find — the genuine reflective boundary — so they **learn** their
yield posterior (Gamma × Exponential cells, #189). Growth ops **can** enumerate their candidates,
so they **compute** the same yield observable exactly, by virtually performing the injection.
One observable (`injection_yield_nats`), one currency, two fidelities — learned where the exact
is unreachable, exact where it is affordable. The first draft's corrector stood exactly where
this symmetry belongs.

**Compute (why this is not dearer).** Today the lookahead runs K *separate* full-mixture replays
(`_grammar_marginal_log_loss` per candidate — Σ_c |G_c| component-events). The virtual injection
runs **one** replay over the deduped union of the candidates' *new* programs (threshold siblings
share most of their language; the union is far smaller than Σ_c |G_c|), and the newcomers-only
replay is the #187 construction (incumbents are already conditioned). For large candidate sets
the scratch state's own `sync_truncate!` bounds the mixture — mass-based truncation as the
metareasoned fidelity knob (T-3.53), an existing discipline, not a new cap. If a host still finds
it dear, the legitimate exit is an approximation *to this object* selected by the compute
cascade — never a different quantity.

**What this move does not touch — the fixed horizon (recorded stationarity assumption).** The
`× H` multiplier is deliberately left unchanged: with an honest, shrunk `flow`, the horizon no
longer over-amplifies a chance window (the flow sits at zero-or-negative there and no multiplier
rescues it), so the winner's-curse scope ends at what flows *into* `growth_value`. But `H ≈ 200`
is a **gain-stationarity assumption wearing a constant's clothes**: `flow · H` asserts that the
per-event gain just measured persists for two hundred events, and in a changing world a regime
change invalidates the feature long before the horizon elapses — every discovery is over-valued
by the ratio of the fixed horizon to the expected time-until-it-stops-paying. Whether `H` is a
hardcoded scalar or a host-given task length does not change the conclusion: the correct
multiplier is the **effective horizon**, `E[events until the next change-point]`, which is
belief-derived, not stipulated. That is the seam where the deferred ρ-as-latent / change-point
machinery attaches — the same anti-stipulation discipline as this document, one level up (regime
as latent structure, marginal likelihood adjudicating switches, horizon falling out of the change
belief rather than a constant). Recorded here as the named successor residual so the fixed `H`
cannot ossify into received truth; this move is not held for it.

## 2. Files touched

- **`src/program_space/exploration.jl`** (modification; net **negative** or near-zero lines —
  the deletion is the point).
  - `_best_threshold_refinement` / `_best_feature_addition` (the per-candidate mll argmax loops)
    are **retired from the scoring path**: replaced by `_virtual_injection(state, candidate_gs,
    buffer; …) → (scratch_state, n_added)` — a thin orchestration of `copy` +
    `add_programs_to_state!` (both existing) over the candidate grammars (`_threshold_candidates`
    → `_refine_grammar` per midpoint; `_feature_candidates` → `_add_feature` — generators
    unchanged), and `_growth_flow(scratch, state, buffer) → Float64` — the `expect`-canalised
    per-event predictive-gain query (§5 Q2 fixes its declared functional form).
  - `exploration_voi` / `exploration_fit` / `feature_discovery_voi` / `feature_discovery_fit`
    keep their signatures (host seam unchanged); internally they project the virtual-injection
    score. `explore_grammar` / `explore_features` (the transition surface) return the union-
    bearing edit rather than the single winner — the host adopts the scratch (§5 Q1 fixes the
    exact hand-off shape; the parallel arrays make "adopt" a field swap).
  - Deleted: the per-candidate argmax bookkeeping and rev 2's would-be logsumexp/π̃ arithmetic
    (never written). `selection_pricing.jl` (draft 1) is never written.
- **`apps/julia/grid_world/host.jl`** (modification, small): the growth branches of
  `execute_gw_meta_action!` adopt the scratch state instead of installing one candidate grammar;
  the per-fire op log gains the `(yield, flow, P_newcomers)` triple (§7's mechanism claim reads
  it). Score seam (`score_gw_meta_actions`) unchanged.
- **`test/test_virtual_injection.jl`** (new, ~6 sections): §1 score/transition identity — the
  scored scratch and the adopted state are the same object (`===` on the belief; the T-3.55 pin
  is now an identity, not an equality); §2 the seed-6 regression — the §4 fixture's chance window
  scores below the fire floor at H = 195 (old score ~26 nats; counter-oracle pinned); §3 the
  informed fire — a genuinely separating candidate's flow within tolerance of its true rate
  (the step-90 case; the headline's fires preserved); §4 multiplicity-by-mixture — K chance
  candidates' union earns no more mass than one chance candidate's (the dedup + prior arithmetic,
  against a hand-built oracle); §5 commutation inheritance — the virtual injection is
  `add_programs_to_state!`, so #187's §1 equality covers it (asserted on the scratch);
  §6 hygiene — a spurious fired union's dead features become #193 replacement candidates once
  their programs' mass collapses (the self-healing loop, end-to-end).
- **`test/test_threshold_explore.jl`, `test/test_feature_discovery.jl`,
  `test/test_grid_world_meta.jl`** (re-baselines): score pins move to the new form; structural
  assertions (gates, floors, no-op identity) unchanged. The §1-of-meta pins re-derive from the
  virtual-injection oracle.

## 3. Behaviour preserved

- **The no-op paths:** empty buffer / no candidates / score below floor ⇒ state untouched,
  bit-identical (the scratch is discarded; `copy` never mutates the original — pinned).
- **Commutation:** the virtual injection is the #187 code path; `test_coherent_injection.jl` §1's
  equality is inherited, asserted again on the scratch (§2's test file, §5).
- **Asymptotic identity:** when one candidate's evidence dominates, the union's posterior
  concentrates on it and the flow approaches that candidate's realised rate — the current score's
  informed-fire behaviour is the limit (pinned §3). The headline's mid-episode fires are expected
  unmoved (gate re-run verifies via the mean-tier CIs).
- All non-growth scoring (escape cells, perturb replacement value, do-nothing floor): bit-stable;
  `test_growth_returns.jl`, `test_replacement.jl`, conjugate/threshold suites pass untouched.
- Skin wire, persistence schema, benchmark harness: untouched (§8 measures fixed; this move is
  judged against them).

## 4. Worked end-to-end example

The tail-seed shape (seed 6: `add_feature` at step 6, final rate gap −0.381), traced through the
virtual injection.

1. **Step 6, buffer n = 4, K = 5 feature candidates.** `_virtual_injection` builds the five
   feature-added grammars' enumerations, dedups against the incumbents and each other, and
   coherently injects the genuinely-new programs (~300 after dedup) into a scratch copy at their
   two-part complexity priors, replaying the 4-event window through Tier-1 `condition`. The
   conjugate smoothing caps every newcomer's Bayes factor at ≈ e^{1.6} on 4 events; the
   complexity prior starts each at `2^{−|G|−|p|}`. Result: the newcomers' collective posterior
   mass barely exceeds their prior counterfactual — `injection_yield_nats(scratch) ≈ 0.3` nats
   (the ratified evidence-relative observable, computed exactly instead of learned).
2. **The flow query.** `_growth_flow` asks, via `expect` on the window's feature records: by the
   union's own posterior, how much better per event does the union predict than the incumbent —
   weighted by who is probably right? The incumbent-dominated posterior (newcomers at ≈ prior
   mass) makes the hedging term dominate: `flow ≈ +0.01` nats/event (the fixture pins the exact
   value). `score = 0.5 · 0.01 · 195 − log 2 ≈ 0.3` nats — **below the floor. The op waits.**
   Under the old score: `0.5 · (1.3/4) · 195 − 2·log 2 ≈ 26` nats — fired, installed `:speed`,
   diluted the posterior, and the seed never recovered.
3. **Step ~90, n = 30, regime 2 (`:speed` genuinely predicts).** The same virtual injection now
   finds the `:speed` programs earning e^{6} Bayes factors over 30 events; the union posterior
   concentrates on them; `flow ≈ 0.2` nats/event ⇒ score ≈ the current score ⇒ **fires** — and
   firing means *adopting the already-computed scratch*: the belief the score priced is the
   belief the agent now holds. No re-enumeration, no re-conditioning, no winner to pick — the
   wrong-but-entertained sibling candidates ride along at their (tiny) earned mass and are pruned
   by the existing hygiene within a few steps; a feature that goes dead later is consumed by
   #193's replacement machinery. The system self-heals through mechanisms that already shipped.
4. Ownership: candidate generation (`exploration.jl`, unchanged generators); the injection and
   its ledgers (`agent_state.jl`, #187, unchanged); the yield read (`growth_returns.jl`, #189,
   unchanged); the flow query (`exploration.jl`, new — the only genuinely new computation, and it
   is an `expect` call); adoption (host, a field swap).

## 5. Open design questions

1. **The adopt hand-off shape.** The scratch state must become the live state without copying
   costs or aliasing hazards. Options: (a) `execute_gw_meta_action!` receives the scratch from
   the scorer via the host's memo (the voi_cache pattern, keyed by epoch — zero recompute,
   recommended); (b) recompute the injection at execute time (pays the replay twice, but
   stateless — the fallback if (a)'s cache-epoch discipline gets hairy). The T-3.55 identity test
   (§2 file, §1) is written against whichever is ratified.
2. **The flow functional's declared form.** `flow = Σ_events w · [log P_union(o|·) −
   log P_inc(o|·)]` under the union's posterior — the event measure is the window's feature
   records (data the host already provides; recommended), vs recency-weighted (adds a tunable —
   smells like a knob) vs a declared feature grid (invents a measure). The inner reads are
   per-component firing evaluations — the `FiringChoice` family; the doc wants ratification that
   the window records are the honest declared measure (they are the same records the injection
   replays — one event measure everywhere).
3. **Union breadth for thresholds.** Feature candidates are few (K ≈ 5); threshold candidates can
   be tens of midpoints. Inject all (the pure form; dedup + truncate bound it) vs a mass-based
   pre-screen (the residual-ordering already ranks candidates; injecting the top-m by residual
   mass with m set by the compute price is a T-3.53 fidelity decision, not a value decision).
   Recommend: all, measured; pre-screen only if the gate's wall-clock says so (and then as a
   priced fidelity, logged).
4. **Does `GW_FEATURE_PRIOR_TERM` survive?** No — the −log 2 prior charge is now carried by each
   newcomer's own complexity prior inside the mixture (the injection arithmetic), so the explicit
   `prior_term` at the score seam would double-charge. It is deleted with a no-double-charge pin
   (the §2 test's §4 oracle covers it). Named here because it retires a ratified #188 coordinate:
   the ratification of THIS doc supersedes it.

## 6. Risk + mitigation

- **Under-exploration (over-shrink).** The flow's hedging term could suppress genuine early
  fires. Mitigation: the asymptotic-identity pin (§3); the headline CI as the acceptance
  tripwire; halt-the-line on degrade. The flow is derived, not tuned — if it over-shrinks, the
  finding is about the declared event measure (§5 Q2), and the response is a derivation fix.
- **State-size growth from union adoption.** Firing injects the union (~hundreds of components),
  not one candidate's (~80). Mitigation: `sync_truncate!`'s existing mass-based cap; junk
  candidates' programs sit at ≈ prior mass and are first out; #193 consumes features that go
  dead. §4 of the test file pins that K chance candidates earn no more collective mass than one.
  Residual risk: transiently larger mixtures between fire and prune — measured in the gate
  re-run's wall-clock, priced via §5 Q3 if real.
- **Cache/epoch discipline for the adopt hand-off** (§5 Q1's risk): a stale scratch adopted after
  the state moved would violate coherence. Mitigation: the voi_cache epoch already invalidates on
  every space change; the scratch memo keys on the same epoch; the §2-file §1 identity test runs
  through the host seam, not just the engine.
- **Compute regression.** One deduped union replay vs K separate replays — expected cheaper, not
  dearer, but *measured* (the benchmark logs per-policy wall-clock; the ~2.5-min suite budget is
  the tripwire). The T-3.53 exit (§5 Q3) is specified in advance.
- **Benchmark comparability.** The growth ops' *transition* changes for every policy (fixed
  schedules and random fire the same ops). Same situation as #193: a fresh baseline, stated in
  the results commit; the falsifiable claims are mechanism-level, measure set fixed by §8.

## 7. Verification cadence

    julia test/test_virtual_injection.jl        # new — score≡transition identity, seed-6
                                                #   regression, informed fire, multiplicity-by-
                                                #   mixture, commutation inheritance, self-healing
    julia test/test_coherent_injection.jl       # untouched pass (the inherited §1 equality)
    julia test/test_threshold_explore.jl        # re-baselined pins
    julia test/test_feature_discovery.jl        # re-baselined pins
    julia test/test_grid_world_meta.jl          # re-baselined score pins
    julia test/test_growth_returns.jl           # untouched pass
    julia test/test_replacement.jl              # untouched pass
    for f in test/test_*.jl; do julia "$f"; done
    JULIA_PROJECT=$PWD PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run python apps/skin/test_skin.py

Gate re-run (20 seeds, background), judged against the §8 measure set **fixed before this design**:

- **Target claims:** minimax worst-seed mean-rate gap vs `random_p005` and `fixed_k50` rises from
  −0.5/−0.381 toward ≥ 0; q10 from −0.476/−0.333 toward ≥ 0. Falsifiable mechanism claims:
  **(i)** no eu_max growth fire whose virtual union was incumbent-dominated (the op log's
  per-fire `(yield, flow, P_newcomers)` triple); **(ii)** post-fire mixtures return to within the
  truncation cap within a bounded number of steps (the self-healing claim, from the component
  logs).
- **Must-not-degrade:** the headline (rate +0.150 CI > 0, 19–1) holds; mean-tier CIs vs the tuned
  baselines do not become significantly negative; the bracket holds; suite wall-clock within
  budget.
- Halt-the-line on any degrade; results commit honestly either way.

## 8. Amendment (revision 4, ratified 2026-07-03): the yield is the score

**Supersedes:** §1's SCORE step and flow bullet (§1b), §5 Q2 (the flow functional's declared
form — the question dissolves with the flow), and — beyond this document — two coordinates of
the belief-derived-valuation design (#188 §2a) at the growth seam: the `plateau · fit · (H /
n_buf)` horizon-completion and the plateau multiplier. Everything else in rev 3 stands
unchanged: the virtual injection as the transition, FIRE = adopt the scratch (score ≡
transition as identity), programs-as-atoms multiplicity pricing, Q1 scratch memo, Q3
inject-all, Q4 no-double-charge, §5-commutation inheritance, §6 self-healing.

### 8.1 The finding (Gibbs kills the flow)

Rev 3 warned in §1 that scoring growth by "the expected predictive gain of the enlarged
belief" is degenerate — and §1b then used exactly that quantity under the union's own
posterior view. For any world-measure the union itself endorses (including the "who is
probably right" group decomposition, which is algebraically the union's own predictive):

    E_world[ log P_union(o) − log P_inc(o) ] = KL(P_union ‖ P_inc) ≥ 0        (Gibbs)

The hedging term is real — `−(1−m)·KL(P_inc ‖ P_union)` — but can never dominate its own KL.
Adding hypotheses is weakly informative from inside, always. So the flow is a fixed-sign
quantity multiplied by a horizon: chronic over-fire, mechanically. Measured on the §4-shaped
fixture (test_virtual_injection.jl §2): retired argmax +5.12 nats; rev-3 flow (retrodictive
read) +12.92; rev-3 flow (group-conditional read) +0.55 — every form fires the chance window.

### 8.2 The score

    score = net_value( injection_yield_nats(scratch), op_compute_cost )

The identity that makes this the honest object: with `m₀` the newcomers' prior-counterfactual
mass and `m` their posterior mass in the scratch,

    injection_yield_nats = log((1−m₀)/(1−m)) = log[ (1−m₀) + m₀ · BF_window ]

— the **marginal-likelihood ratio of the enlarged space over the incumbent space on the
window, under the complexity prior**: the exact quantity the draft-1 review named as what the
machinery already computes. Fire when the realised evidence clears the declared price.

**Why no horizon term — the wait-option argument (what earns the deletion of `× H` rather
than asserting it).** The naive score prices adopt-now against hold-the-incumbent-forever;
that comparison genuinely scales with H. But the real choice is adopt-now against
wait-and-re-decide, and the long-run value is common to both arms: if the signal is real the
wait-arm adopts a few events later (forgoing a few events' gain); if it is junk the adopt-arm
prunes a few events later (paying a few events' hedge). The horizon-extrapolated term
cancels, and what survives is a timing question governed by the realised evidence — a
Gittins-style decomposition: the value of committing now reduces to a near-term index
(yield − cost) because the far-term value factors out.

**Why no plateau.** The yield is realised posterior evidence that the gain is real;
multiplying by P(plateau) would charge that doubt twice. This supersedes the #188 coordinate
that held the reality-of-gain and duration-of-gain multipliers orthogonal to the fit: once
the score is realised yield, the plateau question is subsumed at this seam. (Whether the same
argument retires plateau elsewhere is a separate finding with its own gate — out of scope.)

**T-3.53, exact in form.** Escape ops score `E[next yield] − cost` (learned, GrowthReturns);
growth ops score `computed yield − cost` (exact, the virtual injection). One observable, one
currency, **one score form**, two fidelities.

### 8.3 The two named assumptions (so nothing hides)

1. **The cost slot is a price, not a threshold.** `op_compute_cost` (= log 2 by default) is
   the #189-inherited declared compute price (ratified #188 Q6: prices are utility data),
   passed uniformly to every benchmark policy; it predates this design. The discipline it
   carries forward: any adjustment is a declared-data change ratified in advance — tuning it
   to move the gate would be the threshold this design spent three revisions deleting,
   walking back in through the cost slot. No escape hatch.
2. **The wait-arm regeneration assumption (the named bridge to non-stationarity).** The
   wait-option cancellation holds iff the wait-arm can re-accumulate the evidence — i.e.
   persistent signals re-evidence themselves faster than the buffer scrolls. True for every
   signal worth capturing (a genuinely predictive feature keeps throwing off evidence); false
   only for transients not worth the carry. It is exactly the assumption a regime change
   stresses, and it is where the deferred ρ-as-latent / change-point work attaches. Note the
   direction of fit: the horizon-free yield rule is *better* suited to a changing world than
   the `flow · H` it replaces — it fires on the evidence a signal is currently throwing off
   and lets the existing prune/replacement hygiene retire the signal when the regime moves
   and it stops earning. Fire on present evidence, heal on its absence. The §1 fixed-H
   stationarity residual is thereby DISCHARGED at the growth seam — not by pricing the
   horizon better but by deleting the term that carried it. (`H` bookkeeping remains only
   where a consumer still declares it; nothing at this seam multiplies by a horizon.)

### 8.4 Consequential edits

- `GrowthProposal` carries `(scratch, n_added, yield_nats, p_newcomers)`; the flow and fit
  fields never ship. The per-fire mechanism log is the pair `(yield, P_newcomers)` plus
  `n_added`; §7's mechanism claim (i) reads P_newcomers.
- `_growth_flow` is never written. The host score seam is
  `net_value(prop.yield_nats, op_compute_cost)`; `growth_value` remains in the stdlib for
  consumers with an honest persistent-rate claim, but the grid host's growth tier no longer
  calls it, and the run loop's declared-horizon estimate (`h_events`/`n_cond_events`) retires
  with it (nothing consumed it).
- The `:gw_add_feature` attribution gate re-expresses as "refinement fires first": features
  are gated `-Inf` iff the threshold proposal's own score clears the floor (`yield > cost`),
  not on any positive fit — under the union mechanism a chance-positive threshold yield no
  longer measures an attribution confound, only a competing fire.
- test_virtual_injection.jl §2/§3 pin the yield rule (the chance window WAITS at −0.56; the
  informed window fires from n ≈ 8 at +1.15); the §2 counter-oracle keeps pinning that the
  retired argmax fires (+5.12) on the same fixture.

### 8.5 The boundary framing (why a rule exists at all — recorded from ratification)

The infinite-posterior agent needs none of this: discovery is not an action it takes but a
shape its posterior has. That agent is uncomputable. The finite agent holds a posterior over
an entertained set inside an infinite generable space, and every mechanism in this arc is
bookkeeping at that boundary: the winner's curse was the boundary priced by a max; the flow
degeneracy was the carried set asked to evaluate its own edge; the yield rule is the boundary
priced honestly — a READ of the machinery (what conditioning produced when the candidates
were injected) against the compute the metareasoner already tracks. Discovery is inference,
evaluated at the boundary; the only thing that is not free is deciding when the boundary is
worth moving, and that is a metareasoning question — the one question a Bayesian agent cannot
answer by being more Bayesian. The same shape governs the non-stationarity work to come:
regime hypotheses are more programs, priced across the same boundary by the same yield
against the same cost, no bespoke detector.
