# Removal consumption — replacement semantics for applied compression-class removals

> Exploration-budget arc, follow-up to belief-derived valuation (#188/#189 deviation 3). Design-doc-before-code;
> ratify before any code lands. Master plan: `docs/exploration-budget/master-plan.md`. Predecessors on master:
> coherent injection (#187 — hypothesis addition commutes with conditioning), belief-derived valuation (#189 —
> `:gw_perturb_grammar` provisionally `-Inf` pending exactly this document). Constitutional grounding:
> `CONSTITUTION.md` (Tractatus Credentiae, 2nd ed.) — clauses cited inline as T-x.y. Authored 2026-07-03.

---

## 1. Purpose

Give applied compression-class **removals** (`:remove_rule`, `:remove_feature`) replacement semantics — the
edited grammar **replaces** its ancestor instead of standing beside it — so that the MDL reclaim the score
promises is realised in the belief, the consumed grammar can never re-propose the same edit, and
`:gw_perturb_grammar` re-enters the meta-action argmax it was provisionally `-Inf`-ed out of in #189.

**The bug, constitutionally stated.** Today an applied removal on grammar `g` mints a cleaned **sibling**
`g′` with a fresh id and coherently injects `g′`'s enumeration as *new components*, leaving `g`'s
evidence-rich incumbents untouched. Three violations, one mechanism:

- **T-3.55 (score/transition unity).** `perturbation_voc` scores the edit as `+λ·k` nats of prior reclaim;
  the sibling transition realises *none* of it — the incumbents keep their dirty prior, the newcomers
  duplicate them at near-zero posterior mass. A meta-action valued by one formula and executed by another
  has quietly become two reasoners. This is the same disease #187 cured for injection, one seam over.
- **T-3.52 (compression is re-description).** Compression *re-describes the hypotheses already held* — a
  prior effect. Sibling injection executes a re-description as if it were *exploration* (adding hypotheses):
  a category error, and the source of the duplicate-component dilution.
- **The treadmill (the instrumented failure).** `g` stays registered and evidence-rich, so it stays top-k;
  next step `perturb_grammar(g, …)` proposes the *same* removal, mints *another* sibling, injects *another*
  duplicate set — `+log 2` of phantom VOC forever, 3 ops/step in the smoke run that forced deviation 3.
  Under replacement the ancestor is *consumed*: the candidate disappears because its grammar does. The
  treadmill becomes unrepresentable — the #187 move (make the misspecification inexpressible), applied to
  perturbation.

**The transition (derived, not designed).** Let `g` carry a dead item of payoff `k` symbols,
`g′ = clean(g)`, `Δ = λ·(complexity(g) − complexity(g′)) = k·log 2` (λ pinned to `log 2`, SPEC §1.3;
implementation computes Δ from the two actual complexities, never from assumed `k`). Every live component of
group `G = {i : metadata[i] = (g.id, ·)}` has a program that is expressible unchanged in `g′` (that is what
*dead* means, once the reference count is sound — §2) with an **identical compiled kernel**, hence identical
likelihood on the entire observation history. The counterfactual agent that started with `g′` in place of
`g` therefore holds, after the same history, exactly

    lw′_i = lw_i + Δ·1[i ∈ G]        (unnormalised; Beta posteriors, tags, kernels, programs identical)

because every likelihood term cancels in the ratio and only the grammar-complexity prior term differs. The
replacement transition — re-key `metadata[i ∈ G]` to `g′.id`, shift `log_weights[i ∈ G] += Δ`, delete `g`
from `state.grammars`, register `g′` — is the **unique** map satisfying *replacement commutes with
conditioning*, the same criterion that derived #187's two-ledger injection. It is exact for the **full**
history, not a window: stronger than replaying (which #187 needs because newcomers have no evidence to
cancel), because here the likelihoods cancel identically. Derivation decides the constitution (T-2.81);
§4 traces it concretely and the test asserts it.

**The score (the same function as the transition, T-3.55).** The realised Δ log-evidence of firing, against
the current posterior:

    replacement_value = log(1 + (e^Δ − 1)·W_G)        [nats],   W_G = posterior mass of group G

with `W_G` read canalised as `probability(state.belief, TagSet(group tags))` (the Prevision-level read
landed in #189). Score and transition share one candidate function (§2), so they cannot drift — the
`perturbation_voc`/`perturb_grammar`/`compression_exhausted` shared-`_best_compression_candidate` discipline,
extended to the replacement path. The prior-only `voc = Δ` is this score's `W_G → 1` limit; §5 Q3 argues the
exact form is mandatory here, not optional.

**Resolved decisions** (answers forced; recorded here, not §5):

- **Execution semantics change for every policy, baselines included.** Policies differ in *when* they fire
  meta-actions (selection); the ops themselves are shared machinery (transition). A benchmark whose baselines
  execute a *different, incoherent* op is not comparing policies. The `baseline-comparison` precedent scopes
  Invariant 1 to selection mechanisms, not to op implementations. Gate re-run re-baselines everything.
- **No re-enumeration on replacement.** Removal only shrinks the language (`L(g′) ⊆ L(g)`); there is nothing
  new to enumerate. Replacement adds zero components — `n_added_meta` stays 0 and the host's epoch bump gains
  an explicit `applied_replacement` trigger (§2).
- **No `GrowthReturns` cell for perturb.** The op's consequence is deterministic and exactly computable
  pre-fire; there is nothing to learn (T-2.32: when the exact value is affordable, it *is* the answer).
  Escape ops keep their learned cells; perturb is scored by its realised value directly.
- **`reset_learning_regime!` stays on applied replacement.** Pure re-description leaves likelihoods intact,
  but the cross-group renormalisation jolts the predictive stream the residual regime models; the Q1b
  rationale (a *caused* change-point the agent knows about) applies. Cheap, conservative, and symmetric with
  every other grammar change.
- **The sibling path is retired, not kept beside the replacement path.** One edit, one semantics
  (Invariant 3); a host that wants the old behaviour is a host that wants the treadmill.

**Out of scope, named for the queue:** `:add_rule` consumption (re-expressing incumbents under the new
dictionary — an AST rewrite with per-program Δ; §5 Q2 sequences it); winner's-curse pricing on growth
selection (the remaining worst-seed driver — separate doc); escalate-depth (deepen's re-entry — drafted
separately).

## 2. Files touched

- **`src/program_space/perturbation.jl`** (modification, ~70 lines).
  - `ReplacementCandidate` struct — declared edit: `kind::Symbol` (`:remove_rule | :remove_feature`),
    `payload::Union{ProductionRule, Symbol}`, `payoff_symbols::Int`, `gid::Int`. First-class declared type
    (Invariant 2), mirroring `PerturbationCandidate`.
  - `replacement_candidates(state::AgentState, gid::Int) → Vector{ReplacementCandidate}` — the **sound,
    group-local, live-state** dead-item enumeration: reference sets collected over **every** component of
    group `G` (no weight cut of any kind) unioned with `g.rules` bodies, via the existing
    `collect_nonterminal_refs!`/`collect_feature_refs!` walks. This is the OQ-4 discharge: the
    `SubprogramFrequencyTable` walk cuts support at `w > 1e-15`, which is a *value*-appropriate screen but
    not a soundness guarantee — post-prune components can sit below it (prune keeps relative-to-max
    `> e⁻³⁰ ≈ 9.4e-14`; normalised weight divides by up to `max_components`, so a live component can hold
    `< 1e-15`). Re-keying a component whose program references the removed item would put a program outside
    its grammar's language — unsound, not merely suboptimal — so the soundness predicate must be exact and
    must not share a representation with the frequency estimate (Invariant 3 / T-4.13: one datum, one role).
    Group-local is also *sharper* than the table's global sets: a feature referenced only by *another*
    grammar's programs is dead **for `g`**. The table keeps serving proposal-time value; it no longer
    gates soundness.
  - `replacement_value(state, gid; compute_cost) → Float64` and
    `best_replacement(state, gid) → Union{ReplacementCandidate, Nothing}` — score = `net_value(log1p((exp(Δ)
    − 1)·W_G), compute_cost)` for the best candidate; both route through `replacement_candidates`, so the
    scored candidate and the applied candidate are the same object by construction (T-3.55).
- **`src/program_space/agent_state.jl`** (modification, ~45 lines).
  - `replace_grammar_in_state!(state, cand::ReplacementCandidate) → Grammar` — builds `g′` (threading
    `g.thresholds`, the Move-3 grid-survival discipline), computes `Δ` from the two actual complexities,
    re-keys `metadata`, shifts the group's `log_weights` by `Δ`, reconstructs the `MixturePrevision`
    (constructor normalises — the shift is cross-group, which is the point), deletes `state.grammars[gid]`,
    registers `g′`. Tags, Beta posteriors, kernels, programs untouched — strictly less invasive than
    `sync_prune!`. Docstring carries the §1 commutation derivation and names the test that asserts it
    (executable-documentation discipline). **This is the constitutional write** — see the precedent below.
- **`docs/precedents.md` + `CLAUDE.md` slug index** (modification). New precedent slug
  **`coherent-space-edit`**: a `src/`-resident hypothesis-space edit (inject / re-key / consume) may write
  mixture log-weights **iff** the write is the unique solution of the commutation equation — the state after
  the edit equals the state of the counterfactual agent that held the edited space from the start and
  conditioned on the same history — and the equality is asserted by a named test. Anything short of that
  criterion is a second learning mechanism. Retroactively covers #187's ledger arithmetic (which currently
  leans on `stdlib-composition` by residence); prospectively covers this move. Lands in the same PR that
  relies on it (T-4.5: new escape hatches are constitutional amendments, not inline concessions).
- **`apps/julia/grid_world/host.jl`** (modification, ~30 lines net). Delete the deviation-3 PROVISIONAL
  block; score `:gw_perturb_grammar = max over top-k gids of replacement_value(state, gid;
  compute_cost = op_compute_cost)`; execute via `best_replacement` + `replace_grammar_in_state!` on the
  argmax gid; epoch bump trigger extended to `applied_replacement` (replacement adds 0 components, so the
  current `n_added_meta > 0` trigger alone would miss it); `reset_learning_regime!` on application.
- **`test/test_replacement.jl`** (new, ~6 sections): §1 commutation — replacement then N conditioning steps
  `==` clean-start-with-`g′` then same steps (normalised weights `≤ 1e-12` for summation order; Beta params
  and tags exact `==`); §2 treadmill regression — fire, then `replacement_value` on the successor is `0.0`
  and `best_replacement` is `nothing` (the candidate died with its grammar); §3 the OQ-4 case — a
  sub-`1e-15`-weight live component referencing the feature **blocks** candidacy (this is the test that
  fails against the table-based check and passes against the group-local one); §4 score exactness —
  `replacement_value == log1p((exp(Δ)−1)·W_G) − cost` against a hand-built oracle, and the `W_G → 1`
  degeneration to `voc`; §5 score/transition unity — the scored candidate and the applied candidate are
  the same object (`===`); §6 grammar-registry hygiene — old gid absent, `g′` registered, dedup on a later
  `enumerate_more` of `g′` re-adds nothing.
- **`test/test_grid_world_meta.jl`** (modification): §1 re-baselined — the `-Inf` pin replaced by the
  replacement-value pins; the direct engine-accessor assertion (`perturbation_voc > 0`) retires with the
  sibling path.
- **`test/test_perturb_consumption.jl`** (modification): the no-op/same-id §§ stay (the saturation no-op is
  unchanged); the sibling-injection expectations update to replacement.

## 3. Behaviour preserved

- `perturb_grammar` / `perturbation_voc` / `compression_exhausted` keep their exact current semantics on
  their current signatures — they remain the prior-only, state-free surface (used by `compression_exhausted`
  and any brain-side caller without an `AgentState`). The replacement path is a **new, state-aware** surface
  beside them; no existing test of the trio changes. (The *host* stops calling the sibling path; the
  functions stay.)
- Commutation: `≤ 1e-12` on normalised weights (float summation order), exact `==` on Beta parameters, tags,
  metadata, program identity — the #187 tolerance discipline.
- `test_growth_returns.jl`, `test_coherent_injection.jl`, `test_compression_removal.jl`, the conjugate/
  threshold/complexity suites: untouched passes required.
- The dominance benchmark is expected to **drift** (every policy's transition semantics improve); the gate
  re-run is the acceptance, not bit-stability.

## 4. Worked end-to-end example

State: two grammars. `g1` (features `{a, b}`, no rules) carries group `G = {1, 2, 3}`; component programs
test only `:a` (`:b` is referenced by no program and no rule body). `g2` carries components `{4, 5}`. After
some history, normalised posterior: `W_G = 0.6`, `W_{g2} = 0.4`. Every array position `i` has tag `i`
(re-tag discipline).

1. **Score.** `replacement_candidates(state, g1.id)` (perturbation.jl) walks all three group programs —
   including any sub-`1e-15` stragglers — and `g1`'s (empty) rule bodies: `:b ∉ refs` → one candidate,
   `kind = :remove_feature`, `payoff_symbols = 1`. `g1′ = Grammar({a}, rules, thresholds − b, fresh id)`;
   the 4-arg constructor recomputes complexity; `Δ = λ·(c(g1) − c(g1′)) = log 2`. `W_G` reads canalised:
   `probability(state.belief, TagSet(Interval(0,1), Set([1,2,3]))) = 0.6`. Score
   `= log1p((e^{log 2} − 1)·0.6) = log(1.6) ≈ 0.4700` nats, minus `op_compute_cost`. (The prior-only `voc`
   would have claimed `log 2 ≈ 0.6931` — the surrogate overstates by exactly the mass the group doesn't
   hold.)
2. **Select.** `score_gw_meta_actions` (host) puts `0.4700 − cost` into the same argmax as growth and escape
   scores — one `optimise`, no side-channel (T-3.51).
3. **Transition.** `replace_grammar_in_state!` (agent_state.jl) applies **the same candidate object**:
   `metadata[1..3] ← (g1′.id, ·)`; `log_weights[1..3] += log 2`; mixture reconstructed (normalises);
   `delete!(state.grammars, g1.id)`; `state.grammars[g1′.id] = g1′`. New normalised masses:
   `W′_G = 0.6·2 / (0.6·2 + 0.4) = 0.75`, `W′_{g2} = 0.25`. Realised Δ log-evidence
   `= log(1.6)` — **exactly the score**. Tags 1–5 and all Beta posteriors byte-identical.
4. **Verify (the §1 derivation, concretely).** The counterfactual agent seeded with `g1′` instead of `g1`
   assigns each group component prior `lw_i + log 2` and conditions on the identical history through
   identical kernels — landing on the same normalised weights as step 3. `test_replacement.jl` §1 asserts
   this equality after a further conditioning run (commutation, not just the instant).
5. **Next step (treadmill dead).** `g1` is unregistered — `top_k_grammar_ids` cannot surface it;
   `replacement_candidates(state, g1′.id)` finds no dead item → `replacement_value = 0.0` → perturb scores
   below `do_nothing`'s floor. One edit, once.

Ownership at each step: candidates + score in `perturbation.jl`; the canalised mass read in `ontology.jl`;
the write in `agent_state.jl` beside its siblings (`add_programs_to_state!`, `sync_prune!`); selection in
the host's one argmax. The score/transition dual residency is the §5 Q1 subject; the shared candidate
function is what keeps it one function in two houses.

## 5. Open design questions

1. **Constitutional form of the weight write.** The group shift is a `log_weights` write outside
   `condition` — the letter of "no other function may modify a measure's weights". Three forms considered:
   **(a) the `coherent-space-edit` precedent** (proposed): bless `src/`-resident space edits whose weight
   arithmetic is the unique solution of the commutation equation, each with a named equality test.
   **(b) Route through `condition`** on a fabricated per-component event with likelihood ratio
   `e^{Δ·1[i∈G]}`. **(c) Keep `-Inf`** and accept the treadmill as the price of letter-purity.
   Recommending (a), two reasons: *(i)* the criterion is exactly the one that already blessed #187 —
   injection's ledger arithmetic is a weight computation outside `condition`, sanctioned because commutation
   forces it uniquely; refusing the same criterion here is incoherent case law, and (a) regularises both
   under one named slug. *(ii)* (b) is strictly worse than what it launders: the fabricated "observation"
   is not data — it corrupts every downstream evidence read (injection yields, residual stream, any future
   prequential audit) and misfiles a prior re-declaration as likelihood, violating T-1.3 (the prior is the
   honest statement of ignorance) to satisfy a letter whose purpose is to *prevent* exactly such misfiling.
   (c) leaves a live A3 violation (the duplicate-injection dilution) in place to avoid a derived, tested
   write — the constitution prefers the derivation (T-2.81).
2. **Partial re-entry scope.** Perturb re-enters scored over **removal-class candidates only**;
   `:add_rule` proposals stay dark until add-consumption lands. Alternatives: keep `-Inf` until the full
   class is consumable, or fold add-consumption in now. Recommending partial re-entry: *(i)* the
   instrumented treadmill was removals (`rules 0→0` in the smoke log) — the live pathology is fully cured by
   the removal half, and deepen's precedent (#189 deviation 2: cells tracked, op dark) already establishes
   per-op honest gating; *(ii)* add-consumption needs genuinely new machinery — a semantics-preserving AST
   rewrite (subtree → `NonterminalRef`), per-program non-uniform `Δ_i`, kernel recompilation — whose
   commutation derivation is the same shape but whose blast radius (Program/CompiledKernel identity,
   Invariant 3's no-AST-in-kernels boundary) deserves its own doc rather than a rider. The risk accepted:
   until then, an applied `:add_rule` (fired by *score-blind baselines* — eu_max can't fire it) still
   treadmills; mitigated by baselines' fixed budgets.
3. **Score fidelity.** Exact `log1p((e^Δ − 1)·W_G)` (recommended) vs the prior-only `voc = Δ` surrogate.
   Recommending exact: *(i)* it is the same Δ-log-evidence currency as the escape ops' learned yields and
   growth's `fit` (T-3.53 — one currency), so the meta-argmax compares like with like; the surrogate
   overstates by the held-out mass and would resurrect cleanup churn on low-mass grammars (a zombie lineage's
   hygiene is worth ~nothing, and the exact score *says so*: it self-extinguishes as `W_G → 0`); *(ii)* the
   cascade between fidelities is itself an EU decision on evaluation cost (T-3.53), and here the dearer
   fidelity costs one canalised `probability` read — when exactness is free, choosing the surrogate is
   unpriced error. Counter-position to argue: Move 5 named `voc` *the* cheap fidelity, and consistency of
   the cascade tiers has its own value; answer: Move 5's cascade prices *lookahead re-conditioning*, which
   this score still avoids — the mass read is not a lookahead.

## 6. Risk + mitigation

- **Unsound re-key (a live program outside its new grammar's language).** The catastrophic one: complexity
  bookkeeping silently wrong, any recompile of that component crashes. Cause would be a lossy reference
  count (the `w > 1e-15` cut, or a missed AST node type). Mitigation: group-local full-support walk with
  the no-generic-fallback method style (a new node type fails loud, matching `collect_*_refs!`);
  `test_replacement.jl` §3 pins the sub-threshold case that defeats the table-based check.
- **Score/transition drift** (the 3.55 regression this doc exists to kill). Mitigation: one candidate
  function feeding both; §5 pin (`===` on the candidate object); no numeric re-derivation at apply time.
- **Registry dangling references.** `top_gids` snapshots in the host could hold a consumed gid within a
  step. Mitigation: `haskey` guards already line every host loop (the #189 pattern); §6 of the test pins
  registry hygiene; grep `state.grammars[` across `apps/` and `src/` at build time for unguarded reads.
- **Persistence.** `AgentState` schema is unchanged (same fields, same types) — replacement is a value-level
  operation. Fixture suite must pass untouched; no new fixture needed (no schema version bump).
- **Benchmark comparability.** All policies' transitions change at once; per-policy deltas vs the #189
  gate are not attributable to selection. Mitigation: the gate re-run is a fresh baseline, stated in the
  results commit; the falsifiable claims (§7) are about *mechanism* (treadmill extinction), not AUC
  movement, with AUC/final-window/efficiency reported as before.
- **Cross-lineage duplicates** (two registered grammars structurally equal after independent cleanings, each
  holding evidence on duplicate programs). Pre-existing condition, neither created nor cured by this move;
  consumption *reduces* its incidence (siblings stop being minted). Named non-goal; revisit if the gate's
  op logs show duplicate-grammar mass fragmenting decisions.

## 7. Verification cadence

    julia test/test_replacement.jl                 # new — commutation, treadmill, OQ-4, score oracle
    julia test/test_perturb_consumption.jl         # updated expectations
    julia test/test_grid_world_meta.jl             # re-baselined §1
    julia test/test_compression_removal.jl         # untouched pass
    julia test/test_coherent_injection.jl          # untouched pass
    julia test/test_growth_returns.jl              # untouched pass
    for f in test/test_*.jl; do julia "$f"; done   # full local suite (Julia tests are not CI-gated)
    JULIA_PROJECT=$PWD PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run python apps/skin/test_skin.py
                                                   # optional — no wire change; run for the CI parity habit

Gate re-run (`julia apps/julia/dominance_benchmark/run.jl`, 20 seeds, background, never through head/tail),
with the falsifiable claims: **(i)** perturb fires appear in eu_max's op log *at most once per (grammar,
dead-item)* — zero repeated proposals (the treadmill-extinction claim, checkable from the instrumented op
log); **(ii)** no duplicate-component growth attributable to perturb (component count stable across perturb
fires); **(iii)** worst-seed and AUC-vs-fixed_k25 reported honestly — this move does not claim to close the
winner's-curse gap (that is the next doc); halt-the-line if (i) or (ii) fails. Results commit either way.
