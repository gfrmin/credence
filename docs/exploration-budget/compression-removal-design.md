# Compression-class removal follow-up — design (`:remove_feature` + close `:remove_rule`'s transitive hole)

> Exploration-budget arc, follow-up to Move 4 (issue #174). Design-doc-before-code; ratify before any code
> lands. Master plan: `docs/exploration-budget/master-plan.md` (risk 2 / OQ-4 anticipated a `:remove_rule`
> soundness concern — Move 1 closed the *depth/complexity-1* variant via `collect_nonterminal_refs!`; this
> closes the deeper *transitive-through-rule-body* variant). Predecessors on master: Move 1 (`:remove_rule`
> + sound reference count), Move 2 (saturation signal), Move 3 (threshold lookahead), Move 4 (`:add_feature`
> + the Q4 escalation ladder). Authored 2026-06-29.

---

## 1. Purpose

Two items that share a fix, a home, and a class — closed together because Move 4's grounding showed they are
the same bug in the same place (#174):

1. **Close a shipped `:remove_rule` transitive soundness hole.** Enumerated programs hold `NonterminalRef`s
   *unexpanded* (the body lives in `grammar.rules`, resolved at compile time). `collect_nonterminal_refs!`
   walks only program ASTs, so a rule referenced **only inside another rule's body** — with no direct
   program reference — is invisible: `_removal_payoff` flags it dead, and removing it leaves a dangling
   `NonterminalRef` → `compile_expr` crashes ("Undefined nonterminal"). Reachable under ordinary nested
   abstraction; **verified reproduced** during the Move 4 build. The fail-closed `nothing` default does NOT
   catch it — that default guards the *un-analysed* case; this is *analysed-but-incomplete* (the table is
   populated, it just under-reports). This is the urgent half: a live op that corrupts a rule body.

2. **Add `:remove_feature` as a compression-class meta-action.** Removing a *dead* feature (referenced by no
   support program and no rule body) is prior-only MDL reclamation — the symmetric partner of `:remove_rule`,
   priced by `net_voc` in prior nats, reclaiming exactly **1 symbol** (`length(feature_set)` drops by one;
   thresholds are not charged — Q1(b)). It is **compression, not discovery** (contrast Move 4's `:add_feature`,
   which is generative and lookahead-valued): so it rides inside `perturb_grammar`'s `net_voc` argmax with
   `:add_rule`/`:remove_rule`, **needs no new host meta-action**, and — the tell that it is compression-class —
   **perturbs `compression_exhausted`**, the middle rung of Move 4's Q4 escalation ladder.

Both halves share the fix (the consumer unions rule-body refs) and the home (the compression class), so they
land as one focused follow-up rather than bolted onto the discovery headline. **Resolved decisions** (here,
not §5, per the template — their answers are forced): `:remove_feature` rides inside `perturb_grammar` (no
new meta-action — the clean consequence of Finding 2); its payoff is exactly 1 symbol; removing a *live*
feature is destructive generative change and is never a candidate (only dead features are).

**PR structure (three PRs in sequence — all downstream of the one Q1 fact: `compression_exhausted` is soft
because compression never confounds exploration VOI).** **PR 1 = the `:remove_rule` transitive crash fix**
(`_removal_payoff` unions rule-body refs) + capture-before-refactor + a nested-abstraction regression test —
the isolated soundness fix; lands FIRST regardless (it is a live crash; the rest is suboptimality).
**PR 2 = DROP the compression rung from BOTH exploration gates** (`:gw_explore`, `:gw_add_feature`) — a
self-contained one-sidedness fix (§5 Q1 (i)/(iii)), standalone because once it spans two move surfaces it is
no longer a compression-*removal* concern; lands BEFORE PR 3 (hard ordering: a dead feature feeding
`compression_exhausted` while the gate is still hard would *widen* the soft-cap before it is fixed — drop the
veto, then couple). **PR 3 = `:remove_feature`** (the new op + the `SubprogramFrequencyTable` field + the
argmax wiring), riding on the un-vetoed gate. #174 (the issue) closes on PR 1 + PR 3 — its two stated concerns,
removal soundness and removal mechanics; PR 2 is the sibling fix the Q1 analysis spawned, kept tied to this
doc because all three are downstream of the single fact that compression never confounds exploration VOI.

## 2. Files touched

- **`src/program_space/types.jl`** (modification, ~5 lines). `SubprogramFrequencyTable` gains a 5th field
  `referenced_features::Union{Nothing, Set{Symbol}}` (the same `nothing`-sentinel discipline as
  `referenced_nonterminals` — fail-closed: un-analysed ⇒ no removal). The 3-arg convenience constructor
  (`:259`) defaults BOTH reference fields to `nothing` — so every hand-built test table is unaffected.
- **`src/program_space/perturbation.jl`** (modification):
  - `collect_feature_refs!(acc::Set{Symbol}, e::ProgramExpr)` — NEW. One method per `ProgramExpr` subtype,
    **no generic fallback** (a future node type fails loud — the `collect_nonterminal_refs!` discipline).
    `GTExpr`/`LTExpr` push their `.feature`; `NonterminalRef` contributes nothing *directly* (the rule-body
    union, below, handles transitive feature refs); the rest recurse into all branches. ~13 lines.
  - `analyse_posterior_subtrees` (`:28`–`:58`) — populate `referenced_features` from the SAME support set
    (`w > 1e-15`) the nonterminal walk uses; pass it to the 5-arg constructor at `:58`. ~3 lines.
  - **`_removal_payoff(g, table)` (`:212`) — THE FIX.** Union the program-direct `referenced_nonterminals`
    with the nonterminal refs from **all rule bodies** (walk each `r.body` with `collect_nonterminal_refs!`);
    a rule is removable iff its name is in NEITHER. ~4 lines.
  - `_feature_removal_payoff(g, table)` — NEW, the symmetric partner. Union the program-direct
    `referenced_features` with the feature refs from all rule bodies (`collect_feature_refs!` on each
    `r.body`); a feature in `g.feature_set` in NEITHER is dead → a candidate, payoff 1. `nothing` ⇒ no
    candidates (fail-closed). ~7 lines.
  - `_best_compression_candidate` (`:273`) — fold `:remove_feature` candidates into the `net_voc` argmax
    alongside add/remove-rule. The candidate representation generalises (§5 Q2). ~5 lines.
  - `perturb_grammar` (`:324`) — apply a `:remove_feature` winner: drop the feature from `feature_set` and
    its grid from `thresholds`, fresh id (the 4-arg constructor, threading the surviving grids). ~6 lines.
- **`src/Credence.jl`** — *no change* (corrected at code time). `collect_feature_refs!` stays **internal**:
  the truer mirror of `collect_nonterminal_refs!`, which is itself unexported. Tests reach it via
  `Credence.collect_feature_refs!`.
- **`test/test_compression_removal.jl`** (new) — the `:remove_rule` nested-abstraction soundness fix,
  `collect_feature_refs!` soundness (direct + transitive-via-rule-body), `_feature_removal_payoff`, the
  unified argmax, determinism, and the capture-before-refactor pins (§3).

- **`apps/julia/grid_world/host.jl`** (modification, **PR 2** — §5 Q1's finding, standalone). DROP the
  `compression_exhausted` hard-gate (`... || return -Inf`) entirely from BOTH `:gw_explore` (`:206`) and
  `:gw_add_feature` (`:229`); `threshold_exhausted` (`:233`) stays hard. **No discount, no new constant**
  (the §5 Q1 (ii) resolution): compression is prior-only and never confounds a fit-side VOI, so the soft
  cost-ordering preference is already carried by the existing meta-action cost asymmetry (`GW_PERTURB_COST`
  0.05 < `GW_EXPLORE`/`GW_ADD_FEATURE_COST` 0.10) in the caller's argmax — the host compares PROXY EUs in a
  common scale, so the Q5 currency gap never arises here. Also removes the now-pointless
  `analyse_posterior_subtrees`/`top_k` from `:gw_explore` and the now-unused `compression_exhausted` import.
  New `test/test_grid_world_meta.jl` (the FIRST test of `compute_gw_meta_eu`). Its own PR, not part of #174.

**No new meta-action.** `:remove_feature` rides inside the existing `:perturb_grammar`/`:gw_perturb_grammar`
meta-action (which already calls `perturb_grammar`); the host re-enumerates the returned grammar exactly as
for `:add_rule`/`:remove_rule`. No skin/wire change. The PR-2 host edit modifies existing meta-action
*gates*; it adds no verb.

## 3. Behaviour preserved (capture-before-refactor)

All assertions are **exact `==`** (the affected outputs are structural/discrete — reference sets, candidate
identities, grammar feature sets — not floats; no tolerance class applies).

- **The `:remove_rule` fix is a no-op for non-nested grammars.** For every existing test grammar (none of
  which has a rule body containing a `NonterminalRef` — verified by the §6 grep), the rule-body union adds
  nothing, so `_removal_payoff`, `_best_compression_candidate`, `compression_exhausted`, and `perturb_grammar`
  outputs are **bit-identical**. Pinned PRE-change and asserted `==`: `test_voc_gate.jl`, `test_saturation.jl`,
  `test_program_space.jl`, `test_threshold_explore.jl §4` stay green unchanged.
- **The `SubprogramFrequencyTable` field is additive.** The 3-arg convenience defaults both reference fields
  to `nothing`; every existing construction site (all 3-arg — `test_saturation`, `test_perturb_consumption`,
  `test_voc_gate`, `test_program_space`, `test_threshold_explore`) is unaffected, asserted `==`.
- **`compression_exhausted` / the Q4 ladder shift ONLY where correctness demands it** (§5 Q1): a grammar with
  a removable dead feature, or a nested-abstraction grammar whose false-dead `:remove_rule` candidate the fix
  removes. Both are *corrections*; the nested case is new behaviour exercised only by the new test.
- **The PR-2 gate-DROP is a deliberate behaviour change, NOT preserved** (§5 Q1 finding, §6 risk 4(b)):
  `:gw_add_feature`/`:gw_explore` selection changes when compression is unexhausted (the hard veto was the
  bug). **Correction (verified during PR 2): no existing test pinned these gates** — `compute_gw_meta_eu`
  was untested (`test_grid_world.jl`'s only agent run sets `max_meta_per_step=0`). So PR 2 *adds the first*
  test (`test_grid_world_meta.jl`), a net coverage gain — not a pin migration. There is nothing to capture
  PRE-change; the new test pins the post-drop contract directly. Everything in the bullets above holds `==`.

## 4. Worked end-to-end examples

**(a) The `:remove_rule` transitive fix (the verified repro).** Grammar `g` with rules `NT_B = (gt :x 0.3)`
and `NT_A = AND(NonterminalRef(:NT_B), (gt :y 0.5))`; one support program `IF NonterminalRef(:NT_A) :a :b`
(references `:NT_A` directly, `:NT_B` only transitively).
- `analyse_posterior_subtrees` (owner: perturbation.jl) walks the program AST → `referenced_nonterminals =
  {:NT_A}` (the walk never enters `NT_A`'s body). **Pre-fix:** `_removal_payoff` ⇒ `:NT_B` dead → removed →
  `compile_expr(NT_A.body)` hits `NonterminalRef(:NT_B)` → `error("Undefined nonterminal: NT_B")`.
- **Post-fix:** `_removal_payoff` unions rule-body refs — `collect_nonterminal_refs!(NT_A.body)` → `{:NT_B}`,
  `collect_nonterminal_refs!(NT_B.body)` → `{}` — so `referenced = {:NT_A, :NT_B}`. `:NT_B ∉ referenced` is
  false ⇒ **not a candidate.** `:NT_A` is referenced by the program ⇒ not a candidate either. `_removal_payoff`
  returns `[]`. Sound.

**(b) `:remove_feature` reclaiming a dead feature.** Grammar `g` with `feature_set = {:red, :wall_dist}`, no
rules; after a regime shift every support program uses `:wall_dist` and none uses `:red` (the colour programs
fell below `w > 1e-15` / `min_log_prior`).
- `analyse_posterior_subtrees` → `referenced_features = {:wall_dist}` (only `:wall_dist` appears in a support
  program's `GTExpr`/`LTExpr`).
- `_feature_removal_payoff(g, table)`: program refs `{:wall_dist}` ∪ rule-body refs `{}` (no rules) =
  `{:wall_dist}`. `:red ∉ {:wall_dist}` ⇒ `:red` is a dead-feature candidate, payoff **1 symbol**.
- `_best_compression_candidate`: `net_voc(1, compute_cost) = log2 − compute_cost > 0` (at the host's cost) ⇒
  `:remove_feature :red` enters the argmax; if it is the best compression candidate, `perturb_grammar`
  (owner) returns a fresh grammar with `feature_set = {:wall_dist}`, `:red`'s grid dropped, `complexity`
  down by 1. The host's existing `:perturb_grammar` meta-action applies it — no new wiring.

## 5. Open design questions

### Q1 — Capture-before-refactor scope: the `:remove_rule` fix shifts `compression_exhausted` (the Q4 ladder's middle rung)

The fix correctly *shrinks* `_removal_payoff` for nested-abstraction grammars (a false-dead candidate
disappears), and `:remove_feature` *grows* it (dead features become candidates). Both propagate through
`_best_compression_candidate` → `compression_exhausted` (Move 2) → the Move-4 Q4 ladder
(`plateau ∧ compression_exhausted ∧ threshold_exhausted ⇒ features`). For the `:remove_rule` fix this should
be a **no-op on every existing grammar** (none is nested — §6 grep), so the capture-before-refactor pins
should all hold `==`. **The question:** is that grep-confirmed "no existing nested grammar" assumption one
you want asserted as a test (pin every test grammar's `compression_exhausted` `==`), or is a one-time grep
disposition sufficient? And do you want the `:remove_feature`-induced `compression_exhausted` change (a
removable dead feature now means "not saturated") blessed explicitly, since it couples feature-*removal* into
the gate that admits feature-*addition*? *Recommendation: assert the pins as tests (cheap, and the Q4 ladder
is load-bearing enough to guard mechanically per `executable-documentation`); bless the `:remove_feature`
coupling as correct — a reclaimable symbol is genuine residual compression, so the prior is not saturated.*

> **Ratified (2026-06-29) — coupling blessed; Q1 surfaced a shipped hard-gate bug.**
>
> **Coupling.** Blessed. A reclaimable dead feature genuinely means the prior is not saturated, so it
> *should* drive `compression_exhausted` false. The rejected alternative — keeping feature-removal OUT of
> the signal so the Q4 gate is "unaffected" — makes the signal *lie*: report compression exhausted while a
> symbol is still reclaimable. Don't buy gate-honesty by corrupting the signal that feeds it. Pins asserted
> as tests (every test grammar's `compression_exhausted` pinned `==`, capture-before-refactor).
>
> **What Q1 actually turns on — soft vs hard rungs.** The Q4 ladder's rungs are not the same kind of object.
> `threshold_exhausted` is a **sound HARD deferral**: a feature's Δℓ measured against a coarse-grid baseline
> is *confounded* by the residual threshold-refinement would also capture, so the feature cannot be measured
> correctly until thresholds exhaust — attribution fidelity licenses a hard gate. `compression_exhausted` is
> **NOT** that: compression is prior-only, never touches the residual, so it **never confounds a fit-side
> VOI** (the host's own `:195` comment concedes "orthogonal because compression is a prior effect, the
> residual a fit effect"). There is therefore *no attribution reason* to hard-gate a fit-side exploration
> behind compression — only cheap-before-dear ordering, a **soft, overridable preference** (Move 2 Q3:
> ordering enters as a prior, never a block). The `:remove_feature` coupling is sound **precisely because the
> rung is soft**: a dead feature lowers the compression-saturation prior → lowers the exploration prior →
> which the exploration's own EU overrides if it is worth it. Were the rung hard, a dead feature — irrelevant
> to a beneficial feature's value — would defer that feature for a pass: a soft cap and a one-sidedness
> violation. **`compression_exhausted` enters the feature gate SOFT; `threshold_exhausted` is the hard one.**
>
> **The finding (verified 2026-06-29).** The shipped host hard-gates `compression_exhausted` with
> `... || return -Inf` in **two** places: `:gw_add_feature` (Move 4, `host.jl:229`) and `:gw_explore`
> (Move 3, `host.jl:206`). Both are the soft-prior-as-hard-gate bug this question predicted.
>
> **(i) Both gates — resolved.** One bug, *singular*. The argument is about *compression* (it re-describes
> existing programs without touching the residual, so it cannot confound the VOI of anything downstream of
> the residual), not about features — so it discharges `:gw_explore` (threshold refinement) exactly as it
> discharges `:gw_add_feature`. Softening one and not the other would ship an asymmetry explicable only by
> "they landed in different PRs" — a process fact masquerading as a design decision — and would leave a
> *live* shipped soft-cap in the Move-3 path (a pending compression candidate deferring a correctly-measured
> positive-EU threshold refinement). Soften both.
>
> **(ii) Mechanical form — DROP the gate, no constant (revised from the discount on review).** A first pass
> ratified a multiplicative discount as an interim device. On building it the bare constant was rejected —
> *magic constants are a big no-no* — and, decisively, the discount's stated justification does not hold at
> the host level. That justification was "keep compression (prior nats) and exploration (predictive nats) as
> two separate decisions, sidestepping the cross-currency comparison." But the host never makes that
> comparison: `compute_gw_meta_eu` hands every meta-action a PROXY EU (`plateau · BASE − cost`) in a common
> hand-tuned scale and the caller argmaxes over proxies; the real `explore_grammar`/`explore_features` VOIs
> are computed at *execution*, not selection. So there is no cross-currency comparison at the selection point
> to sidestep — the Q5 currency gap is engine-level, untouched by this host gate. The clean interim is
> therefore to **drop the `compression_exhausted` veto entirely** (no constant): the bug (hard veto) is gone,
> and the soft "compression-first" preference survives via the EXISTING cost asymmetry (`GW_PERTURB_COST`
> 0.05 < `GW_EXPLORE`/`GW_ADD_FEATURE_COST` 0.10), exactly as the un-gated `:gw_enumerate_more`/`:gw_deepen`
> already compete. No correctness claim rides on compression-first (compression never confounds — the whole
> basis of the fix), so a *stronger* nudge would be a preference with no principle behind its strength: the
> magic constant. The principled end-states remain — **(A)** one net-EU argmax over compression and
> exploration once Move 5 closes Q5; **(B)** a graded compression-residual signal for a *real-utility* host
> (the proxy host needs neither now).
>
> **(iii) Sequencing + home — resolved.** Crash-fix → soften → couple. The `:remove_rule` transitive fix is
> **PR 1** (a crash; first regardless). The softening is **PR 2, standalone** — once (i) makes it span both
> the Move-3 and Move-4 gates it is a self-contained one-sidedness fix touching two move surfaces, a cleaner
> legible PR than a third commit forcing #174 to carry three concerns. It lands **before** `:remove_feature`
> (**PR 3**): the moment a dead feature feeds `compression_exhausted` while the gate is still hard, dead
> features start hard-capping exploration — the bug widens before it is fixed. Soften, then couple. The
> through-line: this invents no new gate behaviour — it makes `compression_exhausted` behave the way Move 2
> Q3 already said saturation signals must, in the two places that were quietly treating it as hard.

### Q2 — The feature-removal candidate's representation in the unified `net_voc` argmax

`_best_compression_candidate`'s candidate is the 5-tuple `(voc, is_remove, name::String, kind::Symbol,
rule::ProductionRule)`; `perturb_grammar` dispatches on `kind`. A `:remove_feature` candidate carries a
**`Symbol`** (the feature), not a `ProductionRule`. Two options: **(a)** widen the payload to
`Union{ProductionRule, Symbol}`, dispatched by `kind` (minimal diff, but an untyped-Union smell —
Invariant 3); **(b)** replace the tuple with a small declared `PerturbationCandidate` struct (a `kind` +
a typed payload), which all three candidate sources construct and `perturb_grammar` dispatches on (more
churn, but removes the Union and makes the candidate's structure first-class). *Recommendation: (b)* — the
candidate set is exactly the kind of declared-structure Invariant 3 wants first-class, and three candidate
kinds is the point where a tuple-with-Union starts encoding conventions invisibly. Counter: (a) is a smaller
blast radius on shipped compression machinery if you'd rather keep this follow-up tight.

> **Ratified (2026-06-29) — (b), the declared `PerturbationCandidate` struct.** The decisive reason is
> Move 5: its combined single-currency argmax adds *more* candidate kinds (threshold- and feature-discovery
> join the compression ones), so a typed `kind + payload` candidate is the structure that move needs anyway
> — build it here and Move 5 extends an enum rather than retrofitting one onto a Union that has already
> hardened. Discipline: the struct is a **behaviour-preserving** refactor of the shipped
> `:add_rule`/`:remove_rule` paths, so capture those two kinds' `_best_compression_candidate` winners `==`
> through it (distinct from the `:remove_rule` fix's grep-backed no-op-ness — this pins the *representation*
> change separately).

### Q3 — The transitive-soundness union: ALL rule bodies, or only referenced ones?

The fix unions refs from **every** rule body (cheap, and the conservatism is in the safe direction — a
symbol referenced only inside a *dead* rule's body is kept until that dead rule is removed first, then
reclaimed on a later pass — the existing one-perturbation-per-call cadence). The tighter alternative walks
only the bodies of *referenced* rules (so a dead rule referencing `:NT_B`/`:red` doesn't protect it),
reclaiming in one pass instead of iterating. *Recommendation: union ALL rule bodies* — it is sound, simpler,
and the iterative cleanup of dead chains is consistent with the move's one-candidate-per-call rhythm;
reachability analysis is complexity the saturation cadence doesn't need. Confirm, or take the tighter form
if single-pass reclamation of dead chains matters.

> **Ratified (2026-06-29) — union ALL rule bodies.** Reachability-free and conservative in the safe
> direction. Referenced-only's sole advantage — reclaiming a dead *chain* in one pass — is moot under
> one-perturbation-per-call: a dead chain reclaims one link per call regardless, so the tighter form buys
> reachability complexity to win a race the saturation cadence does not run. Keep a symbol alive while any
> rule body names it; remove the dead rule first, reclaim the orphan next pass.

## 6. Risk + mitigation

1. **The `:remove_rule` fix silently changes a shipped grammar's behaviour.** Failure mode: an existing test
   grammar HAS nested abstraction, so the rule-body union changes its `_removal_payoff` and a
   capture-before-refactor pin flips. Blast radius: `test_voc_gate`, `test_saturation`, `test_program_space`,
   `test_threshold_explore`. **Pre-emptive grep — DONE (2026-06-29):** every `ProductionRule(...)` body in
   `test/` is a leaf predicate (`GTExpr`/`LTExpr`/`AndExpr` of those — `existing_body`, `red_body`, `:LIVE`,
   `:DEAD`, `:BARE`, `:KEEP`, `:DROP`, `:RED`, …); every `NonterminalRef` in tests is in a *program* body
   (`IfExpr(NonterminalRef(...), …)`), never a rule body. **No existing fixture has nested abstraction**, so
   the rule-body union changes nothing for them — the capture-before-refactor pins hold `==` by construction.
   The nested case is exercised only by the new `test_compression_removal.jl`.
2. **`:remove_feature` removes a live feature (the soundness mirror of risk 1).** Failure mode:
   `collect_feature_refs!` misses a feature reference (a missing method, or the transitive rule-body case) →
   a live feature flagged dead → removal deletes support programs. Mitigation: one method per `ProgramExpr`
   subtype with no generic fallback (fails loud on a new node type); the rule-body union (Q3); the `nothing`
   sentinel (un-analysed ⇒ no removal); a test asserting a feature used only inside a rule body is NOT
   removable (the transitive guard) and a depth-1 `GTExpr(:f, t)` reference is seen.
3. **The candidate-representation refactor (Q2(b)) destabilises the argmax order.** Failure mode: replacing
   the tuple changes `_candidate_better`'s total order (e.g. tie-break drift) → a different perturbation wins
   → benchmark drift. Mitigation: preserve the exact order (voc → is_remove → name); capture-before-refactor
   pins `_best_compression_candidate`'s winner `==` on the existing fixtures.
4. **`compression_exhausted` regression to the Q4 ladder + the gate-DROP behaviour change.** Two parts.
   (a) The `:remove_feature` coupling: pin every test grammar's `compression_exhausted` `==`; the only
   intended change is the coupling, blessed in Q1. (b) The **PR-2** gate-DROP **deliberately** changes BOTH
   `:gw_add_feature` and `:gw_explore` selection when compression is unexhausted (the bug fix). No existing
   test pinned the gate (`compute_gw_meta_eu` was untested), so there is nothing to migrate: PR 2's new
   `test_grid_world_meta.jl` pins the post-drop contract directly (`:gw_explore` returns the soft plateau
   proxy, not −Inf, when compression is available; `:gw_add_feature` likewise, with `threshold_exhausted`
   still vetoing). Never reintroduce the hard veto to keep a stale expectation green.

## 7. Verification cadence

```
julia test/test_compression_removal.jl   # the fix, collect_feature_refs!, _feature_removal_payoff, argmax, determinism, pins
julia test/test_voc_gate.jl              # Move 1 reference-count + compression — capture-before-refactor pins ==
julia test/test_saturation.jl            # Move 2 compression_exhausted — pinned ==
julia test/test_threshold_explore.jl     # Move 3 untouched (incl. the refined-grid-survives-compression pin)
julia test/test_feature_discovery.jl     # Move 4 :add_feature untouched
julia test/test_program_space.jl         # enumeration + removal — pinned ==
julia test/test_grid_world_meta.jl       # PR 2: the FIRST test of compute_gw_meta_eu — gate-drop contract
julia test/test_grid_world.jl            # host loads + runs after PR 2; PR 3 dead-feature reclaim (no new meta-action)
```

Per-PR split: **PR 1** (crash fix) = the nested-abstraction regression test + the §3 capture pins (`==`).
**PR 2** (drop the veto, both gates) = `test_grid_world_meta.jl` (NEW — the first test of
`compute_gw_meta_eu`); `test_grid_world.jl` stays green (host still loads + runs).
**PR 3** (`:remove_feature`) = `test_compression_removal.jl` + the remaining `==` pins above. Full
`test/test_*.jl` green before each merge; lint self-test + `check apps/`. **Skin smoke optional** (no wire
verb, no serialised-path change — the `SubprogramFrequencyTable` field is in-memory only). Halt-the-line on
any failure; the branch never sleeps red.

## 8. Status — fully ratified; doc final

**Q1/Q2/Q3 ratified, and Q1's three residual points resolved (2026-06-29)** — see the `> **Ratified**`
blocks inline (§5 Q1 (i)/(ii)/(iii)). Summary:

- **Q1.** Coupling blessed; `compression_exhausted` confirmed **soft**, `threshold_exhausted` the hard one.
  Its finding (the shipped hard-gate at `:gw_add_feature:229` + `:gw_explore:206`) is fixed by **PR 2**.
  - **(i) Both gates** — one bug, one argument ("compression never confounds the residual"); reject the split.
  - **(ii) DROP the gate — no constant** (revised from the discount on review: *magic constants are a big
    no-no*). The discount's "sidestep Q5" rationale does not bite at the host's PROXY-EU layer — no real
    cross-currency comparison happens at *selection* (the real VOIs are computed at execution) — so the clean
    interim is to drop the `compression_exhausted` veto entirely; the soft compression-first preference rides
    the existing cost asymmetry (`GW_PERTURB_COST` < `GW_EXPLORE`/`GW_ADD_FEATURE_COST`). End-state (A) =
    single net-EU argmax once Move 5 closes Q5; (B) = graded compression-residual for a real-utility host.
  - **(iii)** Crash-fix (PR 1) → soften (PR 2, standalone) → `:remove_feature` (PR 3).
- **Q2.** The declared `PerturbationCandidate` struct (the thing Move 5 extends).
- **Q3.** Union **ALL** rule bodies.

No open questions remain; this doc is final. Three PRs land in sequence (PR 1 the crash-fix first, it is a
live crash); #174 closes on PR 1 + PR 3.
