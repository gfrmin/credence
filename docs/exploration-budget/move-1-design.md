# Move 1 design doc — Complete the compression pair: `:remove_rule` + a sound reference count

> Move 1 of the `exploration-budget` arc (`docs/exploration-budget/master-plan.md`). Seven-section
> template. Discharges collapse-towers' **Scope B** named successor. Prior-only, depth-one — extends the
> existing `net_voc` machinery; **no belief, no lookahead yet** (those start at Move 2).

## 1. Purpose

Scope A made `perturb_grammar` monotonic: it only ever *grows* the dictionary (`:add_rule`), never prunes.
Move 1 completes the **MDL compression pair** by adding `:remove_rule` — drop a nonterminal that the
posterior has abandoned — and the **sound nonterminal reference count** that collapse-towers deferred it
on (Scope B was blocked precisely because the lossy `freq_table` could not tell "referenced" from "dead"
soundly; Q4-ratified fix: thread an explicit count through the analysis). Two payoffs:

1. **Dictionary hygiene** — a rule no posterior-support program references is pure dead weight; removing it
   raises the complexity prior (`Δcomplexity_logprior > 0`) at **zero fit cost** (no support program
   changes), so it is genuinely prior-only and `net_voc`-rankable — the symmetric partner of `:add_rule`.
2. **It *enables* Move 2.** "Compression is exhausted" — the prior-side half of the saturation gate — is
   *undefinable* until the compression pair is complete. After Move 1, `perturb_grammar`'s no-op (returns
   `g` unchanged) **is** the prior-saturation signal: no `:add_rule` and no `:remove_rule` raises the
   prior. Move 2 conjoins that with the belief-side residual plateau.

**Strict, by ratified default (`always strict`).** A rule is removable iff **zero posterior-support
programs reference it** — strict structural reference, counted exactly by a full-AST walk. The weighted
"Σ-weight < ε" variant is an unvalidated fit approximation and is out by construction. "Posterior support"
reuses the engine's *existing* support floor (`w > 1e-15`, the filter `analyse_posterior_subtrees` already
applies, `perturbation.jl:28`) — not a new approximation but the engine's own definition of the
ensemble; strict-strict `w > 0` is *degenerate* (enumeration always re-references a dictionary rule with
some sub-floor weight, so it would never fire), so "strict over the posterior support" is the strict
reading that is not vacuous. Removal is therefore prior-only **with respect to the belief's support** —
exact, since only sub-floor programs reference the removed rule and those are not in the belief.

## 2. Files touched

- **`src/program_space/types.jl`** — *modify*: add a fourth field
  `referenced_nonterminals::Union{Nothing, Set{Symbol}}` to `SubprogramFrequencyTable` (the set of NT names
  referenced by ≥1 posterior-support program, or `nothing` when no analysis has run). Add a 3-arg
  convenience constructor defaulting it to **`nothing`** so the existing 3-arg
  `SubprogramFrequencyTable(subtrees, freqs, sources)` call sites (tests that hand-build tables) keep
  compiling and get the **Scope-A-preserving** default. **Encoding correction (grounding, 2026-06-28):**
  the original `Set{Symbol}` default `Set{Symbol}()` does **not** achieve "no rule removable" — the removal
  predicate is `r.name ∉ referenced`, and `r.name ∉ ∅` is *vacuously true*, so an empty set would make
  **every** rule removable (the opposite of intent). An empty set also cannot distinguish "analysed, zero
  references found" (⇒ rules genuinely dead ⇒ remove them) from "not analysed" (⇒ unknown ⇒ remove
  nothing) — opposite behaviours. `Union{Nothing, Set{Symbol}}` with `nothing` = "not analysed" is the
  encoding that realises the doc's own intent: `nothing` ⇒ `_removal_payoff` returns `[]` (no candidates);
  a concrete `Set` (possibly empty — a genuinely-empty analysed set legitimately means "all rules dead") ⇒
  candidates are rules whose name is absent from it. `analyse_posterior_subtrees` **always** populates a
  concrete `Set`.
- **`src/program_space/perturbation.jl`** — *modify*:
  - `analyse_posterior_subtrees`: add a reference-counting pass. For each program with `w > 1e-15` (the
    *same* filter the subtree loop uses), walk the full AST collecting `NonterminalRef` names at **all
    depths** (not the depth-≥`min_complexity` `extract_subtrees` — that lossiness is exactly what made the
    old count unsound). Union into `referenced_nonterminals`. Populate the new field.
  - **Add** `_removal_payoff(g, freq_table) → Vector{Tuple{ProductionRule, Int}}`: for each rule `r` with
    `r.name ∉ freq_table.referenced_nonterminals`, the dictionary-shrink saving is
    `net_payoff = 1 + expr_complexity(r.body)` symbols (the rule's full cost, recovered). Referenced rules
    are **not** candidates (removing them is generative-change — deferred).
  - **Generalise** `perturb_grammar` from "the single `:add_rule` candidate" to "the `argmax`-`net_voc`
    candidate over the **compression class** = {the `:add_rule` candidate} ∪ {`:remove_rule` candidates}".
    Apply the best iff `net_voc > 0`, else the same structural no-op (return `g`, same id). Determinism
    preserved (no `rand`); tiebreak per §5-OQ2.
  - The `:add_rule` path (`_compression_payoff`, the idempotence guard, the no-op-returns-`g` invariant) is
    **unchanged**; `:remove_rule` is a second candidate source feeding the same `argmax`.
- **`src/Credence.jl`** — *modify*: export `_removal_payoff` only if a test references it by name
  (otherwise internal).
- **`test/test_program_space.jl`** — *modify*: extend the perturbation tests (TEST 14 neighbourhood) with
  `:remove_rule` cases (see §7); the existing `:add_rule` / determinism assertions stay green unchanged.
- **`test/test_voc_gate.jl`** — *modify*: add the `:remove_rule` `net_voc` arithmetic + the
  argmax-over-the-pair determinism case.
- **`test/test_perturb_consumption.jl`** — *verify unchanged*: the no-op-returns-same-id invariant is
  untouched (a `:remove_rule` no-op returns `g` identically).

## 3. Behaviour preserved

- **`:add_rule` is bit-exact.** With `referenced_nonterminals` empty (hand-built tables) or with no dead
  rule (the common case: a freshly-added rule is referenced), `_removal_payoff` returns `[]`, the
  candidate set is the singleton `:add_rule`, and the `argmax` reduces to today's gate — `===` on the
  resulting `(feature_set, rules)`. The degenerate reduction.
- **`net_voc` is unchanged** — `:remove_rule` reuses it (`net_voc(1 + expr_complexity(body), compute_cost)`),
  no new currency, no new arithmetic.
- **The no-op invariant holds** — a no-op (no positive-`net_voc` candidate) returns the input `g` (same
  id), so `add_programs_to_state!`'s id-keyed dedup is unaffected (`test_perturb_consumption`).
- **Schema-change safety** — the new `Set{Symbol}` field defaults empty; capture-before-refactor: every
  existing perturbation/program-space test stays green with no edit *except* the new `:remove_rule` cases.
- Tolerance: strata-1 `==`/`===` on grammar structure (id excluded — `next_grammar_id()` is a counter).

## 4. Worked end-to-end example

A grammar `g` with `feature_set = {:red,:green,:blue}` and `rules = [NT_a → And(GT(:red,0.7), LT(:green,0.3)),
NT_b → GT(:blue,0.5)]`. The posterior (after conditioning) is concentrated on programs using `NT_a`; the
programs that used `NT_b` have all dropped below `w = 1e-15`. `compute_cost = 0.0`.

1. `analyse_posterior_subtrees(programs, weights)` walks the support programs (`w > 1e-15`); the
   reference pass finds `NT_a` referenced, `NT_b` not → `referenced_nonterminals = {:NT_a}`. *(owner:
   `perturbation.jl`)*
2. `perturb_grammar(g, freq_table, feats)` → candidate set:
   - `:add_rule`: `_compression_payoff(freq_table)` — say it finds no subtree with `net_payoff > 0` →
     no add candidate.
   - `:remove_rule`: `_removal_payoff(g, freq_table)` — `NT_b ∉ {:NT_a}` → candidate `(remove NT_b,
     net_payoff = 1 + expr_complexity(GT(:blue,0.5)) = 1 + 1 = 2)`. `NT_a` is referenced → not a candidate.
3. `net_voc(2, 0.0) = log(2)·2 = 1.386 > 0` → the singleton `argmax` is "remove `NT_b`". *(owner: `net_voc`
   → `net_value`/`complexity_logprior`)*
4. Apply → `Grammar({:red,:green,:blue}, [NT_a → …], next_grammar_id())`. The dictionary shrank by the dead
   rule; **no support program changed** (none referenced `NT_b`), so the belief is untouched — prior-only.
5. **Determinism:** a second call on the same `(g, freq_table)` runs the identical `argmax` → same
   `(feature_set, rules)`, different id.

Contrast: had `NT_b` been referenced (`:NT_b ∈ referenced_nonterminals`), it would not be a candidate, and
with no add candidate `perturb_grammar` returns `g` unchanged — the **prior-saturation no-op** Move 2 reads.

## 5. Open design questions

> **Resolved by ratified defaults (stated, not asked):** the reference-count *home* is
> `analyse_posterior_subtrees` (Q4 — thread through the analysis); the reference *semantics* is **strict
> over the posterior support** (`always strict`; `w > 0` is degenerate, §1). These are not open.

1. **Schema change vs a sidecar.** Add the field to `SubprogramFrequencyTable` (recommended — the count is
   *of* the analysed ensemble, it belongs with the other per-ensemble summaries; one type, one analysis
   pass) **vs** return a separate `referenced_nonterminals` structure threaded alongside (avoids touching
   the struct, but splits one analysis into two return values and forces every caller to thread both).
   *Recommendation: add the field, with the empty-default constructor for backward-compat.* Counter to
   weigh: does any **serialised** consumer pin `SubprogramFrequencyTable`'s arity? Pre-emptive grep in §6;
   if `test_persistence` serialises it, the schema bump needs a fixture (it almost certainly does not —
   the freq_table is transient, recomputed each step).
2. **Tiebreak when `:add_rule` and a `:remove_rule` have equal `net_voc`.** Determinism (Phase-5) forbids
   `rand` *and* an arbitrary tiebreak — but a tie here is between two ops of *genuinely equal, exactly
   computed* prior value, so a *principled stable* order is not "laundering arbitrariness" (the Phase-5
   concern was tiebreaking *unknown* values). *Recommendation: lexicographic — prefer `:remove_rule` over
   `:add_rule` on ties (hygiene-first: shrink before grow, all else equal), then by `rule.name` hash for a
   total order.* Surface because the *direction* (remove-first vs add-first) is a real, if rarely-exercised,
   choice you may have a view on. Alternative: rank by resulting `grammar.complexity` (identical on a
   `net_voc` tie, so it does not actually break the tie — rejected).
3. **One perturbation per call, or all positive-`net_voc` candidates at once?** *Recommendation: one per
   call (the `argmax`), as today* — the host already loops `perturb_grammar`, each perturbation is
   independently `net_voc`-gated, and one-at-a-time keeps the no-op/saturation signal crisp (a call is a
   no-op iff *nothing* improves the prior). Batching would compound multiple structural edits behind one
   id bump and muddy the saturation read Move 2 depends on.

## 6. Risk + mitigation

- **R1 — unsound reference count removes a live rule.** *Failure mode:* a rule referenced only inside
  depth-1 / low-complexity contexts is missed → wrongly removed → a support program silently breaks.
  *Mitigation:* the count walks the **full AST at all depths** (not `extract_subtrees`'s depth-≥`min_c`
  lossy path — that lossiness is the exact bug Scope B was deferred on); over the **same** `w > 1e-15`
  support set the analysis already uses. Test: a rule referenced *only* at depth 1 (a bare
  `NonterminalRef`) is in `referenced_nonterminals` and is **not** removed (the regression the freq_table's
  `min_complexity=2` would have caused).
- **R2 — schema bump breaks a hand-built `freq_table` call site.** *Blast radius:* every test/app that
  constructs `SubprogramFrequencyTable(...)` positionally. *Mitigation:* empty-default constructor keeps
  the 3-arg form valid (Scope-A-preserving); **pre-emptive grep** `grep -rn 'SubprogramFrequencyTable('
  src/ apps/ test/` — list each hit, confirm it either uses the 3-arg form (gets the default) or is
  `analyse_posterior_subtrees` (populates the field). Also `grep -rn 'SubprogramFrequencyTable' test/test_persistence.jl`
  — confirm it is **not** serialised (no fixture needed); if it is, capture a v-bump fixture.
- **R3 — benchmark drift (grid_world / email_agent).** *Failure mode:* `:remove_rule` now fires where it
  never did → enumerated-program sets shift → `test_email_agent` TEST 12 `n_added` / grid_world counts
  move. *Mitigation:* drift is **intended** (a new capability — dictionary pruning); capture the new
  deterministic values post-change and assert `==` thereafter; name it in the commit. (Likely *no* drift
  in practice — removal only fires on posterior-abandoned rules, which the short benchmark runs rarely
  produce.)
- **R4 — `:remove_rule` + `:add_rule` thrash** (add a rule, next step remove it, re-add…). *Failure mode:*
  a rule oscillates in/out across steps. *Mitigation:* it cannot oscillate within the strict semantics —
  a rule is removed only when *unreferenced by the support*, and `:add_rule` only adds a rule for a
  *frequent* (referenced-to-be) subtree; the two target disjoint conditions. A test conditions a belief
  onto a subtree (so `:add_rule` fires), then away (so the rule goes unreferenced and `:remove_rule`
  fires) and asserts the sequence terminates at a no-op (saturation), not a cycle.
- **Lint:** `perturb_grammar` stays a canalised stdlib composition (the `argmax`-over-`net_voc` *is* the
  canalisation); no pragma, no new precedent. Confirm corpus self-test + `check apps/` stay green.

## 7. Verification cadence

End of Move-1 code (from repo root; Julia tests not CI-gated):
```
julia test/test_program_space.jl      # :remove_rule cases + :add_rule/determinism unchanged
julia test/test_voc_gate.jl           # net_voc on the remove payoff + argmax-over-pair determinism
julia test/test_perturb_consumption.jl
julia test/test_email_agent.jl        # benchmark drift (if any), captured == thereafter
```
Then the **full** `test/test_*.jl` suite + lint corpus self-test (`python tools/credence-lint/credence_lint.py
test`) + `check apps/`, and **stop and report**. Skin smoke **optional** — `perturb_grammar`'s positional
signature is unchanged (the new behaviour is internal candidate generation), so the wire verb
`handle_perturb_grammar` is unaffected; run it anyway as a cheap consumption-surface confirmation.

`test_program_space.jl` / `test_voc_gate.jl` new assertions (repo `check(name, cond, detail)` idiom):
- **Remove a dead rule:** a grammar with an unreferenced rule + a `freq_table` whose
  `referenced_nonterminals` omits it ⇒ `perturb_grammar` removes exactly that rule (`==` on resulting
  rules, id excluded); `net_voc` of the removal `== log(2)·(1 + expr_complexity(body))` (oracle pragma).
- **Keep a referenced rule:** the same rule, now in `referenced_nonterminals` ⇒ **not** removed
  (structural no-op or an `:add_rule` instead).
- **Depth-1 reference is seen (R1):** a rule referenced *only* by a bare `NonterminalRef` is retained.
- **`:add_rule` degenerate reduction:** with no dead rule, `perturb_grammar` is `===` to today's
  `:add_rule` result (capture-before-refactor).
- **Determinism + tiebreak:** two runs on identical `(g, freq_table)` ⇒ structurally identical grammar;
  an add≡remove `net_voc` tie resolves to the §5-OQ2 order deterministically.
- **Saturation no-op (the Move-2 hook):** when neither add nor remove has `net_voc > 0`, `perturb_grammar`
  returns `g` with the **same id** (the prior-saturation signal).

Halt-the-line: any failure at end-of-PR is a halt; the branch never sleeps red.

## 8. Ratification + grounding amendments (2026-06-28)

Ratified by the owner. Four binding amendments folded in (the first is the §2 encoding correction above;
the rest refine §5/§7):

1. **Field encoding → `Union{Nothing, Set{Symbol}}`, default `nothing`** (§2, above). Forced: the
   empty-`Set` default is vacuously all-removable and conflates "analysed-empty" with "unknown."
2. **Q1 — resolved *free*, not a capture.** Pre-emptive grep done: `test_persistence.jl` has **zero**
   references to `SubprogramFrequencyTable` / `freq_table` / `analyse_posterior_subtrees` — the table is
   transient, recomputed each step. The schema bump needs **no fixture**. The four `SubprogramFrequencyTable(`
   call sites are `perturbation.jl:53` (the real builder — populates the field) + three hand-built *empty*
   tables (`test_program_space:530`, `test_perturb_consumption:25`, `test_voc_gate:83`) — all the 3-arg form
   the `nothing`-default constructor preserves.
3. **Q2 — total order is lexicographic `rule.name`, NOT `hash(Symbol)`.** Direction stays remove-first
   (hygiene: shrink before grow), but the within-tie total order is a **string compare on the name**, not a
   hash. Rationale (owner): Julia's `hash` is stable within a session but **not guaranteed across Julia
   versions** (the algorithm has changed between releases); since the entire point of the tiebreak is
   cross-version-reproducible determinism, a lexicographic name compare is stable where the hash is not, at
   no cost. The Phase-5-safety logic is ratified: the prohibition was on tiebreaking *unknown* values
   (feigned indifference you can't compute); here the values are *computed and exactly equal*, so any choice
   is genuinely argmax-optimal and a stable order is honest disambiguation, not laundered arbitrariness.
   *Owner note (recorded, changes nothing): the direction is consequence-free* — removal is prior-only, so
   the add and remove classes are independent (doing one never changes whether the other is positive); on a
   tie **both** fire across successive calls regardless of order, and the saturated fixed point is identical
   either way. Remove-first wins only on the thin Move-2 ground of keeping the saturated grammar minimal (a
   cleaner residual-read baseline). Taken on that basis.
4. **The reference walk is its own full-depth function** `collect_nonterminal_refs!`, recursing to **all
   depths and all branches** (predicate AND action branches of `IfExpr`), with a method per `ProgramExpr`
   subtype and **no generic fallback** (fail-loud on a future node type, matching `_extract!` /
   `expr_complexity`). It is **never** routed through `extract_subtrees(min_complexity)` — that is the exact
   Scope-B unsoundness (the `min_complexity=2` filter drops bare depth-1 references). Named and separate
   precisely so a future refactor cannot silently fold it back into the filtered extraction. The R1 test
   (§7, "depth-1 reference is seen") is the guard that fails against the unsound variant.
5. **Soundness pin — the prior-only claim is an assertion, not prose** (§7 addition). A `:remove_rule` test
   verifies, via an **independent oracle** (`show_expr` string-search for the rule name token, *not* the
   production `collect_nonterminal_refs`), that the removed rule is referenced by **zero** support programs
   — so the support set (hence the belief, on re-conditioning the same data) is bit-identical across the
   removal. "Belief untouched" is the whole soundness argument; it is directly testable.
