# Phase 5 design doc — VOC gate: retire the `rand` breach in `perturb_grammar`

> Seven-section template (`docs/collapse-towers/DESIGN-DOC-TEMPLATE.md`). Master plan:
> `docs/collapse-towers/master-plan.md`. Precedents: `docs/precedents.md`.
>
> **This is the crux phase and a design-doc gate.** The plan's stall clause: "if no cheap estimator
> survives scrutiny, Phase 5 stalls at the doc — do not implement a guess." **Disposition: it does
> NOT fully stall** — a cheap, *exact* estimator survives for a non-empty op set. But grounding the
> code surfaced **two material refinements to the master plan** that must be ratified before any code
> lands (this is the per-phase design doc tightening a master-plan overclaim against the code — the
> 5th instance of that pattern in this arc). They are the substance of §1 and §5. **No code is
> written until this doc is reviewed.**

## 1. Purpose

Phase 5 as scoped in the master plan: retire the `rand`-based meta-action selection in
`perturb_grammar` (`src/program_space/perturbation.jl:153–190`) — the live breach of Invariant 1,
which lists `perturb_grammar` as a *canalised* composition — by replacing it with an `argmax` over
candidate perturbations ranked by a **Value-of-Computation** functional `net_voc`, the structural
twin of `net_voi`. The *selection* becomes a deterministic `argmax`; the *surgery* stays a
meta-action (the boundary is preserved). The signature is unchanged, so the skin wire verb
(`handle_perturb_grammar`, `server.jl:1465`) is unaffected — no protocol bump. The outer
"perturb-or-not" decision is **already** EU-max at the hosts (`compute_gw_meta_eu` + `GW_PERTURB_COST`,
`grid_world/host.jl:173`; `expected_cost(cost_model, :perturb_grammar)`, `email_agent/host.jl:247`);
Phase 5 fixes only the inner "which perturbation" choice.

**The two refinements grounding forced (the heart of this doc):**

**(R1) — `net_voc` is necessarily in LOG-PRIOR currency, not utility currency.** `perturb_grammar`
receives only `(g, freq_table, available_features)` — *no belief, no utilities, no programs, no
re-conditioning* (the value model lives entirely in the hosts; `agent_state.jl` has no score
function, and the new grammar's value is realised only later, when the host *re-enumerates* from it
via `add_programs_to_state!`). The plan's formula `net_voc = E[value(belief after)] − value(now) −
compute_cost` writes `value` as the agent's achievable EU (utils). **But achievable EU is not
computable depth-one from `(g, freq_table)`** — it would need the belief, the utilities, and a
re-conditioning pass (precisely the expensive computation the metalevel is deciding whether to run,
and a signature change that would break the wire). The *only* depth-one-computable value of a grammar
perturbation from these inputs is its effect on the program-space **complexity prior** — the
description-length / log-prior mass over the posterior-weighted ensemble summarised in `freq_table`.
So:

> `net_voc(Δ) = Δcomplexity_logprior(Δ) − compute_cost` — `net_value(Δvalue, cost)` with `Δvalue` in
> **log-prior nats** (the program-space prior's own currency), the same functional FORM as `net_voi`
> (pure linear value − cost, no nonlinearity — it satisfies `net_value`'s invariant) but **not the
> same units** as a domain-utility EU.

**This is not a narrowing of the thesis — it is the thesis's resource-rational instantiation.** You
cannot price the perturbation's *utility* value depth-one from `(g, freq_table)` precisely because
that requires the belief, the utilities, and a re-conditioning pass — the forward inference the
metalevel is deciding whether to spend (the Russell–Wefald regress). So the affordable value-proxy at
the metalevel *is* the change in the complexity prior, and `net_voc` in log-prior nats is the
complexity-prior axiom doing exactly the job it exists for: standing in as the value-proxy when true
value is unaffordable (SPEC §1.3, the Occam/Solomonoff weighting that truncates the
reasoning-about-reasoning tower so it does not regress). It is the *third representation* in the
Phase-3 sense: `net_voi` (utils, scalar) and routing EU (utils, Functional-offset) were "two
representations of one semantics"; `net_voc` is a third — same form, in the prior's currency. For
`:add_rule` it is **exactly** `propose_nonterminal`'s `net_payoff` (scaled to nats by `λ = log(2)`),
folding the compression gate into `net_voc` as the plan asks: `net_payoff > 0` *is* `net_voc > 0` at
`compute_cost = 0`. The `value(now)` baseline is common to all candidates and cancels in the `argmax`
(the same "sunk common baseline cancels" structure as Phase 4's sunk EVPI).

**The honest precision (record it, so the headline is intact rather than overclaimed).** Phase 5
realises the *form* unification — "which perturbation" is now an `argmax` of `net_value`, not a
`rand` — but **not** SPEC's stronger ideal of *one combined `argmax` over the object-and-meta space
in a single currency*. That ideal needs object utility and meta value commensurable, which depth-one
myopia cannot deliver. In the landed architecture the two levels stay *separate and never summed*:
the **host** prices perturb-or-not in utility (`compute_gw_meta_eu` + `GW_PERTURB_COST`),
`perturb_grammar` prices which-perturbation in prior nats, and because they are never added together
they never need a common currency. The combined single-currency `argmax` is the *next* escape-mass
frontier — correctly out of reach here, and named as such rather than papered over. With that
precision recorded, the arc's headline ("one net-value functional; the metalevel is the same `optimise`
as the object level") is honest and intact: it is one *functional*, instanced in the currency each
decision context can afford.

**(R2) — the estimable op set is the COMPRESSION class only; the plan's "5+2 split" is corrected to
"2 + 3" (compression vs generative-change).** R1 has an immediate consequence: depth-one VOC over
`(g, freq_table)` can see *only* the prior, and **only compression ops change the prior.**
- **Compression class — re-describe the SAME hypotheses more/less compactly → `Δcomplexity_logprior
  ≠ 0`, net_voc-rankable:** `:add_rule` (define a frequent subtree as a nonterminal; `net_payoff`),
  `:remove_rule` (drop a nonterminal from the dictionary).
- **Generative-change class — change WHICH hypotheses the grammar generates → a LIKELIHOOD effect
  over programs not in the current ensemble (local or full escape-mass / Cromwell frontier),
  invisible to depth-one prior-only VOC by construction:** `:modify_threshold`, `:add_feature`,
  `:remove_feature`.

The plan put `:modify_threshold` (rands `:183/:187/:190`) in the *dissolving* five. **It belongs in
the deferred class:** a threshold constant is one symbol regardless of value (`expr_complexity` is
threshold-invariant), so changing it leaves `complexity_logprior` **identically zero** — its value
is a fit change over un-enumerated programs, exactly the escape-mass effect that sinks the alphabet
ops. So `:modify_threshold` joins `:add_feature`/`:remove_feature`, not `:add_rule`/`:remove_rule`.
The honest split is **2 net_voc-governed + 3 deferred**, not 5 + 2. (The deeper unification: "what
depth-one VOC can value" = "what changes the prior" = "the compression class" — three names for one
boundary.)

This is conformance, not new doctrine: `complexity_logprior` (Phase 1) and `net_value` (Phase 3) are
the templates; `net_voc` instances them at the metalevel; the `rand` is gone.

## 2. Files touched

- **`src/program_space/perturbation.jl`** — *modify*:
  - **Replace** the `rand`-based body of `perturb_grammar` (`:150–198`, the `ops = [...]; op =
    rand(ops); if/elseif` cascade) with: enumerate the **compression-class** concrete perturbations,
    compute `net_voc` for each, `argmax`, apply iff `net_voc > 0`, else a deterministic no-op. The
    generative-change ops (`:add_feature`/`:remove_feature`/`:modify_threshold` branches) are
    **removed** from the op set (see §5 OQ-3 for their deferral). Zero `rand` calls remain.
  - **Add** `net_voc(...)` (the `Δcomplexity_logprior − compute_cost` functional) + a
    `compression_candidates(g, freq_table)` helper that returns `(Grammar, net_voc)` pairs. New
    `compute_cost::Float64 = 0.0` keyword on `perturb_grammar` (mirrors Phase 4; default recovers
    `propose_nonterminal`'s gate bit-for-bit). The `replace_threshold` / `collect_threshold_nodes`
    machinery (`:248–306`) is retained only if Scope B keeps `:modify_threshold` — under the
    recommended scope it becomes **dead code**, deleted in the same commit (or kept with a `# NOTE:`
    if Guy wants `:modify_threshold` re-enabled by the deferred mechanism). `propose_nonterminal`
    (`:119–136`) is **unchanged** — it remains the `:add_rule` candidate generator; `net_voc` reads
    its `net_payoff`, it is not re-derived.
- **`src/net_value.jl`** — *modify (comment only)*: extend the docstring's instance list to name
  `net_voc` as the **log-prior-currency** instance (R1), so the currency distinction is recorded at
  the definition site, not just here.
- **`src/Credence.jl`** — *modify*: export `net_voc` if tests reference it by name (TBD by the test;
  `compression_candidates` stays internal).
- **`test/test_voc_gate.jl`** — *new* (see §7).
- **`test/test_program_space.jl`** — *modify*: **TEST 14** (`:515–544`, "modify_threshold produces
  grammar with modified thresholds", 200 random trials) asserts a now-retired random op. Rewrite it
  as a **determinism** test on the surviving op set (two runs on identical `(g, freq_table)` →
  structurally identical grammar) — see §6. TESTs 6/13/15 (`propose_nonterminal` requires the table;
  proposed nonterminals are real posterior subtrees) are **unaffected** (`propose_nonterminal` is
  unchanged).
- **`test/test_email_agent.jl`** — *modify if needed*: **TEST 12** (`:538–573`,
  `execute_meta_action!(:perturb_grammar)` adds components) may shift component counts under
  deterministic selection; update the expectation if it drifts (it asserts `n_added`, a count — see
  §6 risk on benchmark drift).

## 3. Behaviour preserved

- **`propose_nonterminal`'s gate is recovered exactly at `compute_cost = 0`.** Under the recommended
  Scope A, `perturb_grammar` becomes "take the `:add_rule` branch deterministically": `proposed =
  propose_nonterminal(freq_table); net_voc = net_payoff·log(2) − 0; net_voc > 0 ? add(proposed) :
  noop`. Since `net_payoff·log(2) > 0 ⟺ net_payoff > 0`, the *decision* (add this rule, or no-op) is
  **bit-identical** to today's `:add_rule` branch (`:168–173`). Tolerance: **exact** (`===` on the
  resulting `(feature_set, rules)`; the grammar `id` differs because `next_grammar_id()` is a
  counter — the test compares structure, not id). This is the degenerate reduction.
- **`propose_nonterminal` itself is untouched** — its frequency-argmax-then-`net_payoff>0` logic is
  preserved verbatim; `net_voc` consumes its output. So TESTs 6/13/15 stay green unchanged
  (strata-1, exact).
- **Skin wire unchanged (no protocol bump).** `perturb_grammar`'s positional signature `(g,
  freq_table, available_features)` is unchanged; the new `compute_cost` is a keyword defaulting to
  `0.0`, and `handle_perturb_grammar` (`server.jl:1478`) passes no keyword — bit-identical call.
- **The outer host EU-max is unchanged** — `compute_gw_meta_eu`, `GW_PERTURB_COST`,
  `expected_cost(:perturb_grammar)` are untouched; only what happens *after* the host decides to
  perturb changes.
- **Intended behaviour change (not preserved, by design):** the three generative-change ops
  (`:modify_threshold`, `:add_feature`, `:remove_feature`) no longer fire. Random grammar
  exploration *is* the breach; retiring it is the conformance fix (§5 OQ-3, §6).

## 4. Worked end-to-end example

Inputs: a grammar `g` with `feature_set = {:red, :green, :blue}`, `rules = []`, and a `freq_table`
whose top entry is the subtree `s = And(GT(:red, 0.7), LT(:green, 0.3))` (`expr_complexity(s) = 3`:
the `And` node + two leaves) appearing in `n_sources = 4` posterior-weighted programs. `compute_cost
= 0.0` (host passes no keyword).

Trace (Scope A — `:add_rule` only):
1. `perturb_grammar(g, freq_table, {:red,:green,:blue})` → `compression_candidates(g, freq_table)`.
   *(owner: `perturbation.jl`)*
2. The `:add_rule` candidate calls `propose_nonterminal(freq_table)` *(unchanged, `perturbation.jl`)*:
   `best = s`, `savings_per_use = expr_c − 1 = 2`, `rule_cost = 1 + expr_c = 4`, `net_payoff =
   n_sources·savings − rule_cost = 4·2 − 4 = 4 > 0` → returns `ProductionRule(:NT_xxxx, s)`.
3. `net_voc = net_value(net_payoff·log(2), compute_cost) = 4·0.6931 − 0.0 = 2.7726` nats. *(owner:
   `net_voc` → `net_value`, `net_value.jl`; `complexity_logprior` currency, `complexity.jl`)*
4. `argmax` over candidates (here the singleton `:add_rule`): best `net_voc = 2.7726 > 0` → apply:
   `Grammar({:red,:green,:blue}, [NT_xxxx → s], next_grammar_id())`. *(owner: `perturbation.jl`)*
5. **Determinism:** a second call on the same `(g, freq_table)` runs the identical `argmax` (no
   `rand`; `propose_nonterminal` is deterministic; `next_grammar_id()` is a counter) → same
   `(feature_set, rules)`, different `id`. The test asserts structural equality.

Contrast — what does **not** happen: with `net_payoff ≤ 0` (e.g. a subtree used by only `n_sources =
1` program: `net_payoff = 1·2 − 4 = −2`), `net_voc = −1.386 < 0` → deterministic **no-op** (today's
`isnothing(proposed)` / gate-fail return). No threshold is jittered, no feature is randomly added.

Scope B would add, in step 1, one `:remove_rule` candidate per existing rule, each with `net_voc =
Δdict_length·log(2) − compute_cost` (positive iff the rule is unreferenced dead weight); the `argmax`
in step 4 then ranges over add-vs-remove. Here `rules = []`, so Scope B is identical to Scope A.

No dual residency is introduced (one function's body is replaced; `net_voc` lives only in
`perturbation.jl`).

## 5. Open design questions

> **All three RESOLVED in review (2026-06-27); rulings inline below.** R1 and R2 both ratified —
> verified against the tree (`perturb_grammar`'s signature carries no belief/utility/programs;
> `expr_complexity(::GTExpr)=::LTExpr=1`, so a threshold node is one symbol regardless of its
> constant). Ship **Scope A** with **OQ-3(a)**; **Scope B** and the **EU-priced exploration budget**
> are tracked as named successors (§6, master plan).

**OQ-1 (the gate question) — RESOLVED: ratify R1, do not stall.** Framed affirmatively in §1: the
log-prior currency is *forced* (utility value is unaffordable depth-one — Russell–Wefald) and is the
complexity-prior axiom doing its job, not a defect; the form unification holds, the combined
single-currency `argmax` is correctly out of reach and named as the next frontier. *Original
position, retained for the record:* **proceed.** The
unification the arc claims is *functional-form* unification — `net_value(Δvalue, cost) = Δvalue −
cost`, pure linear, no nonlinearity — and `net_voc` satisfies it exactly; Phase 3 already established
that one semantics wears multiple representations (scalar `net_voi`, Functional-offset routing EU),
and `net_voc` is a third, in the program-space prior's own currency. The alternative — forcing
`net_voc` into utility currency — provably **cannot** be done depth-one from `(g, freq_table)`
without the belief + utilities + a re-conditioning pass, i.e. without changing the signature (wire
break) and re-incurring the cost the metalevel is deciding to avoid. So the currency is *downstream
of a plan decision* (keep the signature fixed), not a free choice. The counter-position worth your
ruling: if the headline requires a single shared currency across object- and meta-level, then the
honest move is to **stall** and escalate that the depth-one myopic metalevel can only price the
prior, not the utility — which is the escape-mass limit one level up. I do not think we should stall,
but R1 is a real narrowing of the plan's prose and you should rule on it.

**OQ-2 (scope) — RESOLVED: Scope A; Scope B tracked as real debt.** `:remove_rule`'s "is this rule
referenced" read off a lossy `freq_table` is estimable-not-provably-sound — exactly what the stall
gate guards — so ship A (estimator = `net_payoff`, exact and tested). Scope B is **not** someday-maybe:
it is *needed for long-run dictionary hygiene* (Scope A makes `perturb_grammar` monotonic — it only
ever grows the dictionary, never prunes), *blocked on a sound nonterminal reference count* the current
inputs cannot supply. Tracked in §6 + master plan.
- **Scope A (recommended floor):** `net_voc` governs `:add_rule` alone. `perturb_grammar` collapses
  to "deterministically `propose_nonterminal`-or-no-op." Maximal rigor: `:add_rule`'s value is
  `net_payoff`, **exact**, already computed, no imprecision; the degenerate reduction is bit-exact;
  the diff is minimal (delete four random branches). Cost: `perturb_grammar` only ever *grows* the
  dictionary — it never prunes.
- **Scope B:** add `:remove_rule` as a second compression candidate (`Δdict_length`: `+(1+|body|)`
  for an unreferenced rule — pure dictionary shrink, unambiguously prior-improving — vs a negative
  value for a referenced one). Completes the MDL compression *pair* (add/remove). Cost: telling
  "referenced" from "unreferenced" must be read off `freq_table` (scan its depth-≥2 subtrees for
  `NonterminalRef(r.name)`), and `freq_table` is a *lossy* summary (`min_complexity = 2` drops
  bare-reference programs) — so `:remove_rule`'s value is **estimable but not provably sound**
  (a rule referenced only in low-complexity contexts could be misread as dead). That soundness
  question is exactly the kind of thing the stall gate guards; I lean **A** for the floor and propose
  B only if you want the pruning and accept the reference-scan as a heuristic *within* the (already
  EU-gated, reversible) perturbation. Recommendation: **ship A; open an issue for B** once a precise
  reference count is available (it isn't, from the current inputs).

**OQ-3 (the deferred class) — RESOLVED: (a) exclude + defer; exploration budget is the named, adjacent
successor on the *capability* critical path.** R2 puts `:modify_threshold`, `:add_feature`,
`:remove_feature` beyond depth-one VOC by construction. The plan's rule is "never leave them random."
Three dispositions:
- **(a) Exclude + defer (recommended).** Remove all three from the op set; name the deferred
  mechanism (a non-myopic VOC, or an EU-priced *exploration budget* — the resource-rational form,
  consistent with the metareasoned-approximation direction). Honest, conservative, no new doctrine.
- **(b) Deterministic prior-select.** Build a feature-inclusion prior + threshold prior and `argmax`
  them. **Rejected as drafted:** no such prior exists for grammar features/thresholds today (the
  Phase-1 *edge* axis is `structure_bma`'s BN parent-set prior — a *different* consumer; claiming it
  "already supplies" a grammar-feature prior is itself an overclaim I am flagging), and a *static*
  prior + `argmax` is degenerate (it adds the single highest-prior feature, then that op is a no-op
  forever). Building a non-degenerate one is net-new mechanism = new doctrine = out of scope.
- **(c) Deterministic arbitrary tiebreak** (e.g. "first feature alphabetically"). **Rejected:**
  replacing `rand` with `first` is still non-EU action selection — it launders arbitrariness, it
  doesn't retire it. Violates the spirit of Invariant 1 as much as `rand`.
- **The drift you must weigh:** (a) is a *bigger* behaviour change than the plan implied — it cuts
  `perturb_grammar` from 5 ops to ≤2 and makes them deterministic, so grid_world/email_agent lose
  *all* grammar exploration (feature & threshold discovery) until the deferred mechanism lands. The
  constitutional read is that random exploration *was* the breach and any benchmark reliance on it is
  the very thing being retired — but the magnitude is real: the EU-priced exploration budget is now on
  the *capability* critical path, not merely the cleanliness path, because (a) leaves the agent unable
  to discover **any** new feature or threshold until it lands. Resolution: **(a)**, with the budget
  move sequenced as the *immediate, adjacent* successor to this arc — named in the Phase-5 commit and
  the master plan — so the gap between "breach closed" and "exploration restored" is as short as
  possible.

## 6. Risk + mitigation

- **Stall-gate misfire (claiming an estimator survives when it is a guess).** *Mitigation:* the
  surviving estimator is `:add_rule`'s `net_payoff`, which is **not new** — it is `propose_nonterminal`'s
  existing, tested arithmetic, merely re-read through `net_voc`. The degenerate test asserts the
  decision is bit-identical to today's `:add_rule` branch. Anything *beyond* the compression class is
  explicitly deferred (OQ-3), not guessed. So the phase ships exact arithmetic, not an approximation.
- **Benchmark drift (grid_world / email_agent).** *Failure mode:* deterministic compression-only
  perturbation changes which programs get enumerated → `test_email_agent.jl` TEST 12's `n_added`
  count and any grid_world behavioural assertion may move. *Blast radius:* `test_email_agent.jl:538`,
  `test_program_space.jl:515`; grid_world has no committed behavioural oracle on perturbation output
  (the loop is in `host.jl`, exercised but not asserted bit-exact). *Mitigation:* this drift is
  **intended** (greedy `argmax` replaces random surgery); update the expectations to the new
  deterministic values, captured post-change and asserted `==` thereafter — do **not** reintroduce
  randomness to keep an old number. Name the change in the commit message.
- **TEST 14 tests a retired mechanism.** *Failure mode:* TEST 14 (`:515`) loops 200 random trials
  expecting a `:modify_threshold` change; with the op removed it can never pass. *Mitigation:*
  rewrite it as the **determinism** assertion the plan requires — `g1 = perturb_grammar(g, ft); g2 =
  perturb_grammar(g, ft)` on identical inputs ⇒ `g1.feature_set == g2.feature_set && g1.rules ==
  g2.rules` (structural, id excluded). Add a second case: with a `freq_table` carrying a
  `net_payoff>0` subtree, both runs deterministically add the *same* rule; with an empty
  `dummy_table`, both are the same no-op.
- **Pre-emptive grep (done):** `grep -rn 'perturb_grammar\|propose_nonterminal' src/ apps/ test/`
  → callers are `grid_world/host.jl:215`, `email_agent/host.jl:450`, `server.jl:1478` (all pass the
  3 positional args, none pass `compute_cost` — the `0.0` default keeps them bit-identical), and the
  tests above. `rand` in `perturbation.jl`: 7 calls at `:153/:158/:164/:177/:183/:187/:190`, **all**
  removed (verified against the file). No `rand` elsewhere in the perturbation path
  (`next_grammar_id` is a counter, `agent_state.jl`).
- **Lint.** `perturb_grammar` is a stdlib composition on the canalised path (Invariant 1 topological);
  the new `argmax`-over-`net_voc` selection *is* the canalisation that removes the breach. No pragma
  needed — the whole point is that the selection now routes through an EU/`net_value` form. (Confirm
  the lint corpus self-test + `check apps/` stay green; no precedent slug is invoked.)

## 7. Verification cadence

End of Phase-5 code (from repo root):
```
julia test/test_voc_gate.jl
julia test/test_program_space.jl
julia test/test_email_agent.jl
```
Then the full `test/test_*.jl` suite + the lint corpus self-test (`python
tools/credence-lint/credence_lint.py …`) + `check apps/`, and **stop and report**. Whole-arc
end-to-end (the consumption surface, since this is the last phase): `JULIA_PROJECT=. uv run python
apps/skin/test_skin.py` (wire smoke, exercises `perturb_grammar`) + the Python workspace pytest.

`test_voc_gate.jl` (repo `check(name, cond, detail)` idiom):
- **Degenerate reduction:** for a `(g, freq_table)` whose best subtree has `net_payoff > 0`,
  `perturb_grammar(g, ft; compute_cost=0.0)` produces the **same** `(feature_set, rules)` as today's
  `:add_rule` branch (`propose_nonterminal`-then-add); for `net_payoff ≤ 0`, a structural no-op.
  `===`/`==` on structure (id excluded).
- **Directional:** higher `net_payoff` (more sources / larger compressible subtree) preferred at
  equal cost; a large `compute_cost` (> `net_payoff·log(2)`) suppresses the perturbation to a no-op
  (the flip asserted relative to `net_payoff·log(2)`, no magic number). The independent oracle
  recomputing `net_payoff·log(2)` carries the `test-oracle` pragma.
- **Determinism:** two `perturb_grammar` calls on identical `(g, freq_table, features)` ⇒
  structurally identical grammars (the metalevel choice is a deterministic `argmax`, no `rand`).
- **No-`rand` structural guard:** assert (via `methods`/source, mirroring TEST 6's style) that the
  perturbation path contains no `rand` call — the breach is closed at the type/source level, not just
  behaviourally.

Halt-the-line: any failure at end-of-PR is a halt; the branch never sleeps red.
