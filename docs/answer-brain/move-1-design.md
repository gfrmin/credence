# answer-brain — Move 1 design (Stage 1: the brain daemon, native port + parity)

Follows `docs/posture-4/DESIGN-DOC-TEMPLATE.md`. Strategy is settled in `master-plan.md`; this doc
wrestles the tactics of porting the Stage-0 decision math to native Julia/Credence with parity.

## 0. Final-state alignment

This move converges the tip toward `master-plan.md` §"Final-state architecture" by moving the
candidate-posterior + EU decision from Python-builds-spec / skin-evaluates into a **native Julia
brain** that owns the belief. It deliberately leaves transient state: (a) the **HTTP/SSE daemon
surface** (`server.jl`) is *not* built here — its sensor-event schema is defined by the pi-mono body
(Stage 2), so building it now would be speculative and require rework; only the brain math + the
replayable observation log land. (b) life-agent keeps driving the **skin** in production — the brain
is exercised by tests this move, not yet by the body. Both are explicit, not drift: the brain's
public function (observations → posterior/decision) is the Stage-2 cut-over seam and is frozen here.

## 1. Purpose

Port the validated Stage-0 answerer's *decision core* — the tempered candidate posterior
(`lookup_posterior`) and the EU decision (`decide`, plus the new `net_voi`-priced gather/ask gate) —
from `life-agent`'s Python (`src/life_agent/core/lookup.py`, `core/gather.py`) into native Julia
(`apps/answer-brain/brain/answer_brain.jl`) on `src/Credence`, declare its effectors/utility in
`bdsl/*`, and prove **parity**: on shared fixtures the Julia brain produces the same posterior
weights and the same chosen effector as Stage 0, and the observation-log replay reconstructs the
posterior exactly.

## 2. Files touched

Created (this app):
- `apps/answer-brain/brain/answer_brain.jl` — the port. Pure builders (`temper_scales`,
  `observation_densities`, `action_utilities`) + `candidate_posterior` (CategoricalMeasure + tabular
  PushOnly kernels + `condition`) + `terminal_decide` (`optimise` over the action set) + `voi_gather`
  (Credence `net_voi`). ~220 lines.
- `apps/answer-brain/bdsl/capabilities.bdsl` — effectors `answer` / `ask-user` / `abstain` /
  `gather`.
- `apps/answer-brain/bdsl/utility.bdsl` — the stated channel parameters as declared constants
  (`_A_ALTERNATIVES`, `_BETA_ANCESTRY`, `_BETA_MODEL`, `_P_NONE_PRIOR`, `_ORACLE_P`, the §4.1
  covariate priors). The owner's Ū is *input* (from the body), not declared here.
- `apps/answer-brain/bdsl/features.bdsl` — the candidate-set state vocabulary (dispersion band,
  leader credence, era-split, owner-scoped) — declared minimal; extractors are Stage 2.
- `apps/answer-brain/daemon/observation_log.jl` — adapted from `apps/credence-pi/daemon/`: an
  append-only JSONL log of *grounded observations*, with a replay that reconstructs the posterior.
- `apps/answer-brain/daemon/main.jl` — skeleton entrypoint (loads Credence; no HTTP yet).
- `apps/answer-brain/tests/julia/test_answer_brain.jl` — parity + replay tests.
- `apps/answer-brain/tests/fixtures/stage0_parity.json` — generated from `life-agent@4b336db`.
- `apps/answer-brain/tests/fixtures/README.md` — fixture provenance (SHA-pinned).
- `apps/answer-brain/README.md`.

Created (in `life-agent`, the fixture generator — separate repo, separate commit):
- `scripts/dump_parity_fixtures.py` — runs `lookup_posterior` + `decide` on synthetic observation
  sets and dumps `(observations, rho, u_bar) → (candidates, weights, action, eu)`.

Deferred to Stage 2 (named so this move does not silently drop them): `apps/answer-brain/daemon/
server.jl` (HTTP/SSE), `apps/answer-brain/extension/*` (pi-mono TS body), the pi-mono app shell.

No file in `src/` is modified — the port consumes the existing Credence surface only.

## 3. Behaviour preserved

The reference is Stage-0 Python at `life-agent@4b336db`. The fixtures capture, per case,
`(candidates, weights, p_none, action, eu)` from `lookup.lookup_posterior` + `lookup.decide`. The
Julia brain must match:
- **Posterior weights** — to `atol=1e-9` (both call the same `condition(::CategoricalMeasure, …)`,
  ontology.jl:861; the only port surface is the density-matrix construction, so a mismatch is a
  porting bug in `observation_densities` or `temper_scales`).
- **Chosen effector** — exact. With the `report_j` action formulation (Open Q1), `report_{j*}` maps
  to Stage-0 `report`; `hedge`/`ask_clarify`/`abstain` map directly.
- **EU of the chosen action** — to `atol=1e-9`.

Cases (each a fixture): single observation; two observations same document (ancestry temper); two
documents (model temper); covariate `subject_factor`/`time_factor` ≠ 1; NONE-dominant (junk
agreement); dispersed (→ abstain/hedge); clear-leader-below-bar (→ ask_clarify or abstain). Any
divergence is a bug, not a benign reassociation — there is no RNG and no quadrature on this path
(CategoricalMeasure expectations are exact sums, ontology.jl:690).

## 4. Worked end-to-end example

One observation reporting candidate-0; `k=2` candidates + NONE; `rho=0.5`, `authority=0.9`,
`subject_factor=time_factor=1.0`, single obs ⇒ `scale=1.0`, `A=10`, `P_NONE=0.5`. Copy-pasteable
against `src/Credence` (`push!(LOAD_PATH, "…/src"); using Credence`):

```julia
atoms = Float64.(0:2)                       # cand0, cand1, NONE
prior = CategoricalMeasure(Finite(atoms), log.([0.25, 0.25, 0.5]))
r = 0.5 * 0.9                               # = 0.45  (rho · authority · subject · time)
lm = log(r + (1-r)/10)                      # log_match  ≈ -0.6833
lo = log((1-r)/10)                          # log_miss   ≈ -2.9004
dens = [[lm, lo], [lo, lm], [lo, lo]]       # rows: V=cand0, V=cand1, V=NONE ; cols: reported t
src, tgt = Finite(atoms), Finite(Float64.(0:1))
kern = Kernel(src, tgt, _->error("gen"),
              (h,o)->dens[Int(h)+1][Int(o)+1]; likelihood_family = PushOnly())
post = condition(prior, kern, 0.0)          # observation reports candidate 0
weights(post)                               # ≈ [0.754, 0.082, 0.164]  (cand0, cand1, NONE)
```

Decision (illustrative Ū: `u_correct=1, u_wrong=-5, u_hedged=-0.5, u_abstain=0, oracle_p=0.9,
lambda_int=0.05` — NOT the frozen priors, a trace):

```julia
report0 = Tabular([1.0, -5.0, -5.0])        # u_c at cand0, u_w else
report1 = Tabular([-5.0, 1.0, -5.0])
hedge   = Tabular([-0.5, -0.5, -5.0])       # misleads only when truth is NONE
ask     = Tabular(fill(0.9*1.0 - 0.05, 3))  # oracle price, candidate-agnostic
abstain = Tabular([0.0, 0.0, 0.0])
fpa = Dict("0"=>report0, "1"=>report1, "2"=>hedge, "3"=>ask, "4"=>abstain)
optimise(post, ["0","1","2","3","4"], fpa)  # EU(report0)=0.754-5·0.246=-0.476 < EU(abstain)=0 ⇒ "4"
```

`expect(post, report0)` (ontology.jl:690) = `0.754·1 + 0.082·(-5) + 0.164·(-5) = -0.476`; abstain
EU `0` wins — credence 0.754 sits below the `u_wrong`-implied report bar, so the brain abstains
(holding cand0 as the named, withheld leader). This is the Stage-0 behaviour, reproduced natively.
The argmax over `report_j` is done by `optimise` (the single decision mechanism), never by reading
`weights` in the app — see Open Q1.

## 5. Open design questions

**Q1 — `report` formulation under Invariant 1.** Stage-0 Python builds one `report` action whose
utility vector bakes in `j* = argmax(weights)` (`lookup.action_utilities`, lookup.py:606). Porting
that literally puts an *argmax-over-weights to select behaviour* inside the constitution's
jurisdiction — the `compute-on-weights` / `sort-for-display` precedents say comparing weights to
branch action selection is the violation `optimise` exists to absorb. **Recommend (a):** declare K
`report_j` actions, each a `Tabular` rewarding atom `j`; `optimise` over
`{report_0…report_{k-1}, hedge, ask-user, abstain}` picks `report_{j*}` *because* it maximises EU —
provably the same value as Stage-0's `report` EU (`weights[j*]·u_c + (1−weights[j*])·u_w`), so parity
holds, and the argmax lives in the single decision mechanism. The parity test maps `report_j →
report`. Alternative (b): keep one `report` action + an explicit `compute-on-weights` pragma. (a) is
more de Finettian *and* arguably more correct — the action space genuinely contains "report value
j". I propose (a). Push back if the K-way action space is judged to muddy the Stage-2 effector
manifest (the body emits a single `answer` effector regardless).

**Q2 — daemon scope this move.** The plan text says "copy credence-pi's `daemon/*`". `credence-pi`'s
`server.jl` (673 lines) is an HTTP/SSE sensor→signal loop whose event schema (`tool-proposed`,
`user-responded`, `tool-completed`) is the *pi-mono body's*, undefined until Stage 2.
**Recommend:** build only `observation_log.jl` (the replay is a Stage-1 success criterion) + a
`main.jl` skeleton this move; **defer `server.jl` to Stage 2**, where the body fixes the schema. This
is YAGNI-correct and avoids building-then-reworking a wire surface against a guessed schema (the
template's reviewer checklist rejects a move that needs a later move to retract it). Push back if you
want the HTTP surface stubbed now for an earlier end-to-end smoke.

**Q3 — `net_voi` decide: parity-bound or new?** Stage-0's gather loop is *unconditional* — it always
gathers on the top candidates, then decides among the terminal actions; there is no VOI gate in the
Python, so **`net_voi` has no Stage-0 parity counterpart**. The plan names "the `net_voi` decide" as
a Stage-1 deliverable. **Recommend:** build `voi_gather` (Credence `net_voi`, stdlib.jl:156) as the
brain's native forward capability and unit-test it for *correctness* (it prices a discriminating
probe above cost, a non-discriminating one below), but scope **parity strictly to the posterior +
the terminal {report/hedge/ask/abstain} decide**. Stating this honestly keeps the parity claim
clean (no fabricated Python oracle for a path Python never ran). Push back if you'd rather defer
`net_voi` entirely to Stage 2 and make Stage 1 pure terminal-decide parity.

**Resolution (author-approved, in conversation).** Q1 → **(a)** K `report_j` actions; the
argmax lives in `optimise`, no compute-on-weights in the app. Q2 → **defer `server.jl` to
Stage 2**; build only the observation log + replay + a `main.jl` skeleton. Q3 → **build
`voi_gather` + correctness-test it**, with parity scoped strictly to the posterior + terminal
decide. All three are implemented as resolved; the parity + net_voi + replay tests are green
(57 checks, 10 cases, `atol=1e-9`) and `tools/credence-lint check apps/answer-brain` is clean.

## 6. Risk + mitigation

- **Silent port drift** in `observation_densities`/`temper_scales` (the only hand-translated
  arithmetic). *Caught by:* the weights-parity fixtures at `atol=1e-9` across the 7 cases incl.
  ancestry/model tempering and covariates — a transposed index or wrong exponent fails immediately.
- **Tie-break divergence.** Stage-0 routes through the skin's `_sorted_action_keys` numeric sort +
  strict `>` (server.jl:854). *Mitigation:* the brain iterates the action keys in the same
  numeric-sorted order with the stdlib `optimise`'s strict `>` (stdlib.jl:112); a fixture with a
  near-tie pins it.
- **Invariant 1 violation** by computing on weights in the app. *Mitigation:* Q1's `report_j`
  formulation removes the only argmax-on-weights; `tools/credence-lint` `check apps/answer-brain`
  runs at end-of-move; any residual grey-zone gets a named precedent pragma, not silent passage.
- **Float path difference** Python↔Julia. *Mitigation:* both reduce to the same log-sum + exp-
  normalise; `atol=1e-9` is comfortably above FP noise for K≤~20 candidates. If a case ever needs a
  looser tol, that is reported (CLAUDE.md §"Performance problems" discipline), not silently widened.

## 7. Verification cadence

End-of-move: `julia apps/answer-brain/tests/julia/test_answer_brain.jl` (parity + replay) green;
`python tools/credence-lint/credence_lint.py check apps/answer-brain` clean; the life-agent fixture
generator's own unit check green in that repo. Julia core tests are not CI-gated (CLAUDE.md), so the
brain test runs locally and its result is reported with the actual pass count.

## 8. de Finettian discipline self-audit

1. **Every numerical query routed through `expect`?** Yes. Posterior weights come from `condition` +
   `weights`; every action EU is `expect(::CategoricalMeasure, ::Tabular)`; VOI is `net_voi` →
   `value` → `expect`. The builders (`observation_densities`, `action_utilities`) return *declared
   data* (a `Kernel`'s log-density matrix, a `Tabular`'s values) — `declarative-construction`
   precedent — not probabilistic properties of a prevision. No function returns a `Float64`
   describing the posterior without `expect`.
2. **Prevision inside a Measure or vice-versa?** No. The brain holds a `CategoricalMeasure` and
   conditions it; no Prevision is wrapped or unwrapped by the app.
3. **Opaque closure where declared structure fits?** The per-observation kernel uses the same
   closed-over log-density-matrix `Kernel` the skin uses for `tabular_log_density` (server.jl:307) —
   a `PushOnly` leaf, declared, not an `OpaqueClosure`. Utilities are `Tabular`, the declared finite-
   space functional. No `OpaqueClosure` is introduced.
4. **`getproperty` override on a Prevision subtype?** No — the app adds no methods to `src/` types.

## Reviewer checklist

- [x] §0 names transient state explicitly (deferred `server.jl`; skin still drives production).
- [x] §5 holds three non-trivial questions Claude argues a position on.
- [x] §8 returns "yes" on all four with the declarative-construction justification named.
- [x] file:line citations present for every current-state reference.
- [x] The move needs no subsequent move to retract it (deferred files are additive, not rework).
