# Decouple Move 3 — credence-pi separable refit

Design doc per `docs/posture-4/DESIGN-DOC-TEMPLATE.md` (adapted for `decouple`, as
Move 2 did). Goal: make credence-pi **able to live in a separate repo** by moving its
brain math behind the skin wire.

## 0. Final-state alignment

Converges the tip toward `docs/decouple/master-plan.md` §"Final-state architecture":
credence-pi becomes *data + a thin body* that drives the engine over the wire, rather
than a co-released image that embeds Credence in-process. The substrate is already
right — `SparseStructurePrevision` and its conditioning (FiringByTag routing + 2ⁿ
mixture reweight) are Tier-1 today (`src/prevision.jl:310`, `src/sparse_structure.jl`,
exported `src/ontology.jl:45`); `apps/credence-pi/brain/feature_brain.jl` reimplements
no axiom op. This move lifts the **builder/wrapper + decision** layer (StructureBMA,
`build_prior`, `observe`, `belief_at_context`, the EU template) from the app into
engine stdlib and exposes it over the wire, so a non-embedding client can run the
full brain.

**Transient state left explicit (not drift):** (a) the daemon still *embeds*
(`apps/credence-pi/daemon/main.jl` `using Credence`) after this move — the move makes
embedding *optional* by giving a wire path; the literal repo extraction is a
follow-on (named in §6). (b) `feature_brain.jl` does **not** dissolve here — it
becomes a thin shim that keeps genuine credence-pi *domain* wiring (the harm/tail/
latency opt-in plumbing, the ask-vs-block effector policy, the per-request profile
override) and delegates the model/decision math to the lifted stdlib. (c) The
decision verb owns `net_voi` because no `voi`/`net_voi` verb exists on the wire today
(`apps/skin/server.jl:588` `SKIN_METHODS`); this is the only place the EU is assembled.

## 1. Purpose

Promote credence-pi's structure-BMA brain into engine stdlib (`src/structure_bma.jl`)
and a typed EU decision template (`src/stdlib.jl`), then expose both over the skin
wire as `structure_bma` / `structure_observe` / `belief_at_context` / `structure_decide`,
so the credence-pi daemon can drive every belief update and decision through JSON-RPC
with **zero probability/utility arithmetic in the body** — the precondition for
credence-pi living in its own repo.

## 2. Files touched

### Created
- `docs/decouple/move-3-design.md` — this doc.
- `src/structure_bma.jl` — the lifted builder: `StructureBMA` descriptor,
  `build_model`/`build_model_from_decls`, `build_prior`/`build_prior_dense`,
  `context_from_features`, `firing_tags`, `structure_observe` (today's `observe`),
  `belief_at_context`. Compositions over existing `SparseStructurePrevision` /
  `condition` / `with_components` (`src/stdlib.jl:98`); **no new frozen type, no new
  axiom-constrained function.** Included after `stdlib.jl` (`src/ontology.jl:1696`).
- `test/test_structure_bma.jl` — exact-oracle: lifted `build_prior`/`structure_observe`
  reproduce the in-app values bit-for-bit; sparse≡dense (mirrors
  `test/test_sparse_structure_equivalence.jl`); per-context routing
  (mirrors `test/test_product_bma_routing.jl`).
- `test/test_decision_template.jl` — exact-oracle for the EU template: the assembled
  block/ask/proceed EUs equal the hand-computed `c·(1+m)−c·[(1+m)+λ]·θ` / `voi−q` /
  multi-outcome `+H·θ_u` at rtol ~1e-12; reduces to single-outcome at `H=0,m=0`.
- `apps/skin/tests` smoke case (in `apps/skin/test_skin.py`): the §4 wire trace.

### Modified
- `src/stdlib.jl` — the typed EU decision template `decide_with_voi(...)` (peer of
  `voi`:154 / `net_voi`:167), taking typed utility slots (cost, λ, q, H, m, time) +
  the belief view, assembling the `LinearCombination`/`Projection` Functionals,
  folding `net_voi` as the `ask` action, and returning the `optimise` argmax. One op
  with an **optional harm coordinate** (multi-outcome when present); `H=0,m=0` reduces
  to the myopic single-outcome form (see §5 Q3).
- `src/ontology.jl:1696` (+1) — `include("structure_bma.jl")`; export the lifted names.
- `src/Credence.jl` — re-export `StructureBMA`, `build_model`, `build_prior`,
  `structure_observe`, `belief_at_context`, `decide_with_voi`.
- `apps/skin/server.jl` — four verbs in `SKIN_METHODS` (:588) + handlers: a
  **model registry** (descriptor handle, §5 Q1) keyed separately from `STATE_REGISTRY`;
  `structure_bma` (build_model + build_prior → `{model_id, state_id}`);
  `structure_observe` (`{model_id, state_id, context, observation}` → in-place
  `condition`, returns `{state_id, log_marginal}`); `belief_at_context`
  (`{model_id, state_id, context}` → `with_components` view → `{state_id}`);
  `structure_decide` (`{model_id, state_id, context, utility:{cost,lambda,q,harm,...},
  harm_model_id?, harm_state_id?, harm_context?}` → `decide_with_voi` → `{action}`).
  `PROTOCOL_VERSION` (:582) `1.1`→`1.2`.
- `apps/skin/protocol.md` — `Protocol-Version: 1.2` + changelog; document the four
  verbs, the model-handle lifecycle, and the typed utility-slot shape.
- `apps/credence-pi/brain/feature_brain.jl` — becomes a **thin shim**: `build_model`/
  `build_prior`/`observe`/`belief_at_context`/the structure types re-export from the
  engine; `wire_brain!`, the harm/tail/latency opt-in plumbing (:484-570), the
  effector policy (:590-606), and the per-request profile override (:345-347) stay.
  `decide`/`decide_multi` (:358-424) call `decide_with_voi` instead of assembling
  coefficients inline.
- `apps/credence-pi/daemon/server.jl` — `decide-action`/`observe-response` route
  through the wire verbs (or the lifted stdlib for the embedded co-released path);
  no behavioural change.
- `docs/decouple/master-plan.md` — Move-3 row: mark lift + verbs done; note the
  separate-repo extraction as the enabled follow-on.

### Not touched (explicit non-goals)
- No actual extraction of credence-pi to a separate git repo (this move *enables* it).
- No `serve_http` for the skin (stdio engine wire stands; the daemon's own HTTP/SSE is
  the co-released product surface, unchanged).
- No new frozen type; `SparseStructurePrevision` already exists.
- No TS wire-client change (`openclaw-plugin/src/daemon-client.ts` already speaks the
  daemon's HTTP/SSE and carries no math; it is unaffected by the skin-verb additions).

## 3. Behaviour preserved

- **`structure_observe` ≡ `feature_brain.observe` bit-for-bit.** The lifted function
  is the same `condition(top, FiringByTag(BetaBernoulli,Flat), obs)` call; the
  warm-posterior reconstruction from `apps/credence-pi/brain/warm_brain.counts.json`
  (and `harm_brain.counts.json`) via `reconstruct_posterior` must yield identical
  weights to the pre-lift path. `test/test_structure_bma.jl` pins this.
- **Sparse ≡ dense** preserved (`test/test_sparse_structure_equivalence.jl` extended
  to the lifted builders).
- **`decide_with_voi` ≡ inline `decide`/`decide_multi`.** The template assembles the
  identical `block_fn`/`eu_ask`/`fpa` and calls the same `optimise`; the multi-outcome
  joint is the same `ProductMeasure[wrap_in_measure(bxa), wrap_in_measure(bxu)]`.
  `apps/credence-pi/tests/julia/test_feature_brain.jl` +
  `apps/credence-pi/tests/julia/test_harm_governance.jl` must pass unmodified; the
  ClawsBench decisions (`apps/credence-pi/eval/`) are unchanged through the new path.
- **Skin backward-compat:** the four verbs are additive; existing verbs and
  `test_skin.py` cases are unchanged except the `1.1`→`1.2` protocol assertion.

## 4. Worked end-to-end example

Wire trace (stdio; the body holds no math, no belief state):

```
-> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocol":"1","dsl_sources":{"pi":"<features.bdsl + utility.bdsl>"}}}
<- {"jsonrpc":"2.0","id":1,"result":{"version":"0.1.0","protocol":"1.2","methods":[...]}}

; build the structure-BMA model + prior from declared features
-> {"jsonrpc":"2.0","id":2,"method":"structure_bma","params":{"features":[{"name":"tool","values":["bash","read"]},{"name":"repeat","values":["novel","repeated"]}],"p_edge":0.5,"alpha0":2.0,"beta0":2.0}}
<- {"jsonrpc":"2.0","id":2,"result":{"model_id":"m_1","state_id":"s_1","n_components":4}}   ; 2^2 structures

; learn on one (context, approve/deny)
-> {"jsonrpc":"2.0","id":3,"method":"structure_observe","params":{"model_id":"m_1","state_id":"s_1","context":["bash","repeated"],"observation":0}}
<- {"jsonrpc":"2.0","id":3,"result":{"state_id":"s_1","log_marginal":-0.69}}               ; in-place; structures reweighted

; decide on a context, shipping ONLY utility data
-> {"jsonrpc":"2.0","id":4,"method":"structure_decide","params":{"model_id":"m_1","state_id":"s_1","context":["bash","repeated"],"utility":{"cost":0.5,"lambda":1.0,"q":0.02,"expected_repeats":0.0}}}
<- {"jsonrpc":"2.0","id":4,"result":{"action":"block"}}
```

Module ownership of id:4:
1. `server.jl` `structure_decide` handler looks up descriptor `m_1` + belief `s_1`.
2. `belief_at_context(model, top, X)` (`src/structure_bma.jl`) → the transient view via
   `with_components` (`src/stdlib.jl:98`). No arithmetic.
3. `decide_with_voi(view, utility…)` (`src/stdlib.jl`) assembles `block_fn =
   LinearCombination([(-cost·(1+m+λ), Identity)], cost·(1+m))`, folds `net_voi`
   (`src/stdlib.jl:167`) as the `ask` constant, and returns `optimise(view,
   [:proceed,:block,:ask], fpa)`.
4. `:block` symbol serialises out. The body never saw θ, a coefficient, or a Functional.

## 5. Open design questions

1. **Descriptor handle vs. self-describing state.** `structure_observe` needs the
   `StructureBMA` descriptor (2ⁿ structures + `cell_tag`/`tag_lo`/`tag_hi`) to compute
   `firing_tags(X)` and the per-cell tags; the belief itself is the `MixturePrevision`.
   These are the *two Invariant-3 representations* (structural-analysis vs. belief). Do
   we (i) return a **separate `model_id`** (a model registry) alongside the belief
   `state_id` — clean separation, but the wire carries two handles per call — or (ii)
   fold routing metadata into the prevision so the state self-describes — one handle,
   but conflates the two representations? *Lean (i): the Invariant-3 separation is the
   whole reason `CompiledKernel`/`Program` are split; the wire should reflect it. The
   two-handle ergonomic cost is small and the body never inspects either.*

2. **`belief_at_context` as a verb at all.** `structure_decide` builds the view
   *server-side* (no wire round-trip). Is a separate `belief_at_context` *verb* worth
   it? *Lean: yes, but only for read-back* — shadow-mode logging and calibration want
   the per-context approval belief (`weights`/`mean`), and exposing the view as a
   `state_id` lets the existing `weights`/`expect` verbs read it. The hot decision path
   does **not** route through the wire-level verb (avoids a round-trip). Confirm we're
   not paying for a verb only telemetry uses — alternative: fold it into a
   `belief_summary`-style read.

3. **One decision template or two.** `decide`/`decide_multi` are two functions today;
   `decide_multi` provably reduces to `decide` at `H=0, m=0`
   (`feature_brain.jl:399-401`). Should `decide_with_voi` be **one** op with an optional
   harm coordinate (build the `ProductMeasure` joint only when harm features are
   present), or two ops? *Lean: one op.* The reduction is exact, the harm term is one
   extra `Projection(2)` in the same `LinearCombination`, and one op is one verb. Risk:
   the single-outcome path must stay bit-identical (the joint is *not* constructed when
   harm is absent) — pinned by the `H=0,m=0` reduction test.

4. **Where the EU template lives.** `decide_with_voi` as a typed `src/stdlib.jl` op
   (peer of `voi`/`net_voi`) vs. a skin-only helper. *Lean: stdlib op* — it is reusable
   decision machinery (any consumer with a linear-block-EU + VOI-ask shape wants it),
   and the constitution puts decision composition in the canalised stdlib, not in the
   wire layer. The skin verb is then a thin adapter, mirroring how `optimise` is.

## 6. Risk + mitigation

- **Lift drift.** The lifted `build_prior`/`structure_observe` must equal the in-app
  versions exactly. Mitigation: `test/test_structure_bma.jl` asserts bit-for-bit warm-
  posterior reconstruction from the committed `*.counts.json`; the app shim re-exports
  the engine functions (single source), so there is no second copy to drift.
- **Single-outcome regression via the merged template.** Q3's one-op design risks
  perturbing the single-outcome decision. Mitigation: the `H=0,m=0` reduction test +
  unmodified `test_feature_brain.jl`/`test_harm_governance.jl` + ClawsBench decision
  parity.
- **Model-handle lifecycle.** A `model_id` registry adds a second server-side lifetime
  to manage (create/destroy) beside `STATE_REGISTRY`. Mitigation: tie `model_id`
  destruction to `shutdown`/explicit `destroy`; the descriptor is immutable (built
  once), so no mutation hazard.
- **Protocol bump.** `1.1`→`1.2`; the smoke-build version grep is already version-
  agnostic (Move 2 fix), and the header==const unit invariant pins the exact value.
- **Review-process.** The named-template decision (Q2 in the plan) was settled with
  Guy; this doc records it and opens the genuinely-undecided tactics above.

## 7. Verification cadence

`julia test/test_structure_bma.jl` + `test/test_decision_template.jl` +
`test/test_sparse_structure_equivalence.jl` + `test/test_product_bma_routing.jl`;
`julia apps/credence-pi/tests/julia/test_feature_brain.jl` +
`test_harm_governance.jl` + `test_server.jl` (unchanged through the lift);
`python -m apps.skin.test_skin` (four-verb smoke + `1.2` + backward compat);
`credence-lint` corpus + `check apps/` (the shim + daemon read no accessor / do no
arithmetic — they call lifted ops); the ClawsBench eval (`apps/credence-pi/eval/`)
decisions unchanged; `docker build -f apps/credence-pi/daemon/Dockerfile` boots and a
stdio `structure_bma`→`structure_observe`→`structure_decide` round-trip succeeds.

## 8. de Finettian discipline self-audit

1. **Every numerical query through `expect`?** Yes. `structure_observe` →
   `condition`; `belief_at_context` → `with_components` (no arithmetic);
   `decide_with_voi` → `net_voi`/`optimise` (both route through `expect`). The
   `_structure_logweights` prior is declarative *construction* of log-weights (data),
   not a query of a prevision.
2. **Prevision-in-Measure / Measure-in-Prevision?** The multi-outcome path builds
   `ProductMeasure[wrap_in_measure(bxa), wrap_in_measure(bxu)]` — a Measure holding the
   sanctioned Measure *view* over each Prevision (the existing `decide_multi` shape);
   no Prevision holds a Measure. **Yes.**
3. **Opaque closure where declared structure fits?** No. The template takes typed
   utility slots and builds declared `LinearCombination`/`Projection` Functionals;
   `StructureBMA` is a declared descriptor, the kernels are `FiringByTag(BetaBernoulli,
   Flat)`. **Yes.**
4. **`getproperty` override on a Prevision subtype?** No. `StructureBMA` is a plain
   descriptor struct, not a Prevision; no shield added. **Yes.**

---

## Reviewer checklist
- [ ] §0 names the still-embedding daemon, the thin-shim `feature_brain.jl`, and the
      net_voi-in-decision-verb as explicit transient state.
- [ ] §5 questions are tactical (handle shape, verb surface, one-vs-two template), not
      master-plan re-litigation.
- [ ] §8 returns yes-or-justified on all four.
- [ ] file:line citations resolve against the `decouple/credence-pi` tip.
- [ ] No follow-up move needed to retract this one (the repo extraction is *enabled*,
      not *required-to-fix*).
