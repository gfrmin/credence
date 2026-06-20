# Decouple Move 2 — pure-functional MAUT ranker enablement

Design doc per `docs/posture-4/DESIGN-DOC-TEMPLATE.md`. Reprioritized ahead of the
life-agent pilot (Move 1).

## 0. Final-state alignment

Converges the tip toward `docs/decouple/master-plan.md` §"Final-state architecture":
the **skin wire is the only consumption surface**, and an external app (rssfeed)
declares its domain as inline BDSL and carries no probabilistic Julia. The governing
principle is **pure functions over stateful computation, with a clean division of
responsibility** — so the engine/model/skin path is pure and **state is confined to
the consumer**. Transient state left explicit (not drift): (a) the master-plan's
original Move-2 line (program-space PlackettLuce) is **superseded** by MAUT — this
move edits the table; (b) `apps/julia/rss/` stays **reference-only** (read-order is
noise for a swipe-reading user); (c) `call_dsl`'s belief-return semantics are
**specified here for its first consumer** — it is defined but currently invoked
nowhere, so this is greenfield, not a breaking change.

## 1. Purpose

Let rssfeed's parametric MAUT ranker run entirely over the skin wire (stdio) by
making `call_dsl` a **pure belief-function evaluator** — beliefs cross the wire as
declarative `{type, params}` specs — plus a family registry so the model can declare
graded/signed kernels via `:family`. No new transport, no skin state for this path.

## 2. Files touched

### Created
- `docs/decouple/move-2-design.md` — this doc.
- `examples/maut_demo.bdsl` — domain-neutral pure MAUT model.
- `examples/maut_demo_wire.py` — runnable end-to-end stdio example (the §4 trace).
- `test/test_family_registry.jl` — registry dispatch + conjugate values.
- `test/test_prevision_params.jl` — `params`/`build_belief` round-trip per subtype.

### Modified
- `src/prevision.jl` — `params(p::Prevision)` per subtype (adjacent to each struct:
  `BetaPrevision`:233, `GaussianPrevision`:294, `GammaPrevision`:306,
  `DirichletPrevision`:364…), returning `(type_tag, sufficient_stats)`. Export
  `params`. **Not** added to the BDSL `default_env`.
- `src/kernels.jl` — `const FAMILY_REGISTRY` + `register_family!`; self-register
  `bernoulli/flat/soft/weighted/normal` adjacent to the structs (19,26,40,41,44–46).
- `src/eval.jl:548–577` — the `:family` loop + `_parse_family_keyword` dispatch
  through `FAMILY_REGISTRY` (look up keyword, consume declared `arity` of trailing
  numerics, construct).
- `apps/skin/server.jl` — extract pure `build_belief(spec)` from `handle_create_state`
  (608+); `create_state` = `build_belief` + register. `resolve_arg` (489–505): a
  belief-spec dict → `build_belief` (no register). `serialize_value` (511–527): belief
  branch (`Measure`/`Prevision`, and vectors thereof) → `{type, params}` via the
  `params` protocol (no `register_state`). `PROTOCOL_VERSION` `1.0`→`1.1`.
- `apps/skin/protocol.md` — `Protocol-Version: 1.1` + changelog; document the
  `call_dsl` belief round-trip + the shared belief-spec shape.
- `examples/router.bdsl:22–29` — extend the `:family` note.
- `docs/decouple/master-plan.md` — Move-2 row (MAUT supersedes PlackettLuce). *(done)*
- `apps/julia/rss/README.md` — mark **reference-only**.

### Not touched (explicit non-goals)
- No `serve_http.jl` (stdio engine wire). No `read_params` *verb* (the pure round-trip
  replaces it; deferred until a stateful consumer needs it). No soft/weighted *skin*
  kernel specs (graded conditioning is BDSL-side via `:family`). No named/keyed measure.

## 3. Behaviour preserved

- `build_belief` extraction is behaviour-neutral: `create_state` produces the same
  registered states (it now calls `build_belief` then registers).
- The `:family` registry preserves `:bernoulli`→`BetaBernoulli()`, `:flat`→`Flat()`
  bit-identically (`test_family_registry.jl`).
- `serialize_value`'s belief branch has **zero blast radius**: `call_dsl` is invoked
  nowhere today (audited — no apps/tests/examples call it), so no consumer depends on
  the prior `register_state` return for a belief. Numeric/`{value}` paths and the
  opaque-fallback `register_state` are unchanged.
- `params` is new surface (no prior behaviour); the round-trip test pins
  `params(build_belief(serialize(params(p)))) == params(p)` bit-exact.
- Existing `apps/skin/test_skin.py` passes unmodified except the `initialize` protocol
  assertion (`1.0`→`1.1`).

## 4. Worked end-to-end example

`examples/maut_demo.bdsl` (excerpt — real, loadable; the model speaks *beliefs*):

```lisp
; signed Gaussian weight; obs-noise sigma declared on the kernel via :family
(define grade-kernel
  (kernel (space :euclidean 1) (space :euclidean 1)
          (lambda (mu) (lambda (o) o))
          :family normal 0.3))
; pure: belief in -> belief out. No raw params, no state.
(define observe-one (lambda (w obs) (condition w grade-kernel obs)))
; pure: beliefs + feature -> number, via expect (never reads alpha/beta)
(define score-one (lambda (w feat) (* (mean w) feat)))
```

Dispatch trace (module ownership):
1. `:family normal 0.3` → `eval.jl:551` loop → `FAMILY_REGISTRY[:normal] =
   (NormalNormal, 1)` → consume `0.3` → `NormalNormal(0.3)` (kernels.jl:44).
2. Host calls `call_dsl("model","observe-one", [{"type":"gaussian","mu":0.0,"sigma":1.0}, 0.7])`.
3. `resolve_arg` → `build_belief({type:gaussian,…})` → `GaussianMeasure` (no register).
4. closure → `condition` → `maybe_conjugate` (conjugate.jl:87) → posterior
   `GaussianMeasure`.
5. `serialize_value(posterior)` → `params(prevision)` → `{type:"gaussian", mu:0.0644,
   sigma:0.287}` (no register).

Wire (stdio; host holds no math, no state):

```
-> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocol":"1","dsl_sources":{"model":"<maut_demo.bdsl>"}}}
<- {"jsonrpc":"2.0","id":1,"result":{"version":"0.1.0","protocol":"1.1","methods":[...]}}
-> {"jsonrpc":"2.0","id":2,"method":"call_dsl","params":{"env_id":"model","function":"observe-one","args":[{"type":"gaussian","mu":0.0,"sigma":1.0},0.7]}}
<- {"jsonrpc":"2.0","id":2,"result":{"type":"gaussian","mu":0.0644,"sigma":0.287}}   ; belief-spec out, no state_id
```

rssfeed persists `{type,gaussian,mu,sigma}` in Postgres; next request sends it back as
the belief-spec arg. Pure round-trip; the skin registry is never touched.

## 5. Open design questions

1. **Belief-spec round-trip symmetry.** Should `serialize_value` emit exactly the
   `create_state` belief-spec shape (`{type:"beta", alpha, beta}`) so `resolve_arg`
   round-trips it verbatim, or a distinct `{type, params:[…]}` envelope? *Lean: reuse
   the `create_state` shape — one belief-spec format, in and out, built by the shared
   `build_belief`.*
2. **`params` naming.** `params(p::Prevision)` vs `parameters`/`sufficient_stats` —
   the kernel struct has a `k.params` *field* (a Dict); confirm the function name
   doesn't read ambiguously at call sites.
3. **Vector-of-beliefs returns.** `observe-batch` over a positional weight list
   returns a list of beliefs → `serialize_value` maps the belief branch over a vector.
   Confirm the `all(Number)` numeric-vector guard (line 518) is checked *before* the
   belief-vector branch so score vectors still serialize as `{value:[…]}`.

## 6. Risk + mitigation

- **`serialize_value` belief branch ordering.** A score vector (numbers) must hit the
  numeric branch, not the belief-vector branch. Mitigation: keep the `all(x->x isa
  Number)` guard first; the belief-vector branch matches only vectors of
  `Measure`/`Prevision`. Test both.
- **`build_belief` extraction regressions `create_state`.** Mitigation: `create_state`
  = `build_belief` + `register_state`; existing `test_skin.py` `create_state` cases
  (beta/gaussian/product/mixture/program_space) must pass unchanged.
- **Family registry load ordering.** Families must register before BDSL eval.
  Mitigation: `register_family!` calls in `kernels.jl` (loaded before `eval.jl` runs);
  test asserts all five keywords resolve.
- **Lint accessor-exclusion mismatch (latent).** `_ACCESSOR_EXCLUDED_FILES` names
  `previsions.jl` but the file is `prevision.jl`. Harmless (CI lints only `apps/`;
  `server.jl` calls `params()`, never an accessor). **NOTE only, not fixed here.**
- **Review-process.** The pure design was settled in live review; this doc records it.

## 7. Verification cadence

`julia test/test_core.jl` + `test/test_family_registry.jl` + `test/test_prevision_params.jl`;
`python -m apps.skin.test_skin` (belief round-trip + `1.1` + `create_state` backward
compat); `credence-skin-client` smoke (stdio `command` seam); `examples/maut_demo_wire.py`;
`credence-lint` corpus + `check apps/` (`server.jl` reads no accessor — calls `params()`);
`docker build -f Dockerfile.skin` + a stdio `initialize`→`call_dsl` belief round-trip.

## 8. de Finettian discipline self-audit

1. **Every numerical query through `expect`?** `params(p)` returns sufficient
   statistics for *serialization* (the Invariant-3 serialization view, peer to
   `mean`/`expect`), not a decision-feeding `Float64`; scoring routes through `expect`/
   `mean`. **Yes, with justification.**
2. **Prevision-in-Measure / Measure-in-Prevision?** Neither — `params` dispatches on
   `Prevision`; `params(m::Measure)=params(m.prevision)` reads the existing wrap.
   **Yes.**
3. **Opaque closure where declared structure fits?** No — `FAMILY_REGISTRY` is
   declared structure (keyword→constructor+arity); `params` is per-type dispatch.
   **Yes.**
4. **`getproperty` override on a Prevision subtype?** No — `params` is a function over
   existing fields. **Yes.**

---

## Reviewer checklist
- [ ] §0 names the superseded master-plan line + reference-only `apps/julia/rss` as
      explicit transient state.
- [ ] §3 justifies the `serialize_value` change as zero-blast-radius (call_dsl unused).
- [ ] §8 returns yes-or-justified on all four.
- [ ] file:line citations resolve against the `decouple/rssfeed` tip.
- [ ] No follow-up needed to retract this move.
