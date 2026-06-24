# Decouple Move 4 — routing-brain lift

## 0. Final-state alignment

Move 3 lifted credence-pi's **governance** brain into engine stdlib + skin verbs
(`structure_bma`/`structure_observe`/`structure_decide`), making it wire-drivable, and
*enabled* — but explicitly deferred — the literal repo extraction. Move 4 closes the
remaining gap so the **entire** credence-pi reasoning surface is wire-drivable: it lifts
the **routing** brain. Move 5 (separate doc) then extracts credence-pi to its own repo
(`credence-openclaw`) as a pure wire consumer that pins the `credence-skin` image.

After Move 4, `apps/credence-pi/brain/routing_brain.jl` carries **zero** probabilistic
Julia (a shim, like `feature_brain.jl`); the routing math lives in `src/routing.jl` and
is reachable over the wire via six additive skin verbs at protocol **1.5**.

## 1. Purpose

credence-openclaw will be an *external* app → it must consume the engine across the skin
wire only, carrying no probabilistic Julia (`SPEC §6.9`; the principle "credence must
not enable an external app to host its own brain"). Governance is already wire-ready;
routing is not. The routing **decisions** (`route`/`escalation_next`) and the **online
confound-learning** (`decode_correctness` + the coupled-EM `route_outcome!` over ρ/σ
emission Betas + the Gamma latency belief) live only in `routing_brain.jl`, with no
routing skin verbs. Move 4 lifts them.

Routing **reuses** the already-lifted `StructureBMA` substrate
(`build_structure_prior`/`structure_observe`/`structure_observe_soft`/
`belief_at_context`), so this is a builder/decision/learning **lift**, not new inference
— exactly as Move 3 was a lift of an already-Tier-1 substrate.

## 2. Files touched

### Created
- `src/routing.jl` — the lifted routing body: `RoutingState`, `EmissionBelief`,
  `LatencyBelief`, `route`/`route_eu`/`escalation_next`/`posterior_accuracy`, the
  coupled-EM `route_outcome!`, the closed-form `decode_correctness`, the named decision
  resolvers `route_decide`/`escalate_decide` (un-underscored from the app's
  `_route_decide`/`_escalate_decide`), and the data-in `reconstruct_latency_from_data` /
  `reconstruct_routing_tops_from_data`. Composes only Tier-1 ops — no new frozen type,
  no new axiom-constrained fn.
- `test/test_routing.jl` — self-contained engine oracle (no `apps/` dependency).

### Modified
- `src/ontology.jl` — `include("routing.jl")` immediately after
  `include("structure_bma.jl")` (routing depends on it); import the lifted names.
- `src/Credence.jl` — export the lifted names.
- `apps/skin/server.jl` — a new `ROUTING_REGISTRY` (`rt_*` handles) + six verbs
  (`routing_init`/`routing_decide`/`routing_escalate`/`routing_outcome`/
  `routing_belief`/`destroy_routing`) + `SKIN_METHODS` + dispatch; `PROTOCOL_VERSION`
  1.4 → 1.5.
- `apps/skin/protocol.md` — header 1.5 + changelog + verb specs + `rt_*` lifecycle +
  inline-counts shape.
- `apps/skin/test_skin.py` — a routing wire trace.
- `apps/credence-pi/brain/routing_brain.jl` — collapse to a thin shim re-exporting the
  lifted names + keeping the env-reading `wire_routing!` and the path-reading
  reconstruction wrappers.

### Not touched (explicit non-goals)
- **No daemon rewire / repo extraction** — that is Move 5. The daemon still embeds
  (`using Credence`) after this move; embedding is now *optional*, the verbs are the
  wire path.
- **No governance changes** — already lifted in Move 3.
- The body (`openclaw-plugin` TS, HTTP→daemon) is unchanged.

## 3. Behaviour preserved

`apps/credence-pi/tests/julia/test_routing.jl` (40 assertions: exact `==` for
decisions/replay, bounded tolerances for the EM identifiability) passes **byte-for-byte
unmodified** through the shim — the canonical "capture PRE-refactor, assert `==`" pin
(green on `39fa38c` before the lift; must stay green after with zero edits). The daemon
(`server.jl:55 using .RoutingBrain: wire_routing!, route_outcome!`) and every `eval/`
reach-in resolve through the shim's re-export union.

## 4. Worked end-to-end example (the wire path credence-openclaw will use)

```
initialize                                  → {protocol:"1.5", …}
routing_init {feature_names, feature_values, roster:[[name,provider,id,cost]×3],
              reward:0.02, warm_counts:{…inline JSON…}}   → {routing_state_id:"rt_1", n_models:3}
routing_decide {rt_1, features:{prompt-length:"short"}, profile:{reward:0.02}}  → {model:"haiku",…}   # cost-hawk → cheap
routing_decide {rt_1, features:{prompt-length:"short"}, profile:{reward:1.0}}   → {model:"sonnet",…}  # quality-hawk → best-believed (Wald flip)
routing_outcome {rt_1, model_id:"haiku", features:{…}, success:true} ×30        → {routing_state_id:"rt_1"}
routing_belief {rt_1, model_id:"haiku", features:{…}}                           → {theta:↑, rho_bar, sigma_bar}
routing_escalate {rt_1, features, tried:[]}  → tier 1; {…tried:[0]} → tier 2; … → null (STOP)
```
All coefficient assembly (`reward·θ − cost`, the ProductMeasure join, the EM decode)
happens server-side; the consumer ships only data and reads back a model id.

## 5. Open design questions

1. **RoutingState = one opaque handle in a new `ROUTING_REGISTRY`** (recommended,
   adopted). It is a *mutable* bundle (mutated every `routing_outcome`) that grows
   unboundedly (`extra_tops` per unseen model id, ρ/σ cells per unseen context key), and
   the consumer addresses *neither* internal belief — so the Move-3 `m_*`/`s_*` split
   does not transfer (that split exists because the governance consumer addresses both,
   and `MODEL_REGISTRY` is justified by descriptor *immutability*). Pushback welcome on:
   reuse `STATE_REGISTRY` (it already holds the mutable `AgentState`) vs a dedicated
   registry? Chosen: dedicated, so the generic `weights`/`expect`/`destroy_state` verbs
   give a clean typed error on an `rt_*` handle rather than a confusing one.
2. **`decode_correctness` stays closed-form** (do NOT re-express via a transient
   `condition`+`mean`). `conjugate.jl:63-73` computes the identical `π` formula but at the
   *per-cell* level (`θ̄=α/(α+β)`); routing needs `θ̄=E[θ|X]` at the *mixture* level
   (`posterior_accuracy`), because the emission M-step must be weighted by the coherent
   per-turn correctness, not a per-cell responsibility. The transient-condition route
   computes the wrong π and breaks bit-exactness. Is the closed-form `π` (sanctioned
   E-step arithmetic, legal in `src/`) the right call? (Yes — see §8.)
3. **Warm reconstruction = inline counts data** in `routing_init`, reconstructed
   server-side (skin never reads the host FS for external consumers — same rule as
   `dsl_sources`). This is a genuinely new transport shape (its only precedent is
   `dsl_sources`). The path-reading wrappers stay in the shim for the still-embedding
   daemon.
4. **Named decision templates** (`route`/`escalation_next` lifted as engine functions the
   verbs call), NOT a raw-`optimise`-over-Functional-JSON wire path — the latter would
   re-leak `Projection`/`LinearCombination` assembly into the consumer, the anti-pattern
   Move-3's named `decide_with_voi` exists to prevent.
5. **`routing_belief` telemetry verb** — is it worth a verb the hot path never uses? It
   exists so calibration/shadow-mode and the wire parity test can observe the EM-learned
   belief; the hot `routing_decide` builds the per-context view server-side and never
   round-trips. (Mirrors the Move-3 question about `belief_at_context`-as-verb; kept
   because the wire test cannot otherwise see learning.)

## 6. Risk + mitigation

- **Silent re-export miss** → an `eval/` or daemon `RoutingBrain.foo` resolves to
  nothing. *Mitigation:* the shim re-exports the grepped union (`route`, `route_eu`,
  `escalation_next`, `posterior_accuracy`, `route_outcome!`, `decode_correctness`,
  `latency_at`, `reconstruct_latency`, `_reconstruct_routing_tops`, `wire_routing!`,
  `RoutingState`, `EmissionBelief`, `LatencyBelief`, `_ctx_key`); `test_routing.jl` +
  loading every `eval/*.jl` is the gate.
- **`_ctx_key` drift** → emission/latency key mismatch. *Mitigation:* re-export, never
  re-define, the engine `_ctx_key`.
- **Replay non-determinism** → the exact-`==` replay oracle breaks. *Mitigation:*
  `reconstruct_routing_tops_from_data` replays `structure_observe` in the same n1-then-n0
  order the app used; Bayesian updates are order-independent so cold-starts stay exact.
- **Protocol header≠const** (Move-3 risk). *Mitigation:* bump both; CI invariant + the
  new skin-wire-smoke gate.
- **Mutable bundle in the immutable-descriptor registry.** *Mitigation:* `ROUTING_REGISTRY`,
  not `MODEL_REGISTRY`; do not dual-register the inner `StructureBMA`.

## 7. Verification cadence

1. Baseline: `test_routing.jl` green on `39fa38c` (done).
2. After the lift: `test_routing.jl` byte-for-byte green; `test_server.jl`; every
   `eval/*.jl` loads; the engine precompiles.
3. New `test/test_routing.jl` engine oracle green (Wald flip exact, escalation gate
   ±1e-6, `w_time=0` bit-identical, soft↔hard corners, confound-partialling, seeded EM
   bands).
4. Skin: `test_skin.py` routing trace + backward-compat; lint `check apps/` clean;
   header==const(1.5).
5. CI green + author approval → rebase-merge.

## 8. de Finettian discipline self-audit

- **Invariant 1 (one reasoner):** preserved. `route`/`escalation_next` still go through
  the single canonical `optimise`; `route_outcome!` is *learning*, and every belief
  change is `condition` (emission Betas via `WeightedBernoulli`, routing belief via
  `structure_observe`/`structure_observe_soft`). No second optimiser, no hand-rolled
  posterior.
- **Invariant 2 (declared structure / closed-form EU):** preserved. The lifted
  decisions keep typed `LinearCombination`/`Projection` Functionals; the verbs never put
  a Functional or `ProductMeasure` on the wire (consumer ships scalars; the template
  assembles coefficients server-side).
- **Invariant 3 (single-responsibility representations):** the subtle one. `RoutingState`
  *fuses* an immutable `StructureBMA` descriptor with mutable beliefs **under one
  handle**, but they remain **separate objects** — the fusion is at *handle* granularity
  (ergonomics), not *representation* granularity. Not a violation; called out because a
  reviewer conditioned on Move-3's `m_*`/`s_*` split will reach for the flag.
- **"Every numerical query through `expect`?"** — a *weaker* "yes" than Move 3.
  `decode_correctness` does a closed-form Bayes combine `π = r·θ̄/(r·θ̄+w·(1−θ̄))` on
  `θ̄` (an `expect` query) and `(r,w)` (emission `mean` queries). The combine is
  sanctioned E-step arithmetic, legal in `src/` (Tier 1) and structurally identical to
  the posterior-param arithmetic `conjugate.jl` already blesses — **not** Move-3's "no
  raw probability arithmetic at all." This is the one place the audit answer differs from
  Move 3's clean yes.

## Reviewer checklist

- [ ] `apps/credence-pi/tests/julia/test_routing.jl` passes **unmodified**.
- [ ] Every `RoutingBrain.*` reach-in (daemon, `eval/`, tests) resolves via the shim.
- [ ] `decode_correctness` is closed-form (not transient-condition); bit-exact replay.
- [ ] `ROUTING_REGISTRY` (not `MODEL_REGISTRY`); `destroy_routing` present.
- [ ] No `optimise`/`LinearCombination`/`Projection`/`ProductMeasure` import left in the
      shim.
- [ ] Protocol header == `PROTOCOL_VERSION` == 1.5; verbs additive (MINOR).
- [ ] Inline counts (`warm_counts`/`latency_counts`) reconstructed server-side; skin
      reads no host FS.
