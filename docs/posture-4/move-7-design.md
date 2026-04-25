# Move 7 — Skin rewritten

## 0. Final-state alignment

Move 7 carries the Prevision vocabulary through the wire layer. After Move 7, `apps/skin/server.jl` constructs Prevision types internally (via `build_prevision` replacing `build_measure`), the state registry holds Previsions directly (not Measure wrappers), and `AgentState.belief` is typed `MixturePrevision`. Move 7 also completes Move 5's Prevision-primary `condition` story for scalar types — adding `condition(::BetaPrevision, k, obs)` and analogues for Gaussian/Gamma/Dirichlet/NormalGamma — so the skin stores Previsions uniformly (no vocabulary fork between mixture-as-Prevision and scalar-as-Measure). The 28 JSON-RPC methods keep their current names — the semantics generalise cleanly from Measure to Prevision, and renaming signals "new API" without adding meaning. Transient state left: Python callers (`apps/python/`) still speak Measure vocabulary in their client-side types — that's Move 8's scope. The wire format changes in this move but is an internal interface between the skin process and its Python callers; it is not a public API.

## 1. Purpose

The skin is the last Julia-side layer between `src/` and Python. After Move 7, the entire Julia stack — from `src/prevision.jl` through `src/ontology.jl`'s Measure facades through `apps/julia/` hosts through `apps/skin/server.jl` — speaks Prevision as the primary type. Measure facades survive in `src/ontology.jl` for `AgentState` compatibility (now using `MixturePrevision`) and in the BDSL `(measure ...)` evaluator, but the skin constructs and dispatches on Prevision types directly. The `expect-through-accessor` pragmas in `server.jl` (3 sites: lines 614, 623, 626) are expected to retire because the code that reads `.log_weights` and `.components` will operate on `MixturePrevision` fields directly — no longer accessor reads through a Measure shield. The `posterior-iteration` pragmas (2 sites: lines 1109, 1112) may or may not retire depending on whether `eu_interact`'s per-component dispatch can be restructured; this is an open design question.

## 2. Files touched

### `src/program_space/agent_state.jl` — one type change

- Line 25: `belief::Ontology.MixtureMeasure` → `belief::Ontology.MixturePrevision`
- Lines 1–16: docstring updated to reflect Prevision-primary.
- `sync_prune!` and `sync_truncate!` update: these mutate `.components` and `.log_weights` and reindex `TaggedBetaPrevision` tags. After the type change, they operate on `MixturePrevision` fields directly — same mutation semantics, no Measure shield to forward through.

### `src/ontology.jl` — scalar Prevision-level `condition` methods (Move 5 completion)

Add `condition(::BetaPrevision, k, obs)`, `condition(::GaussianPrevision, k, obs)`, `condition(::GammaPrevision, k, obs)`, `condition(::DirichletPrevision, k, obs)`, `condition(::NormalGammaPrevision, k, obs)`. Each delegates to the existing `maybe_conjugate` machinery — the conjugate logic already operates on Previsions internally; the missing piece is the entry-point dispatch. This is explicitly scope expansion beyond the master plan's Move 7 definition, recorded here as Move 5 completion (§5.2).

### `apps/skin/server.jl` (~1140 lines) — the primary migration

**`build_measure` → `build_prevision` (lines 149–184):**
Rename function. Each branch constructs a Prevision instead of a Measure:
- `"categorical"` → `CategoricalPrevision(log_weights)` (but needs carrier space; see §5.1)
- `"beta"` → `BetaPrevision(α, β)`
- `"tagged_beta"` → `TaggedBetaPrevision(tag, BetaPrevision(α, β))`
- `"gaussian"` → `GaussianPrevision(μ, σ)`
- `"gamma"` → `GammaPrevision(α, β)`
- `"dirichlet"` → `DirichletPrevision(α)`
- `"normal_gamma"` → `NormalGammaPrevision(κ, μ, α, β)`
- `"product"` → `ProductPrevision(factors)`
- `"mixture"` → `MixturePrevision(components, log_weights)`

**`handle_create_state` (lines 586–630):**
- Line 587–589: placeholder `MixtureMeasure` → `MixturePrevision`
- Line 599–600: empty `MixtureMeasure` → `MixturePrevision`
- Line 611–612: `TaggedBetaMeasure(…, BetaMeasure())` → `TaggedBetaPrevision(tag, BetaPrevision(1.0, 1.0))`
- Lines 614, 623–626: pragma sites retire — `.log_weights` and `.components` are direct `MixturePrevision` fields, not accessor reads through a shield.

**`handle_condition` (lines 765–794):**
- `condition(state, kernel, obs)` already dispatches on Prevision subtypes (Measure-level methods forward). After the type change, `state` is a Prevision directly; dispatch is identical.

**`handle_weights`, `handle_mean`, `handle_expect` (lines 796–838):**
- `weights(state)`, `mean(state)`, `expect(state, f)` — Move 5's stdlib defines these on Prevision subtypes. No handler code changes beyond the state type.

**`handle_factor`, `handle_replace_factor`, `handle_n_factors` (lines 665–694):**
- `state isa ProductMeasure` checks → `state isa ProductPrevision`
- `factor(state, idx)` / `replace_factor(state, idx, new_factor)` need Prevision-level implementations or Measure-level forwarding. Check whether `factor` and `replace_factor` currently dispatch on Measure only.

**`handle_dispatch_path` (line 707–713):**
- Already calls `state.prevision` — after type change, `state` IS a Prevision; the `.prevision` indirection drops.

**`handle_eu_interact` (lines 1083–1120):**
- `state.belief.components` and `weights(state.belief)` — after `AgentState.belief::MixturePrevision`, these are direct field accesses on MixturePrevision. The `posterior-iteration` pragmas may persist if the per-component dispatch pattern is unavoidable (see §5.3).

**Import updates (lines 14–34):**
- Add Prevision types: `BetaPrevision, TaggedBetaPrevision, GaussianPrevision, GammaPrevision, CategoricalPrevision, DirichletPrevision, NormalGammaPrevision, ProductPrevision, MixturePrevision`
- Remove Measure types from imports where no longer directly constructed.

### `apps/skin/test_skin.py` (~950 lines, 24 tests)

Each test that constructs a `create_state` request with Measure-vocabulary specs (`"type": "beta"`, `"type": "tagged_beta"`, etc.) is unchanged on the wire — the type strings are the wire format, not Julia type names. The handler (`build_prevision`) interprets them into Prevision constructors. Test assertions on `weights()`, `mean()`, `expect()` are numerically identical.

Tests that use `snapshot_state` / `restore_state` may need attention: the serialised form changes from `MixtureMeasure` to `MixturePrevision`. Existing snapshots from pre-Move-7 will not deserialise into the new type. This is acceptable — the skin process is ephemeral (no persistent state across restarts); `test_v1_snapshot_fails_loudly` (line 384) already asserts that old-format snapshots fail cleanly.

### `apps/skin/protocol.md` (~646 lines)

Update type descriptions from "Measure" to "Prevision" vocabulary in prose. Wire format (JSON specs) stays the same — the type strings on the wire are semantic labels (`"beta"`, `"mixture"`), not Julia type names.

## 3. Behaviour preserved

Every JSON-RPC call produces the same numerical result. `condition` on a `BetaPrevision` with a `BetaBernoulli` kernel returns the same updated `α, β`. `weights` returns the same normalised vector. `mean` returns the same scalar. The change is the internal Julia type holding the value, not the value itself.

**Test assertion strategy:** all 24 skin tests pass unchanged. The Move 0 capture is the reference for integration tests. Specifically:
- `python -m skin.test_skin` — 24 tests, all check numerical equality or structural invariants
- Julia domain tests (`test_email_agent.jl`, `test_grid_world.jl`, `test_rss.jl`) pass — they construct `AgentState` which now takes `MixturePrevision`
- `test/test_persistence.jl` — round-trip test for `save_state` / `load_state`; the serialised form changes but the round-trip invariant holds

Divergence: snapshot byte-streams differ (MixturePrevision serialises differently from MixtureMeasure). This is expected and non-breaking — skin state is ephemeral.

## 4. Worked end-to-end example

Trace: Python caller sends `condition` over JSON-RPC for a Beta-Bernoulli update.

**Before (current tip):**
```
→ {"method": "condition", "params": {"state_id": "s_1", "kernel": {...}, "observation": 1}}
```
1. `handle_condition` calls `get_state("s_1")` → returns a `BetaMeasure` (or `AgentState` with `.belief::MixtureMeasure`)
2. `condition(::BetaMeasure, ::Kernel, 1)` dispatches at `src/ontology.jl:726`
3. Internally: `maybe_conjugate(m.prevision, k)` → `ConjugatePrevision{BetaPrevision, Bernoulli}`
4. `update(cp, 1)` → `BetaPrevision(α+1, β)`
5. Returns `BetaMeasure(space, α+1, β)` — re-wrapped in Measure
6. `STATE_REGISTRY["s_1"] = new_state` (a BetaMeasure)

**After (Move 7):**
```
→ {"method": "condition", "params": {"state_id": "s_1", "kernel": {...}, "observation": 1}}
```
1. `handle_condition` calls `get_state("s_1")` → returns a `BetaPrevision` directly
2. `condition(::BetaPrevision, ::Kernel, 1)` dispatches — needs a Prevision-level `condition` method (currently only Measure-level exists at `src/ontology.jl:726`; see §5.2)
3. Internally: `maybe_conjugate(prevision, k)` → `ConjugatePrevision{BetaPrevision, Bernoulli}`
4. `update(cp, 1)` → `BetaPrevision(α+1, β)`
5. Returns `BetaPrevision(α+1, β)` — no Measure re-wrapping
6. `STATE_REGISTRY["s_1"] = new_state` (a BetaPrevision)

Wire response is identical: `{"state_id": "s_1", "log_marginal": -0.693...}`

## 5. Open design questions

1. **CategoricalPrevision and the carrier-space problem.** `CategoricalPrevision` stores only `log_weights` — no `Finite{T}` space. The skin's `build_measure` currently constructs `CategoricalMeasure(space, log_weights)` where `space` is built from the wire spec. After migration to `build_prevision`, the `"categorical"` branch produces a `CategoricalPrevision(log_weights)` — but the carrier space is lost. Downstream, `weights(::CategoricalPrevision)` works fine (it only needs `log_weights`), but `condition(::CategoricalMeasure, k, obs)` iterates over `m.space.values` to compute likelihoods per hypothesis. Without the space, `condition` on a bare `CategoricalPrevision` cannot enumerate hypotheses. **Options:** (a) Keep `CategoricalMeasure` as the sole exception in the skin — same resolution as Move 6 §5.1, consistent with the principle that CategoricalMeasure is the structural boundary. (b) Add a `CategoricalPrevision(space, log_weights)` constructor that carries the space. My prior: (a) — CategoricalMeasure stays in the skin for the same principled reason it stays everywhere. The skin's `build_prevision` has one Measure-returning branch. **The count of Measure-returning branches in `build_prevision` is 1, and that 1 is the checkable invariant for the move's tip.**

2. **`condition` dispatch on scalar Prevision types.** Currently `condition(::BetaMeasure, k, obs)`, `condition(::GaussianMeasure, k, obs)`, etc. are Measure-level methods. There are no `condition(::BetaPrevision, k, obs)` methods. `condition(::MixturePrevision, k, obs)` exists (line 916) and `condition(::MixtureMeasure, k, obs)` forwards to it (line 936). For scalar types, `condition` goes through the Measure. If the skin stores `BetaPrevision` directly, calling `condition(::BetaPrevision, k, obs)` dispatches to... nothing. **Resolution: option (a) — add Prevision-level `condition` methods for scalar types (BetaPrevision, GaussianPrevision, GammaPrevision, DirichletPrevision, NormalGammaPrevision) as Move 5 completion within Move 7.** The alternative — option (c), keeping scalar beliefs as Measures while mixtures are Previsions — creates a vocabulary fork in the skin's state registry that is a worse architectural violation than the scope discipline violation of adding `src/` work to a skin-rewrite move. The scalar `condition` methods are lightweight (each delegates to the existing `maybe_conjugate` machinery that already operates on Previsions); they complete Move 5's Prevision-primary `condition` story rather than introducing new machinery. This is not a separate pre-PR — it lands as substrate-completion work within Move 7's commit sequence.

3. **`eu_interact`'s per-component dispatch.** The handler iterates `state.belief.components`, evaluates each compiled kernel on features, then computes weighted EU. This is the `posterior-iteration` pattern — justified because each component has a different compiled kernel (the EU cannot be expressed as a single `expect` call with a declared Functional, since the functional varies per component). After `AgentState.belief::MixturePrevision`, the `.components` and `.log_weights` reads are direct MixturePrevision fields — no `expect-through-accessor` issue. But the `posterior-iteration` pragmas stay because the weighted sum is still manual mixture arithmetic. **My prior:** pragmas persist with updated reason. No single-Functional `expect` call expresses this *under the current Functional taxonomy*; resolution requires a per-component-dispatch Functional in a future stdlib expansion (post-Posture-4). The pattern is reducible-via-substrate-not-yet-built, not permanently irreducible.

4. **JSON-RPC method names stay unchanged.** Prompt 13 question: rename to `prevision_expect`, etc.? My prior: keep current names. `condition`, `expect`, `weights`, `mean`, `optimise`, `value`, `draw` are the mathematical operations — they are not Measure-specific. Renaming to flag the type change adds noise without adding meaning; callers already don't know the internal type (it's behind opaque IDs).

5. **Handles, not serialised forms.** Prompt 13 question: Prevision encoding on the wire. My prior: handles (opaque IDs) for working state, as currently implemented. `snapshot_state` returns a base64-serialised blob (opaque to the caller); `restore_state` deserialises it. No change to the handle/snapshot split. The skin's value proposition is that callers never see internal types — only IDs and numerical results.

6. **Functional-style mutation over the wire.** Prompt 13 question: `push_component!` vs return-new-handle. Currently `handle_condition` mutates in place (`STATE_REGISTRY[id] = new_state`, line 791). This is already the pattern. `push_component!` and `replace_component!` for MixturePrevision follow the same shape — mutate the in-place state, return the same ID. My prior: keep in-place mutation — it matches the existing condition/prune/truncate pattern and avoids handle proliferation.

## 6. Risk + mitigation

**Risk 1: `condition` dispatch gap for scalar Previsions.** If the skin stores a `BetaPrevision` and calls `condition(::BetaPrevision, k, obs)`, no method exists today. Mitigation: §5.2 resolution — Move 7 adds Prevision-level scalar `condition` methods in `src/ontology.jl` as Move 5 completion, closing the dispatch gap before the skin migration.

**Risk 2: `sync_prune!` / `sync_truncate!` assume MixtureMeasure fields.** After `AgentState.belief::MixturePrevision`, the sync functions operate on MixturePrevision's `.components` and `.log_weights`. These fields exist on MixturePrevision (same names, same types). Risk is in tag reindexing: `sync_prune!` creates new `TaggedBetaMeasure` objects — these need to become `TaggedBetaPrevision`. Mitigation: read `sync_prune!`/`sync_truncate!` before implementation; update the component-reconstruction logic.

**Risk 3: Serialisation format change breaks cross-session state.** Skin state is ephemeral (no persistent state across process restarts). `test_v1_snapshot_fails_loudly` asserts that old-format snapshots fail cleanly. Mitigation: existing test coverage; no persistent state contract to preserve.

**Risk 4: 24 skin tests depend on wire shape.** Tests construct specs with `"type": "beta"` etc. — these wire strings don't change. Tests that assert `weights()` or `mean()` values are numerically invariant. Risk is in tests that use `snapshot_state`/`restore_state` within a single test run (intra-process); these still work because serialise/deserialise round-trips within the same Julia session. Mitigation: run full test suite.

## 7. Verification cadence

After the migration commit:
```bash
# Skin smoke test (primary)
python -m skin.test_skin

# Julia domain tests (AgentState type changed)
julia test/test_email_agent.jl
julia test/test_grid_world.jl
julia test/test_rss.jl

# Persistence round-trip
julia test/test_persistence.jl

# Full test suite
julia test/test_core.jl
julia test/test_host.jl
julia test/test_events.jl
julia test/test_prevision_unit.jl
julia test/test_prevision_conjugate.jl
julia test/test_prevision_mixture.jl
julia test/test_prevision_particle.jl
julia test/test_flat_mixture.jl
julia test/test_program_space.jl

# Lint
python tools/credence-lint/credence_lint.py check apps/
python tools/credence-lint/credence_lint.py test
```

## 8. de Finettian discipline self-audit

1. **Is every numerical query in this move routed through `expect`?** Yes. Move 7 changes internal types, not query paths. `weights`, `mean`, `expect` are stdlib functions over Prevision (Move 5). The `eu_interact` handler's manual weighted sum is the one exception — justified per §5.3 (per-component compiled-kernel dispatch, no single-Functional alternative).

2. **Does this move hold a Prevision inside a Measure, or a Measure inside a Prevision?** `AgentState.belief` changes from `MixtureMeasure` (Measure wrapping Prevision) to `MixturePrevision` (pure Prevision). The skin's scalar beliefs also become Previsions directly (§5.2 option a). The one exception is `CategoricalMeasure` (§5.1) — principled, not migration debris. This removes Measure-wrapping-Prevision from the entire skin layer except the one case that structurally requires it.

3. **Does this move introduce an opaque closure where a declared structure would fit?** No. Move 7 changes types and dispatch, not structure.

4. **Does this move add a `getproperty` override on any Prevision subtype?** No. The MixtureMeasure `getproperty` shield (forwarding `.components` / `.log_weights`) becomes unnecessary when `AgentState.belief::MixturePrevision` — those fields are direct struct fields, no shield needed.
