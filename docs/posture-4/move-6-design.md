# Move 6 — Apps and BDSL stdlib migrated

## 0. Final-state alignment

Move 6 migrates every Measure *construction site* in `apps/julia/` and `examples/` to use Prevision constructors directly. After Move 6, app code constructs `BetaPrevision(1.0, 1.0)` where it previously constructed `BetaMeasure(1.0, 1.0)`, and `TaggedBetaPrevision(idx, BetaPrevision(1.0, 1.0))` where it previously constructed `TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaMeasure(1.0, 1.0))`. This converges the app layer toward `master-plan.md` §"Final-state architecture" where Prevision is the primitive and Measure is a view. Transient state left: `AgentState.belief` remains typed `MixtureMeasure` (unchanged per master plan — "AgentState — unchanged from Posture 3"); the `MixtureMeasure` constructor already accepts Prevision-typed components via its existing Prevision-accepting path, so the Measure wrapper persists at this one structural boundary without requiring app code to think in Measure vocabulary at construction sites. The BDSL surface (`(measure ...)` form) is unchanged per master plan — "eval.jl — unchanged structurally; updated for Prevision surface" — so the four `.bdsl` example files retain their `(measure ...)` calls. The wire layer (`apps/skin/server.jl`) is Move 7's scope; this move does not touch it.

## 1. Purpose

Carry the Prevision vocabulary from `src/` (where Move 5 landed it) outward into the application layer. After Move 6, every host file in scope constructs beliefs using Prevision types directly; Measure types appear only at the `AgentState` and `MixtureMeasure` boundary (where the struct type is `MixtureMeasure` by design, not by accident). The stdlib functions landed in Move 5 (`mean`, `variance`, `probability`, `weights`, `marginal`) are already in use at the few sites that need them; Move 6's contribution is the constructor migration, not the accessor migration (which Move 5 handled).

## 2. Files touched

### Julia host files — constructor migration

**`apps/julia/email_agent/host.jl`** (~780 lines)
- Line 146: `CategoricalMeasure(Finite(action_space), logw)` — stays as-is (CategoricalMeasure is the only Finite-carrier Measure constructor; no CategoricalPrevision equivalent takes a `Finite{T}` space because CategoricalPrevision stores log_weights without a carrier space; `CategoricalMeasure` is the structural boundary here).
- Line 762: `TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaMeasure(1.0, 1.0))` → `TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaPrevision(1.0, 1.0))`
- Line 776: `MixtureMeasure(Interval(0.0, 1.0), components, log_prior_weights)` — stays as-is (populates `AgentState.belief::MixtureMeasure`).

**`apps/julia/email_agent/state_persistence.jl`** (~107 lines)
- Line 81: `Credence.Ontology.BetaMeasure(alphas[i], betas[i])` → `Credence.Ontology.BetaPrevision(alphas[i], betas[i])` inside `TaggedBetaMeasure` constructor call.

**`apps/julia/qa_benchmark/host.jl`** (~130 lines)
- Line 49: `BetaMeasure()` → `BetaPrevision(1.0, 1.0)` (the zero-arg `BetaMeasure()` is a convenience for `BetaMeasure(1.0, 1.0)`; `BetaPrevision` has no zero-arg form — be explicit).
- Line 59: `CategoricalMeasure(Finite(Float64[0, 1, 2, 3]))` — stays as-is (Finite-carrier boundary, same as email_agent line 146).

**`apps/julia/grid_world/host.jl`** (~460 lines)
- Line 275: `TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaMeasure(1.0, 1.0))` → `TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaPrevision(1.0, 1.0))`
- Line 289: `MixtureMeasure(Interval(0.0, 1.0), components, log_prior_weights)` — stays as-is (AgentState boundary).

**`apps/julia/rss/host.jl`** (~310 lines)
- Line 71: `TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaMeasure(1.0, 1.0))` → `TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaPrevision(1.0, 1.0))`
- Line 82: `MixtureMeasure(Interval(0.0, 1.0), components, log_prior_weights)` — stays as-is (AgentState boundary).

**`examples/host_credence_agent.jl`** (~243 lines)
- Line 60: `CategoricalMeasure(Finite([h]))` — stays as-is (Finite-carrier boundary; the Thompson-sampling host creates a point mass on a drawn hypothesis).
- Line 94: `CategoricalMeasure(Finite(Float64.(collect(0:n_categories-1))))` — stays as-is.
- Line 126: `CategoricalMeasure(Finite(answers))` — stays as-is.

### Julia host file — import updates

Each host file's `using Credence:` line adds `BetaPrevision` (and `TaggedBetaPrevision` where used) to its import list. `BetaMeasure` can be removed from the import list at sites where it is no longer directly constructed, but stays if referenced elsewhere (e.g. in type annotations or as a constructor argument to other functions).

### BDSL files — no change

**`src/stdlib.bdsl`**, **`examples/coin.bdsl`**, **`examples/grid_agent.bdsl`**, **`examples/router.bdsl`**, **`examples/credence_agent.bdsl`**: the `(measure ...)` DSL form stays. These files are unchanged in Move 6. The BDSL is consumer-facing surface; `eval.jl`'s `_make_measure` returns Measures, which is correct (Measures are the consumer view per the constitution).

### Files not touched

- `src/` (beyond BDSL): Move 5 already landed the Prevision-primary split. `AgentState.belief::MixtureMeasure` stays per master plan.
- `apps/skin/server.jl`: Move 7 scope.
- `apps/python/`: Move 8 scope.
- `docs/precedents.md`, `CLAUDE.md`: no new precedents or slugs needed.

## 3. Behaviour preserved

Move 6 changes only the type used at *construction* sites — `BetaPrevision(1.0, 1.0)` instead of `BetaMeasure(1.0, 1.0)` — where the receiving constructor (`TaggedBetaMeasure`) already has an overload accepting `BetaPrevision`. No runtime path changes: the `TaggedBetaMeasure(space, tag, ::BetaPrevision)` constructor creates the same `TaggedBetaPrevision` internal state as the `TaggedBetaMeasure(space, tag, ::BetaMeasure)` constructor (the latter extracts the prevision from the Measure's `.prevision` field). The resulting `MixtureMeasure` is bit-identical.

**Test assertion strategy:** all existing tests pass unchanged. The Move 0 capture is the reference. Specifically:
- `test/test_email_agent.jl` — 120 assertions
- `test/test_grid_world.jl` — 36 assertions
- `test/test_rss.jl` — 28 assertions
- `test/test_core.jl`, `test/test_host.jl`, `test/test_persistence.jl` — exercise `host_credence_agent.jl` patterns
- No `test/test_qa_benchmark.jl` exists; qa_benchmark is tested via its main-loop outputs

Divergence: none expected. The constructor migration is semantically transparent — the Measure constructor already delegates to the Prevision constructor internally.

## 4. Worked end-to-end example

Trace: email_agent initialises a new AgentState with program-enumerated mixture components.

**Before (current tip):**
```julia
# apps/julia/email_agent/host.jl:762
push!(components, TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaMeasure(1.0, 1.0)))
# ...
# apps/julia/email_agent/host.jl:776
belief = MixtureMeasure(Interval(0.0, 1.0), components, log_prior_weights)
state = AgentState(belief, metadata, compiled_kernels, all_programs, grammar_dict, program_max_depth)
```

Step by step:
1. `BetaMeasure(1.0, 1.0)` creates `BetaMeasure(prevision=BetaPrevision(1.0, 1.0), space=Interval(0.0, 1.0))`.
2. `TaggedBetaMeasure(Interval, idx, ::BetaMeasure)` extracts `getfield(beta, :prevision)` → `BetaPrevision(1.0, 1.0)`, creates `TaggedBetaPrevision(idx, BetaPrevision(1.0, 1.0))`, wraps in `TaggedBetaMeasure(prevision=TaggedBetaPrevision(...), space=Interval)`.
3. `MixtureMeasure(Interval, components, logw)` creates `MixturePrevision(components, logw)` internally.
4. `AgentState(belief, ...)` stores the `MixtureMeasure`.

**After (Move 6):**
```julia
# apps/julia/email_agent/host.jl:762
push!(components, TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaPrevision(1.0, 1.0)))
# ...
# apps/julia/email_agent/host.jl:776  (unchanged)
belief = MixtureMeasure(Interval(0.0, 1.0), components, log_prior_weights)
state = AgentState(belief, metadata, compiled_kernels, all_programs, grammar_dict, program_max_depth)
```

Step by step:
1. `BetaPrevision(1.0, 1.0)` is created directly (no Measure intermediary).
2. `TaggedBetaMeasure(Interval, idx, ::BetaPrevision)` takes the Prevision directly, creates `TaggedBetaPrevision(idx, BetaPrevision(1.0, 1.0))`, wraps in `TaggedBetaMeasure`.
3. Steps 3–4 identical.

The intermediate `BetaMeasure` allocation is eliminated; the `TaggedBetaPrevision` and `MixturePrevision` internal state is identical. `condition`, `expect`, `weights`, `mean` all dispatch the same way on the resulting `MixtureMeasure`.

## 5. Open design questions

1. **`CategoricalMeasure` stays at every call site.** `CategoricalPrevision` stores only `log_weights`; it has no carrier space. `CategoricalMeasure(Finite(vals))` is the only constructor that binds a `Finite{T}` space to categorical weights. The migration rule "Measure → Prevision at construction" therefore has an exception for `CategoricalMeasure`: it is the structural boundary between the typeless-weight Prevision and the typed-carrier-space Measure. My prior: this is correct and expected — `CategoricalMeasure` is the one Measure constructor that carries information (`Finite{T}` space) that the Prevision does not. State this explicitly so Move 6 code doesn't try to "fix" it.

2. **`grid_world/` and `rss/` are not in the master plan's Move 6 scope.** The master plan (§Move 6) lists `apps/julia/email_agent/`, `apps/julia/qa_benchmark/`, `src/stdlib.bdsl`, `examples/*.bdsl`, `examples/host_credence_agent.jl`. It does not list `grid_world/` or `rss/`. Yet both contain identical `TaggedBetaMeasure(…, BetaMeasure(…))` patterns to email_agent, and their tests (`test_grid_world.jl`, `test_rss.jl`) were migrated in Move 4 alongside `test_email_agent.jl`. The migration is mechanical and the sites are identical. My prior: include them — leaving two hosts on `BetaMeasure` while the others move to `BetaPrevision` creates gratuitous inconsistency and a "why is this one different?" code-review question at Move 7 when the skin migration arrives. If the omission was deliberate (scoping for review size), state the reason so Move 7 inherits the residual cleanly.

3. **BDSL surface stays `(measure ...)`.** The master plan says "BDSL stdlib functions that construct probability objects produce Previsions." But `eval.jl` line 98 says "unchanged structurally." These two statements are in tension only if we read "produce Previsions" as changing the BDSL form. My reading: the BDSL `(measure ...)` form stays; what "produces Previsions" means is that the *internal* representation already is Prevision (since Move 3's wrapping). No new `(prevision ...)` form. The DSL is consumer-facing; Measure is the consumer vocabulary. My prior: correct. State explicitly.

4. **Host file structural cleanup is out of scope.** The Prompt 11 open question: "Move 6 is an opportunity to split — but splitting host files is out of scope per the master plan's 'migration not elaboration' discipline." My prior: defer. The host files are long but functionally correct; splitting them is a separate refactor with its own review surface. Move 6 touches only the construction sites — 1-2 lines per file.

5. **`qa_benchmark/` ablation variants are out of scope.** Prompt 11 open question: "the ablation re-implementation is out of scope per the master plan." Confirmed: `papers/RESULTS.md` lists ablation experiments as "Not Yet Run." Move 6 migrates only the existing code path (Bayesian agent + greedy baseline). Ablation experiments are Paper 1 work, not Move 6 work.

## 6. Risk + mitigation

Low risk. Move 6 is a mechanical rename at 5 construction sites (or 7 if grid_world/rss are included). The rename targets a constructor overload that already exists (`TaggedBetaMeasure(::Interval, ::Int, ::BetaPrevision)`).

**Risk 1: Constructor overload mismatch.** If a `TaggedBetaMeasure` constructor does not accept `BetaPrevision`, the change fails at compile time — immediate, loud. Mitigation: the overload exists at `src/ontology.jl:187-188` (added in Move 3).

**Risk 2: Type-specialised code downstream checks `isa BetaMeasure`.** If any code in `apps/` does `x isa BetaMeasure` on the inner component, passing `BetaPrevision` instead would break the check. Mitigation: `grep -rn "isa BetaMeasure\|typeof.*BetaMeasure" apps/` to verify no such checks exist before implementation.

**Risk 3: Serialisation round-trip.** `state_persistence.jl` constructs `BetaMeasure` during deserialization (line 81). Changing to `BetaPrevision` here means the deserialized `TaggedBetaMeasure` is constructed differently. Since both paths produce the same internal `TaggedBetaPrevision`, the serialised byte-stream is unchanged — but the round-trip test in `test/test_persistence.jl` should pass to confirm. Mitigation: run `julia test/test_persistence.jl` after the migration.

## 7. Verification cadence

Full test suite after the single commit:
```bash
julia test/test_core.jl
julia test/test_email_agent.jl
julia test/test_grid_world.jl
julia test/test_rss.jl
julia test/test_host.jl
julia test/test_persistence.jl
julia test/test_events.jl
julia test/test_prevision_unit.jl
julia test/test_prevision_conjugate.jl
julia test/test_prevision_mixture.jl
julia test/test_prevision_particle.jl
julia test/test_flat_mixture.jl
julia test/test_program_space.jl
```

Plus:
```bash
python tools/credence-lint/credence_lint.py check apps/
python tools/credence-lint/credence_lint.py test
```

CI: the existing `publish-image.yml` workflow runs the credence-lint corpus test and `check apps/` pass.

## 8. de Finettian discipline self-audit

1. **Is every numerical query in this move routed through `expect`?** Yes. Move 6 changes no numerical query paths — only construction-site types. The stdlib (`mean`, `variance`, `probability`, `weights`, `marginal`) was landed in Move 5; Move 6 adds no new functions that return `Float64` describing a probabilistic property.

2. **Does this move hold a Prevision inside a Measure, or a Measure inside a Prevision?** The `MixtureMeasure` constructor (unchanged) wraps `MixturePrevision` internally — this is the existing Measure-as-view relationship, not new. Move 6 does not add new Measure-inside-Prevision or Prevision-inside-Measure nesting. The `TaggedBetaMeasure` holding a `TaggedBetaPrevision` is the existing post-Move-3 pattern. `AgentState.belief::MixtureMeasure` remains the one structural boundary where Measure wraps Prevision; this is noted in §0 as explicit, not drift.

3. **Does this move introduce an opaque closure where a declared structure would fit?** No. Move 6 changes constructor argument types only; no new closures.

4. **Does this move add a `getproperty` override on any Prevision subtype?** No.
