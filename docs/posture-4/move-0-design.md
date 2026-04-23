# Move 0 — Pre-branch invariance capture

## 0. Final-state alignment

Move 0 is the only move that does not converge the current tip toward the final-state architecture. It captures the current tip as the behavioural invariance target against which every subsequent move asserts equivalence. The capture itself is a docs-plus-fixtures-only PR and leaves `src/`, `apps/`, and `test/` untouched. The transient state introduced is a new directory under `test/fixtures/posture-3-capture/` containing the captured assertion values; this directory is read-only throughout the branch and retires at the tip (Move 10) once the paper reconciliation confirms the final behaviour matches.

## 1. Purpose

Capture the behavioural output of every Stratum-1/2/3 assertion in the test suite, pinned at the current master SHA, so that Moves 1–9 can assert bit-exact or tolerance-bounded equivalence against a known ground truth throughout the migration. Without the capture, the ten-move sequence has no invariance anchor and the Move 5 "point of no return" deletion of Measure is taken on faith rather than evidence.

## 2. Files touched

Creates:
- `docs/posture-4/move-0-design.md` — this file.
- `test/fixtures/posture-3-capture/README.md` — capture protocol, SHA pin, tolerance classifications per assertion.
- `test/fixtures/posture-3-capture/strata-1.jls` — Stratum-1 assertions (`==` and `1e-14` tolerances; closed-form posteriors, particle paths under seeded RNG, enumeration posteriors).
- `test/fixtures/posture-3-capture/strata-2.jls` — Stratum-2 assertions (`1e-12` tolerance; numerically-sensitive closed-forms where arithmetic reassociation is legitimate).
- `test/fixtures/posture-3-capture/strata-3.jls` — Stratum-3 assertions (`1e-10` floor; end-to-end integration, paper-claim-supporting numerics).
- `test/fixtures/posture-3-capture/manifest.toml` — per-assertion metadata: test file, test name, assertion line, stratum, captured value, tolerance.

Modifies: none.

Deletes: none.

## 3. Behaviour preserved

Move 0 does not modify behaviour. It records behaviour. The assertion Move 0 makes is meta: "the capture files on disk are a faithful record of what the current master produces under the declared seed discipline."

The meta-assertion is verified by re-running the capture script in a clean checkout immediately before the Move 0 PR lands. If the re-run produces identical fixtures, the capture is stable under the declared protocol. If it produces divergent fixtures, the protocol is under-specified and must be tightened before the capture is checked in.

## 4. Worked end-to-end example

Concrete capture protocol:

```bash
# Clean checkout at the pinned SHA
git checkout <POSTURE-3-TIP-SHA>
git clean -fdx

# Julia version pinned
julia --version  # must match the version recorded in the manifest

# Run the capture script
julia --project=. scripts/capture-invariance.jl \
  --output test/fixtures/posture-3-capture/

# Verify the capture is stable under a second run
julia --project=. scripts/capture-invariance.jl \
  --output /tmp/posture-3-capture-verify/
diff -r test/fixtures/posture-3-capture/ /tmp/posture-3-capture-verify/
# Zero output expected.
```

The capture script `scripts/capture-invariance.jl` walks every `test/test_*.jl` file, wraps each `@test` with an instrumentation hook that records the LHS and RHS of the assertion and the tolerance used, and serialises the result. Assertions marked with `@test_broken` or inside `@test_skip` blocks are captured as broken/skipped and not compared post-capture.

For a single example trace:

```
test/test_prevision_conjugate.jl line 47:
  assertion: @test posterior.alpha ≈ 3.0 atol=1e-14
  stratum: 1 (atol=1e-14 → Stratum 1 by the strata-tolerance mapping)
  captured LHS: 3.0 (exact)
  captured RHS: 3.0
  manifest entry:
    file: test/test_prevision_conjugate.jl
    line: 47
    name: "Beta-Bernoulli conjugate update: two successes"
    stratum: 1
    captured: 3.0
    tolerance: 1e-14
```

At Move 1 (and every subsequent move), the same test is re-run post-refactor; the assertion's LHS is compared against the captured 3.0 at tolerance 1e-14. If it deviates, Move 1 fails verification and does not proceed to the code PR merge.

## 5. Open design questions

1. **Assertion tolerance vs stratum mapping.** Currently the mapping is "tolerance declared in `@test ... atol=X` or `rtol=X` maps to stratum by magnitude." This covers explicit assertions. For `@test` assertions without an explicit tolerance — e.g. `@test length(programs) == 22` — the default is `==`, which is Stratum 1. Does this suffice, or do we need a third classification axis ("structural equality vs numerical equality")? Structural equality assertions (types, lengths, keys) are not affected by reassociation and belong in a separate class for clarity. Argue.

2. **Particle-path seed capture.** The Posture 3 precedent `test/fixtures/particle_canonical_v1.jls` captures sample sequences under `Random.seed!(42)`. Posture 4's capture inherits this but must extend to every particle-consuming test, not just the canonical one. Does the capture protocol iterate every `Random.seed!(...)` call site in the test suite and capture the sample sequence downstream, or does it rely on per-test reseeding being stable across reruns? The conservative answer is the former (capture the sequences); the minimal answer is the latter (trust the reseed). The minimal answer fails silently if a refactor changes seed-consumption order.

3. **Cross-platform reproducibility.** Julia's RNG is platform-independent for the default seed, but floating-point arithmetic in some of the numerically-sensitive paths may produce platform-dependent results at the 1e-15 level. Does the capture pin a platform (Linux x86_64, specific Julia version) as the canonical machine, or does it capture per-platform and compare within-platform? Pinning a canonical machine is simpler; per-platform is more robust. Argue.

4. **What to do with assertions that currently fail or are flaky.** The current test suite has a small number of `@test_broken` and skip-on-RNG-drift assertions. Move 0 captures their state as-is. At Move 10, the paper-reconciliation PR may discover that one of these is now fixable under the cleaner foundation; at that point, the capture is updated and the assertion upgraded from broken to passing. The protocol for this upgrade should be stated in the Move 0 README so Move 10 does not invent it.

## 6. Risk + mitigation

**Risk.** The capture misses an assertion whose value is dependent on a global state the capture script does not control for (e.g. a module-level `const` that changes between the capture run and a later refactor). Later moves see a behavioural "regression" that is actually drift in the global state, and chase a phantom.

**Mitigation.** The capture script records the set of Julia packages and their versions, the Julia version, and the pinned master SHA, all to the manifest. Any Move N verification that produces a tolerance violation first confirms the environment matches the capture environment; if it doesn't, the regression is attributed to environment drift, not the move.

**Risk.** The capture protocol is under-specified and a re-run produces different fixtures.

**Mitigation.** §3's double-run verification catches this before the capture is checked in. If the protocol is not stable under a double run, the fix is to tighten the protocol (e.g. add a missing `Random.seed!`) rather than to proceed with an unstable capture.

**Risk.** The capture takes the full test suite to run, which means every subsequent move's verification also takes the full test suite to run, which slows the review cycle.

**Mitigation.** Accept the cost. The alternative — capturing a subset — creates an under-specified invariance target. Ten moves under a tight invariance is faster in total than ten moves under a loose one.

## 7. Verification cadence

The Move 0 PR runs:

```bash
# Clean checkout at the Move 0 PR tip
git clean -fdx

# Run the capture
julia --project=. scripts/capture-invariance.jl --output test/fixtures/posture-3-capture/

# Verify stability
julia --project=. scripts/capture-invariance.jl --output /tmp/verify/
diff -r test/fixtures/posture-3-capture/ /tmp/verify/
# Output: empty.

# Verify the full test suite still passes (the capture instrumentation is wrap-only; it doesn't modify test behaviour)
julia --project=. -e 'using Pkg; Pkg.test()'
```

The capture does not run on CI as a regular job; it runs once, captures, and is checked in. CI on subsequent moves compares against the captured fixtures.

## 8. de Finettian discipline self-audit

1. **Is every numerical query in this move routed through `expect`?** N/A — Move 0 introduces no new numerical queries. It records existing ones.

2. **Does this move hold a Prevision inside a Measure, or a Measure inside a Prevision, for any reason?** No. Move 0 adds no types.

3. **Does this move introduce an opaque closure where a declared structure would fit?** The capture instrumentation wraps `@test` macros, which involves closure capture of assertion expressions. This is a test-instrumentation concern, not a domain-logic concern; the closures live inside `scripts/capture-invariance.jl` and never enter `src/`. Acceptable.

4. **Does this move add a `getproperty` override on any Prevision subtype?** No.

---

## Reviewer checklist

- [ ] The SHA pinned is the current master tip (Posture 3 Move 8 merge).
- [ ] The capture script is reproducible: a clean checkout at the SHA, run twice, produces identical fixtures.
- [ ] The manifest lists every `@test` call site in `test/` with its stratum and tolerance.
- [ ] The `test/fixtures/posture-3-capture/` directory is checked in as read-only and documented as the invariance target for Moves 1–10.
- [ ] No modifications to `src/`, `apps/`, or `test/test_*.jl` themselves.
