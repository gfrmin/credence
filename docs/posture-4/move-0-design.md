# Move 0 — Pre-branch invariance capture

## 0. Final-state alignment

Move 0 is the only move that does not converge the current tip toward the final-state architecture. It captures the current tip as the behavioural invariance target against which every subsequent move asserts equivalence. The capture itself is a docs-plus-fixtures-only PR and leaves `src/`, `apps/`, and `test/` untouched. The transient state introduced is a new directory under `test/fixtures/posture-3-capture/` containing the captured assertion values; this directory is read-only throughout the branch and retires at the tip (Move 10) once the paper reconciliation confirms the final behaviour matches.

## 1. Purpose

Capture the behavioural output of every Stratum-1/2/3 assertion in the test suite, pinned at the current master SHA, so that Moves 1–9 can assert bit-exact or tolerance-bounded equivalence against a known ground truth throughout the migration. Without the capture, the ten-move sequence has no invariance anchor and the Move 5 "point of no return" deletion of Measure is taken on faith rather than evidence.

## 2. Files touched

Creates:
- `docs/posture-4/move-0-design.md` — this file (amended per the §§2/3/4 reconciliation against the actual assertion surface; see commit history).
- `test/fixtures/posture-3-capture/README.md` — capture protocol, SHA pin, **three** per-idiom manifest schemas, cross-reference to the separate `bad2_*` corpus channel.
- `test/fixtures/posture-3-capture/strata-1.jls` — Stratum-1 assertions (`==`, `isapprox` with `atol <= 1e-14`).
- `test/fixtures/posture-3-capture/strata-2.jls` — Stratum-2 (explicit `atol` / `abs(... - ...) < ε` with `1e-14 < ε <= 1e-12`).
- `test/fixtures/posture-3-capture/strata-3.jls` — Stratum-3 (integration / paper-claim end-to-end; `1e-12 < ε <= 1e-10`; also bare `≈` / `isapprox` without explicit tolerance, classified Stratum 3 by intent with resolved `rtol ≈ sqrt(eps(Float64))` recorded in the tuple).
- `test/fixtures/posture-3-capture/directional.jls` — directional assertions (bare `<`, `<=`, `>`, `>=`) — new shape introduced by this amendment; no tolerance, verified by inequality preservation.
- `test/fixtures/posture-3-capture/structural.jls` — structural assertions (`isa`, membership, `all(pred, ...)`, predicate-form expressions); membership is a distinguished subtype with set-equivalence gating (§3).
- `test/fixtures/posture-3-capture/manifest.toml` — per-assertion metadata, **three schemas** selected by idiom.

Modifies: none.

Deletes: none.

**Assertion surface the walker instruments** (surveyed at SHA `fb880be`, 2026-04-23):

| Idiom | Sites | Files | Manifest key |
|---|---|---|---|
| `@assert <expr>` (Base Julia) | 475 | test_core, test_email_agent, test_flat_mixture, test_grid_world, test_host, test_program_space | `(file, line, content_hash(expr_ast))` |
| `check(name, cond, detail="")` | 144 | test_persistence, test_prevision_{conjugate,mixture,particle,unit} | `name` |
| `@check(name, expr)` | 62 | test_events, test_rss | `name` |

No `Test.@test` / `using Test` imports anywhere. A prior draft of this design doc presumed `Test.@test` was the assertion surface; that presumption was factually wrong and is retired by this amendment. The worked examples in §4 are grounded in real source lines.

**Separate capture channel — not conflated.** The `bad2_*` corpus inventory introduced by Prompt 0 task 3 (landed in PR #42 as a Stage-2 patch) is a **structural-invariance** channel over `tools/credence-lint/corpus/<slug>/bad2_*.{jl,py}` with its own manifest. It asserts that specific patterns remain *caught* by the lint; it does not read `test/` at all. The two channels ship together in this PR but live in different directories (`test/fixtures/posture-3-capture/` for numerical+directional+structural assertion capture; `tools/credence-lint/corpus/` for the `bad2_*` pattern inventory) and are verified independently by Moves 1–10.

## 3. Behaviour preserved

Move 0 records behaviour; it does not modify it. The meta-assertion: *"the capture files on disk are a faithful record of what the current master produces under the declared seed discipline, across all three assertion idioms."*

Verified by the §4 double-run protocol: identical fixtures across two clean-checkout runs.

### Idiom-to-classification mapping

Because the three assertion idioms carry no tolerance metadata as keyword arguments, classification is **expression-AST driven**, not keyword-driven. The walker parses the boolean expression of each assertion site and classifies into one of four shapes:

| Shape | Expression pattern | Captured tuple | Verification semantics |
|---|---|---|---|
| **Exact** | `lhs == rhs` | `(lhs_value, rhs_value)` | post-refactor `lhs_new == rhs_new` bit-exact |
| **Tolerance** | `isapprox(lhs, rhs, atol=ε)`, `isapprox(lhs, rhs, rtol=ρ)`, `abs(lhs - rhs) < ε`, `abs(lhs - rhs) <= ε`, **and** bare `lhs ≈ rhs` / `isapprox(lhs, rhs)` without explicit tolerance — walker resolves `Base.rtoldefault(T)` at capture time and records `rtol = <resolved>`, `atol = 0` explicitly | `(lhs_value, rhs_value, atol, rtol)` | post-refactor `isapprox(lhs_new, rhs_new; atol, rtol)` |
| **Directional** | bare `<`, `<=`, `>`, `>=` (no `abs(…)` wrapper; not pattern-matched as tolerance) | `(lhs_value, op, rhs_value)` | post-refactor `lhs_new op rhs_new` holds; values may drift |
| **Structural** | `lhs isa T`, `x in S` (membership — see special case below), `all(pred, xs)`, `haskey(d, k)`, any predicate-form expression without a numeric binary operator at the root | `(expression_source, true)` plus operand capture for membership | post-refactor expression evaluates to `true` (plus set-equivalence gate for membership) |

Bare `≈` for `Float64` resolves to `rtol ≈ sqrt(eps(Float64)) ≈ 1.49e-8`. The capture records this explicitly so the manifest is self-describing — the verifier doesn't re-derive it, so a future Julia-default change won't silently shift the verification threshold.

The stratum classification then applies only to **Tolerance**-shape assertions, by ε magnitude: `ε <= 1e-14` → Stratum 1, `1e-14 < ε <= 1e-12` → Stratum 2, `1e-12 < ε <= 1e-10` → Stratum 3. **Exact** assertions are Stratum 1 by definition. Bare `≈` sites with resolved-default `rtol ≈ 1.49e-8` fall outside the explicit strata band and are classified **Stratum 3 by intent** (loose-tolerance-by-author-choice) with the resolved `rtol` in the tuple as the authoritative gate. The strata are intent labels, not tight bands. **Directional** and **Structural** assertions are their own fixture files and don't use the stratum label.

This is the "third axis" from the original §5 Q1, generalised into **four shapes** because bare `<` / `<=` / `>` / `>=` need directional treatment — a `<` that holds pre-refactor with values (3.5, 4.0) should still hold post-refactor with values (3.6, 4.0), and asserting specific values freezes in refactor-irrelevant implementation detail. §5 Q1 is revised accordingly by this amendment.

### Oracle precedence under reassociation

Captured LHS and RHS values serve two distinct purposes that the verifier disambiguates by shape.

For **Tolerance** shape, **the post-refactor re-evaluation of both sides is the oracle; the captured pre-refactor LHS is forensic.** The verifier runs `isapprox(lhs_new, rhs_new; atol, rtol)` with both sides freshly computed. If the pre-refactor LHS was `0.8000000001`, the post-refactor LHS is `0.8000000003`, the RHS literal is `0.8`, and `atol = 1e-10`, both values are within tolerance of the RHS and the check passes. The captured `0.8000000001` is recorded so a subsequent investigator can see that the refactor moved the LHS by `2e-10` within the tolerance envelope — useful signal, not a pass/fail gate. This means a legitimate floating-point reassociation that moves the LHS within tolerance does not produce a spurious regression.

For **Exact** shape, both sides are oracles jointly: the invariant is equality of the two expressions, and the verifier checks `lhs_new == rhs_new`. Neither captured value alone is the reference.

For **Directional** shape, both sides may drift within limits; the oracle is the inequality, not the values. The verifier checks `lhs_new op rhs_new` holds. Captured values are forensic (documenting the pre-refactor inequality's "slack") and never gate the pass/fail decision.

For **Structural** shape, the oracle is "expression evaluates to `true`"; see the membership special case below for the one structural subtype where additional gating is required.

### Special case: membership assertions (`x in S`)

`x in S` where `S` is a computed expression — not a literal set — poses a silent-pass risk: a refactor may change `S` in ways that preserve `x in S` by coincidence while breaking the semantic claim the test site was asserting (e.g., "the kept-candidates set contains this particular email"). The walker detects membership as a distinguished Structural subtype and captures both operands:

- `x_old` (the element being tested)
- `S_old` (the collection, serialised under `sort(collect(S))` — a canonical ordering that tolerates `Set` / `Vector` / iterator input as long as elements are orderable)

The verifier checks two conditions post-refactor:

1. `sort(collect(S_new)) == sort(collect(S_old))` — set-equivalence under canonical ordering. If `S` is not orderable, the walker falls back to `Set(S_new) == Set(S_old)` and records this in the manifest.
2. `x_new in S_new` — membership holds.

If (1) fails, the assertion has drifted semantically even when (2) still holds, and the move halts. This prevents the invariance target from degrading to "something plausible is in something" for set-valued assertions.

Escape hatch, for sites where set-equivalence is too strict (e.g., tests asserting membership in a deliberately-large computed set like all valid unicode codepoints): a `# credence-lint: allow — precedent:posterior-iteration — <reason>`-shaped pragma adjacent to the assertion downgrades the site to Directional-forensic semantics (capture both values, gate only on `x_new in S_new`). Not expected for Move 0's capture surface; flagged here so future moves that introduce such sites know the annotation exists.

### Manifest schema per idiom

Stable identifier differs per idiom:

```toml
# check(name, cond, detail) — 144 sites
[[check_assertions]]
name = "BetaMeasure(2, 3) × Identity = 2/5 (exact)"
file = "test/test_prevision_unit.jl"
line = 41
shape = "exact"
captured_lhs = 0.4
captured_rhs = 0.4

# @check(name, expr) — 62 sites
[[check_macro_assertions]]
name = "log_density(tag=1, true) == 0.0  [tag in fires]"
file = "test/test_events.jl"
line = 53
shape = "exact"
captured_lhs = 0.0
captured_rhs = 0.0

# @assert <expr> — 475 sites; unnamed, keyed on file:line + AST hash
[[assert_assertions]]
file = "test/test_core.jl"
line = 41
expr_hash = "sha256:..."  # hash of Meta.parse(expr) to survive re-ordering
expr_source = "abs(result - 0.8) < 1e-10"
shape = "tolerance"
captured_lhs = 0.8
captured_rhs = 0.8
atol = 1e-10
rtol = 0
stratum = 3
```

AST hashing for `@assert` means a re-indentation or a few-line shift doesn't invalidate the capture; a genuine expression change (different variable, different constant) does. Move 4 will shift hundreds of `@assert` sites as it rewrites tests against Prevision constructors; the AST-hash-plus-file locality is the verifier's anchor for matching pre-refactor to post-refactor sites.

`check` and `@check` are name-keyed because the name is the stable identifier the author wrote intending it to survive code edits — this is why those idioms exist in the codebase in the first place, and the manifest respects the authorial intent.

### The meta-assertion post-amendment

Unchanged from the original: the capture files on disk are a faithful record. The expansion: faithful **across three manifest schemas and four assertion shapes** (plus the membership Structural subtype), all verifiable by the same post-refactor walker running against the post-refactor source.

## 4. Worked end-to-end example

Capture protocol:

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

The capture script `scripts/capture-invariance.jl` walks every `test/test_*.jl` file with a bespoke macro-walker (`@assert`, `check`, `@check` each have a distinct recognition pass), classifies every assertion by shape per §3, and serialises the result. Three real examples follow — one per idiom — covering tolerance / exact-named / directional shapes.

### Example 1 — `@assert` with tolerance-in-expression (tolerance shape, Stratum 3)

Source at `test/test_core.jl:41`:

```julia
@assert abs(result - 0.8) < 1e-10
```

Walker parses this as:
- Shape: **tolerance** (matches `abs(lhs - rhs) < ε` pattern)
- `lhs` = `result`, evaluated at the assertion site → captured value
- `rhs` = `0.8` (literal) → captured value
- `atol` = `1e-10`, `rtol` = `0` → classified Stratum 3 by magnitude

Manifest entry:

```toml
[[assert_assertions]]
file = "test/test_core.jl"
line = 41
expr_hash = "sha256:<hash of Meta.parse(\"abs(result - 0.8) < 1e-10\")>"
expr_source = "abs(result - 0.8) < 1e-10"
shape = "tolerance"
captured_lhs = 0.8  # result evaluates to 0.8 at pre-refactor time
captured_rhs = 0.8
atol = 1e-10
rtol = 0
stratum = 3
```

Post-refactor verification: re-execute the test; at the matched site (by `(file, expr_hash)`, with `line` as a locality hint), re-evaluate both sides and verify `isapprox(result_new, 0.8; atol=1e-10, rtol=0)`. The captured `0.8` LHS is forensic — if the refactor moves `result` by 2e-11 within the tolerance envelope, the check passes and the drift is recorded in the verification log for investigation. If the refactor moves `result` by 2e-9, the check fails as a Stratum-3 regression; design-doc halt.

### Example 2 — `check(name, cond, detail)` (exact shape, Stratum 1, name-keyed)

Source at `test/test_prevision_unit.jl:41`:

```julia
let m = BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0)
    expected = 2.0 / 5.0
    actual = expect(m, Identity())
    check("BetaMeasure(2, 3) × Identity = 2/5 (exact)",
          actual == expected,
          "got $actual, expected $expected")
end
```

Walker parses the boolean condition `actual == expected` as:
- Shape: **exact** (matches `==` pattern, no tolerance)
- `lhs` = `actual` → captured value
- `rhs` = `expected` → captured value

Manifest entry:

```toml
[[check_assertions]]
name = "BetaMeasure(2, 3) × Identity = 2/5 (exact)"
file = "test/test_prevision_unit.jl"
line = 41
shape = "exact"
captured_lhs = 0.4
captured_rhs = 0.4
```

Post-refactor verification: re-execute; at the matched site (by `name`, no locality hint needed — the author's name is authoritative even if the file is renamed or the assertion moves), re-evaluate both sides and verify `actual_new == expected_new` bit-exactly. Move 4 is the first move to exercise name-matching under rewrite — `BetaMeasure(...)` becomes `BetaPrevision(...)` but the `check` name stays.

### Example 3 — `@assert` with bare inequality (directional shape)

Source at `test/test_core.jl:701`:

```julia
@assert post.sigma < sig0 "Variance must shrink: σ_post=$(post.sigma) ≥ σ_prior=$(sig0)"
```

Both operands are computed values (posterior and prior σ in a Gaussian-Gaussian conjugate update), which is the canonical use case for directional capture — neither side is a literal that the invariance ought to be pinned to.

Walker parses as:
- Shape: **directional** (bare `<`, no `abs(...)` wrapper, not a pattern-matched tolerance)
- `lhs` = `post.sigma` → evaluated at pre-refactor time, captured as a concrete `Float64`
- `op` = `<`
- `rhs` = `sig0` → evaluated at pre-refactor time, captured as a concrete `Float64`

Manifest entry populated by the capture run (values shown are illustrative of the Bayesian shrinkage setup at `test_core.jl:701`; the walker writes whatever the test produces):

```toml
[[assert_assertions]]
file = "test/test_core.jl"
line = 701
expr_hash = "sha256:<hash of Meta.parse(\"post.sigma < sig0\")>"
expr_source = "post.sigma < sig0"
shape = "directional"
captured_lhs = 0.3162  # post.sigma under the test's specific conjugate update
captured_op = "<"
captured_rhs = 0.5     # sig0, the prior σ
```

Post-refactor verification: re-execute the test; verify `post.sigma_new < sig0_new` **holds**. A refactor that produces `post.sigma_new = 0.32`, `sig0_new = 0.5` is fine (inequality preserved, `σ_post` even moved); a refactor that produces `post.sigma_new = 0.51`, `sig0_new = 0.5` (inequality flipped) is a directional regression and halts the move. The captured `0.3162` is recorded for drift diagnosis — "the refactor shifted `post.sigma` by +3%, inequality still preserved" — not for the pass/fail decision. The semantic claim the test is encoding is "posterior variance shrinks under conditioning," not "posterior variance equals `0.3162` to ten decimal places."

## 5. Open design questions

1. **Shape classification (four shapes orthogonal to stratum) — resolved by this amendment.** The original question ("structural vs numerical" as a third axis) is resolved: introduce a **shape** axis with four values — `exact`, `tolerance`, `directional`, `structural` — orthogonal to the stratum axis. Stratum applies only to `tolerance` shape, by ε magnitude. `exact` assertions are Stratum 1 by definition. `directional` and `structural` assertions are their own fixture files and do not use the stratum label. Membership (`x in S`) is a distinguished `structural` subtype requiring set-equivalence gating per §3. Bare `≈` / `isapprox` without explicit tolerance is classified `tolerance` with the resolved `Base.rtoldefault(T)` recorded explicitly; Stratum 3 by intent. Details and worked examples in §§3–4. The question persists on the reviewer's checklist only as a consistency check that §§3–4 carry the four-shape model through — no further open choice.

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

3. **Does this move introduce an opaque closure where a declared structure would fit?** The capture instrumentation walks the assertion ASTs of `@assert`, `check`, and `@check` sites and, for each matched site, evaluates operands via metaprogramming. This is a test-instrumentation concern, not a domain-logic concern; the walker lives inside `scripts/capture-invariance.jl` and never enters `src/`. The shape classification (exact / tolerance / directional / structural) is itself a declared type per §3 — not a closure-in-disguise — so the invariance target is expressed as typed tuples rather than opaque thunks. Acceptable.

4. **Does this move add a `getproperty` override on any Prevision subtype?** No.

---

## Reviewer checklist

- [ ] The SHA pinned is the current master tip (Posture 3 Move 8 merge).
- [ ] The capture script is reproducible: a clean checkout at the SHA, run twice, produces identical fixtures.
- [ ] The manifest lists every assertion site in `test/` across the three idioms (`@assert`, `check`, `@check`) with its shape (exact / tolerance / directional / structural), stratum (where applicable — Tolerance shape only), and per-shape captured tuple per §3.
- [ ] The `test/fixtures/posture-3-capture/` directory is checked in as read-only and documented as the invariance target for Moves 1–10.
- [ ] No modifications to `src/`, `apps/`, or `test/test_*.jl` themselves.
