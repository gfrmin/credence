# Posture 3 capture fixtures — Move 0 invariance target for `de-finetti/complete`

**Captured at SHA:** `5c6a94e464225776e996d6f1f690219a0728ff35` (master tip after PR #43 merge; Posture 3 complete + Move 0 design-doc amendment).
**Capture tool:** `scripts/capture-invariance.jl`.
**Julia version:** recorded in `manifest.toml[capture.julia_version]`.
**Platform:** Linux x86_64 per Prompt 0 task 4 Q3 resolution.

## What this is

The behavioural invariance target for every move of the Posture 4 branch
(`de-finetti/complete`). Every assertion in the test suite was captured at
pre-branch time; Moves 1–10 verify that the same assertions re-hit the same
shapes with the same values (per their tolerance semantics per `move-0-design.md`
§3).

**Two complementary capture channels** ship together in Move 0:

- **Numerical/behavioural** (this directory): 6118 unique site×value tuples
  across the three assertion idioms (`@assert`, `check(name, cond, detail)`,
  `@check(name, expr)`). See the shape-per-file breakdown below.
- **Structural-invariance** (`tools/credence-lint/corpus/<slug>/bad2_*.{jl,py}`
  files): 3 files inventoried in this manifest's `bad2_corpus` section. Each
  asserts that the `credence-lint` pass-two taint analysis still triggers on a
  specific pattern. Not in this directory; cross-referenced here for Moves 1–10
  verifier convenience.

## Files

| File | Content | Shape(s) |
|---|---|---|
| `strata-1.jls` | Exact + Tolerance-Stratum-1 assertions | Exact (==) and `atol <= 1e-14` |
| `strata-2.jls` | Tolerance-Stratum-2 assertions | `1e-14 < atol <= 1e-12` |
| `strata-3.jls` | Tolerance-Stratum-3 assertions | `1e-12 < atol <= 1e-10`; also default-isapprox sites (Stratum 3 by intent) |
| `directional.jls` | Directional assertions (bare `<`, `<=`, `>`, `>=`) | Directional |
| `structural.jls` | Structural assertions (isa, membership, predicate-form) | Structural |
| `failing.jls` | Latent broken assertions (conditions that returned false, threw, or produced non-bool) — per Q4 | Failing |
| `manifest.toml` | Per-idiom sorted listing + capture metadata + `bad2_corpus` inventory | — |

## Manifest schema

Three schemas, one per idiom. The verifier selects by `idiom` field.

### `@assert <expr>` (475 sites in source → after dedup, most collapse)

```toml
[[assert_assertions]]
idiom = "assert"
file = "test/test_core.jl"
line = 41
expr_hash = "sha256:<16-hex>"         # AST content hash — survives re-indentation
expr_source = "abs(result - 0.8) < 1e-10"
shape = "tolerance"                   # exact | tolerance | directional | structural | failing
captured_lhs = 0.8                    # forensic for tolerance/directional; oracle for exact
captured_rhs = 0.8
atol = 1e-10
rtol = 0
stratum = 3
```

### `check(name, cond, detail="")` (144 sites; name-keyed)

```toml
[[check_assertions]]
idiom = "check"
name = "BetaMeasure(2, 3) × Identity = 2/5 (exact)"
file = "test/test_prevision_unit.jl"
line = 41
expr_source = "actual == expected"
shape = "exact"
captured_lhs = 0.4
captured_rhs = 0.4
```

### `@check(name, expr)` (62 sites; name-keyed)

```toml
[[check_macro_assertions]]
idiom = "check_macro"
name = "log_density(tag=1, true) == 0.0 [tag in fires]"
file = "test/test_events.jl"
line = 53
expr_source = "k.log_density(tbm_1, true) == 0.0"
shape = "exact"
captured_lhs = 0.0
captured_rhs = 0.0
```

## Verification semantics per shape

Recapping `move-0-design.md` §3 for the verifier:

- **Exact:** `lhs_new == rhs_new` bit-exactly.
- **Tolerance:** `isapprox(lhs_new, rhs_new; atol, rtol)`. The captured LHS
  and RHS are forensic; the post-refactor re-evaluation of both sides is
  the oracle.
- **Directional:** `lhs_new op rhs_new` holds. Values may drift; the oracle
  is the inequality. The manifest records `"<forensic — see .jls>"` for the
  captured values to keep the manifest stable under timing-dependent
  directional assertions; the `.jls` retains the actual values for drift
  diagnosis.
- **Structural:** expression evaluates to `true` post-refactor. For
  membership subtype (`x in S`), additional gate: `sort(collect(S_new)) ==
  sort(collect(S_old))` or `Set(S_new) == Set(S_old)` if unorderable.
- **Failing:** the assertion was captured in a failing state (returned-false
  / threw / returned-non-bool). Per Q4, documented truthfully; Move 10 has
  permission to upgrade broken → passing when the cleaner foundation fixes.

## Open-design-question resolutions (Prompt 0 task 4)

- **Q1 (shape axis):** Four shapes — exact / tolerance / directional / structural —
  orthogonal to the stratum axis. Stratum applies only to tolerance. See
  `move-0-design.md` §3.
- **Q2 (particle seed capture):** Minimal answer adopted with a tightening —
  capture script calls `Random.seed!(42)` at the start of each test file's
  eval. This gives stable per-file RNG state regardless of what ran earlier.
  Per-`Random.seed!`-call-site downstream-sequence capture is a follow-up
  (not landed in Move 0).

  **Fidelity caveat.** The capture runs under per-file `Random.seed!(42)`
  rather than the suite's natural RNG inheritance. This is a pragmatic
  determinism choice, not a fidelity claim: the captured behaviour is what
  the suite does *under this forced-clean seed regime*, which may differ
  bit-for-bit from what the suite does under its own normal
  `julia test/test_*.jl` invocation (where prior files' RNG consumption
  determines subsequent files' starting state). The invariance target for
  Moves 1–10 is "post-refactor execution under the same forced-clean seed
  regime produces the same captured values," not "behaviour under natural
  RNG inheritance is preserved." If any Move-N refactor changes the
  sequence of `Random.seed!` calls *within* a test file, the per-file seed
  pin will not catch the resulting drift — this is a known limitation
  (deferred until the per-seed-site sequence-capture follow-up lands), not
  a silent failure mode.
- **Q3 (platform):** Linux x86_64. CPU string and Julia version recorded
  in manifest.
- **Q4 (broken assertions):** Captured truthfully as shape `failing` with
  reason and operand values. Move 10 may upgrade broken → passing.

## Double-run verification

The Move 0 PR that lands this directory ran `scripts/capture-invariance.jl --verify`
and confirmed stability:

```
✓ Verified: manifests identical (modulo timestamp)
```

## Read-only contract

**Do not regenerate these fixtures to fix a load bug in a later move.**
The Posture 3 fixture-capture precedent (see `test/fixtures/README.md`
for the Move 0 SHA pinning) applies: the fixtures are pinned at a
specific SHA; loading code in later moves must accommodate the schema
here, not the other way around.

Regeneration is permitted only when:

1. A Move 0 follow-up PR amends the capture protocol (with design-doc
   amendment per the Posture 4 cadence), OR
2. Move 10's paper-reconciliation PR upgrades broken → passing
   assertions per Q4.

Otherwise, every Move N verifier compares against this directory and
halts on divergence.

## Size / scalability note

The `manifest.toml` at this capture is ~1.4 MB / 55k lines — at the upper
bound of what "human-inspectable" can honestly mean. Future moves may
grow this: Move 4 in particular rewrites `test/` against Prevision
constructors and will likely regenerate a capture under the same
protocol, potentially with different per-site value counts. **If any
future capture produces a `manifest.toml` exceeding ~10 MB or ~250k
lines, the shape-partitioning should promote from flat `.jls` +
monolithic `manifest.toml` to per-shape TOML files (one per strata /
directional / structural / failing) with a top-level `manifest.toml`
that becomes a directory index.** This threshold is a promotion trigger,
not a hard limit — the intent is to keep `grep`-over-the-manifest
practical for archaeological review.
