# Lint corpus

Labeled examples that drive the `credence-lint` design and serve as its
regression suite. Each subdirectory is named for a precedent slug from
`CLAUDE.md`. Files within are classified by filename:

- `good_*.{py,jl}` — clean code. Lint must report zero violations.
- `bad_*.{py,jl}` — expected pass-one violation. Lint's grep-based pass
  must flag at least one line with a diagnostic that references the
  appropriate precedent slug.
- `bad2_*.{py,jl}` — expected pass-two violation. The grep-based pass
  is allowed to miss; the AST-based pass (Python) or the stateful
  scanner (Julia) must flag.

Every file carries a `# Role:` header so the lint can determine which
sub-layer rule applies. Any brain/skin/body role yields the same rule
in pass one; the headers are a dispatch affordance for future
differentiation and for matching the real tree's conventions.

The corpus is the canonical place to make judgement calls explicit —
a novel case proposed in a PR should show up here with a filename that
names the precedent, not as an ad-hoc regex tweak.

## Layout

Directories correspond 1:1 with the seven slugs declared in
`CLAUDE.md` (Precedents section):

- `compute-on-weights/` — no-escape: any arithmetic on posterior access
- `sort-for-display/` — escape-hatch: display-only comparisons
- `display-arithmetic/` — escape-hatch: formatting percentages
- `test-oracle/` — escape-hatch: hand-computed truth in tests
- `posterior-iteration/` — no-escape: loops over support + weights
- `declarative-construction/` — legal-by-construction: struct building
- `stdlib-composition/` — legal-by-construction: stdlib calling stdlib
- `pragma-malformed/` — corpus for validating the pragma parser itself
  (missing slug, missing reason, unknown slug, etc.)
