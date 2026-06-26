# Phase 1 design doc — Extract the complexity log-prior

> Seven-section template (`docs/collapse-towers/DESIGN-DOC-TEMPLATE.md`). Arc master plan:
> `docs/collapse-towers/master-plan.md`. Precedents: `docs/precedents.md`.

## 1. Purpose

Phase 1 as scoped in the master plan: introduce the single structural complexity log-prior
`complexity_logprior(L; λ, offset)` (the operational form of SPEC §1.3, `P(program) = 2^{-|program|}`)
and recover the two existing open-coded forms — the structure-BMA edge prior and the program
node-count prior — as instances under their **own per-axis** `λ`. This is a refactor + generalisation:
no behaviour changes (the priors compute the same weights), but the three-becomes-one consolidation
unblocks Phase 2 (Family-BMA = `complexity_logprior` on a *family* index) and Phase 5 (the VOC value
estimate is a `complexity_logprior` difference). `λ` is per-axis and stays so: program axis `ln 2`
(pinned by §1.3), edge axis `log((1−p_edge)/p_edge)` (`0` at `p_edge=0.5`). A single shared `λ` is
ruled out by the axiom (`0 ≠ ln 2`).

## 2. Files touched

- **`src/complexity.jl`** — *new file*. Defines `complexity_logprior(L::Real; λ::Real, offset::Real=0.0) = -λ*L + offset` plus a short docstring tying it to SPEC §1.3. ~15 lines.
- **`src/ontology.jl`** (~line 2023, the `include` block) — *modification*. Add `include("complexity.jl")` **before** `include("structure_bma.jl")`. Because `ontology.jl` is included before `program_space/*` in `Credence.jl`, the symbol is also visible to `enumeration.jl`/`agent_state.jl`. 1 line.
- **`src/structure_bma.jl:96-99`** (`_structure_logweights`) — *modification*. Replace the open-coded `k·log(p) + (n−k)·log(1−p)` with `complexity_logprior(length(parents); λ=log((1−p_edge)/p_edge), offset=n_features·log(1−p_edge))`.
- **`src/program_space/enumeration.jl:170`** (`enumerate_programs_as_measure`) — *modification*. Replace `-g.complexity*log(2) - p.complexity*log(2)` with `complexity_logprior(g.complexity; λ=log(2)) + complexity_logprior(p.complexity; λ=log(2))` (two-call sum — see §3 for the bit-exactness argument).
- **`src/program_space/agent_state.jl:131`** (`add_programs_to_state!`) — *modification*. Same two-call substitution (`lw = …`).
- **`test/test_complexity.jl`** — *new file*. Unit + equivalence tests (see §7).

`src/Credence.jl` export: `complexity_logprior` is engine-internal stdlib; export only if a test needs
it unqualified (tests already do `using Credence: …`). Decide at code time — default **export** for
symmetry with `voi`/`net_voi`.

## 3. Behaviour preserved

Tolerance classes for this refactor (the master plan's strata, specialised):

- **Program axis — bit-exact (`==`).** The two-call sum reproduces the old literal
  `-g.complexity*log(2) - p.complexity*log(2)` **bit-for-bit** in IEEE-754, for all `complexity > 0`
  (always true — `test_program_space.jl:39,48` assert `complexity > 0`):
  - `complexity_logprior(c; λ=log(2)) = -log(2)*c + 0.0`. Multiplication commutes and `(-a)*b == -(a*b)`
    exactly, so `-log(2)*c == -c*log(2)` (== the old per-term product). `c > 0 ⇒ -log(2)*c ≠ -0.0`, so
    `(… + 0.0) == (…)` exactly.
  - The sum `(-log(2)*g + 0.0) + (-log(2)*p + 0.0) == (-g*log(2)) + (-(p*log(2)))`, which is exactly the
    old `-g*log(2) - p*log(2)` (subtraction = addition of the exact negation, same two addends, same order).
  - **Disposition:** `test_program_space.jl` rebuilds the expected `log_prior` from that *same literal*
    (lines 277, 283, 327, 431) and compares to `weights(belief)`; bit-identity ⇒ it stays green unchanged.
- **Edge axis — up-to-`offset`, differences exact to FP.** `k·log(p)+(n−k)·log(1−p)` and
  `−k·log((1−p)/p)+n·log(1−p)` are algebraically equal; in FP they may differ by ≤ a few ULPs per
  structure (a `log(a)−log(b)` vs `log(a/b)` reassociation) plus the structure-independent `offset`.
  Neither matters downstream:
  - **Every** `test_structure_bma.jl` weight assertion is *relative* — sparse≡dense (`:38,48,56`),
    soft≡hard at the certain corners (`:64-68`), warm≡manual (`:91,93`) — with both sides flowing
    through the **same** rewired `_structure_logweights`. So each stays exact `==` by construction,
    independent of how `_structure_logweights` is computed. **No absolute-weight oracle exists** (grep
    in §6). `test_sparse_structure_equivalence.jl:61` is `isapprox(atol=1e-12)` — trivially preserved.
  - The new `test_complexity.jl` asserts the *differences between structures* match the old hand-formula
    to `rtol=1e-12` (not `==`, because of the log-ratio reassociation — named, not hidden).

## 4. Worked end-to-end example

**Edge axis.** `build_structure_model(["a","b"], [["0","1"],["0","1"]]; p_edge=0.3)` ⇒ 4 structures with
parent counts `k ∈ {0,1,1,2}`, `n=2`. `_structure_logweights` (now `complexity.jl`-routed):
`λ = log((1−0.3)/0.3) = log(0.7/0.3) ≈ 0.8473`, `offset = 2·log(0.7) ≈ −0.7133`.
- `k=0`: `−0.8473·0 + (−0.7133) = −0.7133`. Old: `0·log(0.3) + 2·log(0.7) = −0.7133`. ✓
- `k=1`: `−0.8473·1 − 0.7133 = −1.5606`. Old: `1·log(0.3) + 1·log(0.7) = −1.204 − 0.357 = −1.5606`. ✓
- `k=2`: `−0.8473·2 − 0.7133 = −2.4079`. Old: `2·log(0.3) = −2.4079`. ✓
Difference `k=2` minus `k=0` is `−1.6946` both ways (= `−2·λ`); the `MixturePrevision` renormalises the
common part away. Module ownership: `structure_bma._structure_logweights` calls
`complexity.complexity_logprior`; the mixture is assembled by `build_structure_prior`.

**Program axis.** Grammar complexity `3`, program complexity `2`: log-weight
`complexity_logprior(3; λ=log2) + complexity_logprior(2; λ=log2) = −3log2 + −2log2 = −5log2 ≈ −3.4657`,
**bit-identical** to the old `-3*log(2) - 2*log(2)`. Owner: `enumeration.enumerate_programs_as_measure`
builds the `CategoricalPrevision(log_weights)`.

## 5. Open design questions

1. **Program-axis `L` — RESOLVED: keep `g.complexity + p.complexity`; reject `expanded_complexity` on the
   merits (this is not a deferred correction).** The move prompt names `L = expanded_complexity`, but that
   is an **error in the prompt**, not pending work. `g + p` is the correct §1.3 instance: it is the
   **two-part MDL code** — `g.complexity = length(features) + Σ(1 + |body|)` (`types.jl:150`) pays once to
   *define* the dictionary, and `p.complexity` charges each nonterminal reference a single symbol
   (`expr_complexity(NonterminalRef)=1`, `enumeration.jl:16`), describing the program *given* the dictionary.
   That is the honest Solomonoff/MDL setup: charge for abstraction, then let reuse pay it back.
   `expanded_complexity` inlines every nonterminal — the degenerate **one-part** code that forbids a
   dictionary and gives reuse *no* discount (a program using `F` three times costs as much as writing `F`'s
   body out three times). Rejected for two architectural reasons, both already in the tree:
   - **It would make `propose_nonterminal` optimise a phantom.** `net_payoff = n_sources·savings_per_use −
     rule_cost` with `savings_per_use = expr_c − 1` (`perturbation.jl:128-130`) is a saving *only* because a
     reference costs 1 under `p.complexity`. Under `expanded_complexity`, `savings_per_use` collapses to 0 —
     the prior grants no compression for the `net_payoff > 0` gate to detect. The two-part prior and that
     gate are two halves of one mechanism: propose a nonterminal iff it compresses.
   - **It would falsify Phase 5's central identity.** Phase 5 requires the `:add_rule` VOC estimate to
     recover `net_payoff` *exactly* (closing the loop between the two towers). That is possible only under
     the two-part code, where adding a rule shifts prior mass (more `g`, less `p` for users). Under
     `expanded_complexity`, naming a sub-expression is prior-invariant, so `net_payoff` would track no prior
     difference and the loop could not close.
   Clincher: `expanded_complexity` is currently **only a test helper** — used in exactly two
   `test_program_space.jl` assertions (`:177`, `:506`) that verify the savings logic, never as a prior
   anywhere (grep in §6). Adopting it would introduce a brand-new prior contradicting the compression
   semantics the grammar machinery already runs on. The question is closed here — no "separate change later".
2. **`offset` in the signature.** `complexity_logprior(L; λ, offset=0.0)` carries the edge axis's
   `n·log(1−p)` normaliser so the edge axis is a *single* call (the move's pedagogical "edge axis = one
   `complexity_logprior` on `L=|parents|`"). Alternative: drop `offset`, make the edge axis a sum of two
   calls like the program axis (`complexity_logprior(k; λ=−log p) + complexity_logprior(n−k; λ=−log(1−p))`)
   which would be bit-exact too. **Recommendation: keep `offset`** — it states the edge axis as one prior
   over one description length `|parents|`, which is the conceptual content; bit-exactness on the edge axis
   is not needed (no oracle depends on it). Reviewer may prefer the uniform "always a sum, never an offset"
   shape for cross-axis symmetry.
3. **`p_edge`: fixed hyperparameter (confirm out of scope).** Giving `p_edge` its own (hyper)prior =
   conditioning over the edge-inclusion rate = another BMA axis. **Recommendation: out of scope for this
   arc** (master plan agrees); `p_edge` stays a fixed `StructureBMA` field. Flagging only to close it.

## 6. Risk + mitigation

- **FP reassociation silently shifts a prior and a downstream posterior test breaks.**
  - *Blast radius:* `test_program_space.jl`, `test_structure_bma.jl`, `test_sparse_structure_equivalence.jl`.
  - *Mitigation + pre-emptive grep (done):*
    - `grep -nE 'weights\(|== |isapprox' test/test_structure_bma.jl` → all 6 weight assertions are
      relative (sparse≡dense / soft≡hard / warm≡manual) through the shared `_structure_logweights`;
      **disposition: no edit, green by construction.**
    - `grep -nE 'log\(2\)|weights\(' test/test_program_space.jl` → oracle rebuilds `log_prior` from the
      *literal* `-g.complexity*log(2) - p.complexity*log(2)` (lines 277/283/327/431);
      **disposition: bit-identity (§3) ⇒ green unchanged.**
    - `test_sparse_structure_equivalence.jl:61` is `isapprox(atol=1e-12)`; **disposition: preserved.**
  - *Captured invariant:* `test_complexity.jl` asserts the program-axis two-call sum `==` the old literal
    for representative `(g,p)` — a canonical pre-refactor value captured and asserted bit-exact
    (precedent: capture-canonical-before-refactor).
- **Include ordering — `complexity_logprior` undefined when `structure_bma.jl` loads.**
  - *Mitigation:* insert the `include("complexity.jl")` line *before* `include("structure_bma.jl")` in
    `ontology.jl`; `program_space/*` loads after `ontology.jl` (per `Credence.jl`), so it is visible there
    too. A failed precompile is loud (caught by the first `using Credence`).
- **A future reader resurrects `expanded_complexity` as the program-axis length (OQ-1).** *Mitigation:*
  OQ-1 rejects it on the merits — it breaks `propose_nonterminal` and Phase 5 (see §5) — so there is no
  deferred correction to resurrect. The `test_program_space.jl` bit-identity guard fails loudly if `L` is
  silently changed; `expanded_complexity` stays a test-only savings-verification helper.

## 7. Verification cadence

End of Phase-1 code (from repo root):
```
julia test/test_complexity.jl
julia test/test_structure_bma.jl
julia test/test_program_space.jl
julia test/test_sparse_structure_equivalence.jl
```
Then the full suite before the phase-boundary stop: `for f in test/test_*.jl; do julia "$f"; done` (39
files), plus the lint self-test `python tools/credence-lint/credence_lint.py` and the `check apps/` pass.
Skin smoke test is **optional** for Phase 1 (no JSON-RPC boundary change — the prior weights are
server-internal). Halt-the-line on any red.

`test_complexity.jl` contents (repo `check(name, cond, detail)` idiom; no `using Test`):
- `complexity_logprior(L; λ=0)` is constant in `L` (uniform) — assert two `L` give equal values.
- Program axis: two-call sum `==` `-g*log(2) - p*log(2)` for `(g,p) ∈ {(1,1),(3,2),(5,4)}` (bit-exact).
- Edge axis: for `p_edge ∈ {0.3,0.5,0.7}`, pairwise *differences* of `complexity_logprior(k;…)` match the
  old hand-formula differences (`isapprox rtol=1e-12`); at `p_edge=0.5`, all `k` give equal weight (`λ=0`).
- Monotonicity (directional): `p_edge<0.5 ⇒` more parents ⇒ strictly lower weight; `p_edge>0.5 ⇒` higher.
- The independent hand-oracle comparisons carry `# credence-lint: allow — precedent:test-oracle — …`.
