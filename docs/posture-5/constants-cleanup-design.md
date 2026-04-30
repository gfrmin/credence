# Constants cleanup — design doc

## Strategic context

v0.1 shipped with three hardcoded constants tracked for principled replacement during the Posture 5 closure ritual (PR #97). All three violate the architectural discipline established by Amendment 1 to Move 2's design doc (PR #87): *intervention thresholds derive from posterior precision, not hardcoded values*. The discipline applies to §5.1's intervention thresholds (where Amendment 1 replaced the original hardcoded escalation floor), to §5.4's instruction-decay retirement condition, and to §5.6's no-confidence span — but the latter two shipped with discipline gaps, and the §5.6 stationarity threshold shipped with a proxy that was forced by the absence of `SpecialFunctions.jl` at the time of sub-PR 6 (PR #94).

This cleanup replaces each constant with a posterior-derived quantity. No new functionality; no architectural ambition; the detection logic stays the same. After this PR, v0.1 has principled Bayesian foundations throughout its detector and decay machinery.

## Scope

Three constant replacements, each with a derivation, implementation surface, and test plan. No changes to the four-intervention vocabulary, persistence machinery, compaction-survival mechanism, fail-open behaviour, or plugin code.

One new project dependency: `SpecialFunctions.jl` (for KL divergence via digamma and log-beta).

## Out of scope

- New detectors for failure modes beyond the three in sub-PR 6.
- Changes to the EU function form (`EU(p) = 2p − 1`), the escalation threshold, or the downgrade threshold — those were settled in Amendment 1 and sub-PR 1.
- Re-running Move 3's demonstration scenarios — the implementation PR does that, not this design doc.
- Any other library dependencies beyond `SpecialFunctions.jl`.

## §5.1 — Retirement ratio: posterior-symmetric retirement condition

### Current code

At `brain.jl:292`:

```julia
if precision > 10.0 * prior_strength && approvals > denials
    push!(retired, i)
end
```

The `10.0` multiplier is the discipline gap. `precision` is `2.0 + approvals + denials` (the Beta posterior's α + β, offset by the uniform prior's contribution). The condition says "retire when we've seen 10× as much evidence as the prior contributed AND more approvals than denials." The 10× ratio has no derivation.

### Replacement

Two clauses joined by AND:

```julia
α_posterior = 1.0 + approvals   # uniform prior α=1 plus approvals
β_posterior = 1.0 + denials     # uniform prior β=1 plus denials
α_prior = 1.0                  # prior on approval rate (uniform)
β_prior = prior_strength       # prior on denial rate (instruction-shaped)

# Clause 1: posterior approves with symmetric force to prior's denial
ratio_inverted = α_posterior * α_prior > β_posterior * β_prior

# Clause 2: evidence at least equal to prior contribution
sufficient_evidence = (α_posterior + β_posterior) > 2.0 * (α_prior + β_prior)

if ratio_inverted && sufficient_evidence
    push!(retired, i)
end
```

**Derivation.** The prior for a registered instruction is asymmetric: strong on the deny side (high `β_prior` = `prior_strength`, low `α_prior` = 1). The instruction says "this action class is probably harmful until evidence says otherwise."

Retirement should fire when the posterior's approval/denial ratio inverts to roughly the magnitude the prior denied. The symmetric condition `α_post / β_post > β_prior / α_prior` says "the evidence has flipped the ratio" — rearranged to avoid division: `α_post × α_prior > β_post × β_prior`.

The precision clause `α_post + β_post > 2 × (α_prior + β_prior)` requires evidence at least equal to the prior's information content. The `2×` multiplier means "twice the prior's contribution" — the minimum to trust a ratio inversion rather than treating it as noise.

**What the `2×` replaces.** The original 10× required 10 times the prior's evidence. At `prior_strength=5`, retirement required `precision > 50`, meaning ~48 observations. The new `2×` requires `precision > 2 × (1 + 5) = 12`, meaning ~10 observations plus the ratio inversion. The new condition is less conservative on evidence quantity but more specific on evidence quality (the ratio inversion is a stronger directional signal than `approvals > denials`).

**What the first clause subsumes.** The original `approvals > denials` guard is subsumed: if `α_post × 1 > β_post × prior_strength` and `prior_strength ≥ 1`, then `α_post > β_post × prior_strength ≥ β_post`, so `approvals + 1 > (denials + 1) × prior_strength`, which implies `approvals > denials` when `prior_strength ≥ 1`.

### Implementation surface

Replace the body of the retirement check in `update_instruction_decay!` (`brain.jl:287–294`). The function signature and the surrounding iteration logic stay the same. The prior's α and β values are implicit in the current code (the uniform prior on approval rate is Beta(1, prior_strength)); they should be made explicit in the replacement.

### Test plan

The canonical test case: with `prior_strength=5`, the prior is Beta(1, 5). Under the old condition, retirement fires at `precision > 50 AND approvals > denials`. Under the new condition:

- `sufficient_evidence` fires when `α_post + β_post > 12`, i.e., when `approvals + denials > 10`.
- `ratio_inverted` fires when `α_post > β_post × 5`, i.e., when `approvals + 1 > 5 × (denials + 1)`.

With 0 denials: retirement fires when `approvals + 1 > 5` AND `approvals > 10`, so first fires at `approvals = 10` (precision = 12, ratio = 11/1 > 5). With the old condition, retirement fired at `approvals = 48` (precision = 50).

Tests should verify:
1. Retirement fires at `approvals=10, denials=0` with `prior_strength=5` (new condition met).
2. Retirement does NOT fire at `approvals=9, denials=0` (insufficient evidence).
3. Retirement does NOT fire at `approvals=10, denials=3` (ratio check fails: 11 > 20? No).
4. With `prior_strength=1` (symmetric prior), retirement fires at `approvals=2, denials=0` (ratio: 3 > 1, evidence: 4 > 4).

If any test produces different behaviour than these predictions, halt and report.

## §5.2 — No-confidence span: posterior-noise-derived span

### Current code

At `detectors.jl:85`:

```julia
const NO_CONFIDENCE_SPAN = 5
```

Used at `detectors.jl:118`:

```julia
state.no_confidence_consecutive >= NO_CONFIDENCE_SPAN
```

The `5` is the discipline gap. The CV threshold (`1/√(α+β)`) and the EU window size (10) are both posterior-derived or configurable; the span alone is hardcoded.

### Replacement

The span scales with the noise level of EU measurements under the current posterior:

```julia
function no_confidence_span(m::BetaMeasure)::Int
    concentration = m.alpha + m.beta
    # EU = 2p - 1, so variance(EU) = 4 × variance(p) = 4αβ / ((α+β)²(α+β+1))
    eu_var = 4.0 * m.alpha * m.beta / (concentration^2 * (concentration + 1.0))
    threshold = 1.0 / sqrt(concentration)
    raw = ceil(Int, 2.0 * eu_var / threshold^2)
    clamp(raw, 3, 20)
end
```

**Derivation.** Under the null hypothesis "EU(proceed) is genuinely stable at its posterior mean", each EU measurement has variance `variance(EU) = 4 × variance(p)` (since `EU = 2p − 1` is linear in p with slope 2, and `variance(p) = αβ / ((α+β)²(α+β+1))` for Beta(α, β)). The CV exceeds the threshold `1/√(α+β)` when the sample standard deviation of EU values in the window is large relative to their mean.

The span should be long enough that sustained threshold-exceedance is unlikely under the null. The quantity `2 × variance(EU) / threshold²` measures how many consecutive exceedances the noise can sustain: when `variance(EU)` is large relative to `threshold²`, the detector needs more consecutive exceedances to distinguish signal from noise. The `2×` multiplier has the same meaning as in §5.1: "twice the noise-to-threshold ratio" — the minimum span that provides evidence beyond noise.

**Behaviour at extremes.** For a diffuse posterior Beta(2, 2): `eu_var = 4 × 2 × 2 / (16 × 5) = 0.2`, `threshold = 1/2 = 0.5`, `raw = ceil(2 × 0.2 / 0.25) = ceil(1.6) = 2`, clamped to floor 3. For a concentrated posterior Beta(50, 50): `eu_var = 4 × 2500 / (10000 × 101) ≈ 0.0099`, `threshold = 1/10 = 0.1`, `raw = ceil(2 × 0.0099 / 0.01) = ceil(1.98) = 2`, clamped to floor 3.

Both extremes hit the floor. This is expected: the span formula produces small values because concentrated posteriors have both small EU variance and small thresholds (they scale together). The floor `3` is the meaningful minimum — the smallest span that distinguishes "sustained" from "transient". The ceiling `20` prevents impractical spans under pathological posteriors.

**Why the floor is 3, not 2.** Two consecutive exceedances can result from a single transient event (one unusual EU value affects two consecutive window computations via overlap). Three exceedances require at least two distinct events, which is the minimum for "sustained".

**Why the ceiling is 20.** At 20 consecutive evaluations, the agent has been in a no-confidence state for at least 20 tool-call decisions. If the detector hasn't fired by then under any posterior, the detection window is too long to provide timely intervention. The ceiling is a pragmatic bound: "even under maximum noise, halt before 20 consecutive uncertain decisions."

### Implementation surface

Replace the `NO_CONFIDENCE_SPAN` constant with a function call. The function takes the current posterior and returns the span. The `check_no_confidence` function receives the posterior already (it calls `get_posterior`); the span computation adds one function call per evaluation.

The `no_confidence_consecutive` counter on `BrainState` still tracks consecutive exceedances; the change is that the comparison target is dynamic rather than static. The counter resets to 0 when CV drops below threshold, same as before.

One subtlety: the span is computed from the posterior at the time of each evaluation, but the posterior changes between evaluations (via `update_posterior!`). The span should be computed from the current posterior at each check, not cached — the detector's sensitivity should track the posterior's evolution.

### Test plan

Tests should verify:
1. Under Beta(2, 2): span is 3 (floor).
2. Under Beta(50, 50): span is 3 (floor).
3. Under an asymmetric posterior where `eu_var / threshold²` is large enough to produce span > 3 (e.g., Beta(1, 100): `eu_var ≈ 4 × 100 / (101² × 102) ≈ 0.000384`, `threshold ≈ 1/√101 ≈ 0.0995`, `raw = ceil(2 × 0.000384 / 0.0099) = ceil(0.0776) = 1`, clamped to 3). This case also hits the floor — the formula tends to produce small values because `eu_var` and `threshold²` scale together.
4. Verify that the floor of 3 is reachable and the ceiling of 20 is reachable (construct a posterior where `raw > 20` if one exists, or document that the formula structurally can't exceed the floor under legitimate posteriors, which would mean the ceiling is precautionary).
5. Move 3's Scenario 3 (#65550) should still fire — the span at Beta(6, 1) and Beta(1, 3) should be ≤ the number of consecutive exceedances the scenario produces.

If the formula structurally produces `raw ≤ 3` for all legitimate Beta posteriors, that is a finding worth noting: the span is effectively always 3, and the `NO_CONFIDENCE_SPAN=5` was more conservative than the posterior warranted.

## §5.3 — Stationarity threshold: KL divergence with `SpecialFunctions.jl`

### Current code

At `detectors.jl:35–38`:

```julia
function stationarity_threshold(m::BetaMeasure)::Float64
    p = mean(m)
    p * (1.0 - p) * 0.1
end
```

The `0.1` calibration is the discipline gap. The function computes 10% of the posterior predictive variance as a proxy for "no meaningful shift." The proxy was forced by the absence of `SpecialFunctions.jl` at the time of sub-PR 6 — KL divergence between Beta distributions requires the digamma function ψ.

### Replacement

Add `SpecialFunctions.jl` to `Project.toml`. Replace the proxy with KL divergence between consecutive outcome windows:

```julia
using SpecialFunctions: digamma, logbeta

function beta_kl(α₁, β₁, α₂, β₂)::Float64
    logbeta(α₂, β₂) - logbeta(α₁, β₁) +
    (α₁ - α₂) * digamma(α₁) +
    (β₁ - β₂) * digamma(β₁) +
    (α₂ - α₁ + β₂ - β₁) * digamma(α₁ + β₁)
end

function stationarity_threshold(m::BetaMeasure)::Float64
    concentration = m.alpha + m.beta
    log(1.0 + 1.0 / concentration)
end
```

The detection logic changes from variance-based comparison to KL-based comparison:

```julia
# Current: variance of boolean outcomes in window vs proxy threshold
ov = outcome_variance(window)
fires = ov <= thresh

# Replacement: KL between empirical Beta fits of consecutive half-windows vs KL threshold
(α₁, β₁) = fit_beta(first_half)
(α₂, β₂) = fit_beta(second_half)
kl = beta_kl(α₁, β₁, α₂, β₂)
fires = kl < threshold
```

**Derivation of the threshold.** `log(1 + 1/(α+β))` is the KL distance corresponding to one observation's worth of new information. A Beta(α, β) posterior that absorbs one new observation shifts to Beta(α+1, β) or Beta(α, β+1); the KL between these is approximately `1/(α+β)` for large concentration (first-order Taylor expansion of the digamma terms). Taking `log(1 + 1/(α+β))` rather than the raw `1/(α+β)` gives the correct form for small concentrations too (where the linear approximation breaks down). A KL below this threshold means "the windows differ by less than one observation's typical contribution."

**Why path A (KL with dependency) over path B (better-calibrated proxy).** The proxy `p(1-p) × c` for any calibration constant `c` conflates two distinct quantities: the posterior predictive variance and the information-theoretic shift between distributions. A well-calibrated `c` fixes the threshold numerically but not structurally — the proxy still measures the wrong thing (outcome spread instead of distributional shift). KL directly measures what the detector needs: "have the recent outcomes shifted the effective posterior?" Using the right quantity eliminates calibration as a concern. The dependency cost is one well-established package (`SpecialFunctions.jl`, standard in the Julia ecosystem, no transitive issues with HTTP/JSON3/LibPQ/Serialization).

**Empirical Beta fit from outcome windows.** The `fit_beta` function computes the method-of-moments Beta fit from a boolean outcome window: `α = successes + 1, β = failures + 1` (adding pseudocounts for numerical stability at boundaries). This is the posterior under a uniform prior, which is the natural choice when the window's outcomes are the only information.

**Detection logic change.** The current detector compares outcome variance in a single window against a threshold. The replacement compares two consecutive half-windows via KL. This changes the detector's semantics slightly: the current detector asks "are the outcomes uniform?" (low variance = all same = stationary); the replacement asks "have the outcomes shifted between the first half and second half of the window?" (low KL = no shift = stationary). Both are valid stationarity tests; the KL version is more sensitive to distributional shifts that preserve variance (e.g., a shift from 60% success to 40% success has the same variance but nonzero KL).

The window K is split into two halves: `K₁ = floor(K/2)` and `K₂ = K - K₁`. With the current minimum K=2, this gives two windows of size 1 each — which is the boundary case where the Beta fit is Beta(2,1) or Beta(1,2). The KL between these is well-defined (`beta_kl(2,1,1,2)` is finite). At K=2, the detector has minimal discriminative power, same as the current variance approach.

### Implementation surface

1. Add `SpecialFunctions.jl` to `Project.toml`.
2. Add `using SpecialFunctions: digamma, logbeta` to `detectors.jl`.
3. Replace `stationarity_threshold` with the `log(1 + 1/concentration)` form.
4. Add `beta_kl` function.
5. Add `fit_beta` function (method of moments from boolean window).
6. Change `check_stationarity` to split the window, fit Betas, compute KL, and compare against the threshold.
7. The `outcome_variance` function becomes unused and can be removed.

### Test plan

1. `beta_kl(α, β, α, β) ≈ 0.0` for any valid α, β (KL of identical distributions is zero).
2. `beta_kl(2, 1, 1, 2) > 0` and finite (minimal distributional shift is detectable).
3. `beta_kl(50, 50, 51, 50) < log(1 + 1/100)` (one observation's shift against a concentrated posterior is below threshold — the detector does NOT fire on a single new observation).
4. `beta_kl(50, 50, 60, 50) > log(1 + 1/100)` (ten observations' shift exceeds threshold — the detector fires).
5. Threshold scales correctly: `stationarity_threshold(BetaMeasure(50, 50)) < stationarity_threshold(BetaMeasure(2, 2))` (concentrated posteriors have smaller thresholds, i.e., more sensitive detector).
6. Boundary case: Beta(1, 1) posterior with K=2 window of [true, true] — KL between Beta(2,1) and Beta(2,1) is 0, detector fires (stationary). Window of [true, false] — KL between Beta(2,1) and Beta(1,2) is nonzero, detector does not fire if KL exceeds threshold.
7. Move 3's Scenario 2 (#34574) should still fire: the scenario produces identical failing outcomes, so both half-windows produce the same Beta fit, KL ≈ 0, below threshold.

If `SpecialFunctions.jl` cannot be added cleanly to `Project.toml` (version conflict, CI breakage), halt and report. Path B (better-calibrated proxy with documented calibration) becomes the fallback.

## Risks

**Risk 1: Retirement timing changes the eager/conservative balance.** The new symmetric condition retires earlier than the original 10× ratio for typical priors (retirement at ~10 observations vs ~48 observations at `prior_strength=5`). This means registered instructions are retired sooner. The implementation PR re-runs Move 3's demonstration scenarios; if any scenario's retirement behaviour changes (e.g., Scenario 1 or 4's compaction-survival produces retirement where it previously didn't), that's a finding for conversation. Mitigation: the test plan above predicts the exact firing points; if predictions match, the balance shift is intentional and documented.

**Risk 2: No-confidence span may be structurally constant.** The derivation in §5.2 produces `raw ≤ 3` for most legitimate Beta posteriors because `variance(EU)` and `threshold²` scale together. If the formula always produces the floor value 3, the cleanup's effect is to change the span from 5 to 3 — a less conservative detector that fires sooner. The implementation PR verifies this empirically across a range of posteriors. If span=3 is structurally universal, that should be documented as a finding (the original NO_CONFIDENCE_SPAN=5 was more conservative than the posterior warranted).

**Risk 3: KL divergence numerical issues at posterior boundaries.** For Beta(1,1) priors, `digamma(1.0) = -γ ≈ -0.5772` (the Euler–Mascheroni constant), which is finite. For very small α or β approaching 0, `digamma` approaches `-∞`. But legitimate Beta posteriors have α ≥ 1 and β ≥ 1 (the uniform prior ensures this), so the boundary case is `digamma(1.0)`, which is well-defined. The implementation should assert α ≥ 1 and β ≥ 1 at the `beta_kl` call site as a guard.

**Risk 4: `SpecialFunctions.jl` dependency and CI.** `SpecialFunctions.jl` is a standard Julia package with no transitive conflicts against HTTP, JSON3, LibPQ, or Serialization. CI runs `Pkg.instantiate` + `Pkg.precompile`; the new dependency adds compile time but no architectural risk. The implementation PR verifies CI stays green.

**Risk 5: Move 3 demonstration evidence timing changes.** The three threshold/span replacements may shift the exact turn at which detectors fire in Move 3's five scenarios. The structure of the evidence stays the same (same detectors, same decisions); only per-scenario turn numbers may change. The implementation PR re-runs `julia evaluations/move-3/run_scenarios.jl`, updates the scenario fixtures' expected outcomes if needed, and updates the evidence document's per-scenario logs.

**Risk 6: Half-window KL changes stationarity detector semantics.** The current detector asks "are outcomes uniform within one window?" The replacement asks "did outcomes shift between two consecutive half-windows?" These detect different things: the current detector fires on any uniform window (including "all successes"), while the replacement fires on any non-shifting window. For the exec-repetition use case (all failures), both detect stationarity equally — both half-windows are identical. But for a window of [success, success, failure, failure], the current detector has variance 0.25 (likely above threshold, doesn't fire), while the replacement has nonzero KL (likely doesn't fire either). The semantic change is narrow and aligned with the detector's purpose (catching repeated identical outcomes), but worth verifying against the Move 3 scenarios.

## References

- Move 2 design doc (PR #86), §5.1 (intervention thresholds), §5.4 (instruction decay), §5.6 (detector specifications).
- Move 2 design doc amendments (PR #87), Amendment 1: posterior-derived escalation threshold.
- Move 2 implementation design (PR #88), sub-PR 4 (compaction-survival decay), sub-PR 6 (failure-mode detectors).
- Move 3 design doc (PR #95), §5.2 (demonstration mechanism), §5.5 (honest scope).
- Posture 5 closure ritual (PR #97), queued constants-cleanup item.

## Acceptance criteria for the implementation PR

- `brain.jl`: retirement condition uses posterior-symmetric derivation (§5.1).
- `detectors.jl`: no-confidence span uses posterior-noise-derived computation (§5.2).
- `detectors.jl`: stationarity threshold uses KL divergence with `SpecialFunctions.jl` (§5.3).
- `Project.toml`: `SpecialFunctions.jl` added as a dependency.
- All existing tests pass with updated expected values.
- Move 3's five demonstration scenarios re-run and produce correct decisions (fixture expected outcomes updated if turn numbers changed).
- Evidence document updated if any scenario timing changed.
- No changes to the four-intervention vocabulary, persistence machinery, compaction-survival mechanism, or fail-open behaviour.
- CI green.
