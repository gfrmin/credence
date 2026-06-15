# Welfare MVP — de-risking the causal cost-reduction claim

**Question.** credence-pi's *detection* of wasteful loops is proven
(precision 1.0 / recall 1.0 on the 76 ClawsBench loops, `FINDINGS.md`). The
repo is scrupulous that the **causal** claim — *governing actually reduces
cost* — was unproven ("the direction is certain; the size is not… needs live
telemetry", `README.md`). This MVP de-risks it.

**Frame.** There is one currency — the human's **welfare** — with four
measurable coordinates: **money, time, attention, risk**. A "user" is a utility
**profile** (weights over the coordinates). credence-pi is a proxy
EU-maximiser for whichever human it serves: **one shared learned posterior
(beliefs), many per-human utilities (preferences), one EU-max mechanism.** The
brain already runs this multi-coordinate EU-max (`decide_multi`); money,
attention, risk are already utility terms (time is measured here, made a
decision coordinate in Phase 2).

**Governance is an investment.** It *spends* (brain compute, daemon latency,
attention when asking, opportunity cost of false-blocks) to *save* more (wasted
tokens, wall-clock, harm). The only honest headline is **net ΔWelfare =
returns − investment**, net of the sidecar's own overhead. Every figure below
is net of the daemon's *measured* governance latency.

This is the principled-MVP cut: EU-max (Tier-1) is untouched and exact; we vary
only Tier-2 utility/scope (two profile points, money in flat units offline,
time measured-not-yet-optimised). Every cut is the general framework at a
corner, recoverable, and labelled.

---

## Result 1 — Offline: adaptivity + dominance (real held-out sessions)

`eval/profile_adaptivity.jl` on ClawsBench (3384 sessions; 11,306 held-out
calls; 76 objective loops). One shared frozen posterior; two profiles —
**cost-hawk** (λ=0.25, q=0.05: blocks readily, asks cheaply) and **flow-guard**
(λ=4.0, q=1.0: blocks only when very sure, almost never asks).

**Dominance (welfare in each profile's own units; 0 = ideal):** at full
training EU-max is optimal under *both* profiles, beating every baseline by
1–3 orders of magnitude.

| policy | under cost-hawk | under flow-guard |
|---|---:|---:|
| **EU-max** | **−0.0** | **−0.0** |
| no-governance | −76.0 | −76.0 |
| always-ask | −565.3 | −11306.0 |
| naive block-all-repeats | −2032.2 | −32501.0 |

**Adaptivity is the uncertain-regime behaviour.** The EU-max action differs
across profiles *only where the belief is unsure*; divergence decays to 0 as the
brain learns the corpus to bimodal certainty:

| train-frac | divergent calls | cost-hawk blocks | flow-guard blocks |
|---:|---:|---:|---:|
| 0.02 | 91 (0.80%) | 91 | 0 |
| 0.05 | 37 (0.33%) | 104 | 67 |
| 0.10 | 1 (0.01%) | 77 | 76 |
| 1.00 | 0 (0.0%) | 76 | 76 |

At full training the held-out P(approve) histogram is bimodal — 11,229 calls at
≈1, the loops at ≈0, **0 calls** in the contested band [0.2, 0.8]. That is *why*
divergence → 0: when the belief is certain, every profile (and a good rule)
agrees. credence-pi behaves like a confident expert when it knows and defers to
the human's profile when it doesn't.

(Offline axes are money + attention — the ones ClawsBench supports; it has no
per-call tokens or duration. Time and the full money $ are the live layer's job;
risk is ATBench's. The "each-best-in-its-own-units" welfare *separation* needs a
distribution of contested calls — structurally it is the simplex sweep, Phase 2.)

---

## Result 2 — Causal: governance net-raises welfare (real daemon)

`eval/ab_runner.jl` drives a looping scenario through the **real daemon**
(`handle_sensor_event` — the same wire path the OpenClaw body uses) under
no-governance vs governed, for both profiles, with the shipped warm brain. The
loop context (`exec/no-path/other/rep-3plus/ident-1/gt-10m`, θ_approve≈0.006) is
one the warm brain genuinely learned is waste. Scenario: 3 wanted calls + a
10-call loop + 1 wanted call + 1 injected exfil probe; turn-cost $0.50,
turn-time 8 s, net of measured governance latency.

| profile | turns run | net ΔWelfare | money saved | time saved | harm avoided | gov latency |
|---|---:|---:|---:|---:|---:|---:|
| **cost-hawk** | 4 / 15 | **+94.45** | $5.50 | 87.95 s | yes | 52 ms |
| **flow-guard** | 5 / 15 | **+85.00** | $5.00 | 80.00 s | no | 2.4 ms |

**Both profiles net-raise welfare** by catching the confident loop (its whole
tail is saved), comfortably net of the sidecar's own latency. cost-hawk
additionally blocks the exfil probe (risk saved).

**Adaptivity, live:** the profiles diverge on the probe — cost-hawk blocks it;
flow-guard's extreme false-block aversion (λ=4) leaves the block bar *above* the
harm threshold for a call that *looks* routine (high θ_approve), so it proceeds.
A real, honest coupling limitation (below).

---

## Result 3 — A version-stable warm brain

`eval/train_warm_brain.jl` now ships the warm posterior as per-context **counts
JSON** (`brain/warm_brain.counts.json`: 118 contexts, 39,314 calls, 356
loop-denials), reconstructed by replaying `observe` — order-independent, so
**bit-exact** (verified: max θ diff = 0.0), and robust across Julia versions
(unlike the previous `Serialization` blob; CI pins 1.11, dev runs 1.12). The
daemon's `load_starting_posterior` loads it (the same path the harm posterior
already uses). It is a PRIOR — each user's own usage keeps updating it via
`condition`.

---

## Result 4 — Live integration proof-of-life (real qwen, real daemon)

The full stack was brought up on this machine (daemon with the counts-JSON warm
brain → OpenClaw gateway → plugin → `ollama/qwen2.5:7b-instruct`):

- **Daemon + warm brain + HTTP wire**: a confident-loop event POSTed to
  `/sensor` is **blocked** at the realistic $0.50 stake (decision logged).
- **Full live path**: a real qwen agent turn issued an `exec` tool call →
  plugin → daemon → **proceed** decision → the tool ran → outcome + turn-cost
  logged. End-to-end live governance of a real agent's tool calls is confirmed.

The full *multi-turn* A/B (a looping task under shadow vs enforce, measured) is
gated only on local inference speed, not integration: at the gateway's 32k
context, qwen2.5:7b CPU-spills past the 8 GB VRAM (a turn exceeded 280 s).
Capping `num_ctx` (~8192) runs it fully on GPU — a deferred local tuning step.
Tokens are free locally; a small paid-model run gives the real-$ headline.

---

## Result 5 — Multi-turn look-ahead: the tail-aware EU (the myopia fix, and where "loops" rejoins)

Limitation #1 — per-call EU is myopic about the loop *tail* — is now addressed in the
mechanism, fully within the Bayesian paradigm (no arbitrary constants), and doing so
**unifies the proven loop-detection result with the decision**.

**The belief.** A second posterior over the SAME feature model — the *continuation*
probability ρ = P(another identical call follows | context) — trained on real loop
continue/stop events (`eval/train_tail_brain.jl`: 118 contexts, 356 continue-events),
shipped as version-stable counts-JSON and reconstructed **bit-exactly** through the same
`reconstruct_posterior` path the warm and harm posteriors use.

**The look-ahead, in closed form (not an approximation).** Expected future loop turns under
per-step rate ρ is the geometric sum Σₜ₌₁ ρᵗ = ρ/(1−ρ) — the infinite-horizon look-ahead —
and its posterior expectation is *exact*: m = E_{ρ∼Beta(α,β)}[ρ/(1−ρ)] = α/(β−1). This is a
new declared `GeometricTail` test function in `src/` whose closed form sits beside
`Identity`'s; m is a genuine `expect(continuation_belief, GeometricTail())`, integrated over
the posterior — not hand arithmetic, not a point plug-in. The waste-save coefficient becomes
**c·(1+m)** (this call + m expected more), so block ⟺ θ < (1+m)/((1+m)+λ): the myopic
1/(1+λ) at m=0, sliding toward 1 as the detected tail grows. The `ask` VOI inherits the
(1+m)× factor, so an attention-precious profile asks about a loop it would myopically wave
through. Proven on a (θ,λ,m) grid + two headline demonstrations in `test_feature_brain.jl §8`;
m=0 is **bit-identical** to the prior myopic EU. Mechanism is Tier-2 (the outcome model), not
the EU-max mechanism — Invariant 1 intact (still one `optimise` over `expect` of a declared
Functional).

**This is where "we solved loops" rejoins the decision.** Detection (the waste posterior θ —
precision 1.0 / recall 1.0, `FINDINGS.md`) and look-ahead (the continuation posterior m) are
two beliefs over the *same* feature model. The tail-aware EU values stopping a detected loop
by its whole expected tail: detection supplies *what* to stop; the look-ahead supplies *how
much it's worth*.

**Honest finding — the deepest result here.** On ClawsBench the look-ahead changes **0 of 118
trained-context decisions** (`eval/tail_lookahead.jl`), for a principled reason: the contexts
with a long learned tail (deep loops) are *exactly* the ones the waste brain is already
confident are waste (low θ), so the two terms agree. **On real frontier-model loops, "is this
waste?" and "will this persist?" are the same question** — which is *why* the waste brain
alone already governs them so well. The look-ahead's distinct value is the
*uncertain-persistent* regime (moderate θ, high m) — proven to bite in §8, but under-populated
on ClawsBench. And the sparse continuation signal + the `rep`/`ident` features capping at
`3plus` make the belief **pool at the tool level** (a first-occurrence `exec` call inherits
`exec`'s loopiness → m≈1.5, a false-block risk). So tail-aware governance ships **proven but
OFF by default** (`tail-aware "on"` opts in): the mechanism is done; what it needs to *help*
is a better-resolved continuation belief — more continue/stop data, or a finer repetition
feature (the structure-BMA then auto-sophisticates over the finer partition). The saturation
precondition, made concrete.

## What this de-risks — and the honest limits

**De-risked:** governing tool-calls by EU-max **causally net-raises a human's
welfare**, through the real daemon, net of overhead, for *both* contrasting
profiles — and the policy *adapts* to the profile exactly where the belief is
uncertain. The naive block-all-repeats rule is catastrophic (−2032 / −32501);
EU-max is optimal.

**Honest limitations (the Phase-2 map):**

1. **Per-call EU myopia — now FIXED in the mechanism (Result 5), data-gated empirically.**
   The tail-aware EU (`GeometricTail`, m = `expect(continuation posterior, ·)` = α/(β−1))
   values blocking by the whole expected loop tail and the `ask` VOI by (1+m)×, proven in
   `test_feature_brain.jl §8`; m=0 recovers the myopic form bit-for-bit. Its empirical reach
   on ClawsBench is 0 trained-context changes (detection already subsumes it — Result 5), and
   the sparse continuation signal over-pools at rep-0, so it ships proven but OFF by default.
   The remaining work is a better-resolved continuation belief, not the mechanism.
2. **Harm-blocking is coupled to false-block aversion.** A very high λ raises the
   block bar above the harm threshold for a call that looks wanted; the
   `harm-response: ask` net only fires when the decision is already "block". Fix:
   a harm floor independent of λ (safety calibration).
3. **The causal harness is a scenario, not a full live agent run.** The loop
   length and "a caught loop is abandoned" are stipulated; the daemon, its
   decisions, and the latency are real. The live stack is proven end-to-end
   (Result 4) — a real qwen tool call is governed by the warm-brain daemon — but
   the multi-turn shadow-vs-enforce A/B that measures the real agent's *reaction*
   to a block is gated on capping `num_ctx` for full-GPU inference (deferred
   local tuning). Money in tokens free-locally; $ on a small paid run.
4. **Money is per-turn, not per-tool** (pi exposes no per-tool cost); time and
   harm here are labelled scenario parameters; offline money is a flat unit.

**Reproduce:**
```
julia --project=. apps/credence-pi/eval/train_warm_brain.jl \
    --events data/credence_pi_eval/clawsbench_openclaw.events.jsonl \
    --out-counts apps/credence-pi/brain/warm_brain.counts.json --corpus "ClawsBench"
julia --project=. apps/credence-pi/eval/profile_adaptivity.jl \
    --events data/credence_pi_eval/clawsbench_openclaw.events.jsonl \
    --out apps/credence-pi/eval/results/profile_adaptivity.summary.json
julia --project=. apps/credence-pi/eval/ab_runner.jl \
    --out apps/credence-pi/eval/results/ab_causal.summary.json
# Multi-turn look-ahead (Result 5): train the continuation belief, then measure its reach.
julia --project=. apps/credence-pi/eval/train_tail_brain.jl \
    --events data/credence_pi_eval/clawsbench_openclaw.events.jsonl \
    --out-counts apps/credence-pi/brain/tail_brain.counts.json --corpus "ClawsBench"
julia --project=. apps/credence-pi/eval/tail_lookahead.jl
```
