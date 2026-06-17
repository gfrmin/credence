# credence-pi routing proof — Terminal-Bench capability×cost matrix

**Goal.** Prove that a Bayesian router (credence-pi) which learns, per task
context X, beliefs over `P(resolved | model, X)` and `E[cost | model, X]`, then
selects the model maximising expected utility

    EU(model | X) = value · P(resolved | model, X) − E[cost | model, X]

**dominates every fixed single-model policy** (always-haiku / always-sonnet /
always-opus) and naive heuristics — on *real* agentic terminal tasks, not toy
coding. The dominance comes from exploiting the capability×cost spread: a cheap
model that thrashes on a hard task burns more turns/tokens and can cost **more**
than a dearer model that one-shots it, while still failing.

This is grounded in real-world OpenClaw-class usage: the instrument is
`claude-code` (native tool-calling, same agentic class as OpenClaw — Bash/Edit/
Write/Read/Grep loop), run on **Terminal-Bench 2.0** tasks (89 human-verified
end-to-end terminal tasks: compile/debug/configure/sysadmin/security/data-science,
each with a Docker env + deterministic test suite). OpenClaw-live is the
deployment grounding (separate, follow-on).

## Why Terminal-Bench (ROI argument)

A router's return = (spread in outcomes across model choices) × (frequency the
spread is decision-relevant). Saturated easy coding (Exercism) → all frontier
models solve everything → spread collapses → router ROI ≈ 0. Terminal-Bench
maximises spread: local fails, haiku thrashes/fails on hard tasks, sonnet is the
workhorse, opus is overkill on easy. High spread ⇒ high router ROI. Honest
boundary: ROI is realised on *divergent* tasks, not on uniformly-trivial or
uniformly-impossible tails.

## Instrument (validated 2026-06-17)

- Dataset: `terminal-bench-core==0.1.1` (80 tasks: 12 easy / 44 medium / 24 hard;
  9 categories; incl. 4 real SWE-bench issues and several `.easy`/`.hard` variants).
- Agent: `tb_agent.py::ClaudeCodeStreamJSONL` — subclasses tb's `claude-code`
  agent, redirecting `claude --output-format stream-json` to a mounted JSONL file
  for clean, cache-aware cost capture (the stock agent's asciinema capture mangles
  the JSON). Native tool-calling — NOT terminus (terminus's brittle text-JSON
  schema confounds capability with schema-compliance and records no usage).
- Runtime: podman (docker-CLI shim) + podman-compose; `--n-concurrent` for fan-out.
- Per run we record: `resolved` (tb's deterministic grade), `cost_usd`
  (claude's own `total_cost_usd` — the real bill, priced live; cross-checked to
  reproduce exactly from the verified token rates), `num_turns`, full token
  breakdown (input/output/cache_read/cache_create), `duration_s`, `failure_mode`.

### Prices (verified 2026-06-17, USD/MTok: input / output / cache_read / 5m_write)

| haiku | sonnet | opus |
|---|---|---|
| 1 / 5 / 0.10 / 1.25 | 3 / 15 / 0.30 / 3.75 | 5 / 25 / 0.50 / 6.25 |

NB Opus 4.8 is **$5/$25**, ~3× cheaper than the retired Opus-4.1 $15/$75 — only
~1.67× Sonnet. The cost claim must use current prices (a stale table is a
first-read killer); we report claude's realized bill, which encodes them.

## Method

1. **Matrix** (`tb_matrix.py`): for each tier × task (R reps), one `tb run` per
   tier fanning over tasks. → `results/tb_matrix.jsonl`. This is the empirical
   ground truth the router is a decision OVER.
2. **Beliefs** (credence-pi brain, all via `condition`/`expect` — no arbitrary
   constants): per (tier, context-cell) a Beta over `P(resolved)` [BetaBernoulli]
   and a Gamma over turns [Poisson-Gamma, already in `src/`]; `E[cost|tier,X]`
   from `E[tokens|tier,X]` × verified per-token price. Context X from declared
   task features (difficulty / category / instruction length — plus a
   prompt-only feature set to refute "you used the label as the answer").
3. **Split**: warm beliefs on a train split; route on held-out test tasks;
   tally realized (cost, correctness) of the router vs each fixed policy and vs
   naive heuristics (always-cheapest, always-best, random).
4. **Dominance test**: router's cost-per-solve and total cost ≤ every fixed
   policy at equal-or-better resolve rate, on held-out tasks.
5. **Adversarial verification** (Workflow tool): independent checks for
   confounds — train/test leakage, cost-accounting errors, cherry-picked tasks,
   feature = relabelled answer, variance/seed sensitivity.

## Honest limits (to state up front in the report)

- ROI is realised on divergent tasks; uniformly-easy/impossible tails give the
  router nothing to exploit (and we will show that, not hide it).
- The matrix instrument is `claude-code`, not OpenClaw; the routing math is
  agent-agnostic, but the OpenClaw-live A/B is the deployment proof (follow-on).
- Per-run cost has cold/warm-cache variance; reps + the Gamma belief absorb it.
- `local` (ollama) is excluded from this matrix (the claude CLI can't drive it);
  it is a separate, OpenClaw-only arm — and prior work shows local can't reliably
  do agentic tool-use, so it is the free-but-incapable floor, not a contender.

## Status

- [x] Stack validated (oracle + 3 tiers on hello-world; clean cost capture)
- [x] Prices verified
- [ ] Pilot matrix (6 tasks × 3 tiers) — measure spread + per-tier cost
- [ ] Full matrix (~18–30 tasks × 3 tiers, reps on decision-relevant tasks)
- [ ] Beliefs + train/test split + dominance test
- [ ] Adversarial verification
- [ ] Report: measured savings vs every fixed policy + honest limits
