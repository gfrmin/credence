# answer-brain

A credence "brain" that governs **answering** the way `credence-pi` governs tool-call safety.
Sibling app to `credence-pi`; the durable plan is `docs/answer-brain/master-plan.md`, the
current move is `docs/answer-brain/move-1-design.md`.

Belief: a `CategoricalMeasure` over K candidate values + an explicit NONE atom. The brain
holds this posterior, conditions it on grounded observations (tempered noisy-channel kernels),
and chooses a terminal effector (answer / ask-user / abstain) by EU under the owner's utility
posterior Ū — pricing the forward `gather` steer by `net_voi`. It reuses `life-agent`'s
retrieval/extraction/probes (it never derives); same Credence primitives the skin evaluates,
now held statefully and decided natively.

## Status — Stage 1 (the brain daemon)

Landed: the native port of `life-agent`'s validated Stage-0 decision core, parity-tested
against it.

- `brain/answer_brain.jl` — the posterior (`candidate_posterior`), the terminal EU decide
  (`terminal_decide`, K `report_j` actions + hedge/ask/abstain, argmax via `optimise`), and
  the `net_voi`-priced gather/ask gate (`voi_gather`).
- `bdsl/{capabilities,utility,features}.bdsl` — the effector manifest, the operator-set
  channel priors (the brain's `CANONICAL_CHANNEL` mirrors them), the candidate-set features.
- `daemon/observation_log.jl` — append-only JSONL + deterministic replay.
- `daemon/main.jl` — load-and-ready skeleton.

## Status — Stage 2a (the decision wire)

Landed (`docs/answer-brain/move-2-design.md`): a **stateless** HTTP surface over the Stage-1 brain.

- `daemon/server.jl` — `POST /decide` (`observations + candidates + rho + u_bar → effector +
  report_index + value + credences + p_none + eu`) + `GET /ready`. Request/response, not SSE — the
  decision is synchronous (move-2 §5 Q2). Rebuilds the posterior per call, so no per-question state.
  Wire parity proven against `stage0_parity.json` (`tests/julia/test_server.jl`, 48 checks).
- `brain/answer_brain.jl` — adds `decide_full` (returns the report index `optimise` chose, for the
  wire), `terminal_decide` behaviour preserved (Stage-1 regression green).
- the parity boundary's Python side is `life-agent`'s `bridge/observations.py` (separate repo).

## Status — Move 3 (the capability bridge)

Landed (`docs/answer-brain/move-3-design.md`, in the **life-agent** repo): a **stateless**
JSON-over-HTTP service (`life_agent/bridge/server.py`) — the second backend the body drives,
beside the daemon's `/decide`.

- `POST /route /retrieve /extract /probe/{recency,subject,authority,corroborate}`, `GET /utility
  /ready` — each a thin wrapper of an existing tested read. The bridge gathers + shapes evidence;
  the daemon decides. It holds no posterior and no per-question state; the owner profile + utility
  are read server-side (loopback-bound), never over the wire.
- `/extract` returns exactly `to_abstract_observations`'s output, so its `{candidates,
  observations, rho}` + the body's `u_bar` (`GET /utility`) compose directly into `POST /decide`.
- Hermetic contract tests + one opt-in live round-trip (`retrieve → extract → probe → extract`);
  the retrieval seam was promoted to `life_agent/core/retrieval.py` (the bridge reuses it, no
  src↛scripts import).

Deferred to **Move 4** (body-coupled): `extension/*` (the pi-mono TS body + the minimal answering
app), then the end-to-end eval + gate, and with it the model-choice-under-PII resolution.

## Test

```
julia --project=. apps/answer-brain/tests/julia/test_answer_brain.jl
```

Asserts, on `tests/fixtures/stage0_parity.json` (life-agent `c1a781f`): the native brain
reproduces Stage-0's posterior weights, chosen effector, and EU to `atol=1e-9` across 10 cases
(ancestry/model tempering, the subject/time covariates, every terminal action as a winner);
the `net_voi` gate prices a perfect probe above a useless one; and observation-log replay
reconstructs the posterior exactly.
