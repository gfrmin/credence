# answer-brain test fixtures

Per the repo CLAUDE.md ("Test fixtures are commit-pinned"): regenerate only by re-running
the named generator at a recorded SHA and updating this note. Never hand-edit a fixture to
fix a loading bug — fix the loader / the brain.

## `stage0_parity.json` — schema `answer-brain/parity/v1`

The Stage-1 parity oracle (`move-1-design.md` §3). The reference `(weights, p_none, action,
eu)` per case is the **Stage-0 lookup answerer's** output — `life_agent.core.lookup`'s
`lookup_posterior` + `decide`, run through the credence skin.

- **Generator:** `scripts/dump_parity_fixtures.py` in the **life-agent** repo.
- **Provenance SHA:** life-agent `c1a781f` (branch `capability-sprint-extraction`).
- **Regenerate:** `uv run python scripts/dump_parity_fixtures.py --out <path>` in life-agent,
  then copy here and update the SHA above.
- **Content:** 10 synthetic cases (no PII — every value is a synthetic label) covering
  ancestry/model tempering, the subject/time covariates, and each terminal action
  (report / hedge / ask_clarify / abstain) as a winner. Observations are integer-indexed:
  the brain reasons over abstract candidates/groups, so candidate identity stays Python-side.
