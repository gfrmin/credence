# answer-brain demo app

A minimal pi-mono agent, driven by a local **Ollama** model, that answers the owner's
point-fact questions through the answer-brain body. **This is the demo, not the gate** —
the gate (`life-agent/scripts/answer_brain_gate.py`) certifies the decision math
deterministically; this *shows* the govern+steer loop end-to-end with a real LLM driving
the tools and the brain governing the answer. Its result is reported, not assumed.

Everything runs on-machine (the corpus is the owner's real PII — no cloud model).

## check.ts — deterministic, no Ollama

Proves the app↔daemon plumbing: loads the extension against the live daemon, fetches
`GET /manifest`, and verifies the body implements it + registers its tools.

```bash
npm install
# daemon up on :8799 (julia --project=$HOME/git/credence apps/answer-brain/daemon/main.jl)
npm run check
```

## main.ts — the full agent (opt-in)

```bash
# 1. daemon:  julia --project=$HOME/git/credence $HOME/git/credence/apps/answer-brain/daemon/main.jl
# 2. bridge:  from a life-agent checkout WITH bridge era_split (branch bridge-era-split):
#             uv run python -m life_agent.bridge.server
# 3. Ollama:  the model pulled (default qwen2.5:7b-instruct)
npm run ask -- "what is my mobile phone number?"
```

The agent calls `retrieve_documents → extract_candidates → answer`; on `answer` the brain
posts the accumulated evidence to `/decide` and either lets the answer through (rewritten to
the evidence-backed value), steers a recency gather (re-extract, re-decide), or withholds.

Env: `ANSWER_BRAIN_DAEMON_URL` (:8799), `LIFE_AGENT_BRIDGE_URL` (:8798),
`ANSWER_BRAIN_MODEL` (qwen2.5:7b-instruct), `OLLAMA_BASE_URL` (http://localhost:11434/v1).
