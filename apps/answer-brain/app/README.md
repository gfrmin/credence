# answer-brain app

A pi-mono agent, driven by a local **Ollama** model, that answers the owner's point-fact
questions through the answer-brain body and **captures a one-bit good/bad verdict** that folds
into `u_wrong` (Stage B — the dogfood driver). **This is the driver, not the gate** — the gate
(`life-agent/scripts/answer_brain_gate.py`) certifies the decision math deterministically; this
runs the govern+steer loop end-to-end with a real (nondeterministic) LLM driving the tools and
the brain governing the answer, and accrues the owner's verdicts.

Everything runs on-machine (the corpus is the owner's real PII — no cloud model).

## check.ts — deterministic, no Ollama

Proves the app↔daemon plumbing: loads the extension against the live daemon, fetches
`GET /manifest`, and verifies the body implements it + registers its tools.

```bash
npm install
# daemon up on :8799 (julia --project=$HOME/git/credence apps/answer-brain/daemon/main.jl)
npm run check
```

## main.ts — the full agent (REPL + verdict capture)

The easy way is the launcher (in life-agent), which starts the daemon + bridge, waits for
both, runs this app, and tears the services down on exit:

```bash
answer-brain                      # REPL: ask, react [g]ood/[b]ad, repeat
answer-brain "what is my X?"      # one question, react, exit
```

Or run the app directly, with the three prereqs already up (daemon :8799, bridge :8798,
Ollama with the model pulled):

```bash
npm run ask                       # REPL
npm run ask -- "what is my X?"    # one-shot
```

The agent calls `retrieve_documents → extract_candidates → answer`; on `answer` the brain
posts the accumulated evidence to `/decide` and either lets the answer through (rewritten to
the evidence-backed value), steers a recency gather (re-extract, re-decide), or withholds.
After each answer the app prompts a **one-bit** verdict and posts it to the bridge
(`POST /log_reaction`); an abstain verdict folds into `u_wrong` on the next gate run, a report
verdict is recorded-not-folded. The verdict is one bit (good/bad) — never free text.

Env: `ANSWER_BRAIN_DAEMON_URL` (:8799), `LIFE_AGENT_BRIDGE_URL` (:8798),
`ANSWER_BRAIN_MODEL` (qwen2.5:7b-instruct), `OLLAMA_BASE_URL` (http://localhost:11434/v1).
