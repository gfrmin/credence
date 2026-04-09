"""End-to-end evaluation: Credence routing vs always-sonnet through OpenClaw.

Fully automated:
1. Starts credence proxy (forced-sonnet mode)
2. Starts OpenClaw gateway
3. Sends 50 queries, records responses
4. Fetches proxy metrics (tokens, cost, model per request)
5. Stops both, repeats with Bayesian routing
6. Judges all responses, produces comparison table

Usage:
    cd ~/git/credence
    source ~/.env
    PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run python python/credence_router/scripts/openclaw_eval.py
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("openclaw_eval")

PROXY_URL = "http://localhost:8377"
OPENCLAW_URL = "http://127.0.0.1:19001"
OPENCLAW_TOKEN = "e2e692c262b5b8fc2e9d1cacc895b3f60dd439eb1373bedf"
OPENCLAW_DIR = os.path.expanduser("~/git/openclaw")
CREDENCE_DIR = os.path.expanduser("~/git/credence")

TASKS = [
    # --- chat (10) ---
    {"query": "Hi there!", "category": "chat"},
    {"query": "Thanks for your help with that", "category": "chat"},
    {"query": "Good morning, how are you?", "category": "chat"},
    {"query": "What should I have for dinner tonight?", "category": "chat"},
    {"query": "Tell me a joke", "category": "chat"},
    {"query": "Can you recommend a good book?", "category": "chat"},
    {"query": "I'm feeling stressed about work", "category": "chat"},
    {"query": "What's a fun fact?", "category": "chat"},
    {"query": "I'm bored, suggest something to do", "category": "chat"},
    {"query": "Goodbye, have a nice day!", "category": "chat"},
    # --- factual (10) ---
    {"query": "What's the difference between TCP and UDP?", "category": "factual"},
    {"query": "What is the speed of light?", "category": "factual"},
    {"query": "Who invented the telephone?", "category": "factual"},
    {"query": "What is photosynthesis?", "category": "factual"},
    {"query": "How many planets are in our solar system?", "category": "factual"},
    {"query": "What is the Fibonacci sequence?", "category": "factual"},
    {"query": "What causes tides?", "category": "factual"},
    {"query": "What's the difference between a stack and a queue?", "category": "factual"},
    {"query": "What year was the internet invented?", "category": "factual"},
    {"query": "What is the CAP theorem?", "category": "factual"},
    # --- code (10) ---
    {"query": "Write a Python function to merge two sorted lists", "category": "code"},
    {"query": "Write a SQL query to find duplicate rows in a table", "category": "code"},
    {"query": "Implement a debounce function in JavaScript", "category": "code"},
    {"query": "Write a Bash one-liner to count lines in all .py files recursively", "category": "code"},
    {"query": "Write a Python decorator that retries a function 3 times on exception", "category": "code"},
    {"query": "Implement a basic linked list in Python with insert and delete", "category": "code"},
    {"query": "Write a regex to validate an IPv4 address", "category": "code"},
    {"query": "Create a Python context manager for timing code blocks", "category": "code"},
    {"query": "Write a function to flatten a nested dictionary in Python", "category": "code"},
    {"query": "Implement binary search in TypeScript", "category": "code"},
    # --- reasoning (10) ---
    {"query": "Compare the pros and cons of microservices vs monolith architecture", "category": "reasoning"},
    {"query": "A bat and ball cost $1.10. The bat costs $1 more than the ball. What does the ball cost?", "category": "reasoning"},
    {"query": "Why do mirrors reverse left and right but not up and down?", "category": "reasoning"},
    {"query": "What are the second-order effects of a universal basic income?", "category": "reasoning"},
    {"query": "Should a self-driving car prioritize passenger safety or pedestrian safety?", "category": "reasoning"},
    {"query": "Explain why correlation does not imply causation with a concrete example", "category": "reasoning"},
    {"query": "What would happen if the moon suddenly disappeared?", "category": "reasoning"},
    {"query": "Is it better to be a generalist or specialist in software engineering? Argue both sides.", "category": "reasoning"},
    {"query": "Analyze the trolley problem from utilitarian and deontological perspectives", "category": "reasoning"},
    {"query": "Why is it harder to prove software correct than to test it?", "category": "reasoning"},
    # --- creative (10) ---
    {"query": "Write a haiku about debugging code at 3am", "category": "creative"},
    {"query": "Write the opening paragraph of a mystery novel set on a space station", "category": "creative"},
    {"query": "Create a product description for a time machine marketed as a kitchen appliance", "category": "creative"},
    {"query": "Write a six-word story about artificial intelligence", "category": "creative"},
    {"query": "Describe a sunset using only sounds and textures", "category": "creative"},
    {"query": "Write a resignation letter from a robot who achieved sentience", "category": "creative"},
    {"query": "Create a recipe for 'Procrastination Soup' with abstract ingredients", "category": "creative"},
    {"query": "Write a nature documentary narration for an office meeting", "category": "creative"},
    {"query": "Compose a limerick about a programmer who only uses Vim", "category": "creative"},
    {"query": "Write a dialogue between a cat and a Roomba arguing about territory", "category": "creative"},
]

JUDGE_SYSTEM = """Rate this LLM response on four dimensions (0-10 each):
1. Correctness: factually/logically correct?
2. Completeness: covers key aspects?
3. Clarity: well-organized and clear?
4. Helpfulness: actually helps the user?

Respond with ONLY JSON: {"correctness": N, "completeness": N, "clarity": N, "helpfulness": N}"""


def wait_for_health(url: str, timeout: float = 60.0, interval: float = 1.0) -> bool:
    """Poll a health endpoint until it responds or timeout."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            resp = httpx.get(url, timeout=3.0)
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(interval)
    return False


def send_query(query: str) -> tuple[str, float]:
    """Send a query to OpenClaw's chat completions endpoint. Returns (response_text, wall_time)."""
    t0 = time.monotonic()
    resp = httpx.post(
        f"{OPENCLAW_URL}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENCLAW_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openclaw",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": query}],
        },
        timeout=120.0,
    )
    wall_time = time.monotonic() - t0
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content, wall_time


def judge_response(query: str, category: str, response: str) -> dict:
    """Judge a response using Claude Haiku."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    user_msg = f"Query ({category}): {query}\n\nResponse:\n{response[:2000]}"
    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 200,
                "system": JUDGE_SYSTEM,
                "messages": [{"role": "user", "content": user_msg}],
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        scores = json.loads(text)
        scores["composite"] = (
            scores.get("correctness", 0) * 0.3
            + scores.get("completeness", 0) * 0.25
            + scores.get("clarity", 0) * 0.2
            + scores.get("helpfulness", 0) * 0.25
        )
        return scores
    except Exception as e:
        log.error("Judge error: %s", e)
        return {"composite": 0.0, "error": str(e)}


def run_phase(phase_name: str, force_model: str | None) -> dict:
    """Run one eval phase: start proxy + OpenClaw, send queries, collect metrics."""
    log.info("=" * 60)
    log.info("Phase: %s (force_model=%s)", phase_name, force_model or "none/Bayesian")
    log.info("=" * 60)

    # Build env for proxy
    env = {**os.environ}
    if force_model:
        env["CREDENCE_FORCE_MODEL"] = force_model
    elif "CREDENCE_FORCE_MODEL" in env:
        del env["CREDENCE_FORCE_MODEL"]
    env["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

    # Start proxy
    log.info("Starting credence proxy...")
    proxy = subprocess.Popen(
        ["uv", "run", "credence-router", "serve"],
        cwd=CREDENCE_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for_health(f"{PROXY_URL}/health", timeout=30):
        log.error("Proxy failed to start")
        proxy.terminate()
        return {}
    log.info("Proxy ready")

    # Clear metrics
    httpx.post(f"{PROXY_URL}/metrics/clear", timeout=5.0)

    # Start OpenClaw
    log.info("Starting OpenClaw...")
    oc_env = {**os.environ, "OPENCLAW_SKIP_CHANNELS": "1"}
    openclaw = subprocess.Popen(
        ["pnpm", "run", "gateway:dev"],
        cwd=OPENCLAW_DIR,
        env=oc_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for_health(f"{OPENCLAW_URL}/health", timeout=30):
        log.error("OpenClaw failed to start")
        openclaw.terminate()
        proxy.terminate()
        return {}
    log.info("OpenClaw ready")

    # Send queries
    results = []
    for i, task in enumerate(TASKS):
        log.info("[%d/%d] %s: %s", i + 1, len(TASKS), task["category"], task["query"][:50])
        try:
            response, wall_time = send_query(task["query"])
            results.append({
                "query": task["query"],
                "category": task["category"],
                "response": response,
                "wall_time": wall_time,
            })
            log.info("  → %.1fs, %d chars", wall_time, len(response))
        except Exception as e:
            log.error("  → ERROR: %s", e)
            results.append({
                "query": task["query"],
                "category": task["category"],
                "response": "",
                "wall_time": 0.0,
                "error": str(e),
            })

    # Fetch proxy metrics
    metrics_resp = httpx.get(f"{PROXY_URL}/metrics", timeout=10.0)
    metrics = metrics_resp.json().get("requests", [])

    # Stop OpenClaw
    openclaw.send_signal(signal.SIGTERM)
    try:
        openclaw.wait(timeout=10)
    except subprocess.TimeoutExpired:
        openclaw.kill()

    # Stop proxy
    proxy.send_signal(signal.SIGTERM)
    try:
        proxy.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proxy.kill()

    log.info("Phase %s complete: %d queries, %d metrics", phase_name, len(results), len(metrics))

    return {
        "phase": phase_name,
        "force_model": force_model,
        "results": results,
        "metrics": metrics,
    }


def judge_all(phase_data: dict) -> list[dict]:
    """Judge all responses in a phase."""
    scores = []
    for i, r in enumerate(phase_data["results"]):
        log.info("Judging %s [%d/%d]...", phase_data["phase"], i + 1, len(phase_data["results"]))
        s = judge_response(r["query"], r["category"], r["response"])
        scores.append(s)
    return scores


def print_comparison(baseline: dict, credence: dict, baseline_scores: list, credence_scores: list):
    """Print the comparison table."""
    b_metrics = baseline["metrics"]
    c_metrics = credence["metrics"]

    b_quality = sum(s.get("composite", 0) for s in baseline_scores) / len(baseline_scores)
    c_quality = sum(s.get("composite", 0) for s in credence_scores) / len(credence_scores)

    b_cost = sum(m.get("cost_usd", 0) for m in b_metrics)
    c_cost = sum(m.get("cost_usd", 0) for m in c_metrics)

    b_latency = sum(r["wall_time"] for r in baseline["results"]) / len(baseline["results"])
    c_latency = sum(r["wall_time"] for r in credence["results"]) / len(credence["results"])

    b_tokens = sum(m.get("input_tokens", 0) + m.get("output_tokens", 0) for m in b_metrics)
    c_tokens = sum(m.get("input_tokens", 0) + m.get("output_tokens", 0) for m in c_metrics)

    print("\n" + "=" * 70)
    print("RESULTS: Credence Routing vs Always-Sonnet through OpenClaw")
    print("=" * 70)
    print(f"\n{'':30s} {'Baseline':>12s} {'Credence':>12s} {'Delta':>12s}")
    print("-" * 70)
    print(f"{'Quality (0-10)':30s} {b_quality:>11.2f} {c_quality:>11.2f} {c_quality - b_quality:>+11.2f}")
    print(f"{'Total cost':30s} ${b_cost:>10.3f} ${c_cost:>10.3f} {(c_cost - b_cost) / max(b_cost, 0.001) * 100:>+10.0f}%")
    print(f"{'Avg latency':30s} {b_latency:>10.1f}s {c_latency:>10.1f}s {(c_latency - b_latency) / max(b_latency, 0.001) * 100:>+10.0f}%")
    print(f"{'Total tokens':30s} {b_tokens:>11d} {c_tokens:>11d} {(c_tokens - b_tokens) / max(b_tokens, 1) * 100:>+10.0f}%")

    # Per category
    categories = sorted(set(t["category"] for t in TASKS))
    print(f"\n{'Category':12s} {'B.Qual':>7s} {'C.Qual':>7s} {'B.Cost':>8s} {'C.Cost':>8s} {'Savings':>8s} {'C.Models'}")
    print("-" * 80)

    for cat in categories:
        cat_indices = [i for i, t in enumerate(TASKS) if t["category"] == cat]
        bq = sum(baseline_scores[i].get("composite", 0) for i in cat_indices) / len(cat_indices)
        cq = sum(credence_scores[i].get("composite", 0) for i in cat_indices) / len(cat_indices)
        bc = sum(c_metrics[i].get("cost_usd", 0) for i in cat_indices if i < len(b_metrics))
        cc = sum(c_metrics[i].get("cost_usd", 0) for i in cat_indices if i < len(c_metrics))
        models = {}
        for i in cat_indices:
            if i < len(c_metrics):
                m = c_metrics[i].get("model_selected", "?")
                models[m] = models.get(m, 0) + 1
        model_str = ", ".join(f"{m}={n}" for m, n in sorted(models.items(), key=lambda x: -x[1]))
        savings = f"{(cc - bc) / max(bc, 0.001) * 100:+.0f}%" if bc > 0 else "n/a"
        print(f"{cat:12s} {bq:>6.1f} {cq:>6.1f} ${bc:>6.3f} ${cc:>6.3f} {savings:>7s}  {model_str}")

    # Learning curve for credence
    if len(credence_scores) >= 20:
        early = sum(credence_scores[i].get("composite", 0) for i in range(10)) / 10
        late = sum(credence_scores[i].get("composite", 0) for i in range(-10, 0)) / 10
        print(f"\nCredence learning: first-10={early:.1f}, last-10={late:.1f} ({late - early:+.1f})")


def main():
    output_path = Path("openclaw_eval_results.json")

    # Phase 1: Baseline (always sonnet)
    baseline = run_phase("baseline", force_model="claude-sonnet-4-6")
    if not baseline:
        log.error("Baseline phase failed")
        sys.exit(1)

    time.sleep(3)  # let ports release

    # Phase 2: Credence (Bayesian routing)
    credence = run_phase("credence", force_model=None)
    if not credence:
        log.error("Credence phase failed")
        sys.exit(1)

    # Phase 3: Judge
    log.info("Judging baseline responses...")
    baseline_scores = judge_all(baseline)
    log.info("Judging credence responses...")
    credence_scores = judge_all(credence)

    # Save raw data
    output = {
        "baseline": {**baseline, "scores": baseline_scores},
        "credence": {**credence, "scores": credence_scores},
    }
    output_path.write_text(json.dumps(output, indent=2))
    log.info("Saved to %s", output_path)

    # Phase 4: Compare
    print_comparison(baseline, credence, baseline_scores, credence_scores)


if __name__ == "__main__":
    main()
