# Role: body
"""Full benchmark: Bayesian model-tier routing cost savings on coding workloads.

Executes the Move 1 methodology: 20 workloads × N=3 repetitions × 2 phases
(baseline forced-Sonnet + Bayesian routing). Reports regime-tagged results
(cold-start / transition / warm-state), variance decomposition, per-task-type
breakdown, and Opus-judged quality scores.

Usage:
    cd ~/git/credence
    PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run python \
        apps/python/credence_router/scripts/benchmark.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("benchmark")

PROXY_URL = "http://localhost:8377"
CREDENCE_DIR = os.path.expanduser("~/git/credence")
WORKLOADS_DIR = Path(__file__).parent / "feasibility_workloads"

N_REPS = 3
JUDGE_DELAY_SECONDS = 4

# Regime boundaries (1-indexed workload position within a repetition)
REGIME_COLD = range(1, 6)       # workloads 1–5
REGIME_TRANSITION = range(6, 11)  # workloads 6–10
REGIME_WARM = range(11, 21)      # workloads 11–20

# Pricing snapshot (2026-04-28, per 1M tokens)
PRICING_SNAPSHOT = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
}
PRICING_DATE = "2026-04-28"


def load_workloads() -> list[dict]:
    workloads = []
    for path in sorted(WORKLOADS_DIR.glob("*.json")):
        workloads.append(json.loads(path.read_text()))
    return workloads


def wait_for_ready(url: str, timeout: float = 180.0, interval: float = 2.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(interval)
    return False


def send_conversation_turn(messages: list[dict], max_tokens: int = 1024) -> tuple[str, dict]:
    content_parts: list[str] = []
    model_selected = "unknown"

    with httpx.Client(timeout=180.0) as client:
        with client.stream(
            "POST",
            f"{PROXY_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={"model": "auto", "messages": messages, "max_tokens": max_tokens},
        ) as resp:
            model_selected = resp.headers.get("X-Credence-Model", "unknown")
            if resp.status_code >= 400:
                resp.read()
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp,
                )
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        content_parts.append(text)
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass

    return "".join(content_parts), {"model": model_selected}


def check_outcome_regex(workload: dict, assistant_responses: list[str]) -> bool:
    patterns = workload.get("expected_patterns", [])
    if not patterns:
        return True
    combined_text = "\n".join(assistant_responses)
    for pattern in patterns:
        if re.search(pattern, combined_text, re.IGNORECASE):
            return True
    return False


async def judge_quality_opus(
    user_messages: list[str],
    assistant_responses: list[str],
    model_used: str,
) -> float | None:
    """External Opus judge: scores overall conversation quality 0-10."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None

    conversation_excerpt = []
    for i, (u, a) in enumerate(zip(user_messages, assistant_responses)):
        conversation_excerpt.append(f"Turn {i+1} user: {u[:300]}")
        conversation_excerpt.append(f"Turn {i+1} assistant ({model_used}): {a[:500]}")
    text = "\n\n".join(conversation_excerpt)
    if len(text) > 6000:
        text = text[:6000] + "\n[truncated]"

    system = (
        "You are evaluating an AI coding assistant's performance across a multi-turn "
        "conversation. Rate the overall quality 0-10 on three dimensions:\n"
        "- Correctness: Are the code samples and explanations technically correct?\n"
        "- Completeness: Does the assistant address all parts of each request?\n"
        "- Helpfulness: Is the response well-structured and actionable?\n\n"
        "Respond with EXACTLY three lines:\n"
        "correctness: N\n"
        "completeness: N\n"
        "helpfulness: N\n"
        "where N is 0-10. Nothing else."
    )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-opus-4-6",
                    "max_tokens": 50,
                    "system": system,
                    "messages": [{"role": "user", "content": text}],
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            judge_text = resp.json()["content"][0]["text"].strip()
            scores = []
            for line in judge_text.split("\n"):
                if ":" in line:
                    try:
                        val = float(line.split(":")[1].strip())
                        scores.append(max(0.0, min(10.0, val)))
                    except ValueError:
                        pass
            if scores:
                return sum(scores) / len(scores)
            return None
    except Exception as e:
        log.error("Opus judge failed: %s", e)
        return None


def run_workload(workload: dict, phase: str, wait_for_judge: bool) -> dict:
    name = workload["name"]
    system_prompt = workload.get("system_prompt", "You are a helpful coding assistant.")
    user_turns = workload["turns"]

    messages = [{"role": "system", "content": system_prompt}]
    turns = []
    all_assistant_text = []
    all_user_text = []

    for i, user_msg in enumerate(user_turns):
        messages.append({"role": "user", "content": user_msg})
        all_user_text.append(user_msg)
        t0 = time.monotonic()

        try:
            content, info = send_conversation_turn(messages)
        except Exception as e:
            log.error("  [%s] Turn %d failed: %s", name, i + 1, e)
            turns.append({
                "turn": i + 1,
                "model": "error",
                "error": str(e),
                "wall_time": time.monotonic() - t0,
            })
            messages.append({"role": "assistant", "content": f"[Error: {e}]"})
            continue

        wall_time = time.monotonic() - t0
        messages.append({"role": "assistant", "content": content})
        all_assistant_text.append(content)

        log.info(
            "  [%s] Turn %d/%d -> %s (%.1fs, %d chars)",
            name, i + 1, len(user_turns), info["model"], wall_time, len(content),
        )
        turns.append({
            "turn": i + 1,
            "model": info["model"],
            "wall_time": wall_time,
            "response_length": len(content),
        })

        if wait_for_judge and i < len(user_turns) - 1:
            time.sleep(JUDGE_DELAY_SECONDS)

    regex_pass = check_outcome_regex(workload, all_assistant_text)

    opus_score = asyncio.run(judge_quality_opus(
        all_user_text, all_assistant_text,
        turns[-1]["model"] if turns else "unknown",
    ))

    return {
        "workload": name,
        "task_type": workload["task_type"],
        "phase": phase,
        "turns": turns,
        "n_turns": len(user_turns),
        "outcome_pass": regex_pass,
        "opus_score": opus_score,
    }


def fetch_metrics() -> list[dict]:
    try:
        resp = httpx.get(f"{PROXY_URL}/metrics", timeout=10.0)
        return resp.json().get("requests", [])
    except Exception as e:
        log.error("Failed to fetch metrics: %s", e)
        return []


def _refresh_api_keys(env: dict) -> dict:
    try:
        for key_name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            result = subprocess.run(
                ["secret-tool", "lookup", "service", "env", "key", key_name],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                env[key_name] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return env


def start_proxy(
    force_model: str | None = None,
    providers: tuple[str, ...] = ("anthropic",),
    state_dir: str | None = None,
) -> subprocess.Popen:
    env = _refresh_api_keys({**os.environ})
    provider_keys = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
    for provider, key in provider_keys.items():
        if provider not in providers and key in env:
            del env[key]
    if force_model:
        env["CREDENCE_FORCE_MODEL"] = force_model
    elif "CREDENCE_FORCE_MODEL" in env:
        del env["CREDENCE_FORCE_MODEL"]
    env["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
    env["CREDENCE_JULIA"] = "julia"

    if state_dir is None:
        state_dir = tempfile.mkdtemp(prefix="credence-bench-")
    env["CREDENCE_STATE_PATH"] = os.path.join(state_dir, "search-state.json")
    env["CREDENCE_LLM_STATE_PATH"] = os.path.join(state_dir, "llm-state.bin")

    log_file = tempfile.NamedTemporaryFile(
        prefix="credence-bench-", suffix=".log", delete=False, mode="w",
    )
    log.info("Proxy log: %s", log_file.name)
    proc = subprocess.Popen(
        ["uv", "run", "credence-router", "serve"],
        cwd=CREDENCE_DIR,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc


def stop_proxy(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def compute_costs(metrics: list[dict]) -> dict[str, float]:
    total_cost = sum(m.get("cost_usd", 0.0) for m in metrics)
    total_input = sum(m.get("input_tokens", 0) for m in metrics)
    total_output = sum(m.get("output_tokens", 0) for m in metrics)
    return {
        "total_cost": total_cost,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
    }


def assign_metrics_to_workloads(
    phase_results: list[dict], metrics: list[dict],
) -> list[dict]:
    enriched = []
    metric_idx = 0
    for result in phase_results:
        n_turns = result["n_turns"]
        workload_metrics = metrics[metric_idx: metric_idx + n_turns]
        metric_idx += n_turns

        costs = compute_costs(workload_metrics)
        models_used = [m.get("model_selected", "?") for m in workload_metrics]
        model_counts: dict[str, int] = {}
        for m in models_used:
            model_counts[m] = model_counts.get(m, 0) + 1

        enriched.append({
            **result,
            "cost": costs,
            "models_used": models_used,
            "model_distribution": model_counts,
        })
    return enriched


def compute_opus_baseline_cost(metrics: list[dict]) -> float:
    """Reprice Sonnet baseline metrics at Opus rates (secondary baseline)."""
    opus_prices = PRICING_SNAPSHOT["claude-opus-4-6"]
    total = 0.0
    for m in metrics:
        inp = m.get("input_tokens", 0)
        out = m.get("output_tokens", 0)
        total += (inp * opus_prices["input"] + out * opus_prices["output"]) / 1_000_000
    return total


def regime_tag(workload_position: int) -> str:
    if workload_position in REGIME_COLD:
        return "cold-start"
    elif workload_position in REGIME_TRANSITION:
        return "transition"
    else:
        return "warm-state"


def run_repetition(
    rep_id: int,
    workloads: list[dict],
    force_model: str | None,
    phase_name: str,
) -> dict:
    """Run one complete repetition (all workloads, fresh proxy state)."""
    log.info("=" * 60)
    log.info(
        "Rep %d/%d — Phase: %s (force=%s)",
        rep_id + 1, N_REPS, phase_name, force_model or "Bayesian routing",
    )
    log.info("=" * 60)

    state_dir = tempfile.mkdtemp(prefix=f"credence-bench-{phase_name}-rep{rep_id}-")
    proxy = start_proxy(force_model=force_model, state_dir=state_dir)

    if not wait_for_ready(f"{PROXY_URL}/ready"):
        log.error("Proxy failed to start within 180s")
        stop_proxy(proxy)
        return {"error": "proxy_start_failed", "results": [], "metrics": []}
    log.info("Proxy ready")

    httpx.post(f"{PROXY_URL}/metrics/clear", timeout=5.0)

    results = []
    wait_for_judge = force_model is None
    belief_snapshots = []

    for idx, workload in enumerate(workloads):
        position = idx + 1
        tag = regime_tag(position)
        log.info(
            "[Rep %d] Workload %d/%d: %s (%s) [%s]",
            rep_id + 1, position, len(workloads),
            workload["name"], workload["task_type"], tag,
        )
        result = run_workload(workload, phase_name, wait_for_judge)
        result["position"] = position
        result["regime"] = tag
        result["rep_id"] = rep_id
        results.append(result)

        if not force_model and position in (5, 10, 15, 20):
            try:
                state_resp = httpx.get(f"{PROXY_URL}/state", timeout=10.0)
                belief_snapshots.append({
                    "after_workload": position,
                    "state": state_resp.json(),
                })
            except Exception:
                pass

    metrics = fetch_metrics()

    try:
        state_resp = httpx.get(f"{PROXY_URL}/state", timeout=10.0)
        final_state = state_resp.json()
    except Exception:
        final_state = {}

    stop_proxy(proxy)

    return {
        "phase": phase_name,
        "rep_id": rep_id,
        "force_model": force_model,
        "results": results,
        "metrics": metrics,
        "final_state": final_state,
        "belief_snapshots": belief_snapshots,
        "state_dir": state_dir,
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def analyze_results(
    workloads: list[dict],
    baseline_reps: list[dict],
    routing_reps: list[dict],
) -> dict:
    """Compute all analysis: per-workload stats, regime breakdown, variance decomposition."""

    # Enrich each rep's results with metrics
    for rep in baseline_reps + routing_reps:
        rep["enriched"] = assign_metrics_to_workloads(rep["results"], rep["metrics"])

    per_workload_stats = []
    for w_idx, workload in enumerate(workloads):
        name = workload["name"]
        task_type = workload["task_type"]
        position = w_idx + 1
        tag = regime_tag(position)

        # Baseline stats
        b_costs = [rep["enriched"][w_idx]["cost"]["total_cost"] for rep in baseline_reps]
        b_tokens = [rep["enriched"][w_idx]["cost"]["total_tokens"] for rep in baseline_reps]
        b_outcomes = [rep["enriched"][w_idx]["outcome_pass"] for rep in baseline_reps]
        b_opus_scores = [
            rep["enriched"][w_idx].get("opus_score")
            for rep in baseline_reps
            if rep["enriched"][w_idx].get("opus_score") is not None
        ]

        # Routing stats
        r_costs = [rep["enriched"][w_idx]["cost"]["total_cost"] for rep in routing_reps]
        r_tokens = [rep["enriched"][w_idx]["cost"]["total_tokens"] for rep in routing_reps]
        r_outcomes = [rep["enriched"][w_idx]["outcome_pass"] for rep in routing_reps]
        r_opus_scores = [
            rep["enriched"][w_idx].get("opus_score")
            for rep in routing_reps
            if rep["enriched"][w_idx].get("opus_score") is not None
        ]
        r_model_dists = [rep["enriched"][w_idx].get("model_distribution", {}) for rep in routing_reps]

        # Savings
        savings_pcts = [
            ((b - r) / b * 100) if b > 0 else 0.0
            for b, r in zip(b_costs, r_costs)
        ]

        # Quality degradation: baseline passes but routing fails
        quality_degradations = sum(
            1 for b, r in zip(b_outcomes, r_outcomes) if b and not r
        )

        # Opus score degradation (>1.0 point drop)
        opus_degradations = 0
        if b_opus_scores and r_opus_scores:
            b_mean_opus = _mean(b_opus_scores)
            r_mean_opus = _mean(r_opus_scores)
            if b_mean_opus - r_mean_opus > 1.0:
                opus_degradations = 1

        # Opus secondary baseline: reprice at Opus rates
        opus_costs = []
        for rep in baseline_reps:
            rep_metrics = rep["metrics"]
            start = sum(r["n_turns"] for r in rep["results"][:w_idx])
            end = start + rep["results"][w_idx]["n_turns"]
            opus_cost = compute_opus_baseline_cost(rep_metrics[start:end])
            opus_costs.append(opus_cost)

        per_workload_stats.append({
            "workload": name,
            "task_type": task_type,
            "position": position,
            "regime": tag,
            "n_turns": workload.get("turns", []) and len(workload["turns"]),
            "baseline_cost_mean": _mean(b_costs),
            "baseline_cost_std": _std(b_costs),
            "routing_cost_mean": _mean(r_costs),
            "routing_cost_std": _std(r_costs),
            "savings_pct_mean": _mean(savings_pcts),
            "savings_pct_std": _std(savings_pcts),
            "baseline_pass_rate": sum(b_outcomes) / len(b_outcomes) if b_outcomes else 0,
            "routing_pass_rate": sum(r_outcomes) / len(r_outcomes) if r_outcomes else 0,
            "baseline_opus_score_mean": _mean(b_opus_scores) if b_opus_scores else None,
            "routing_opus_score_mean": _mean(r_opus_scores) if r_opus_scores else None,
            "quality_degradations": quality_degradations,
            "opus_degradations": opus_degradations,
            "opus_baseline_cost_mean": _mean(opus_costs),
            "opus_savings_pct": (
                (_mean(opus_costs) - _mean(r_costs)) / _mean(opus_costs) * 100
                if _mean(opus_costs) > 0 else 0.0
            ),
            "routing_model_distributions": r_model_dists,
            "baseline_tokens_mean": _mean(b_tokens),
        })

    # Regime aggregates
    regime_stats = {}
    for tag_name in ("cold-start", "transition", "warm-state"):
        regime_workloads = [w for w in per_workload_stats if w["regime"] == tag_name]
        if not regime_workloads:
            continue
        savings = [w["savings_pct_mean"] for w in regime_workloads]
        tokens = [w["baseline_tokens_mean"] for w in regime_workloads]
        total_tokens = sum(tokens)
        weighted_savings = (
            sum(s * t for s, t in zip(savings, tokens)) / total_tokens
            if total_tokens > 0 else 0.0
        )
        regime_stats[tag_name] = {
            "n_workloads": len(regime_workloads),
            "mean_savings_pct": _mean(savings),
            "weighted_savings_pct": weighted_savings,
            "savings_std": _std(savings),
            "quality_degradations": sum(w["quality_degradations"] for w in regime_workloads),
            "opus_degradations": sum(w["opus_degradations"] for w in regime_workloads),
        }

    # Task-type aggregates
    task_types = sorted(set(w["task_type"] for w in per_workload_stats))
    task_type_stats = {}
    for tt in task_types:
        tt_workloads = [w for w in per_workload_stats if w["task_type"] == tt]
        savings = [w["savings_pct_mean"] for w in tt_workloads]
        task_type_stats[tt] = {
            "n_workloads": len(tt_workloads),
            "mean_savings_pct": _mean(savings),
            "savings_std": _std(savings),
        }

    # Overall aggregates
    all_savings = [w["savings_pct_mean"] for w in per_workload_stats]
    all_tokens = [w["baseline_tokens_mean"] for w in per_workload_stats]
    total_tokens = sum(all_tokens)
    weighted_savings = (
        sum(s * t for s, t in zip(all_savings, all_tokens)) / total_tokens
        if total_tokens > 0 else 0.0
    )
    total_quality_degradations = sum(w["quality_degradations"] for w in per_workload_stats)
    total_opus_degradations = sum(w["opus_degradations"] for w in per_workload_stats)

    # Variance decomposition: baseline std is output variance, routing std is output+routing
    baseline_cost_stds = [w["baseline_cost_std"] for w in per_workload_stats]
    routing_cost_stds = [w["routing_cost_std"] for w in per_workload_stats]
    output_variance = _mean([s ** 2 for s in baseline_cost_stds])
    total_variance = _mean([s ** 2 for s in routing_cost_stds])
    routing_variance = max(0.0, total_variance - output_variance)

    return {
        "per_workload": per_workload_stats,
        "regime_stats": regime_stats,
        "task_type_stats": task_type_stats,
        "aggregate": {
            "n_workloads": len(per_workload_stats),
            "n_reps": N_REPS,
            "mean_savings_pct": _mean(all_savings),
            "savings_std": _std(all_savings),
            "weighted_savings_pct": weighted_savings,
            "quality_degradation_count": total_quality_degradations,
            "opus_degradation_count": total_opus_degradations,
            "pricing_date": PRICING_DATE,
        },
        "variance_decomposition": {
            "output_variance": output_variance,
            "total_variance": total_variance,
            "routing_variance_contribution": routing_variance,
            "routing_fraction": routing_variance / total_variance if total_variance > 0 else 0.0,
        },
    }


def format_report(analysis: dict, routing_reps: list[dict]) -> str:
    agg = analysis["aggregate"]
    lines = [
        "# Move 2 — Benchmark results",
        "",
        "## Executive summary",
        "",
        f"Bayesian model-tier routing achieved **{agg['weighted_savings_pct']:.1f}%** "
        f"token-volume-weighted cost savings vs always-Sonnet across {agg['n_workloads']} "
        f"coding workloads (N={agg['n_reps']} repetitions). "
        f"Quality degradation rate: {agg['quality_degradation_count']}/{agg['n_workloads'] * agg['n_reps']} "
        f"workload-repetition pairs (regex), "
        f"{agg['opus_degradation_count']}/{agg['n_workloads']} workloads (Opus judge, >1pt drop). "
        "Scope: model-tier routing on Anthropic models only; cache savings explicitly "
        "not measured (Move 0 finding: prompt caching unsupported on OAI-compatible endpoint).",
        "",
        "## Methodology",
        "",
        f"- **Workloads:** {agg['n_workloads']} hand-curated multi-turn coding workloads",
        f"- **Repetitions:** N={agg['n_reps']} per phase, fresh routing state per repetition",
        "- **Baseline:** always-Sonnet (`CREDENCE_FORCE_MODEL=claude-sonnet-4-6`)",
        "- **Secondary baseline:** always-Opus (repriced from Sonnet baseline token counts)",
        "- **Quality measurement:** regex pattern matching (binary) + Claude Opus 4.6 judge (0-10 scale)",
        f"- **Pricing snapshot:** {PRICING_DATE} (Haiku $1/$5, Sonnet $3/$15, Opus $5/$25 per 1M tokens)",
        "- **Regime tags:** cold-start (workloads 1-5), transition (6-10), warm-state (11-20)",
        "",
        "## Per-workload results",
        "",
        "| # | Workload | Type | Turns | Regime | Sonnet cost | Routing cost | Savings % | Quality (regex) | Opus score |",
        "|---|----------|------|-------|--------|-------------|--------------|-----------|-----------------|------------|",
    ]

    for w in analysis["per_workload"]:
        b_cost = f"${w['baseline_cost_mean']:.4f} +/- {w['baseline_cost_std']:.4f}"
        r_cost = f"${w['routing_cost_mean']:.4f} +/- {w['routing_cost_std']:.4f}"
        savings = f"{w['savings_pct_mean']:.1f}% +/- {w['savings_pct_std']:.1f}%"
        regex_q = f"{w['baseline_pass_rate']:.0%}/{w['routing_pass_rate']:.0%}"
        opus_b = f"{w['baseline_opus_score_mean']:.1f}" if w['baseline_opus_score_mean'] is not None else "-"
        opus_r = f"{w['routing_opus_score_mean']:.1f}" if w['routing_opus_score_mean'] is not None else "-"
        opus_q = f"{opus_b}/{opus_r}"
        degraded = " **DEGRADED**" if w["quality_degradations"] > 0 or w["opus_degradations"] > 0 else ""
        lines.append(
            f"| {w['position']} | {w['workload']} | {w['task_type']} | {w['n_turns']} "
            f"| {w['regime']} | {b_cost} | {r_cost} | {savings} | {regex_q}{degraded} | {opus_q} |"
        )

    lines += [
        "",
        "## Regime-tagged results",
        "",
        "| Regime | Workloads | Mean savings % | Weighted savings % | Quality degradations |",
        "|--------|-----------|----------------|--------------------|----------------------|",
    ]
    for tag in ("cold-start", "transition", "warm-state"):
        rs = analysis["regime_stats"].get(tag, {})
        if rs:
            lines.append(
                f"| {tag} | {rs['n_workloads']} "
                f"| {rs['mean_savings_pct']:.1f}% +/- {rs['savings_std']:.1f}% "
                f"| {rs['weighted_savings_pct']:.1f}% "
                f"| {rs['quality_degradations']} (regex) / {rs['opus_degradations']} (opus) |"
            )

    lines += [
        "",
        "## Task-type breakdown",
        "",
        "| Task type | N | Mean savings % |",
        "|-----------|---|----------------|",
    ]
    for tt, ts in sorted(analysis["task_type_stats"].items()):
        lines.append(f"| {tt} | {ts['n_workloads']} | {ts['mean_savings_pct']:.1f}% +/- {ts['savings_std']:.1f}% |")

    # Variance decomposition
    vd = analysis["variance_decomposition"]
    lines += [
        "",
        "## Variance decomposition",
        "",
        "Baseline repetitions isolate output variance (model stochasticity). "
        "Routing repetitions combine output variance with routing variance "
        "(cold-start learning, model-selection stochasticity).",
        "",
        f"- Mean output variance (from baseline): {vd['output_variance']:.8f}",
        f"- Mean total variance (from routing): {vd['total_variance']:.8f}",
        f"- Routing variance contribution: {vd['routing_variance_contribution']:.8f}",
        f"- Routing fraction of total variance: {vd['routing_fraction']:.1%}",
        "",
    ]

    # Routing distribution
    lines += [
        "## Routing distribution",
        "",
    ]
    for w in analysis["per_workload"]:
        dists = w.get("routing_model_distributions", [])
        if dists:
            merged: dict[str, int] = {}
            for d in dists:
                for m, c in d.items():
                    merged[m] = merged.get(m, 0) + c
            total = sum(merged.values())
            dist_str = ", ".join(
                f"{m}: {c}/{total} ({c/total:.0%})"
                for m, c in sorted(merged.items(), key=lambda x: -x[1])
            )
            lines.append(f"**{w['workload']}** ({w['regime']}): {dist_str}")

    # Belief snapshots
    if any(rep.get("belief_snapshots") for rep in routing_reps):
        lines += ["", "## Belief state snapshots", ""]
        for rep in routing_reps:
            if rep.get("belief_snapshots"):
                lines.append(f"### Repetition {rep['rep_id'] + 1}")
                for snap in rep["belief_snapshots"]:
                    lines.append(f"\nAfter workload {snap['after_workload']}:")
                    llm_state = snap["state"].get("llm", {})
                    if isinstance(llm_state, dict) and "error" not in llm_state:
                        for model, cats in llm_state.items():
                            if isinstance(cats, dict):
                                vals = " ".join(
                                    f"{k}={v:.3f}" for k, v in sorted(cats.items())
                                )
                                lines.append(f"  {model}: {vals}")
                lines.append("")

    # Always-Opus secondary baseline
    lines += [
        "## Secondary baseline: always-Opus",
        "",
        "| Workload | Opus cost | Routing cost | Savings vs Opus |",
        "|----------|-----------|--------------|-----------------|",
    ]
    for w in analysis["per_workload"]:
        lines.append(
            f"| {w['workload']} | ${w['opus_baseline_cost_mean']:.4f} "
            f"| ${w['routing_cost_mean']:.4f} | {w['opus_savings_pct']:.1f}% |"
        )

    # Aggregate
    lines += [
        "",
        "## Aggregate",
        "",
        f"- **Token-volume-weighted savings: {agg['weighted_savings_pct']:.1f}%**",
        f"- Unweighted mean savings: {agg['mean_savings_pct']:.1f}% +/- {agg['savings_std']:.1f}%",
        f"- Quality degradations (regex): {agg['quality_degradation_count']}/{agg['n_workloads'] * agg['n_reps']}",
        f"- Quality degradations (Opus judge >1pt): {agg['opus_degradation_count']}/{agg['n_workloads']}",
        "",
        "## Honest limitations",
        "",
        "- Cache savings not measured (Move 0 finding: prompt caching unsupported on OAI-compatible endpoint).",
        "- All 20 workloads are hand-curated, not production transcripts or SWE-bench traces.",
        f"- N={agg['n_reps']} repetitions; statistical power is limited.",
        "- Cold-start learning curve affects workloads 1-5 in routing phase.",
        "- Only Anthropic models tested (OpenAI excluded due to `max_tokens` / `max_completion_tokens` incompatibility).",
        "- Routing is based on keyword-category inference, not semantic understanding.",
        "- Opus judge truncates conversations to ~6000 chars; long conversations lose context.",
        "",
        "## Move 1 input absorbed",
        "",
        "- Move 0 audit: no cache measurement (OAI-compatible endpoint only).",
        "- Move 1 methodology: N=3, regime-tagged, variance decomposition, Opus judging.",
        "- Proxy cost table verified against official pricing (snapshot " + PRICING_DATE + ").",
    ]

    return "\n".join(lines) + "\n"


def main():
    workloads = load_workloads()
    if len(workloads) < 20:
        log.warning("Expected 20 workloads, found %d", len(workloads))
    log.info("Loaded %d workloads", len(workloads))

    _refresh_api_keys(os.environ)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Phase 1: Baseline repetitions (forced Sonnet)
    baseline_reps = []
    for rep_id in range(N_REPS):
        rep = run_repetition(rep_id, workloads, "claude-sonnet-4-6", "baseline")
        baseline_reps.append(rep)
        time.sleep(5)

    # Phase 2: Routing repetitions (fresh priors each time)
    routing_reps = []
    for rep_id in range(N_REPS):
        rep = run_repetition(rep_id, workloads, None, "routing")
        routing_reps.append(rep)
        time.sleep(5)

    # Analysis
    analysis = analyze_results(workloads, baseline_reps, routing_reps)

    # Report
    report = format_report(analysis, routing_reps)
    report_path = Path(CREDENCE_DIR) / "docs" / "posture-5" / "move-2-benchmark-results.md"
    report_path.write_text(report)
    log.info("Report written to %s", report_path)

    # Raw data
    raw_path = Path(CREDENCE_DIR) / "data" / "benchmark-raw.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_data = {
        "baseline_reps": [
            {
                "rep_id": rep["rep_id"],
                "enriched": rep.get("enriched", []),
                "metrics": rep["metrics"],
            }
            for rep in baseline_reps
        ],
        "routing_reps": [
            {
                "rep_id": rep["rep_id"],
                "enriched": rep.get("enriched", []),
                "metrics": rep["metrics"],
                "final_state": rep.get("final_state", {}),
                "belief_snapshots": rep.get("belief_snapshots", []),
            }
            for rep in routing_reps
        ],
        "analysis": analysis,
    }
    raw_path.write_text(json.dumps(raw_data, indent=2, default=str))
    log.info("Raw data saved to %s", raw_path)

    # Print summary
    agg = analysis["aggregate"]
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for w in analysis["per_workload"]:
        status = "+" if w["quality_degradations"] == 0 else "X"
        print(
            f"  {status} {w['workload']:35s} "
            f"${w['baseline_cost_mean']:.4f} -> ${w['routing_cost_mean']:.4f} "
            f"({w['savings_pct_mean']:+.1f}%) [{w['regime']}]"
        )
    print("-" * 60)
    print(f"  Weighted savings: {agg['weighted_savings_pct']:.1f}%")
    print(f"  Mean savings: {agg['mean_savings_pct']:.1f}% +/- {agg['savings_std']:.1f}%")
    print(f"  Quality degradations (regex): {agg['quality_degradation_count']}")
    print(f"  Quality degradations (opus): {agg['opus_degradation_count']}")
    for tag, rs in analysis["regime_stats"].items():
        print(f"  {tag}: {rs['weighted_savings_pct']:.1f}% savings")
    vd = analysis["variance_decomposition"]
    print(f"  Routing variance fraction: {vd['routing_fraction']:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
