#!/usr/bin/env python3
"""SWE-bench live A/B runner — credence-pi routing on REAL GitHub issues.

Authentic agentic tasks (vs the saturated Exercism toys): each instance is a real
SWE-bench Lite issue — a real repo at a real base commit with a real hidden test
(`FAIL_TO_PASS`) that the fix must make pass. The capability ladder is genuine here:
haiku fails issues that sonnet/opus resolve. This is real-world OpenClaw use (the
agent reads a real codebase, edits source, runs commands — the dominant `exec` loop).

Official SWE-bench grade protocol (the model never sees the hidden test):
  1. checkout base_commit, `pip install -e .` in a per-repo venv
  2. run OpenClaw on the issue text ONLY; it edits source
  3. reset test paths, apply the gold `test_patch` (adds the hidden tests)
  4. instance RESOLVED iff every FAIL_TO_PASS passes AND every PASS_TO_PASS still passes

Cost/turns harvested per turn from the session log, priced by the real model used —
identical accounting to run_ab.py, so SWE-bench and Exercism numbers are comparable.

Usage:
    python3 run_swebench.py --precheck --instances psf__requests-2674       # env sanity
    python3 run_swebench.py --arms haiku,sonnet --instances <ids> --timeout 600
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time, venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SWE = ROOT / "swebench"
REPOS = SWE / "repos"          # cached clones (one per repo)
VENVS = SWE / "venvs"          # cached venvs (one per repo)
INSTANCES = SWE / "instances_light.json"
RESULTS_DIR = ROOT / "results"
WS_LINK = Path("/tmp/credence-swebench/ws")   # the swebench agent workspace (symlink → active repo)
OPENCLAW = ["node", "/home/g/git/openclaw/openclaw.mjs"]
OPENCLAW_CWD = "/home/g/git/openclaw"
SESSIONS = Path.home() / ".openclaw/agents/swebench/sessions"

REPO_URL = {
    "psf/requests":      "https://github.com/psf/requests",
    "pallets/flask":     "https://github.com/pallets/flask",
    "pylint-dev/pylint": "https://github.com/pylint-dev/pylint",
    "pytest-dev/pytest": "https://github.com/pytest-dev/pytest",
    "sympy/sympy":       "https://github.com/sympy/sympy",
}

# Real published prices, USD per 1M tokens (input, output, cacheRead, cacheWrite).
PRICES = {
    "local":  dict(i=0.0,  o=0.0,  cr=0.0,  cw=0.0),
    "haiku":  dict(i=0.80, o=4.0,  cr=0.08, cw=1.0),
    "sonnet": dict(i=3.0,  o=15.0, cr=0.30, cw=3.75),
    "opus":   dict(i=15.0, o=75.0, cr=1.50, cw=18.75),
}
ARMS = {
    "local-9b": "ollama/qwen3.5:9b",
    "local-7b": "ollama/qwen2.5:7b-instruct",
    "haiku":    "anthropic/claude-haiku-4-5-20251001",
    "sonnet":   "anthropic/claude-sonnet-4-6",
    "opus":     "anthropic/claude-opus-4-8",
    "routing":  None,
}


def tier_of(model):
    if not model:
        return None
    m = model.lower()
    if "haiku" in m:  return "haiku"
    if "sonnet" in m: return "sonnet"
    if "opus" in m:   return "opus"
    if any(k in m for k in ("qwen", "llama", "gemma", "ollama")): return "local"
    return None


def load_key():
    out = subprocess.run(["secret-tool", "lookup", "service", "env", "key", "ANTHROPIC_API_KEY"],
                         capture_output=True, text=True)
    return out.stdout.strip()


def sh(cmd, cwd=None, env=None, timeout=None):
    return subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout, capture_output=True, text=True)


# Exact per-(repo,version) environment specs distilled from the swebench package
# (correct Python version + pinned deps — the reason naive `pip install -e .` fails PtP).
SPECS = json.load(open(SWE / "specs.json"))

def repo_dir(repo):  return REPOS / repo.replace("/", "__")
def _key(inst):      return f"{inst['repo'].replace('/', '__')}__{inst.get('version', 'x')}"
def venv_dir(inst):  return VENVS / _key(inst)
def venv_py(inst):   return str(venv_dir(inst) / "bin" / "python")
def spec_for(inst):  return SPECS.get(inst["repo"], {}).get(str(inst.get("version", "")), {})


def as_list(v):
    if isinstance(v, list): return v
    if isinstance(v, str):
        try: return json.loads(v)
        except Exception: return [v]
    return []


def ensure_clone(repo):
    rd = repo_dir(repo)
    if not (rd / ".git").exists():
        REPOS.mkdir(parents=True, exist_ok=True)
        r = sh(["git", "clone", "--quiet", REPO_URL[repo], str(rd)], timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"clone failed for {repo}: {r.stderr[-400:]}")
    return rd


# SWE-bench targets Python 3.8–3.11; the host (3.14) breaks old setuptools. uv pins 3.11.
SWE_PY = "3.11"

def ensure_venv(repo):
    vd = venv_dir(repo)
    if not (vd / "bin" / "python").exists():
        VENVS.mkdir(parents=True, exist_ok=True)
        r = sh(["uv", "venv", "--python", SWE_PY, str(vd)], timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"uv venv failed: {r.stderr[-300:]}")
    return vd


def clean_checkout(rd, base_commit):
    sh(["git", "reset", "--hard", "-q"], cwd=rd)
    sh(["git", "clean", "-fdq"], cwd=rd)
    r = sh(["git", "checkout", "-q", "-f", base_commit], cwd=rd)
    if r.returncode != 0:
        # base_commit may be unfetched on a shallow clone — fetch then retry
        sh(["git", "fetch", "--quiet", "origin", base_commit], cwd=rd, timeout=300)
        r = sh(["git", "checkout", "-q", "-f", base_commit], cwd=rd)
        if r.returncode != 0:
            raise RuntimeError(f"checkout {base_commit} failed: {r.stderr[-300:]}")


def pip_install(repo, rd):
    r = sh(["uv", "pip", "install", "--python", venv_py(repo), "-q", "-e", "."], cwd=rd, timeout=900)
    sh(["uv", "pip", "install", "--python", venv_py(repo), "-q", "pytest"], cwd=rd, timeout=300)
    return r.returncode, (r.stderr[-800:] if r.returncode else "")


def setup_instance(inst):
    repo = inst["repo"]
    rd = ensure_clone(repo); ensure_venv(repo)
    clean_checkout(rd, inst["base_commit"])
    rc, err = pip_install(repo, rd)
    if rc != 0:
        raise RuntimeError(f"pip install -e . failed for {inst['instance_id']}: {err}")
    return rd


def reset_test_paths(rd, test_patch):
    """Discard any agent edits to files the gold test_patch touches, so it applies cleanly."""
    for ln in test_patch.splitlines():
        if ln.startswith("diff --git "):
            # 'diff --git a/x b/x'
            parts = ln.split()
            if len(parts) >= 4:
                path = parts[2][2:]  # strip 'a/'
                sh(["git", "checkout", "-q", "--", path], cwd=rd)


def apply_test_patch(rd, test_patch):
    p = rd / "_credence_testpatch.diff"
    p.write_text(test_patch)
    r = sh(["git", "apply", "--whitespace=nowarn", str(p)], cwd=rd)
    if r.returncode != 0:
        r = sh(["git", "apply", "--whitespace=nowarn", "-3", str(p)], cwd=rd)  # 3-way fallback
    p.unlink(missing_ok=True)
    return r.returncode == 0, r.stderr[-300:]


def run_tests(repo, rd, tests, timeout=600):
    if not tests:
        return True, ""
    r = sh([venv_py(repo), "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", *tests],
           cwd=rd, timeout=timeout)
    return r.returncode == 0, (r.stdout[-1500:] + r.stderr[-500:])


def grade(inst, rd):
    """Official protocol: reset test paths, apply gold test_patch, FAIL_TO_PASS must pass
    AND PASS_TO_PASS must still pass."""
    reset_test_paths(rd, inst["test_patch"])
    ok, err = apply_test_patch(rd, inst["test_patch"])
    if not ok:
        return False, f"test_patch did not apply: {err}"
    ftp_ok, ftp_log = run_tests(inst["repo"], rd, as_list(inst["FAIL_TO_PASS"]))
    if not ftp_ok:
        return False, "FAIL_TO_PASS not satisfied"
    ptp_ok, _ = run_tests(inst["repo"], rd, as_list(inst["PASS_TO_PASS"]))
    if not ptp_ok:
        return False, "PASS_TO_PASS regressed"
    return True, "resolved"


def point_workspace(rd):
    WS_LINK.parent.mkdir(parents=True, exist_ok=True)
    if WS_LINK.is_symlink():
        WS_LINK.unlink()
    elif WS_LINK.exists():
        shutil.rmtree(WS_LINK)
    WS_LINK.symlink_to(rd)


def harvest_cost(session_id):
    f = SESSIONS / f"{session_id}.jsonl"
    usd = 0.0
    tot = dict(turns=0, input=0, output=0, cacheRead=0, cacheWrite=0)
    models_used = {}
    if f.exists():
        for ln in f.read_text().splitlines():
            try: r = json.loads(ln)
            except Exception: continue
            msg = r.get("message") or {}
            u = msg.get("usage") or r.get("usage")
            if not isinstance(u, dict): continue
            model = msg.get("model") or r.get("model") or ""
            tier = tier_of(model)
            ti, to = u.get("input", 0) or 0, u.get("output", 0) or 0
            cr, cw = u.get("cacheRead", 0) or 0, u.get("cacheWrite", 0) or 0
            tot["turns"] += 1; tot["input"] += ti; tot["output"] += to
            tot["cacheRead"] += cr; tot["cacheWrite"] += cw
            if tier:
                p = PRICES[tier]
                usd += (ti*p["i"] + to*p["o"] + cr*p["cr"] + cw*p["cw"]) / 1e6
                models_used[model] = models_used.get(model, 0) + 1
    tot["usd"] = round(usd, 5)
    tot["models_used"] = models_used
    return tot


PROMPT_TMPL = """Resolve this GitHub issue in the current repository ({repo}).

Read the relevant source, make the necessary code changes to fix it, and verify your
fix. Edit source files only — do NOT modify or add test files. When done, the project's
existing test suite for this area should pass.

--- ISSUE ---
{problem}
"""


def run_one(inst, arm, env, timeout):
    rd = setup_instance(inst)
    point_workspace(rd)
    prompt = PROMPT_TMPL.format(repo=inst["repo"], problem=inst["problem_statement"])
    sid = f"{inst['instance_id']}__{arm}"
    model = ARMS[arm]
    cmd = OPENCLAW + ["agent", "--agent", "swebench", "--session-id", sid,
                      "--message", prompt, "--json", "--timeout", str(timeout - 20)]
    if model:
        cmd += ["--model", model]
    # The agent's own python/pytest must be the instance venv (3.11), not host 3.14.
    aenv = dict(env)
    vbin = str(venv_dir(inst["repo"]) / "bin")
    aenv["PATH"] = vbin + os.pathsep + aenv.get("PATH", "")
    aenv["VIRTUAL_ENV"] = str(venv_dir(inst["repo"]))
    t0 = time.time()
    try:
        subprocess.run(cmd, cwd=OPENCLAW_CWD, env=aenv, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass
    dur = round(time.time() - t0, 1)
    resolved, detail = grade(inst, rd)
    cost = harvest_cost(sid)
    # cleanup for the next arm/instance
    sh(["git", "reset", "--hard", "-q"], cwd=rd); sh(["git", "clean", "-fdq"], cwd=rd)
    row = dict(instance=inst["instance_id"], repo=inst["repo"], arm=arm,
               success=resolved, detail=detail, duration_s=dur, **cost)
    print(f"  [{arm:>8}] {inst['instance_id']:<28} "
          f"{'RESOLVED' if resolved else 'failed  '}  ${cost['usd']:<8} "
          f"{cost['turns']}t {dur}s  {detail if not resolved else ''}")
    return row


def summarize(rows):
    arms = sorted({r["arm"] for r in rows})
    summ = {}
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        n = len(rs); passes = sum(1 for r in rs if r["success"])
        total = sum(r["usd"] for r in rs)
        summ[a] = dict(n=n, resolved=passes,
                       resolve_rate=round(passes/n, 3) if n else None,
                       total_usd=round(total, 4),
                       cost_per_resolved=round(total/passes, 5) if passes else None)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="", help="comma list of instance_ids; default=all light")
    ap.add_argument("--repos", default="", help="comma list of repos to filter (e.g. psf/requests)")
    ap.add_argument("--arms", default="haiku,sonnet,opus")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--precheck", action="store_true", help="env sanity only: FAIL_TO_PASS must FAIL at base")
    ap.add_argument("--out", default=str(RESULTS_DIR / "swebench.jsonl"))
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    allinst = json.load(open(INSTANCES))
    by_id = {i["instance_id"]: i for i in allinst}
    if args.instances:
        sel = [by_id[x] for x in args.instances.split(",") if x in by_id]
    elif args.repos:
        repos = set(args.repos.split(","))
        sel = [i for i in allinst if i["repo"] in repos]
    else:
        sel = allinst
    if not sel:
        sys.exit("no instances selected")

    if args.precheck:
        for inst in sel:
            try:
                rd = setup_instance(inst)
                reset_test_paths(rd, inst["test_patch"])
                ok, err = apply_test_patch(rd, inst["test_patch"])
                if not ok:
                    print(f"  {inst['instance_id']:<28} PATCH-FAIL  {err}"); continue
                ftp_ok, _ = run_tests(inst["repo"], rd, as_list(inst["FAIL_TO_PASS"]))
                ptp_ok, _ = run_tests(inst["repo"], rd, as_list(inst["PASS_TO_PASS"]))
                verdict = "OK (issue real)" if (not ftp_ok and ptp_ok) else \
                          ("WARN ftp-already-passes" if ftp_ok else "WARN ptp-fails-at-base")
                print(f"  {inst['instance_id']:<28} FtP_pass={ftp_ok} PtP_pass={ptp_ok}  {verdict}")
                sh(["git", "reset", "--hard", "-q"], cwd=rd); sh(["git", "clean", "-fdq"], cwd=rd)
            except Exception as e:
                print(f"  {inst['instance_id']:<28} SETUP-ERROR  {str(e)[:200]}")
        return

    env = dict(os.environ)
    key = load_key()
    if not key:
        sys.exit("ANTHROPIC_API_KEY not found in keyring")
    env["ANTHROPIC_API_KEY"] = key

    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS: sys.exit(f"unknown arm {a}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"instances={[i['instance_id'] for i in sel]} arms={arms}")
    for inst in sel:
        for arm in arms:
            rows.append(run_one(inst, arm, env, args.timeout))
    with open(args.out, "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    print("\n=== SUMMARY ===")
    print(json.dumps(summarize(rows), indent=2))
    print(f"\nrows -> {args.out}")


if __name__ == "__main__":
    main()
