#!/usr/bin/env python3
"""Fetch Exercism Python exercises (MIT-licensed) and convert to live_ab task format.

Each produced task: tasks/<slug>/{prompt.txt, seed/<module>.py (stub),
seed/<module>_test.py, verify.sh, TIER}. The agent gets prompt.txt (the exercise
instructions) + the stub + the test in its workspace, and must make pytest pass.

Every task is PRE-VALIDATED with the exercise's reference solution (.meta/example.py)
against its own test on THIS Python, so we never ship a task that is unsolvable here.
Source: github.com/exercism/python (MIT). The reference solution is NEVER placed in
the agent workspace — it is used only to validate solvability.
"""
from __future__ import annotations
import subprocess, tempfile, urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/exercism/python/main/exercises/practice"
TASKS = Path(__file__).resolve().parent / "tasks"

# (slug, tier) — tiers per Exercism difficulty + Aider polyglot membership.
# easy=floor (all pass), medium=Haiku borderline, hard/stretch=expected crossover band.
SPEC = [
    ("leap", "easy"), ("two-fer", "easy"), ("anagram", "easy"),
    ("change", "medium"), ("wordy", "medium"), ("circular-buffer", "medium"),
    ("forth", "hard"), ("book-store", "hard"), ("zebra-puzzle", "hard"), ("react", "hard"),
    ("pov", "stretch"), ("sgf-parsing", "stretch"),
    # stretch2: the fiddliest remaining polyglot slugs (most edge-cases / subtle state),
    # to probe whether ANY standard coding task separates Haiku 4.5 from Sonnet 4.6.
    ("poker", "stretch2"), ("zipper", "stretch2"), ("rest-api", "stretch2"),
    ("dominoes", "stretch2"), ("connect", "stretch2"), ("two-bucket", "stretch2"),
]


def mod(slug: str) -> str:
    return slug.replace("-", "_")


def get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode()


def fetch(slug: str, tier: str) -> str | None:
    m = mod(slug)
    try:
        stub = get(f"{RAW}/{slug}/{m}.py")
        test = get(f"{RAW}/{slug}/{m}_test.py")
        instr = get(f"{RAW}/{slug}/.docs/instructions.md")
        ref = get(f"{RAW}/{slug}/.meta/example.py")
    except Exception as e:
        print(f"  SKIP {slug}: fetch failed ({e})")
        return None
    # Pre-validate: the reference solution must pass the test on this Python.
    with tempfile.TemporaryDirectory() as d:
        Path(d, f"{m}.py").write_text(ref)
        Path(d, f"{m}_test.py").write_text(test)
        rc = subprocess.run(["python3", "-m", "pytest", "-q", f"{m}_test.py"],
                            cwd=d, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode
    if rc != 0:
        print(f"  SKIP {slug}: reference solution does not pass here (rc={rc})")
        return None
    td = TASKS / slug
    (td / "seed").mkdir(parents=True, exist_ok=True)
    (td / "seed" / f"{m}.py").write_text(stub)
    (td / "seed" / f"{m}_test.py").write_text(test)
    (td / "prompt.txt").write_text(
        instr.strip()
        + f"\n\nImplement your solution in {m}.py so that all tests in "
        f"{m}_test.py pass. After editing, run `python3 -m pytest -q` to confirm.\n")
    (td / "verify.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# Deterministic oracle (Exercism " + slug + ", MIT). Exit 0 = tests pass.\n"
        "set -u\nWS=\"${1:?usage: verify.sh <workspace_dir>}\"\n"
        "cd \"$WS\" || exit 2\n"
        f"python3 -m pytest -q {m}_test.py >/dev/null 2>&1\n")
    (td / "verify.sh").chmod(0o755)
    (td / "TIER").write_text(tier + "\n")
    print(f"  OK   {slug:<16} [{tier}]  (reference validated)")
    return slug


def main() -> None:
    TASKS.mkdir(exist_ok=True)
    ok = [s for s, t in SPEC if fetch(s, t)]
    print(f"\nfetched {len(ok)}/{len(SPEC)} -> {TASKS}")
    print("tiers:", {t: [s for s, tt in SPEC if tt == t and s in ok]
                     for t in ("easy", "medium", "hard", "stretch")})


if __name__ == "__main__":
    main()
