#!/usr/bin/env python3
# Role: tooling
"""credence-lint — enforce the single-reasoner invariant outside src/.

Pass one (this file): grep-based heuristic. Flags direct, same-line
arithmetic/comparison on the return values of DSL functions (expect,
weights, mean, density, …) in any file that carries a `# Role:` header.

Escape-hatch pragma (same line as the violation):

    # credence-lint: allow — precedent:<slug> — <one-line reason>

Both slug and reason are mandatory. Slugs are extracted from CLAUDE.md's
Precedents section — unknown slugs fail the lint. Em-dashes (—) and
plain double-hyphens (--) are both accepted as separators.

Indirection through an intermediate binding (r = expect(...); r * 2) is
out of scope for pass one and will be caught by pass two (AST-based).
Files without a `# Role:` header under apps/ are flagged (the role
header is the lint's dispatch key).

Usage:
    credence_lint.py check PATH [PATH ...]   # lint files / dirs
    credence_lint.py test                    # run against corpus/
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ── DSL return-value functions ────────────────────────────────────────
# Calling these returns a probability, utility, density, or posterior
# accessor value. Arithmetic on the return is the canonical violation.
DSL_FNS: tuple[str, ...] = (
    "expect",
    "density", "log_density_at", "log_predictive", "log_marginal",
    "push_measure",
    "weights", "mean", "variance",
    "voi", "value",
)

# Aggregators that, applied to a DSL return, perform probability arithmetic.
AGGREGATORS: tuple[str, ...] = (
    "sum", "prod", "max", "min",
    "np.sum", "np.prod", "np.mean",
    "logsumexp",
)

# Binary operators that make a DSL return causal. Deliberately excluding
# single `=` (assignment) — that's read, not compute. The `-(?!>)` guard
# keeps `->` in return-type annotations from matching as subtraction.
_OP = r"(?:==|!=|<=|>=|\+|-(?!>)|\*\*|\*|/|%|<|>)"
_DSL = "|".join(DSL_FNS)
_AGG = "|".join(re.escape(a) for a in AGGREGATORS)

# Violation patterns (same-line only; indirection is pass two).
#
# `(?<![\w.])` guards against arbitrary module qualifiers — `Dates.value`,
# `np.mean`, `math.prod` are Julia/Python stdlib that happen to share a
# name with a DSL fn and are not the target. Allowed prefixes are
# `self.`, `skin.`, `brain.` — common bindings for objects that hold
# real Measures. Object-style qualifiers like `m.mean()` (where `m` is
# a Measure variable) are missed here and left to pass two.
_PRE = r"(?<![\w.])(?:(?:self|skin|brain)\.)?"
_CALL = rf"{_PRE}\b(?:{_DSL})\s*\([^)]*\)(?:\[[^\]]*\])?"
PATTERN_CALL_LEADS = re.compile(rf"{_CALL}\s*{_OP}")
PATTERN_CALL_TRAILS = re.compile(rf"{_OP}\s*{_PRE}\b(?:{_DSL})\s*\(")
PATTERN_AGG_WRAP = re.compile(rf"(?<![\w.])\b(?:{_AGG})\s*\(\s*{_PRE}\b(?:{_DSL})\s*\(")
PATTERN_SORT_DSL = re.compile(rf"(?<![\w.])\b(?:sort|sort!|sorted)\s*\([^)]*{_PRE}\b(?:{_DSL})\s*\(")

# The pragma. Accept em-dash, hyphen-hyphen, or a single hyphen as separator.
_SEP = r"\s*(?:—|--|-)\s*"
PRAGMA_RE = re.compile(
    rf"#\s*credence-lint:\s*allow"
    rf"(?:{_SEP}precedent:(\S+))?"
    rf"(?:{_SEP}(.+?))?\s*$",
    re.MULTILINE,
)
PRAGMA_PRESENT = re.compile(r"#\s*credence-lint:\s*allow")

# Slug extraction from CLAUDE.md.
SLUG_RE = re.compile(r"^\*\*Slug:\*\*\s*`([^`]+)`")

# Role header.
ROLE_RE = re.compile(r"^#\s*Role:\s*(.+?)\s*$")

LANG_SUFFIXES = {".py", ".jl"}
SKIP_DIR_PARTS = {"__pycache__", ".venv", ".git", "node_modules", ".pytest_cache"}

# Declaration prefixes — function/class signatures look like DSL calls but
# aren't callsites. Match leading whitespace then keyword.
DECL_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class|function|macro|struct|abstract\s+type|mutable\s+struct)\s+")

# Triple-quoted string boundaries (Python docstrings, Julia """..."""
# multi-line docstrings). Minimal tracker: count """ and ''' occurrences
# on each line; odd count toggles "in-docstring" state.
TRIPLE_RE = re.compile(r'"""|\'\'\'')


@dataclass
class Violation:
    path: Path
    line_no: int
    line: str
    message: str


# ── CLAUDE.md slug table ──────────────────────────────────────────────
def extract_slugs(claude_md: Path) -> set[str]:
    if not claude_md.is_file():
        return set()
    return {
        m.group(1)
        for line in claude_md.read_text().splitlines()
        if (m := SLUG_RE.match(line))
    }


# ── role header parsing ───────────────────────────────────────────────
def read_role(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            head = [f.readline() for _ in range(5)]
    except OSError:
        return None
    for line in head:
        if not line:
            break
        m = ROLE_RE.match(line.strip())
        if m:
            return m.group(1).strip()
    return None


# ── pragma parsing ────────────────────────────────────────────────────
def check_pragma(line: str, valid_slugs: set[str]) -> tuple[bool, str | None]:
    """Return (ok, error). ok=True means 'pragma permits this violation'."""
    if not PRAGMA_PRESENT.search(line):
        return False, None  # no pragma — caller reports the original violation
    m = PRAGMA_RE.search(line)
    if not m or m.group(1) is None:
        return False, "malformed pragma: expected `# credence-lint: allow — precedent:<slug> — <reason>`"
    slug = m.group(1).strip()
    reason = (m.group(2) or "").strip()
    if not reason:
        return False, f"pragma missing reason (slug `{slug}`)"
    if slug not in valid_slugs:
        return False, f"pragma references unknown precedent slug `{slug}`"
    return True, None


# ── per-file lint ─────────────────────────────────────────────────────
def _violates(line: str) -> str | None:
    if PATTERN_AGG_WRAP.search(line):
        return "aggregator applied to DSL return"
    if PATTERN_SORT_DSL.search(line):
        return "sort over DSL return"
    if PATTERN_CALL_LEADS.search(line):
        return "arithmetic/comparison on DSL return"
    if PATTERN_CALL_TRAILS.search(line):
        return "DSL return as operand of arithmetic/comparison"
    return None


def check_file(path: Path, valid_slugs: set[str], require_role: bool) -> list[Violation]:
    violations: list[Violation] = []
    role = read_role(path)
    if require_role and role is None:
        violations.append(Violation(path, 1, "", "missing `# Role:` header"))
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return violations

    in_docstring = False
    for i, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        # Triple-quoted string tracker: flip state on odd triple-quote count.
        triples = len(TRIPLE_RE.findall(line))
        was_in_docstring = in_docstring
        if triples % 2 == 1:
            in_docstring = not in_docstring
        # Skip lines that are entirely inside a docstring (entered before,
        # still in after) — DSL names in prose are not callsites.
        if was_in_docstring and in_docstring:
            continue
        if stripped.startswith("#") or stripped.startswith(";"):
            continue  # comment-only
        if DECL_RE.match(line):
            continue  # function/class signature, not a callsite
        kind = _violates(line)
        if kind is None:
            # Still need to validate a pragma if one is present but no
            # violation — a dangling allow is either malformed or confusing.
            if PRAGMA_PRESENT.search(line):
                ok, err = check_pragma(line, valid_slugs)
                if not ok and err is not None:
                    violations.append(Violation(path, i, line.rstrip(), err))
            continue
        ok, err = check_pragma(line, valid_slugs)
        if ok:
            continue
        if err is not None:
            violations.append(Violation(path, i, line.rstrip(), err))
        else:
            violations.append(Violation(
                path, i, line.rstrip(),
                f"{kind} — declare a Functional and call expect(), or add a"
                f" `# credence-lint: allow — precedent:<slug> — <reason>` pragma",
            ))
    return violations


# ── file walking ──────────────────────────────────────────────────────
def walk(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_file():
            if p.suffix in LANG_SUFFIXES:
                out.append(p)
            continue
        if not p.is_dir():
            continue
        for sub in p.rglob("*"):
            if not sub.is_file():
                continue
            if sub.suffix not in LANG_SUFFIXES:
                continue
            if any(part in SKIP_DIR_PARTS for part in sub.parts):
                continue
            out.append(sub)
    return sorted(out)


# ── commands ──────────────────────────────────────────────────────────
def _repo_root(arg: str | None) -> Path:
    return Path(arg).resolve() if arg else Path.cwd().resolve()


def cmd_check(args: argparse.Namespace) -> int:
    root = _repo_root(args.repo_root)
    slugs = extract_slugs(root / "CLAUDE.md")
    files = walk([Path(p).resolve() for p in args.paths])
    viols: list[Violation] = []
    for f in files:
        # Files under apps/ are required to carry a role header (#3).
        require_role = "apps" in f.relative_to(root).parts if f.is_relative_to(root) else False
        viols.extend(check_file(f, slugs, require_role))
    for v in viols:
        try:
            rel = v.path.relative_to(root)
        except ValueError:
            rel = v.path
        print(f"{rel}:{v.line_no}: {v.message}")
        if v.line:
            print(f"    {v.line.strip()}")
    if viols:
        print(f"\n{len(viols)} violation(s) across {len(files)} file(s)", file=sys.stderr)
        return 1
    print(f"OK: {len(files)} files, 0 violations")
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Run against the corpus; good_* must be clean, bad_* must be flagged."""
    root = _repo_root(args.repo_root)
    corpus = root / "tools" / "credence-lint" / "corpus"
    if not corpus.is_dir():
        print(f"corpus not found at {corpus}", file=sys.stderr)
        return 2
    slugs = extract_slugs(root / "CLAUDE.md")
    files = walk([corpus])
    failures: list[str] = []
    n_good = n_bad1 = n_bad2 = 0
    for f in files:
        # Corpus files don't live under apps/, so role header is
        # informational but not required by the lint's path rule.
        viols = check_file(f, slugs, require_role=False)
        stem = f.stem
        if stem.startswith("good_"):
            n_good += 1
            if viols:
                failures.append(
                    f"FAIL good: {f.relative_to(root)} flagged ({len(viols)} violation(s))"
                )
                for v in viols:
                    failures.append(f"    L{v.line_no}: {v.message}")
        elif stem.startswith("bad_"):
            n_bad1 += 1
            if not viols:
                failures.append(
                    f"FAIL bad: {f.relative_to(root)} not flagged — pass one should catch"
                )
        elif stem.startswith("bad2_"):
            n_bad2 += 1
            # Pass two only — informational. Pass one is allowed to miss.
            status = "caught by pass one" if viols else "deferred to pass two"
            print(f"  (bad2 {status}): {f.relative_to(root)}")
    for msg in failures:
        print(msg)
    print(
        f"\ncorpus: {n_good} good / {n_bad1} bad (pass one) / {n_bad2} bad (pass two only)"
    )
    if failures:
        print(f"{len(failures)} corpus classification failure(s)", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="credence-lint")
    ap.add_argument("--repo-root", default=None, help="repo root (defaults to cwd)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("check", help="lint files and/or directories")
    pc.add_argument("paths", nargs="+")
    pc.set_defaults(func=cmd_check)

    pt = sub.add_parser("test", help="run the corpus self-test")
    pt.set_defaults(func=cmd_test)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
