#!/usr/bin/env python3
# Role: tooling
"""credence-lint — enforce the single-reasoner invariant outside src/.

Pass one: grep-based heuristic. Flags direct, same-line arithmetic or
comparison on the return values of DSL functions (expect, weights,
mean, density, …) in any file that carries a `# Role:` header.

Pass two: taint analysis. Catches indirection through variable
bindings (`r = expect(…); r * 2`) that pass one cannot see from a
single line. Python files are parsed with the stdlib `ast` module;
Julia has no Python-accessible AST, so `.jl` files run through a
stateful line scanner that reuses the same DSL-call regex and pragma
machinery. Both share the seed rule (DSL call → tainted return) and
the flag rule (arithmetic / comparison-to-branch / aggregator / sort
on a tainted value is a violation; reads, argument passing, and
string formatting are not). Taint does not chase through opaque
function calls.

Escape-hatch pragma:

    # credence-lint: allow — precedent:<slug> — <one-line reason>

Accepted on the same line as the violation or on the immediately
preceding comment-only line. Both slug and reason are mandatory.
Slugs are extracted from CLAUDE.md's Precedents section — unknown
slugs fail the lint. Em-dashes (—) and plain double-hyphens (--) are
both accepted as separators.

Files without a `# Role:` header under apps/ are flagged (the role
header is the lint's dispatch key).

Usage:
    credence_lint.py check PATH [PATH ...]   # lint files / dirs
    credence_lint.py test                    # run against corpus/
"""
from __future__ import annotations

import argparse
import ast
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
# Per issue #5 ("out of scope: Extending to CI for apps/julia/pomdp_agent/
# — that package has its own src/ and its own invariants, evaluate
# separately"), skip the pomdp_agent subtree from apps/ scans.
SKIP_REL_DIRS = (("apps", "julia", "pomdp_agent"),)

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


def _pragma_allows(lines: list[str], lineno: int,
                   valid_slugs: set[str]) -> tuple[bool, str | None]:
    """Does a pragma on this line — or the immediately preceding comment-
    only line — permit a violation here?

    Returns (ok, err). ok=True means 'pragma permits'. err non-None means
    a pragma was found but is malformed (missing slug/reason, unknown
    slug) and should itself be reported.
    """
    cur = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
    ok, err = check_pragma(cur, valid_slugs)
    if ok:
        return True, None
    if err is not None:
        return False, err
    if lineno >= 2:
        prev = lines[lineno - 2]
        stripped = prev.strip()
        if stripped.startswith("#") or stripped.startswith(";"):
            ok, err = check_pragma(prev, valid_slugs)
            if ok:
                return True, None
            if err is not None:
                return False, err
    return False, None


# ── pass two: taint analysis ──────────────────────────────────────────
#
# A value is TAINTED if it is the return of a DSL function (optionally
# through `self./skin./brain.`) or derives from one through assignment,
# tuple unpacking, `for`-loop targets, or a small set of identity-like
# passthroughs (list, tuple, enumerate, reversed, sorted, collect, iter,
# zip). A VIOLATION is arithmetic, comparison-in-branch, aggregator
# application, or sort applied to a tainted value. Reads (argument
# passing, string formatting, return, attribute access) do not flag.
#
# Taint does not chase through opaque function calls. If `y = f(tainted)`
# with `f` unknown, `y` is not tainted — the author is responsible for
# pragma-tagging or fixing inside `f`. This bounds the lint's ambition.

_DSL_NAMES = set(DSL_FNS)
_PREFIX_NAMES = {"self", "skin", "brain"}
_PASSTHROUGH_NAMES = {
    "list", "tuple", "enumerate", "reversed", "sorted",
    "iter", "zip", "collect",
}
# Stdlib aggregators. Unlike DSL fns, calling these on a tainted arg is
# itself the violation — they are not seeds.
_AGG_NAMES = {"sum", "prod", "max", "min", "logsumexp"}
# numpy-style aggregators reachable via `np.*`.
_NP_AGG_ATTRS = {"sum", "prod", "mean"}


# Python AST helpers ────────────────────────────────────────────────────

def _py_is_dsl_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Name) and f.id in _DSL_NAMES:
        return True
    if (isinstance(f, ast.Attribute) and f.attr in _DSL_NAMES
            and isinstance(f.value, ast.Name) and f.value.id in _PREFIX_NAMES):
        return True
    return False


def _py_expr_tainted(node: ast.AST, tainted: set[str]) -> bool:
    if isinstance(node, ast.Call):
        if _py_is_dsl_call(node):
            return True
        f = node.func
        if isinstance(f, ast.Name) and f.id in _PASSTHROUGH_NAMES:
            return any(_py_expr_tainted(a, tainted) for a in node.args)
        return False
    if isinstance(node, ast.Name):
        return node.id in tainted
    if isinstance(node, ast.Subscript):
        return _py_expr_tainted(node.value, tainted)
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_py_expr_tainted(e, tainted) for e in node.elts)
    if isinstance(node, ast.Starred):
        return _py_expr_tainted(node.value, tainted)
    return False


def _py_bind_targets(target: ast.AST, rhs_tainted: bool,
                     tainted: set[str]) -> None:
    if isinstance(target, ast.Name):
        (tainted.add if rhs_tainted else tainted.discard)(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for t in target.elts:
            _py_bind_targets(t, rhs_tainted, tainted)
    elif isinstance(target, ast.Starred):
        _py_bind_targets(target.value, rhs_tainted, tainted)
    # Subscript/Attribute targets don't rebind a simple name — leave alone.


def _py_bind_for(target: ast.AST, iter_node: ast.AST,
                 tainted: set[str]) -> None:
    """Precise for `for a, b in zip(x, y)`: each target inherits taint from
    its positionally-matched zip arg. Otherwise taint all targets iff the
    iterator is tainted."""
    if (isinstance(iter_node, ast.Call)
            and isinstance(iter_node.func, ast.Name)
            and iter_node.func.id == "zip"
            and isinstance(target, (ast.Tuple, ast.List))
            and len(target.elts) == len(iter_node.args)):
        for t, src in zip(target.elts, iter_node.args):
            _py_bind_targets(t, _py_expr_tainted(src, tainted), tainted)
        return
    _py_bind_targets(target, _py_expr_tainted(iter_node, tainted), tainted)


class _TaintVisitor(ast.NodeVisitor):
    def __init__(self, lines: list[str], valid_slugs: set[str],
                 path: Path) -> None:
        self.lines = lines
        self.valid_slugs = valid_slugs
        self.path = path
        self.violations: list[Violation] = []
        self.tainted_stack: list[set[str]] = [set()]

    @property
    def tainted(self) -> set[str]:
        return self.tainted_stack[-1]

    def _flag(self, lineno: int, kind: str) -> None:
        if not (0 < lineno <= len(self.lines)):
            return
        line = self.lines[lineno - 1]
        ok, err = _pragma_allows(self.lines, lineno, self.valid_slugs)
        if ok:
            return
        if err is not None:
            self.violations.append(Violation(self.path, lineno, line.rstrip(), err))
        else:
            self.violations.append(Violation(
                self.path, lineno, line.rstrip(),
                f"{kind} (pass two: indirection through a prior binding)"
                f" — declare a Functional and call expect(), or add a"
                f" `# credence-lint: allow — precedent:<slug> — <reason>` pragma",
            ))

    def _enter_scope(self) -> None:
        self.tainted_stack.append(set())

    def _exit_scope(self) -> None:
        self.tainted_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_scope()
        self.generic_visit(node)
        self._exit_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_scope()
        self.generic_visit(node)
        self._exit_scope()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._enter_scope()
        self.generic_visit(node)
        self._exit_scope()

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        rhs_tainted = _py_expr_tainted(node.value, self.tainted)
        for t in node.targets:
            _py_bind_targets(t, rhs_tainted, self.tainted)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            rhs_tainted = _py_expr_tainted(node.value, self.tainted)
            _py_bind_targets(node.target, rhs_tainted, self.tainted)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        target_tainted = (isinstance(node.target, ast.Name)
                          and node.target.id in self.tainted)
        rhs_tainted = _py_expr_tainted(node.value, self.tainted)
        if target_tainted or rhs_tainted:
            self._flag(node.lineno, "augmented arithmetic on DSL return")
        if isinstance(node.target, ast.Name) and rhs_tainted:
            self.tainted.add(node.target.id)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        _py_bind_for(node.target, node.iter, self.tainted)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)  # type: ignore[arg-type]

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if (_py_expr_tainted(node.left, self.tainted)
                or _py_expr_tainted(node.right, self.tainted)):
            self._flag(node.lineno, "arithmetic on DSL return")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        operands: list[ast.AST] = [node.left, *node.comparators]
        if any(_py_expr_tainted(o, self.tainted) for o in operands):
            self._flag(node.lineno, "comparison-to-branch on DSL return")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        if _py_is_dsl_call(node):
            return  # seed
        f = node.func
        if isinstance(f, ast.Name) and f.id in _AGG_NAMES:
            if any(_py_expr_tainted(a, self.tainted) for a in node.args):
                self._flag(node.lineno, "aggregator on DSL return")
                return
        if (isinstance(f, ast.Attribute) and f.attr in _NP_AGG_ATTRS
                and isinstance(f.value, ast.Name) and f.value.id == "np"):
            if any(_py_expr_tainted(a, self.tainted) for a in node.args):
                self._flag(node.lineno, "aggregator on DSL return")
                return
        if isinstance(f, ast.Name) and f.id in {"sort", "sorted"}:
            if any(_py_expr_tainted(a, self.tainted) for a in node.args):
                self._flag(node.lineno, "sort on DSL return")
                return
        if (isinstance(f, ast.Attribute) and f.attr == "sort"
                and _py_expr_tainted(f.value, self.tainted)):
            self._flag(node.lineno, "sort on DSL return")


def check_file_ast_py(path: Path, valid_slugs: set[str]) -> list[Violation]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []  # let the type checker / CI complain about broken syntax
    visitor = _TaintVisitor(source.splitlines(), valid_slugs, path)
    visitor.visit(tree)
    return visitor.violations


# Julia scanner ─────────────────────────────────────────────────────────
# Julia has no stdlib-accessible AST from Python. We run a stateful line
# scanner that tracks a per-function set of tainted names and flags the
# same kinds of violations the Python AST visitor does. Less precise,
# but adequate for the narrow indirection patterns `bad2_*.jl` exercises.

_JL_IDENT = r"[A-Za-z_][A-Za-z0-9_!]*"
_JL_OPENER_RE = re.compile(
    r"^\s*(function|for|while|if|begin|let|struct|mutable\s+struct"
    r"|macro|module|quote|try)\b"
)
_JL_END_RE = re.compile(r"^\s*end\b")
_JL_ASSIGN_RE = re.compile(
    rf"^\s*(?P<tgt>{_JL_IDENT})(?:\s*::\s*[^\n=]+?)?\s*=(?!=)\s*(?P<rhs>.+?)\s*$"
)
_JL_TUPLE_ASSIGN_RE = re.compile(
    rf"^\s*\(?\s*(?P<tgts>{_JL_IDENT}(?:\s*,\s*{_JL_IDENT})+)\s*\)?\s*=(?!=)\s*(?P<rhs>.+?)\s*$"
)
_JL_FOR_RE = re.compile(
    rf"^\s*for\s+(?P<tgts>.+?)\s+in\s+(?P<iter>.+?)\s*$"
)
_JL_AUG_RE = re.compile(r"[+\-*/%^\\]=(?!=)")
_JL_SORT_RE = re.compile(rf"\bsort!?\s*\(\s*(?P<arg>{_JL_IDENT})")

_JL_PASSTHROUGH_NAMES = {
    "collect", "enumerate", "zip", "reverse", "reverse!",
    "sort", "sort!", "first", "last", "view", "filter", "filter!",
}


def _jl_split_commas(s: str) -> list[str]:
    """Split a comma-separated list respecting ()[]{} nesting."""
    out, depth, buf = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return [x for x in out if x]


def _jl_strip_outer_parens(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        depth, closed_early = 0, False
        for j, c in enumerate(s):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and j != len(s) - 1:
                    closed_early = True
                    break
        if closed_early:
            break
        s = s[1:-1].strip()
    return s


def _jl_rhs_value_tainted(rhs: str, tainted: set[str]) -> bool:
    """Narrow: would this Julia expression evaluate to a probability-derived
    value? Used for taint propagation through `x = rhs`. Stops at opaque
    function calls (non-DSL, non-passthrough)."""
    s = _jl_strip_outer_parens(rhs)
    # Call form `FUNC(args)`.
    m = re.match(rf"^({_JL_IDENT})\s*\(", s)
    if m and s.endswith(")"):
        open_paren = m.end() - 1
        fname = m.group(1)
        args_str = s[open_paren + 1:-1]
        if fname in _DSL_NAMES:
            return True
        if fname in _JL_PASSTHROUGH_NAMES:
            for arg in _jl_split_commas(args_str):
                if _jl_rhs_value_tainted(arg, tainted):
                    return True
        return False
    # Subscript: `name[...]`.
    m = re.match(rf"^({_JL_IDENT})\s*\[", s)
    if m:
        return m.group(1) in tainted
    # Bare identifier.
    if re.fullmatch(_JL_IDENT, s):
        return s in tainted
    return False


def _jl_arithmetic_on_tainted(line: str, tainted: set[str]) -> bool:
    """Broad: does a tainted name appear as an operand of an arithmetic or
    comparison operator on this line? Used for violation detection.

    The `(?<!=)` lookbehinds on `<` / `>` exclude Julia's `=>` pair
    constructor (not arithmetic) from matching."""
    if not tainted:
        return False
    names = "|".join(re.escape(n) for n in tainted)
    op = (r"(?:==|!=|<=|>=|\*\*|\+|-(?!>)|\*|/|%|\^"
          r"|(?<!=)<(?![=<])|(?<!=)>(?![=>]))")
    name_sub = rf"(?<![\w.])\b(?:{names})\b(?:\[[^\]]*\])?"
    return bool(
        re.search(rf"{name_sub}\s*{op}", line)
        or re.search(rf"{op}\s*{name_sub}", line)
    )


def check_file_scanner_jl(path: Path, valid_slugs: set[str]) -> list[Violation]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = content.splitlines()
    violations: list[Violation] = []

    tainted: set[str] = set()
    fn_stack: list[set[str]] = []
    block_stack: list[str] = []

    dsl_call_re = re.compile(rf"(?<![\w.])\b(?:{_DSL})\s*\(")

    def _flag(lineno: int, kind: str) -> None:
        line = lines[lineno - 1]
        ok, err = _pragma_allows(lines, lineno, valid_slugs)
        if ok:
            return
        if err is not None:
            violations.append(Violation(path, lineno, line.rstrip(), err))
        else:
            violations.append(Violation(
                path, lineno, line.rstrip(),
                f"{kind} (pass two: indirection through a prior binding)"
                f" — declare a Functional and call expect(), or add a"
                f" `# credence-lint: allow — precedent:<slug> — <reason>` pragma",
            ))

    in_docstring = False
    for i, raw_src in enumerate(lines, start=1):
        triples = len(TRIPLE_RE.findall(raw_src))
        was_in_docstring = in_docstring
        if triples % 2 == 1:
            in_docstring = not in_docstring
        if was_in_docstring and in_docstring:
            continue
        stripped_src = raw_src.strip()
        if not stripped_src or stripped_src.startswith("#"):
            continue
        # Strip trailing `#…` comment (best-effort — doesn't respect string
        # literals, but # inside "…" is rare in practice).
        raw = _jl_strip_trailing_comment(raw_src)
        stripped = raw.strip()
        if not stripped:
            continue

        # Block tracking: openers push, `end` pops.
        om = _JL_OPENER_RE.match(raw)
        if om:
            kind = om.group(1).split()[-1]
            if kind == "function":
                fn_stack.append(tainted)
                tainted = set()
                block_stack.append("function")
                continue
            block_stack.append("other")
        if _JL_END_RE.match(raw):
            if block_stack:
                popped = block_stack.pop()
                if popped == "function" and fn_stack:
                    tainted = fn_stack.pop()
            continue

        # Flag: augmented arithmetic touching a tainted operand.
        if _JL_AUG_RE.search(raw):
            split = _JL_AUG_RE.split(raw, maxsplit=1)
            lhs_text = split[0]
            rhs_text = split[1] if len(split) > 1 else ""
            lhs_match = re.search(rf"({_JL_IDENT})\s*$", lhs_text.rstrip())
            lhs_name = lhs_match.group(1) if lhs_match else ""
            lhs_tainted = lhs_name in tainted
            rhs_val_tainted = _jl_rhs_value_tainted(rhs_text, tainted)
            rhs_has_arith = _jl_arithmetic_on_tainted(rhs_text, tainted)
            if lhs_tainted or rhs_val_tainted or rhs_has_arith:
                _flag(i, "augmented arithmetic on DSL return")
                continue

        # Flag: sort on a tainted first-arg name.
        sm = _JL_SORT_RE.search(raw)
        if sm and sm.group("arg") in tainted:
            _flag(i, "sort on DSL return")
            continue

        # Flag: `if <expr>` / `elseif <expr>` with arithmetic on tainted.
        if stripped.startswith("if ") or stripped.startswith("elseif "):
            cond = stripped.split(None, 1)[1]
            if _jl_arithmetic_on_tainted(cond, tainted):
                _flag(i, "comparison-to-branch on DSL return")
                continue

        # Flag: a line that is a bare expression with arithmetic on tainted
        # (not an assignment line — those we analyse specifically below).
        if not _JL_ASSIGN_RE.match(raw) and not _JL_TUPLE_ASSIGN_RE.match(raw):
            if _jl_arithmetic_on_tainted(raw, tainted):
                _flag(i, "arithmetic on DSL return")
                continue

        # Propagate taint: for-loop.
        fm = _JL_FOR_RE.match(raw)
        if fm:
            targets_raw, iter_raw = fm.group("tgts"), fm.group("iter")
            t_stripped = targets_raw.strip()
            if t_stripped.startswith("(") and t_stripped.endswith(")"):
                t_stripped = t_stripped[1:-1]
            t_names = [t.strip() for t in _jl_split_commas(t_stripped)]
            t_names = [t for t in t_names if re.fullmatch(_JL_IDENT, t)]

            zip_m = re.match(r"^\s*zip\((.+)\)\s*$", iter_raw)
            if zip_m and t_names:
                zip_args = _jl_split_commas(zip_m.group(1))
                if len(zip_args) == len(t_names):
                    for tn, src in zip(t_names, zip_args):
                        src_t = _jl_rhs_value_tainted(src, tainted) or bool(
                            dsl_call_re.search(src))
                        (tainted.add if src_t else tainted.discard)(tn)
                    continue
            iter_t = _jl_rhs_value_tainted(iter_raw, tainted) or bool(
                dsl_call_re.search(iter_raw))
            for tn in t_names:
                (tainted.add if iter_t else tainted.discard)(tn)
            continue

        # Propagate taint: simple assignment.
        am = _JL_ASSIGN_RE.match(raw)
        if am:
            tgt, rhs = am.group("tgt"), am.group("rhs")
            if _jl_arithmetic_on_tainted(rhs, tainted):
                _flag(i, "arithmetic on DSL return")
            rhs_t = _jl_rhs_value_tainted(rhs, tainted)
            (tainted.add if rhs_t else tainted.discard)(tgt)
            continue
        tm = _JL_TUPLE_ASSIGN_RE.match(raw)
        if tm:
            tgts, rhs = tm.group("tgts"), tm.group("rhs")
            if _jl_arithmetic_on_tainted(rhs, tainted):
                _flag(i, "arithmetic on DSL return")
            names = [t.strip() for t in tgts.split(",") if t.strip()]
            rhs_t = _jl_rhs_value_tainted(rhs, tainted)
            for n in names:
                (tainted.add if rhs_t else tainted.discard)(n)
            continue

    return violations


def _jl_strip_trailing_comment(line: str) -> str:
    """Remove `#…` from a Julia line, respecting `"…"` string literals.
    Best-effort — doesn't handle triple-quoted strings or escaped quotes
    inside strings, but adequate for source code under apps/."""
    in_str = False
    for j, c in enumerate(line):
        if c == '"':
            in_str = not in_str
        elif c == "#" and not in_str:
            return line[:j]
    return line


# ── per-file lint ─────────────────────────────────────────────────────
def check_file(path: Path, valid_slugs: set[str], require_role: bool) -> list[Violation]:
    violations: list[Violation] = []
    role = read_role(path)
    if require_role and role is None:
        violations.append(Violation(path, 1, "", "missing `# Role:` header"))
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return violations

    lines = content.splitlines()
    in_docstring = False
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        triples = len(TRIPLE_RE.findall(line))
        was_in_docstring = in_docstring
        if triples % 2 == 1:
            in_docstring = not in_docstring
        if was_in_docstring and in_docstring:
            continue
        if stripped.startswith("#") or stripped.startswith(";"):
            continue
        if DECL_RE.match(line):
            continue
        kind = _violates(line)
        if kind is None:
            if PRAGMA_PRESENT.search(line):
                ok, err = check_pragma(line, valid_slugs)
                if not ok and err is not None:
                    violations.append(Violation(path, i, line.rstrip(), err))
            continue
        ok, err = _pragma_allows(lines, i, valid_slugs)
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

    # Pass two — dedup by line number with pass one (same-line coverage
    # overlap is frequent for `bad_*` files that both passes catch).
    p1_lines = {v.line_no for v in violations}
    if path.suffix == ".py":
        p2 = check_file_ast_py(path, valid_slugs)
    elif path.suffix == ".jl":
        p2 = check_file_scanner_jl(path, valid_slugs)
    else:
        p2 = []
    for v in p2:
        if v.line_no not in p1_lines:
            violations.append(v)

    return sorted(violations, key=lambda v: v.line_no)


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
            if any(
                any(
                    sub.parts[k:k + len(rel)] == rel
                    for k in range(len(sub.parts) - len(rel) + 1)
                )
                for rel in SKIP_REL_DIRS
            ):
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
            if not viols:
                failures.append(
                    f"FAIL bad2: {f.relative_to(root)} not flagged —"
                    f" pass two should catch"
                )
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
