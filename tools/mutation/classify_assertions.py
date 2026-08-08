#!/usr/bin/env python3
"""Assertion-strength classifier for the imp GTest suite.

Splits every tests/*.cpp|*.cu into TEST/TEST_F/TEST_P bodies and classifies each
body by its STRONGEST assertion, using the A0-A4 ladder from the test-hardening
dispatch:

  A0 smoke      - only "it did not crash / returned success / pointer non-null"
  A1 shape/type - only sizes, dtypes, non-empty, isfinite
  A2 weak value - inequalities / ranges / non-zero, no fixed expected value
  A3 tolerance  - EXPECT_NEAR / FLOAT_EQ against a fixed expected number
  A4 oracle     - compared against an independent reference computed in-test

A4 cannot be detected from the macro alone; it is inferred from the presence of
a reference computation in the body (a cpu_reference/naive/golden/ref_ symbol)
feeding the comparison. Everything reported here is a *coarse* screen: the
dispatch requires 40 hand-read tests on top, and the hand-read set is what the
report quotes for the hot paths.

Usage: classify_assertions.py <repo-root>
"""
import re
import sys
import json
from pathlib import Path

TEST_RE = re.compile(r'^\s*(TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P)\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)', re.M)
ASSERT_RE = re.compile(r'\b(ASSERT|EXPECT)_([A-Z_]+)\s*\(')

# Reference-implementation markers: a body that computes its own expected value
# from an independent path is an oracle test, not a golden-constant test.
ORACLE_RE = re.compile(r'\b(cpu_ref|ref_|reference|naive_|golden|Golden|GOLDEN|expected_ref|oracle|Oracle)\w*', re.M)
# A golden *header* include is a committed reference tensor -> still A4-ish, but
# we separate it because the generation provenance matters.
GOLDEN_INC_RE = re.compile(r'#include\s+"refs/')

TOL_MACROS = {'NEAR', 'FLOAT_EQ', 'DOUBLE_EQ'}
INEQ_MACROS = {'LT', 'GT', 'LE', 'GE'}
SMOKE_MACROS = {'NO_THROW', 'THROW'}


def body_of(src: str, start: int) -> str:
    """Extract the brace-balanced body that follows the TEST(...) header.

    Must skip string literals, char literals, raw strings and comments: a body
    containing `doc.back() = (last == '}') ? ']' : '}';` closes the brace count
    six lines in and truncates the test to nothing, which then reports as
    "zero assertions". Every literal-blind brace counter produces that lie.
    """
    i = src.find('{', start)
    if i < 0:
        return ''
    depth = 0
    j = i
    n = len(src)
    while j < n:
        c = src[j]
        if c == '/' and j + 1 < n and src[j + 1] == '/':
            j = src.find('\n', j)
            if j < 0:
                break
            continue
        if c == '/' and j + 1 < n and src[j + 1] == '*':
            j = src.find('*/', j + 2)
            if j < 0:
                break
            j += 2
            continue
        if c == 'R' and src[j:j + 2] == 'R"':
            m = re.match(r'R"([^(]*)\(', src[j:])
            if m:
                end = src.find(')' + m.group(1) + '"', j + m.end())
                if end < 0:
                    break
                j = end + len(m.group(1)) + 2
                continue
        if c in '"\'':
            quote = c
            j += 1
            while j < n:
                if src[j] == '\\':
                    j += 2
                    continue
                if src[j] == quote:
                    break
                j += 1
            j += 1
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return src[i:j + 1]
        j += 1
    return src[i:]


# A one-line `TEST_F(X, Y) { run_config(...); }` carries its assertions in the
# helper, not in the body. Classifying such a test from the body alone reports
# a golden-checked kernel test as A0 — the exact false "the suite asserts
# nothing" claim this audit must not make. So resolve callees one level deep
# (transitively, bounded) against every function defined in the same file or in
# any tests/ header it includes.
FUNC_HEAD_RE = re.compile(
    r'^[ \t]*(?:static\s+|inline\s+|constexpr\s+|template\s*<[^>]*>\s*)*'
    r'(?:[A-Za-z_][\w:<>,\s\*&]*?)\s+([A-Za-z_]\w*)\s*\(', re.M)
CALL_RE = re.compile(r'\b([A-Za-z_]\w*)\s*\(')
NOT_A_CALL = {'if', 'for', 'while', 'switch', 'return', 'sizeof', 'catch',
              'TEST', 'TEST_F', 'TEST_P', 'TYPED_TEST', 'static_cast',
              'reinterpret_cast', 'const_cast', 'dynamic_cast'}


def collect_functions(src: str) -> dict:
    """name -> body, for every brace-balanced function definition in src.

    The parameter list is scanned with balanced parens, not a regex: a default
    argument like `MxRecipe recipe = {}` contains a brace, and a `[^;{]*`
    parameter pattern silently drops every helper that has one — which is how
    30 golden-checked FMHA tests first reported as "zero assertions".
    """
    out = {}
    n = len(src)
    for m in FUNC_HEAD_RE.finditer(src):
        name = m.group(1)
        if name in NOT_A_CALL:
            continue
        # Balanced scan of the parameter list.
        j, depth = m.end() - 1, 0
        while j < n:
            if src[j] == '(':
                depth += 1
            elif src[j] == ')':
                depth -= 1
                if depth == 0:
                    break
            elif src[j] == ';':
                j = -1
                break
            j += 1
        if j < 0 or j >= n:
            continue
        k = j + 1
        while k < n and (src[k].isspace() or src.startswith('const', k)
                         or src.startswith('noexcept', k) or src.startswith('override', k)):
            k += 5 if src.startswith('const', k) else (
                8 if src.startswith('noexcept', k) else (
                    8 if src.startswith('override', k) else 1))
        if k >= n or src[k] != '{':  # declaration, not a definition
            continue
        out.setdefault(name, body_of(src, k))
    return out


def expand(body: str, funcs: dict, depth: int = 3) -> str:
    """Body plus the bodies of same-file helpers it calls, transitively."""
    seen = set()
    text = [body]
    frontier = [body]
    for _ in range(depth):
        nxt = []
        for chunk in frontier:
            for m in CALL_RE.finditer(chunk):
                name = m.group(1)
                if name in NOT_A_CALL or name in seen or name not in funcs:
                    continue
                seen.add(name)
                nxt.append(funcs[name])
                text.append(funcs[name])
        if not nxt:
            break
        frontier = nxt
    return '\n'.join(text)


def classify(body: str, has_golden_include: bool):
    macros = [m.group(2) for m in ASSERT_RE.finditer(body)]
    if not macros:
        return 'A0', macros  # no assertion at all: pure smoke
    has_tol = any(m in TOL_MACROS for m in macros)
    has_ineq = any(m in INEQ_MACROS for m in macros)
    has_eq = any(m in ('EQ', 'STREQ') for m in macros)
    has_oracle = bool(ORACLE_RE.search(body)) or has_golden_include

    # Shape/type-only heuristic: every EQ compares against .size()/.dtype/dims.
    shape_only = has_eq and not has_tol and all(
        re.search(r'\.(size|length|numel|ndim|rows|cols)\(\)|shape|dtype|dim',
                  line) for line in re.findall(r'(?:ASSERT|EXPECT)_EQ\s*\(([^;]*)\)', body)
    ) if has_eq else False

    if has_tol and has_oracle:
        return 'A4', macros
    if has_tol:
        return 'A3', macros
    if has_oracle and has_eq:
        return 'A4', macros
    if has_eq and not shape_only:
        # EQ against a literal/expected value is a fixed-value check without a
        # tolerance: for integer/token/string outputs that is as strong as A3.
        return 'A3', macros
    if shape_only:
        return 'A1', macros
    if has_ineq:
        return 'A2', macros
    if all(m in ('TRUE', 'FALSE', 'NE', 'NO_THROW', 'THROW') for m in macros):
        return 'A0', macros
    return 'A2', macros


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    rows = []
    for f in sorted((root / 'tests').rglob('*')):
        if f.suffix not in ('.cpp', '.cu'):
            continue
        src = f.read_text(errors='replace')
        has_golden = bool(GOLDEN_INC_RE.search(src))
        scope = src
        # Pull in tests/*.h helpers this file includes: several suites keep the
        # whole assertion body in a shared header (scoped_engine_arena.h, ...).
        for inc in re.findall(r'#include\s+"([^"]+\.h)"', src):
            p = (root / 'tests' / inc)
            if p.exists():
                scope += '\n' + p.read_text(errors='replace')
        funcs = collect_functions(scope)
        for m in TEST_RE.finditer(src):
            body = body_of(src, m.end())
            direct = len(ASSERT_RE.findall(body))
            full = body if direct else expand(body, funcs)
            cls, macros = classify(full, has_golden)
            rows.append({
                'file': str(f.relative_to(root)),
                'line': src[:m.start()].count('\n') + 1,
                'suite': m.group(2),
                'name': m.group(3),
                'class': cls,
                'n_assert': len(macros),
                'direct_assert': direct,
                'delegated': direct == 0 and len(macros) > 0,
            })

    dist = {}
    for r in rows:
        dist[r['class']] = dist.get(r['class'], 0) + 1
    out = {'total': len(rows), 'distribution': dist, 'tests': rows}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
