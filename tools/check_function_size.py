#!/usr/bin/env python3
"""Function-size gate for imp.

Sibling of tools/check_filesize.py. That gate measures a FILE; this one measures
the largest thing inside it, because the two are not the same number and the
difference is where the debt hid:

  src/exec/executor_workspace_buffers.cu   1534 code LOC, allowlisted "(c) one
  concern" — and 1252 of those lines are a SINGLE function body. A cohesion
  argument about a file says nothing about a function that long.

  src/exec/executor_attention.cu            542 code LOC to the file gate, but
  GraphExecutor::run_attention textually #include's three .cu fragments INSIDE
  its own body, so the function the compiler sees is ~1300 code LOC. Both gates
  now expand those includes (see EXPANSION below).

METRIC: CODE LOC of the body (blank + comment lines stripped by the same state
machine check_filesize.py uses — imported, not re-implemented, so the two gates
can never disagree about what a line is). Braces of the signature and the
closing brace are not counted.

WHAT IS A FUNCTION HERE: a definition whose signature starts in column 0 and
whose body closes with a `}` in column 0. That is the whole tree's style for
free functions and out-of-line members, and it deliberately skips class-inline
methods, lambdas and kernels defined inside other constructs — a body that is
already nested is charged to its enclosing top-level definition.

EXPANSION: a `#include "foo.cu"` inside a body is expanded in place with the
fragment's code LOC. The three executor_attention fragments say in their own
headers that they are not translation units; a gate that counts them separately
measures four files that do not exist as compilation objects.

THRESHOLDS come from the measured distribution (see the toml header), not from
what sounds tidy.

ALLOWLIST is a two-way ceiling with a mandatory reason, identical in shape and
intent to check_filesize.py's — an entry records that a long function is
accepted AND how long it was when that was decided, so it cannot grow silently.

Exit codes:
  0  no non-allowlisted function over hard, no allowlist drift
  1  a violation or a drifted allowlist entry
  2  malformed config

Usage:
  python3 tools/check_function_size.py              # blocking gate
  python3 tools/check_function_size.py --warn-only  # advisory
  python3 tools/check_function_size.py --list       # every function over warn
  python3 tools/check_function_size.py --stats      # distribution, for thresholds
  python3 tools/check_function_size.py --update     # re-pin [allow] code_loc
"""
import argparse
import importlib.util
import os
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    sys.stderr.write("check_function_size.py needs Python 3.11+ (tomllib)\n")
    sys.exit(2)

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS)
DEFAULT_CONFIG = os.path.join(TOOLS, "function_size_thresholds.toml")

# Share the comment stripper with the file-size gate rather than copying it: two
# gates that disagree about what a code line is would produce two baselines.
_spec = importlib.util.spec_from_file_location("_cfs", os.path.join(TOOLS, "check_filesize.py"))
_cfs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfs)
code_loc = _cfs.code_loc

SRC_EXT = (".cu", ".cpp")

# A signature starts in column 0 and is not one of the block openers that are not
# functions. `if`/`for`/`while` cannot appear in column 0 outside a body, so they
# need no exclusion. A parameter list is what separates a function from a data
# table: `static const uint32_t T[256] = {` opens a 256-line brace block in column
# 0 and is not a function, and requiring `)` is what tells them apart.
NOT_A_FUNCTION = re.compile(
    r"^(namespace|extern|struct|class|enum|union|using|typedef|template|#|//|/\*|\}|else)\b")
SIG_START = re.compile(r"^[A-Za-z_~][^=]*\(")
INCLUDE_CU = re.compile(r'^\s*#include\s+"([^"]+\.cu)"')
# A signature may wrap over several lines; the continuations are indented, so the
# opening `{` can be up to this many lines below the line that starts in column 0.
MAX_SIG_LINES = 12


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def fragment_code_loc(rel, cache):
    """code LOC of a textually included .cu fragment, resolved from src/ or the repo root."""
    if rel in cache:
        return cache[rel]
    for base in ("src", ""):
        full = os.path.join(REPO_ROOT, base, rel)
        if os.path.isfile(full):
            with open(full, "r", errors="replace") as fh:
                cache[rel] = code_loc(fh.read())[1]
            return cache[rel]
    cache[rel] = 0
    return 0


def functions_in(path, text, frag_cache):
    """Yield (name, start_line, code_loc, included_fragments) per top-level definition."""
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not SIG_START.match(line) or NOT_A_FUNCTION.match(line):
            i += 1
            continue
        # Walk the (possibly wrapped) signature to its opening brace. A `;` first
        # means this was a declaration, not a definition.
        sig, k, one_liner = [], i, False
        while k < len(lines) and k - i < MAX_SIG_LINES:
            sig.append(lines[k])
            s = lines[k].rstrip()
            if "{" in s:
                # `T f(a) { return x; }` closes on its own line — a body of zero
                # counted lines, and swallowing it would merge the next function in.
                one_liner = "}" in s[s.index("{") + 1:]
                break
            if s.endswith(";") or (k > i and not lines[k][:1].isspace() and lines[k].strip()):
                k = -1
                break
            k += 1
        if k < 0 or k >= len(lines) or k - i >= MAX_SIG_LINES or one_liner \
                or ")" not in " ".join(sig):
            i += 1
            continue
        body, frags = [], []
        j = k + 1
        while j < len(lines) and not lines[j].startswith("}"):
            m = INCLUDE_CU.match(lines[j])
            if m:
                frags.append(m.group(1))
            body.append(lines[j])
            j += 1
        if j >= len(lines):  # no closing brace in column 0 — not a top-level body
            i += 1
            continue
        n = code_loc("\n".join(body))[1]
        n += sum(fragment_code_loc(f, frag_cache) for f in frags)
        name = re.sub(r"\s*\{[ \t]*$", "", " ".join(x.strip() for x in sig)).strip()
        out.append((name, i + 1, n, frags))
        i = j + 1
    return out


def scan(roots, skip_dirs):
    frag_cache = {}
    # A file that is only ever #include'd into another body is not a TU; its
    # functions are charged to the includer, so do not scan it standalone.
    included = set()
    files = []
    for r in roots:
        base = os.path.join(REPO_ROOT, r)
        if not os.path.isdir(base):
            continue
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d not in skip_dirs and d != "__pycache__"]
            for fn in fns:
                if os.path.splitext(fn)[1] in SRC_EXT:
                    files.append(os.path.join(dp, fn))
    texts = {}
    for full in files:
        with open(full, "r", errors="replace") as fh:
            texts[full] = fh.read()
        for m in INCLUDE_CU.finditer(texts[full]):
            included.add(os.path.normpath(os.path.join(REPO_ROOT, "src", m.group(1))))

    rows = []
    for full, text in texts.items():
        if os.path.normpath(full) in included:
            continue
        rel = os.path.relpath(full, REPO_ROOT)
        for name, ln, n, frags in functions_in(rel, text, frag_cache):
            rows.append({"path": rel, "line": ln, "name": name, "code": n, "frags": frags})
    return rows


def selftest():
    """Plant what this parser must get right, including what it got wrong first.

    Every case below except the last two was written after the detector produced
    a wrong answer on the real tree: `static const uint32_t BYTE_TO_CODEPOINT[256]
    = {` was reported as a 256-line function (a brace block in column 0 is not a
    function; a parameter list is what tells them apart), and a one-line body
    `size_t round_up(...) { return ...; }` swallowed the next function whole
    because the walk kept looking for a line ending in `{`. #1858 measured 5 of
    13 static gates missing their own violations, so a gate without this is not
    a gate.
    """
    cases = [
        ("plain function", "void f(int a) {\n    int x = 1;\n    g(x);\n}\n", [2]),
        ("blank + comment lines are not code",
         "void f() {\n    int x = 1;\n\n    // note\n    /* block */\n}\n", [1]),
        ("wrapped signature",
         "void f(int a,\n       int b,\n       int c) {\n    g();\n}\n", [1]),
        ("declaration only", "void f(int a);\nvoid g() {\n    h();\n}\n", [1]),
        ("data table in column 0",
         "static const int T[3] = {\n    1,\n    2,\n    3,\n};\n", []),
        ("one-line body does not swallow the next function",
         "int r(int v) { return v; }\nvoid f() {\n    g();\n    h();\n}\n", [2]),
        ("class-inline method is charged to nothing",
         "struct S {\n    void m() {\n        g();\n    }\n};\n", []),
        ("lambda inside a body does not end it",
         "void f() {\n    auto l = [] {\n        g();\n    };\n    l();\n}\n", [4]),
        ("included .cu fragment is expanded in place",
         'void f() {\n    a();\n#include "exec/frag.cu"\n    b();\n}\n', [3 + 40]),
        ("two functions in one file",
         "void f() {\n    a();\n}\nvoid g() {\n    b();\n    c();\n}\n", [1, 2]),
    ]
    failures = 0
    for name, text, want in cases:
        got = [n for _, _, n, _ in functions_in("t.cu", text, {"exec/frag.cu": 40})]
        ok = got == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}: expected {want}, got {got}")
    print(f"selftest: {len(cases) - failures}/{len(cases)} cases")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description="imp function-size gate")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--warn-only", action="store_true")
    ap.add_argument("--list", action="store_true", help="print every function over warn")
    ap.add_argument("--stats", action="store_true", help="print the distribution and exit")
    ap.add_argument("--keys", action="store_true",
                    help="print the full allowlist key of every function over hard")
    ap.add_argument("--update", action="store_true", help="re-pin every [allow] code_loc")
    ap.add_argument("--selftest", action="store_true",
                    help="plant each parse case the gate must get right")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    cfg = load_config(args.config)
    warn_t = cfg["thresholds"]["warn"]
    hard_t = cfg["thresholds"]["hard"]
    allow = cfg.get("allow", {})

    bad = [k for k, v in allow.items()
           if not isinstance(v, dict) or not str(v.get("reason", "")).strip()
           or not isinstance(v.get("code_loc"), int)]
    if bad:
        print("ERROR: allowlist entries are `{ code_loc = N, reason = \"...\" }` tables")
        print("       with a non-empty reason. Offenders:")
        for k in bad:
            print(f"  {k}")
        return 2

    rows = scan(cfg["scan"]["roots"], set(cfg["scan"].get("skip_dirs", [])))

    if args.stats:
        vals = sorted(r["code"] for r in rows)
        if not vals:
            print("no functions found")
            return 2

        def pct(p):
            return vals[min(len(vals) - 1, int(len(vals) * p))]

        print(f"functions={len(vals)}  p50={pct(.50)}  p90={pct(.90)}  p95={pct(.95)} "
              f"p99={pct(.99)}  max={vals[-1]}")
        for t in (200, 300, 400, 500, 600, 800, 1000):
            print(f"  > {t:<5} {sum(1 for v in vals if v > t):>4}")
        return 0

    # key = "path:name" — line numbers move with every edit above the function,
    # so pinning on them would make the allowlist stale by construction.
    def key(r):
        return f"{r['path']}::{r['name']}"

    if args.keys:
        for r in sorted((x for x in rows if x["code"] > hard_t), key=lambda x: -x["code"]):
            print(f'"{key(r)}" = {{ code_loc = {r["code"]}, reason = "" }}')
        return 0

    warns, hards, allowed = [], [], []
    for r in rows:
        if r["code"] > hard_t:
            (allowed if key(r) in allow else hards).append(r)
        elif r["code"] > warn_t:
            warns.append(r)

    def table(title, items, limit):
        if not items:
            return
        print(f"\n{title}")
        print(f"  {'code':>5} {'+/-':>6}  file:line  function")
        for r in sorted(items, key=lambda x: -x["code"]):
            frag = f"  (+{len(r['frags'])} included .cu)" if r["frags"] else ""
            print(f"  {r['code']:>5} {r['code'] - limit:>+6}  {r['path']}:{r['line']}  "
                  f"{r['name'][:78]}{frag}")

    if args.list or warns:
        table(f"WARN ({len(warns)}) — soft smell, not blocking", warns, warn_t)
    table(f"ALLOWLISTED ({len(allowed)}) — over hard but accepted in baseline", allowed, hard_t)
    table(f"HARD VIOLATIONS ({len(hards)}) — NOT allowlisted", hards, hard_t)

    print(f"\nscanned {len(rows)} top-level functions in {', '.join(cfg['scan']['roots'])} | "
          f"warn={len(warns)} allowlisted={len(allowed)} violations={len(hards)}")

    measured = {key(r): r["code"] for r in rows}
    drift = [(k, e["code_loc"], measured[k]) for k, e in sorted(allow.items())
             if k in measured and measured[k] != e["code_loc"]]
    gone = [k for k in allow if k not in measured]

    if args.update:
        text = open(args.config, encoding="utf-8").read()
        for k, _, actual in drift:
            pat = re.compile(r'(^"' + re.escape(k) + r'"\s*=\s*\{\s*code_loc\s*=\s*)\d+', re.M)
            text, n = pat.subn(lambda m: m.group(1) + str(actual), text)
            if n != 1:
                print(f"ERROR: --update could not re-pin {k} (matched {n} lines)")
                return 2
        open(args.config, "w", encoding="utf-8").write(text)
        print(f"\nallowlist re-pinned: {len(drift)} entr(y/ies) updated")
        return 0

    if gone:
        print(f"\nNOTE: {len(gone)} allowlist entr(y/ies) no longer match a function "
              f"(renamed, moved or split — remove them from [allow]):")
        for k in sorted(gone):
            print(f"  {k}")

    if drift and not args.warn_only:
        print(f"\nFAIL: {len(drift)} allowlisted function(s) drifted from their pinned code_loc.")
        print(f"  {'pinned':>7} {'actual':>7} {'+/-':>6}  function")
        for k, pinned, actual in drift:
            print(f"  {pinned:>7} {actual:>7} {actual - pinned:>+6}  {k}")
        print("\nAn allowlist entry is a ceiling, not an exemption. Re-pin with")
        print("  python3 tools/check_function_size.py --update")
        print("and say in the PR body which way it moved and why.")
        return 1

    if hards and not args.warn_only:
        print(f"\nFAIL: {len(hards)} function(s) over the hard threshold ({hard_t} code LOC). "
              "Extract a step, or — if the body is one indivisible sequence — add it to "
              "[allow] in tools/function_size_thresholds.toml WITH a reason.")
        return 1
    print("\nOK" + (" (warn-only)" if args.warn_only else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
