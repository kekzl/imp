#!/usr/bin/env python3
"""Cyclomatic-complexity ratchet for imp (#2210).

Sibling of tools/check_function_size.py, which gates LENGTH only: on 467bd5f8
148 of the 203 functions with CCN > 25 were <= 200 NLOC, below its warn line.

METRIC: McCabe CCN with lizard's C/C++ rule: 1 + one per `if`, `for`, `while`,
`case`, `catch`, `&&`, `||`, `?`, comments and string/char literals stripped,
`#else`/`#elif` branches skipped (lizard does the same). Every function body is
measured: free functions, class-inline methods, kernels, header definitions.
Lambdas and nested blocks are charged to the enclosing function; a
`#include "x.cu"` fragment inside a body is charged to that body.

RATCHET: tools/complexity_baseline.toml pins every function over the threshold
at its measured CCN, keyed `path::signature` like function_size_thresholds.toml.
  FAIL  a function over the threshold that is not in the baseline (new)
  FAIL  a baseline function whose CCN grew past its pin (no slack: +1 fails)
  NOTE  a baseline function that shrank or is gone (re-pin with --update)

Exit codes: 0 pass, 1 violation, 2 malformed config.

Usage:
  python3 tools/check_complexity.py             # blocking gate
  python3 tools/check_complexity.py --list      # every function over threshold
  python3 tools/check_complexity.py --update    # rewrite [baseline] from the tree
  python3 tools/check_complexity.py --selftest  # plant each case the gate must catch
"""
import argparse
import json
import os
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    sys.stderr.write("check_complexity.py needs Python 3.11+ (tomllib)\n")
    sys.exit(2)

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS)
DEFAULT_CONFIG = os.path.join(TOOLS, "complexity_baseline.toml")
SRC_EXT = (".cu", ".cpp", ".h", ".cuh", ".hpp")

DECISION = re.compile(r"\b(?:if|for|while|case|catch)\b|&&|\|\||\?")
RAW_OPEN = re.compile(r'(?<![A-Za-z0-9_])(?:u8|u|U|L)?R"([^(\s]{0,16})\(')
INCLUDE_CU = re.compile(r'^\s*#\s*include\s+"([^"]+\.cu)"')
PP_IF = re.compile(r"^\s*#\s*if")
PP_ELSE = re.compile(r"^\s*#\s*(?:else|elif)\b")
PP_ENDIF = re.compile(r"^\s*#\s*endif\b")
ACCESS = re.compile(r"^\s*(?:(?:public|private|protected)\s*:(?!:)\s*)+")
TYPE_KW = re.compile(r"\b(?:class|struct|union|namespace|extern)\b")
TAIL_WORD = re.compile(r"(?:const|volatile|noexcept|override|final|mutable|try)\b|&&|&")


def tail_ok(t):
    """True if `t` may follow a function's parameter list before its `{` (linear scan)."""
    t = t.strip()
    while t:
        m = TAIL_WORD.match(t)
        if m:
            t = t[m.end():].lstrip()
        elif t.startswith("->") or re.match(r"requires\b", t):
            return not re.search(r"[;=]", t)
        elif t.startswith("[[") and "]]" in t:
            t = t[t.index("]]") + 2:].lstrip()
        else:
            return False
    return True
CTOR_INIT = re.compile(r"\)\s*(?:noexcept\s*)?:(?!:)")


def _blank(s):
    """Same length, newlines kept, everything else a space: line numbers survive."""
    return re.sub(r"[^\n]", " ", s)


def strip_code(text):
    """Comments, string literals and char literals blanked; newlines and offsets kept.

    A `'` right after an alphanumeric is a digit separator (1'000'000), not a char literal.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            e = text.find("\n", i)
            e = n if e < 0 else e
            out.append(_blank(text[i:e]))
            i = e
            continue
        if text.startswith("/*", i):
            e = text.find("*/", i + 2)
            e = n if e < 0 else e + 2
            out.append(_blank(text[i:e]))
            i = e
            continue
        m = RAW_OPEN.match(text, i) if c in "uULR" else None
        if m:
            e = text.find(")" + m.group(1) + '"', m.end())
            e = n if e < 0 else e + len(m.group(1)) + 2
            out.append(_blank(text[i:e]))
            i = e
            continue
        if c == '"' or (c == "'" and not (i > 0 and text[i - 1].isalnum())):
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(_blank(text[i:j]))
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out)


def preprocess(text):
    """(code, fragments): stripped code with directives and #else/#elif branches blanked.

    fragments = [(line_no, "x.cu")] for each `#include "x.cu"`.
    """
    lines = strip_code(text).split("\n")
    raw = text.split("\n")
    frags, stack, cont = [], [], False
    for i, ln in enumerate(lines):
        directive = cont or ln.lstrip().startswith("#")
        cont = directive and ln.rstrip().endswith("\\")
        if directive and not cont and ln.lstrip().startswith("#"):
            if PP_IF.match(ln):
                stack.append(False)
            elif PP_ELSE.match(ln) and stack:
                stack[-1] = True
            elif PP_ENDIF.match(ln) and stack:
                stack.pop()
            m = INCLUDE_CU.match(raw[i])
            if m and not any(stack):
                frags.append((i + 1, m.group(1)))
        if directive or any(stack):
            lines[i] = _blank(ln)
    return "\n".join(lines), frags


def _strip_template(h):
    """Drop leading `template <...>` groups (balanced angles)."""
    while True:
        m = re.match(r"\s*template\s*<", h)
        if not m:
            return h
        depth, i = 1, m.end()
        while i < len(h) and depth:
            depth += {"<": 1, ">": -1}.get(h[i], 0)
            i += 1
        h = h[i:]


def classify(header):
    """'func', 'scope' (class/namespace: look for functions inside) or 'opaque'."""
    h = _strip_template(ACCESS.sub("", header))
    paren = h.find("(")
    pre = h if paren < 0 else h[:paren]
    if re.search(r"\benum\b", pre):
        return "opaque"
    if TYPE_KW.search(pre):
        return "scope"
    if paren < 0 or ("=" in pre and "operator" not in pre):
        return "opaque"
    depth, last_close = 0, -1
    for i, ch in enumerate(h):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                last_close = i
    if depth != 0 or last_close < 0:
        return "opaque"
    return "func" if tail_ok(h[last_close + 1:]) else "opaque"


def signature(header):
    h = " ".join(_strip_template(ACCESS.sub("", header)).split())
    h = re.sub(r"\s+\)", ")", re.sub(r"\(\s+", "(", h))
    depth = 0
    for i, ch in enumerate(h):  # a ctor's init list is not part of its signature
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if ch == ")" and depth == 0:
            m = CTOR_INIT.match(h, i)
            if m:
                return h[:i + 1]
    return h


def functions(text, frag_decisions=lambda rel: 0):
    """[(signature, line, complexity)] for every function body in one file's text."""
    code, frags = preprocess(text)
    out, stack, hstart = [], [], 0   # stack of (kind, open_pos, header, header_pos)
    for m in re.finditer(r"[{};]", code):
        ch, pos = m.group(), m.start()
        in_func = any(k == "func" for k, *_ in stack)
        in_opaque = any(k in ("opaque", "init") for k, *_ in stack)
        if ch == ";":
            if not in_func and not in_opaque:
                hstart = pos + 1
            continue
        if ch == "{":
            if in_func or in_opaque:
                stack.append(("inner", pos, "", 0))
                continue
            header = code[hstart:pos]
            # Brace inside an open parameter list (`T p = {}`) or a ctor init list (`a_{x}`).
            if header.count("(") > header.count(")") or (
                    CTOR_INIT.search(header) and re.search(r"[\w>]\s*$", header)):
                stack.append(("init", pos, "", 0))
                continue
            kind = classify(header)
            stack.append((kind, pos, header, hstart))
            if kind != "func":
                hstart = pos + 1
            continue
        if not stack:  # unbalanced `}`: resync
            hstart = pos + 1
            continue
        kind, open_pos, header, hpos = stack.pop()
        if kind == "func":
            first = hpos + len(header) - len(_strip_template(ACCESS.sub("", header)).lstrip())
            line = code.count("\n", 0, first) + 1
            end_line = code.count("\n", 0, pos) + 1
            body = code[open_pos + 1:pos]
            c = 1 + len(DECISION.findall(body))
            c += sum(frag_decisions(rel) for ln, rel in frags if line <= ln <= end_line)
            out.append((signature(header), line, c))
        if kind not in ("inner", "init") and not any(k in ("func", "opaque", "init")
                                                      for k, *_ in stack):
            hstart = pos + 1
    return out


def scan(roots, skip_dirs):
    files, texts, included = [], {}, set()
    for r in roots:
        for dp, dns, fns in os.walk(os.path.join(REPO_ROOT, r)):
            dns[:] = sorted(d for d in dns if d not in skip_dirs and d != "__pycache__")
            files += [os.path.join(dp, f) for f in sorted(fns) if os.path.splitext(f)[1] in SRC_EXT]
    for full in files:
        with open(full, "r", errors="replace") as fh:
            texts[full] = fh.read()
        for ln in texts[full].split("\n"):
            m = INCLUDE_CU.match(ln)
            if m:
                included.add(os.path.normpath(os.path.join(REPO_ROOT, "src", m.group(1))))
    cache = {}

    def frag_decisions(rel):  # a body fragment has no function of its own: count its decisions
        if rel not in cache:
            full = os.path.normpath(os.path.join(REPO_ROOT, "src", rel))
            cache[rel] = len(DECISION.findall(preprocess(texts.get(full, ""))[0]))
        return cache[rel]

    rows = []
    for full, text in texts.items():
        if os.path.normpath(full) in included:
            continue
        rel = os.path.relpath(full, REPO_ROOT)
        for sig, line, c in functions(text, frag_decisions):
            rows.append({"key": f"{rel}::{sig}", "path": rel, "line": line, "cyclomatic": c})
    return rows, len(files)


def by_key(rows):
    """key -> max CCN; two definitions with one signature text share a pin."""
    out = {}
    for r in rows:
        out[r["key"]] = max(out.get(r["key"], 0), r["cyclomatic"])
    return out


def evaluate(measured, baseline, limit):
    """(new, grown, notes) for key->complexity maps."""
    new = sorted((k, c) for k, c in measured.items() if c > limit and k not in baseline)
    grown = sorted((k, baseline[k], c) for k, c in measured.items()
                   if k in baseline and c > baseline[k])
    notes = sorted((k, p, measured.get(k)) for k, p in baseline.items()
                   if k not in measured or measured[k] < p)
    return new, grown, notes


def write_baseline(path, measured, limit):
    text = open(path, encoding="utf-8").read()
    head = text[:text.index("[baseline]\n") + len("[baseline]\n")]
    body = "".join(f"{json.dumps(k)} = {c}\n" for k, c in sorted(measured.items()) if c > limit)
    open(path, "w", encoding="utf-8").write(head + body)


def selftest():
    """Every rule the gate relies on, planted. A gate without this is not a gate (#1858)."""
    ifs = "".join(f"    if (a == {i}) g();\n" for i in range(29))
    long_sig = "__global__ void k(\n" + "".join(f"    int a{i},\n" for i in range(20)) + "    int z) {\n"
    cases = [
        ("straight line", "void f() {\n    g();\n}\n", [1]),
        ("each decision token counts once",
         "void f() {\n    if (a && b || c) x();\n    for (;;) {}\n    while (d) {}\n"
         "    switch (e) { case 1: case 2: break; default: break; }\n"
         "    try { y(); } catch (...) {}\n    int z = p ? 1 : 2;\n}\n", [10]),
        ("else if is one if", "void f() {\n    if (a) x();\n    else if (b) y();\n    else z();\n}\n", [3]),
        ("comments and strings do not count",
         "void f() {\n    // if (a)\n    /* while && */\n    s = \"if || ?\";\n    c = '?';\n}\n", [1]),
        ("raw string does not count", 'void f() {\n    s = R"x(if (a) && b ?)x";\n}\n', [1]),
        ("digit separator is not a char literal", "void f() {\n    n = 1'000'000;\n    if (a) x();\n}\n", [2]),
        ("identifiers containing keywords do not count",
         "void f() {\n    notify(); casein(); forward(); whileX();\n}\n", [1]),
        ("lambda is charged to its enclosing function",
         "void f() {\n    auto l = [](int v) { return v ? 1 : 2; };\n    if (l(1)) x();\n}\n", [3]),
        ("decision on the brace line counts", "void f() { if (a) {\n    x();\n}\n}\n", [2]),
        ("29 ifs is CCN 30", "void f(int a) {\n" + ifs + "}\n", [30]),
        ("class-inline method is a function",
         "namespace n {\nclass C : public B {\npublic:\n    int m(int a) const {\n        return a ? 1 : 2;\n"
         "    }\n    int x = 3;\n};\n}\n", [2]),
        ("kernel with a 21-line signature", "template <int N>\n" + long_sig + "    if (z) g();\n}\n", [2]),
        ("ctor init-list braces are not the body",
         "C::C(int a) : a_{a}, b_(a ? 1 : 2) {\n    if (a) g();\n}\n", [2]),
        ("brace-init default argument is not the body",
         "__global__ void k(int a, Args p = {}) {\n    if (a) g();\n}\n", [2]),
        ("#else branch is skipped",
         "#if X\nvoid f(int a) {\n#else\nvoid f(long a) {\n#endif\n    if (a) g();\n}\n", [2]),
        ("initializer tables and enums are not functions",
         "enum class E { A, B };\nstatic const int T[] = { 1, 2 };\nauto l = [](int v) { if (v) g(); };\n"
         "void f() {\n    g();\n}\n", [1]),
        ("two functions in one file", "void f() {\n    if (a) g();\n}\nvoid h() {\n    g();\n}\n", [2, 1]),
    ]
    failures = 0
    for name, text, want in cases:
        got = [c for _, _, c in functions(text)]
        ok = got == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}: expected {want}, got {got}")
    frag = functions('void f() {\n    a();\n#include "exec/frag.cu"\n}\n', lambda rel: 7)
    ok = [c for _, _, c in frag] == [8]
    failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  included .cu fragment is charged to its body: "
          f"expected [8], got {[c for _, _, c in frag]}")
    sig = functions("C::C(int a) : a_(a) {\n}\n")[0][0]
    ok = sig == "C::C(int a)"
    failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  ctor key stops at its params: expected 'C::C(int a)', got {sig!r}")
    gate_cases = [
        ("new function over the limit fails", {"a::f": 30}, {}, (1, 0)),
        ("pinned function at its pin passes", {"a::f": 30}, {"a::f": 30}, (0, 0)),
        ("pinned function +1 fails", {"a::f": 31}, {"a::f": 30}, (0, 1)),
        ("new function at the limit passes", {"a::f": 25}, {}, (0, 0)),
        ("shrink is a note, not a failure", {"a::f": 27}, {"a::f": 30}, (0, 0)),
    ]
    for name, measured, base, want in gate_cases:
        new, grown, _ = evaluate(measured, base, 25)
        got = (len(new), len(grown))
        ok = got == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}: expected {want}, got {got}")
    total = len(cases) + 2 + len(gate_cases)
    print(f"selftest: {total - failures}/{total} cases")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description="imp cyclomatic-complexity ratchet")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--list", action="store_true", help="print every function over the threshold")
    ap.add_argument("--update", action="store_true", help="rewrite [baseline] from the tree")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    limit = cfg["thresholds"]["cyclomatic"]
    baseline = cfg.get("baseline", {})
    bad = [k for k, v in baseline.items() if not isinstance(v, int) or v <= limit]
    if bad:
        print(f"ERROR: [baseline] values are integers above {limit}. Offenders:")
        for k in bad:
            print(f"  {k}")
        return 2

    rows, n_files = scan(cfg["scan"]["roots"], set(cfg["scan"].get("skip_dirs", [])))
    measured = by_key(rows)
    over = sorted((r for r in rows if r["cyclomatic"] > limit), key=lambda r: -r["cyclomatic"])

    if args.update:
        write_baseline(args.config, measured, limit)
        print(f"baseline rewritten: {sum(1 for c in measured.values() if c > limit)} entries")
        return 0
    if args.list:
        for r in over:
            print(f"  {r['cyclomatic']:>4}  {r['path']}:{r['line']}  {r['key'].split('::', 1)[1][:90]}")

    new, grown, notes = evaluate(measured, baseline, limit)
    print(f"scanned {len(rows)} functions in {n_files} files | CCN > {limit}: {len(over)} "
          f"| baseline {len(baseline)} | new {len(new)} | grown {len(grown)}")
    if notes:
        print(f"NOTE: {len(notes)} baseline entr(y/ies) shrank or are gone; re-pin with --update:")
        for k, p, c in notes:
            print(f"  {p:>4} -> {c if c is not None else 'gone':>4}  {k[:110]}")
    for k, c in new:
        print(f"FAIL  new  CCN {c} > {limit}  {k[:110]}")
    for k, p, c in grown:
        print(f"FAIL  grew CCN {p} -> {c}  {k[:110]}")
    if new or grown:
        print("\nFAIL: split the function. A new pin needs --update and a reason in the PR body.")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
