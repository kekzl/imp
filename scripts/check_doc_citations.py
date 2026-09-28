#!/usr/bin/env python3
"""Verify `path:line anchor` and bare `docs/*.md` citations in the LIVING docs.

Cite format: `path:LINES anchor`, one backtick span.
  LINES  = N | N-M (range) | N,M,... or N/M/... (list)
  anchor = literal text that must appear on the cited line (range: within it;
           list: on every listed line). No anchor = DEAD (#2185: existence-only
           checks passed two Makefile cites that pointed 16 and 81 lines off).
A bare basename is resolved by the anchor when several files share the name.
Records (docs/archive/, docs/plans/, docs/audit/) are out of scope.
"""
import re, sys, os

EXT = r"cpp|cu|cuh|h|hpp|c|py|sh|md|txt|cmake|json|toml|yml|yaml|conf"
PATH = r"((?:[\w./-]+/)?((?:[\w.-]+\.(?:" + EXT + r"))|Makefile|Dockerfile))"
CITE_START_RE = re.compile(PATH + r":\d")
CITE_RE = re.compile(PATH + r":(\d+(?:[-\u2013]\d+)?(?:[,/]\d+(?:[-\u2013]\d+)?)*)(?: (.*))?")


def _spans(spec):
    out = []
    for part in re.split(r"[,/]", spec):
        lo, _, hi = part.replace("\u2013", "-").partition("-")
        out.append((int(lo), int(hi or lo)))
    return out


def _lines(path, cache={}):
    if path not in cache:
        with open(path, encoding="utf-8", errors="replace") as fh:
            cache[path] = fh.read().split("\n")
    return cache[path]


def _verify(full, spans, anchor):
    """'' if every span holds the anchor, else the reason."""
    src = _lines(full)
    n = len(src) - (1 if src and src[-1] == "" else 0)
    for a, b in spans:
        if b > n or a < 1 or a > b:
            return f"file has only {n} lines"
        if anchor not in "\n".join(src[a - 1:b]):
            got = src[a - 1].strip()[:70]
            return f"anchor '{anchor}' not on line {a}{'-' + str(b) if b != a else ''} (line {a}: '{got}')"
    return ""


def _index(root):
    index = {}
    for base, _dirs, files in os.walk(root):
        if any(x in base for x in (".git", "build", "third_party", "node_modules")):
            continue
        for f in files:
            index.setdefault(f, []).append(os.path.join(base, f))
    return index


def check(doc, root, index=None):
    index = index if index is not None else _index(root)
    text = open(doc, encoding="utf-8").read()
    bad, ambiguous = [], []

    for sm in re.finditer(r"`([^`\n]+)`", text):
        span, ln = sm.group(1), text.count("\n", 0, sm.start()) + 1
        if not CITE_START_RE.match(span):
            continue
        m = CITE_RE.fullmatch(span)
        if not m:
            bad.append(f"{ln}: {span} - malformed cite: write `path:N anchor`, N as N, N-M, N,M or N/M")
            continue
        path, basename, spec, anchor = m.group(1), m.group(2), m.group(3), (m.group(4) or "").strip()
        cite = f"{path}:{spec}"
        if not anchor:
            bad.append(f"{ln}: {cite} - no anchor: write `{cite} <text on that line>`")
            continue
        spans = _spans(spec)
        cands = [p for p in (os.path.join(root, path), os.path.join(os.path.dirname(doc), path)) if os.path.isfile(p)]
        cands = cands[:1] or index.get(basename, [])
        if not cands:
            bad.append(f"{ln}: {cite} - no file of that name in the tree")
            continue
        results = [(c, _verify(c, spans, anchor)) for c in cands]
        ok = [c for c, why in results if not why]
        if len(cands) == 1 and not ok:
            bad.append(f"{ln}: {cite} - {results[0][1]}")
        elif not ok:
            bad.append(f"{ln}: {cite} - anchor '{anchor}' matches none of the {len(cands)} files named {basename}; cite the path")
        elif len(ok) > 1:
            ambiguous.append(f"{ln}: {cite} - anchor matches {len(ok)} files named {basename}; cite the path")

    # bare docs/*.md names (markdown links are already covered elsewhere)
    for m in re.finditer(r"`(docs/[\w./-]+\.md)`", text):
        if not os.path.exists(os.path.join(root, m.group(1))):
            bad.append(f"{text.count(chr(10), 0, m.start()) + 1}: {m.group(1)} - referenced file does not exist")
    return bad, ambiguous

def living_docs(root):
    """Every doc whose citations must stay live. Records are excluded on
    purpose: docs/archive/, docs/plans/ and docs/audit/ cite the line numbers
    of the commit they describe, and rewriting those would destroy the record
    (same reason docs_lint.py excludes roadmap.md)."""
    import glob
    docs = [os.path.join(root, "docs/roadmap.md")]
    docs += sorted(glob.glob(os.path.join(root, "docs/*.md")))
    docs += sorted(glob.glob(os.path.join(root, "docs/internals/*.md")))
    for extra in ("README.md", "CONTRIBUTING.md", "AGENTS.md", "AUDIT.md"):
        p = os.path.join(root, extra)
        if os.path.exists(p):
            docs.append(p)
    seen, out = set(), []
    for d in docs:
        rp = os.path.realpath(d)
        if rp not in seen:
            seen.add(rp)
            out.append(d)
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    docs = [sys.argv[2]] if len(sys.argv) > 2 else living_docs(root)
    total_bad, index = 0, _index(root)
    for doc in docs:
        bad, ambiguous = check(doc, root, index)
        rel = os.path.relpath(doc, root)
        for b in sorted(set(bad)):
            print(f"  DEAD      {rel}:{b}")
        for a in sorted(set(ambiguous)):
            print(f"  AMBIGUOUS {rel}:{a}")
        total_bad += len(set(bad))
    print(f"{'FAIL' if total_bad else 'PASS'}: {total_bad} dead citation(s) across {len(docs)} living doc(s)")
    sys.exit(1 if total_bad else 0)
