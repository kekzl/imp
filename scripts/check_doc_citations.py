#!/usr/bin/env python3
"""Verify `path:line` and bare `*.md` citations in a doc still resolve.

docs/roadmap.md is deliberately excluded from docs_lint.py (it is a record, and
stripping it for provenance blocks would destroy that record) — so nothing
checks it at all. The two drift classes that costs are mechanical:

  * a `src/foo.cpp:1234` citation whose file has since shrunk, or whose lines
    moved, so the reader lands on unrelated code;
  * a bare `docs/thing.md` name that was renamed, which the markdown-link
    checker never sees because it is not a link.

Neither needs judgement, so neither should need a human to notice.
"""
import re, sys, os

def check(doc, root):
    text = open(doc, encoding="utf-8").read()
    bad, ambiguous = [], []

    # Build a basename index once: the doc cites most files by bare name
    # (`engine_spec_ngram.cpp:1072`), not by path.
    index = {}
    for base, _dirs, files in os.walk(root):
        if any(x in base for x in (".git", "build", "third_party", "node_modules")):
            continue
        for f in files:
            index.setdefault(f, []).append(os.path.join(base, f))

    # path:line, with or without a leading directory
    for m in re.finditer(r'`((?:[\w./-]+/)?([\w.-]+\.(?:cpp|cu|cuh|h|py|sh))):(\d+)', text):
        path, basename, line = m.group(1), m.group(2), int(m.group(3))
        full = os.path.join(root, path)
        if not os.path.exists(full):
            hits = index.get(basename, [])
            if len(hits) != 1:
                # Not a dead citation — the doc cites a basename that exists in
                # several places (or none). Worth reporting as "cite the path",
                # but it must not be conflated with a line that points past EOF.
                ambiguous.append(f"{path}:{line} — {len(hits)} files share that name; cite the path")
                continue
            full = hits[0]
        n = sum(1 for _ in open(full, encoding="utf-8", errors="replace"))
        if line > n:
            bad.append(f"{path}:{line} — file has only {n} lines")

    # bare docs/*.md names (markdown links are already covered elsewhere)
    for m in re.finditer(r'`(docs/[\w./-]+\.md)`', text):
        if not os.path.exists(os.path.join(root, m.group(1))):
            bad.append(f"{m.group(1)} — referenced file does not exist")
    return bad, ambiguous

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    doc = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "docs/roadmap.md")
    bad, ambiguous = check(doc, root)
    for b in sorted(set(bad)):
        print("  DEAD      " + b)
    for a in sorted(set(ambiguous)):
        print("  AMBIGUOUS " + a)
    print(f"{'FAIL' if bad else 'PASS'}: {len(set(bad))} dead citation(s) in docs/roadmap.md")
    sys.exit(1 if bad else 0)
