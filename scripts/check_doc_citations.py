#!/usr/bin/env python3
"""Verify `path:line` and bare `*.md` citations in the LIVING docs resolve.

Default scope (since 2026-08-26): docs/roadmap.md plus every doc under docs/
and docs/internals/ and the root docs — records (docs/archive/, docs/plans/,
docs/audit/) are excluded because their line numbers describe the commit they
document. The two drift classes that cost are mechanical:

  * a `src/foo.cpp:1234` citation whose file has since shrunk, or whose lines
    moved, so the reader lands on unrelated code;
  * a bare `docs/<name>.md` reference that was renamed, which the markdown-link
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
    total_bad = 0
    for doc in docs:
        bad, ambiguous = check(doc, root)
        rel = os.path.relpath(doc, root)
        for b in sorted(set(bad)):
            print(f"  DEAD      {rel}: {b}")
        for a in sorted(set(ambiguous)):
            print(f"  AMBIGUOUS {rel}: {a}")
        total_bad += len(set(bad))
    print(f"{'FAIL' if total_bad else 'PASS'}: {total_bad} dead citation(s) across {len(docs)} living doc(s)")
    sys.exit(1 if total_bad else 0)
